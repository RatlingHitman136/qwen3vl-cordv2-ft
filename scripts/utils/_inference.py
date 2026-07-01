# inference.py -- generation, schema-constrained generation, and adapter selection
import os
import gc
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

from ._constants import FIXED_PROMPT
from ._metrics import aggregate_generation_metrics
from ._receipt_schema import Receipt


def load_model_for_inference(
    base_path,
    adapter_path=None,
    merge=False,
    max_pixels=1600 * 28 * 28,
    dtype=torch.bfloat16,
):
    """Load the base model (+ optional LoRA adapter) and its processor for eval."""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_path, dtype=dtype, device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(base_path, max_pixels=max_pixels)

    if adapter_path is not None:
        cfg = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(cfg):
            raise FileNotFoundError(
                f"No adapter_config.json in '{adapter_path}'. "
                "If this is a FULL fine-tune (not LoRA), pass it as base_path instead."
            )
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge:
            model = model.merge_and_unload()

    model.eval()
    return model, processor


def _build_batch(processor, rows, prompt):
    """Chat-template the prompt + collect images/labels for one batch of {"image", "label"} rows."""
    texts, images, labels = [], [], []
    for ex in rows:
        user_turn = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        texts.append(processor.apply_chat_template(
            user_turn, tokenize=False, add_generation_prompt=True))
        images.append(ex["image"])
        labels.append(ex["label"])
    return texts, images, labels


@torch.no_grad()
def run_inference(model, processor, dataset, prompt=FIXED_PROMPT, batch_size=8, max_new_tokens=512):
    """Batched greedy decoding over dataset. Returns (metrics, preds, labels)."""
    model.eval()
    model.config.use_cache = True               # generation needs the KV cache back on
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()  # checkpointing silently disables the cache

    # left padding so generation continues from real tokens, not pad tokens;
    # restore afterward so the (right-padded) training collator stays correct
    orig_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    preds, labels = [], []
    try:
        for start in tqdm(range(0, len(dataset), batch_size), desc="inference", unit="batch"):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            texts, images, batch_labels = _build_batch(processor, rows, prompt)
            labels.extend(batch_labels)

            # one processor call builds the whole padded batch (text + vision together)
            inputs = processor(text=texts, images=images,
                               padding=True, return_tensors="pt").to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                                    # greedy -> reproducible
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,       # stop early when done
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            # left padding keeps the prompt block the same width across the batch,
            # so slicing off exactly the input length keeps only generated tokens
            gen = out[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(gen, skip_special_tokens=True)
            preds.extend(d.strip() for d in decoded)
    finally:
        processor.tokenizer.padding_side = orig_side

    return aggregate_generation_metrics(preds, labels), preds, labels


@torch.no_grad()
def run_constrained_inference(model, processor, dataset, prompt=FIXED_PROMPT, batch_size=8, max_new_tokens=320):
    """Batched greedy decoding constrained to the Receipt JSON schema via xgrammar.
    Forces structurally schema-valid JSON (json_validity -> ~1.0); field_f1 still
    reflects reading quality. Returns (metrics, preds, labels). Requires xgrammar."""
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    tokenizer = processor.tokenizer

    # lm_head width, NOT config.vocab_size: VLM configs nest the text vocab and the
    # lm_head is often padded to a multiple, so the embedding shape is the only
    # reliable match for the grammar's token mask width.
    vocab_size = model.get_output_embeddings().weight.shape[0]

    # compile the grammar ONCE -- compilation is the expensive part
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_json_schema(Receipt)

    preds, labels = [], []
    for start in tqdm(range(0, len(dataset), batch_size), desc="constrained", unit="batch"):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        texts, images, batch_labels = _build_batch(processor, rows, prompt)
        labels.extend(batch_labels)

        # per-call left padding only (does NOT mutate the global padding_side)
        inputs = processor(text=texts, images=images,
                           padding=True, padding_side="left",
                           return_tensors="pt").to(model.device)

        # fresh processor per batch: xgrammar's HF LogitsProcessor is stateful -- it
        # tracks each row's matcher progress, so reusing one across generate() calls
        # corrupts state
        xgr_processor = XGrammarLogitsProcessor(compiled_grammar)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[xgr_processor],
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True)
        preds.extend(d.strip() for d in decoded)

    return aggregate_generation_metrics(preds, labels), preds, labels


def select_best_by_f1(base_path, adapter_paths, select_ds, decode="plain", max_pixels=1600 * 28 * 28):
    """Rank adapters (plus optionally the bare base model, via a `None` entry) on
    select_ds by field F1. Returns results sorted best-first."""
    if decode not in ("plain", "constrained"):
        raise ValueError("decode must be 'plain' or 'constrained'")
    infer = run_inference if decode == "plain" else run_constrained_inference

    results = []
    for ap in adapter_paths:
        label = "BASE(no-adapter)" if ap is None else os.path.basename(os.path.normpath(ap))

        model, processor = load_model_for_inference(
            base_path=base_path, adapter_path=ap, merge=False, max_pixels=max_pixels,
        )
        metrics, _, _ = infer(model, processor, select_ds)
        results.append({
            "adapter":       label,
            "path":          ap,
            "field_f1":      metrics["field_f1"],
            "json_validity": metrics["json_validity"],
            "exact_match":   metrics["exact_match"],
        })
        print(f"{label:<24} f1={metrics['field_f1']:.4f}  "
              f"valid={metrics['json_validity']:.4f}  exact={metrics['exact_match']:.4f}")

        # unload before the next adapter so VRAM doesn't stack
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()

    results.sort(key=lambda r: r["field_f1"], reverse=True)
    best = results[0]
    print(f"\nBEST: {best['adapter']}  (field_f1={best['field_f1']:.4f}, decode={decode})")
    return results
