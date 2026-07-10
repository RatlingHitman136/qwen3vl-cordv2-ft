from ._constants import DATASET_ID, FIXED_PROMPT

# One README.md per published repo. Every number here is copied from the notebook's own runs
# (sections 7.2 / 7.3 / 8.2b) or from ../results/; nothing is recomputed.

CARD_LICENSE = "apache-2.0"
CARD_CODE_URL = "https://github.com/RatlingHitman136/qwen3vl-cordv2-ft"
CARD_CODE_REPO = CARD_CODE_URL.split("github.com/", 1)[1]

# CORD-v2 test split, n=100, greedy decode, max_pixels = 1600*28*28
CARD_METRICS = {
    "master": {"field_f1": 0.8862, "normalized_ted": 0.9105, "json_validity": 0.99, "exact_match": 0.45,
               "source": "`results/finetune-result.json` (`eval_script.py`, plain decode)"},
    "gptq":   {"field_f1": 0.8840, "normalized_ted": 0.9143, "json_validity": 0.98, "exact_match": 0.41,
               "source": "section 7.2 of `finetune_qwen3vl.ipynb`"},
    "gguf":   {"field_f1": 0.8737, "normalized_ted": 0.9051, "json_validity": 0.97, "exact_match": 0.40,
               "source": "section 8.2b of `finetune_qwen3vl.ipynb`"},
    "2b":     {"field_f1": 0.8857, "normalized_ted": 0.8990, "json_validity": 0.99, "exact_match": 0.43,
               "source": "section 7.3 of `finetune_qwen3vl.ipynb`"},
}

_TRANSFORMERS_USAGE = """```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

REPO = "<your-hf-username>/@@REPO_NAME@@"
model = Qwen3VLForConditionalGeneration.from_pretrained(REPO, dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(REPO, max_pixels=1600 * 28 * 28)

messages = [{"role": "user", "content": [
    {"type": "image", "image": "receipt.jpg"},
    {"type": "text", "text": "@@PROMPT@@"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```"""

_GPTQ_USAGE = """Needs [GPTQModel](https://github.com/modelcloud/gptqmodel) (plain transformers would need `optimum`).

```python
import torch
from gptqmodel import GPTQModel
from transformers import AutoProcessor

REPO = "<your-hf-username>/@@REPO_NAME@@"
model = GPTQModel.load(REPO, device_map="auto").model   # .model is the underlying HF model
processor = AutoProcessor.from_pretrained(REPO, max_pixels=1600 * 28 * 28)

messages = [{"role": "user", "content": [
    {"type": "image", "image": "receipt.jpg"},
    {"type": "text", "text": "@@PROMPT@@"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

It also serves under vLLM, which reads the same GPTQ weight format."""

_GGUF_USAGE = """Two files: the quantized language model and its `mmproj` vision tower. You need both.

```bash
huggingface-cli download <your-hf-username>/@@REPO_NAME@@ --local-dir gguf

llama-server -m gguf/*-Q4_K_M.gguf --mmproj gguf/mmproj-*-f16.gguf -ngl 99 -c 8192
```

Then query the OpenAI-compatible endpoint with an image and the prompt:

```
@@PROMPT@@
```

Your llama.cpp build must support the Qwen3-VL `qwen3vl_merger` projector. LM Studio and
Ollama load the same pair."""

_GPTQ_QUANT = """Quantized with GPTQModel (4-bit, `group_size=128`, symmetric, `desc_act=false`), which works
out to roughly 4.29 bits per weight. Only the language-model decoder linears are quantized;
the vision tower, the embeddings and `lm_head` stay in bf16.

Calibration used 256 multimodal chat samples from the CORD-v2 train split, each one a receipt
image plus the fixed prompt plus the gold JSON answer. That makes this a data-aware
quantization, and it shows: against the bf16 model it loses 0.0022 field_f1 and actually gains
0.0038 nTED, which is inside the noise of a 100-receipt split. A data-free bitsandbytes NF4
quantization of the same model came out measurably worse on both metrics.

Inference runs through the Marlin 4-bit kernel."""

_GGUF_QUANT = """The language model is quantized to **Q4_K_M** by llama.cpp's `llama-quantize`. This is
data-free: no calibration set, no receipts involved. The vision tower is exported separately
as an f16 `mmproj` file and is left unquantized.

It converts from the bf16 master rather than the GPTQ build, because llama.cpp cannot
ingest GPTQ-packed weights. Being data-free, it gives up more than the calibrated GPTQ build
does: field_f1 0.8737 here against 0.8840 for GPTQ, on the same test split."""

CARD_SPECS = {
    "master": {
        "repo_name": "qwen3vl-4b-receipt-extraction",
        "title": "Qwen3-VL-4B Receipt Extraction (bf16)",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "library_name": "transformers",
        "extra_tags": ["lora", "bf16"],
        "summary": "Reads a receipt image and returns it as structured JSON. Full-precision master: "
                   "the merged bf16 model every other artifact in this project is derived from.",
        "glance": [("Format", "bf16 safetensors"), ("Size on disk", "8.3 GB"), ("Runtime", "transformers")],
        "usage": _TRANSFORMERS_USAGE,
        "quant": None,
    },
    "gptq": {
        "repo_name": "qwen3vl-4b-receipt-extraction-gptq",
        "title": "Qwen3-VL-4B Receipt Extraction (GPTQ 4-bit)",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "library_name": "transformers",
        "extra_tags": ["lora", "gptq", "4-bit", "quantized"],
        "summary": "Reads a receipt image and returns it as structured JSON. Calibrated 4-bit GPTQ build, "
                   "the one this project recommends for GPU serving.",
        "glance": [("Format", "GPTQ 4-bit, `group_size=128`, ~4.29 bpw"), ("Size on disk", "3.3 GB"),
                   ("VRAM (weights)", "3.51 GB"), ("Runtime", "gptqmodel, vLLM")],
        "usage": _GPTQ_USAGE,
        "quant": _GPTQ_QUANT,
    },
    "gguf": {
        "repo_name": "qwen3vl-4b-receipt-extraction-gguf",
        "title": "Qwen3-VL-4B Receipt Extraction (GGUF Q4_K_M)",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "library_name": "gguf",
        "extra_tags": ["lora", "gguf", "llama.cpp", "ollama", "quantized"],
        "summary": "Reads a receipt image and returns it as structured JSON. GGUF build for llama.cpp, "
                   "Ollama and LM Studio.",
        "glance": [("Format", "Q4_K_M language model + f16 `mmproj` vision tower"),
                   ("Files", "`*-Q4_K_M.gguf` (2.50 GB), `mmproj-*-f16.gguf` (0.84 GB)"),
                   ("Size on disk", "3.3 GB"), ("Runtime", "llama.cpp, Ollama, LM Studio")],
        "usage": _GGUF_USAGE,
        "quant": _GGUF_QUANT,
    },
    "2b": {
        "repo_name": "qwen3vl-2b-receipt-extraction",
        "title": "Qwen3-VL-2B Receipt Extraction (bf16)",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "library_name": "transformers",
        "extra_tags": ["lora", "bf16"],
        "summary": "Reads a receipt image and returns it as structured JSON. Smaller 2B fine-tune: it "
                   "matches the 4B on field_f1 and runs faster, but structures the JSON less reliably.",
        "glance": [("Format", "bf16 safetensors"), ("Size on disk", "4.0 GB"),
                   ("VRAM (weights)", "4.33 GB"), ("Runtime", "transformers")],
        "usage": _TRANSFORMERS_USAGE,
        "quant": None,
    },
}

_OUTPUT_EXAMPLE = """{"menu": [{"nm": "Es Teh Manis", "cnt": "2", "price": "8,000"}],
 "sub_total": {"subtotal_price": "8,000"},
 "total": {"total_price": "8,000", "cashprice": "10,000", "changeprice": "2,000"}}"""

_TRAINING = """Fine-tuned from `@@BASE_MODEL@@` with LoRA on [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
(800 train / 100 validation / 100 test receipts). Labels were losslessly normalized before
training.

| | |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Adapted modules | language tower only (`q,k,v,o,gate,up,down_proj`); vision tower frozen |
| Precision | bf16 |
| Learning rate | 1e-4 |
| Batch size | 2 per device x 4 gradient accumulation |
| Epochs | 20 |
| Image budget | 1600 patches of 28x28 |

Training ran well past convergence on purpose, so there was a wide range of adapters to pick
from. Validation loss and the task metrics peak at different times here: loss starts rising
around epoch 3, while the best field_f1 and nTED land at epoch 9 or later. The adapter
published here is the one that scored best on the **validation** split by field_f1, and it was
then scored once on the held-out test split. Early-stopping on loss would have picked a
noticeably worse one.

The whole pipeline, from data prep and training through evaluation, quantization and export,
is at [@@CODE_REPO@@](@@CODE_URL@@)."""

_LIMITATIONS = """- The test split is 100 receipts. Gaps of a point or two in field_f1 are inside the noise band
  and should be read as ties.
- CORD-v2 is photographed Indonesian retail and restaurant receipts. Expect worse results on
  other layouts, languages, or document types (invoices, handwriting).
- Field names mirror the raw CORD keys (`nm`, `cnt`, `unitprice`), not friendly names.
- Values are transcribed text, not validated arithmetic. Nothing checks that the line items
  sum to the total. Do not use the output for financial decisions without review.
- Schema-constrained decoding was tried and scored far worse than plain decoding (field_f1
  falls roughly 35 points), so plain greedy decoding is what these numbers use and what is
  recommended.
- Any timing or throughput figure depends heavily on hardware. Measure on the machine you
  intend to deploy on."""

_BODY = """# @@TITLE@@

@@SUMMARY@@

## At a glance

| | |
|---|---|
| Base model | [`@@BASE_MODEL@@`](https://huggingface.co/@@BASE_MODEL@@) |
| Fine-tuning | LoRA r=16, language tower only |
| Dataset | [`@@DATASET@@`](https://huggingface.co/datasets/@@DATASET@@) |
| Code | [@@CODE_REPO@@](@@CODE_URL@@) |
@@GLANCE_ROWS@@

## Intended use

Turning a photo or scan of a retail receipt into structured JSON, for bookkeeping and expense
pipelines, or as a starting point for research on document understanding.

### Out of scope

Invoices, handwritten notes, and non-receipt images. The model was never trained on them. It
also should not be trusted to make financial decisions on its own; treat the output as an
extraction to be reviewed.

## How to use

The model expects one image and this exact prompt, which is the prompt it was trained with:

```
@@PROMPT@@
```

@@USAGE@@

## Output

A single JSON object with three top-level keys: `menu` (the line items), `sub_total` and
`total`. Every value is a string, copied from the receipt.

```json
@@OUTPUT_EXAMPLE@@
```

## Evaluation

CORD-v2 `test` split, 100 receipts, greedy decoding, images capped at 1600 patches.

| field_f1 | nTED | json_validity | exact_match |
|---:|---:|---:|---:|
| @@F1@@ | @@NTED@@ | @@JV@@ | @@EM@@ |

Measured in @@SOURCE@@.

- **field_f1** flattens the predicted and gold JSON into (field-path, value) leaf pairs and
  scores precision/recall over them. It asks whether the right values landed at the right
  paths.
- **nTED** (normalized tree-edit distance, following the CORD paper) compares the two JSON
  trees directly. It catches structural mistakes, like wrong nesting or a missing key, that
  leaf-flattening misses.

Neither subsumes the other, which is why both are here.

## Training

@@TRAINING@@
@@QUANT_SECTION@@
## Limitations

@@LIMITATIONS@@

## License

`@@LICENSE@@`, inherited from the base model. The CORD-v2 dataset carries its own terms; see
its dataset card.
"""


def _frontmatter(key):
    """YAML frontmatter, including a model-index block so the Hub renders a metrics box."""
    spec, m = CARD_SPECS[key], CARD_METRICS[key]
    tags = ["receipt", "ocr", "document-understanding", "structured-extraction",
            "qwen3-vl", "image-text-to-text"] + spec["extra_tags"]
    lines = [
        "---",
        f"license: {CARD_LICENSE}",
        f"base_model: {spec['base_model']}",
        f"library_name: {spec['library_name']}",
        "pipeline_tag: image-text-to-text",
        "datasets:",
        f"- {DATASET_ID}",
        "language:",
        "- en",
        "tags:",
        *[f"- {t}" for t in tags],
        "model-index:",
        f"- name: {spec['repo_name']}",
        "  results:",
        "  - task:",
        "      type: image-text-to-text",
        "      name: Receipt JSON extraction",
        "    dataset:",
        "      name: CORD-v2",
        f"      type: {DATASET_ID}",
        "      split: test",
        "    metrics:",
    ]
    for mtype, mname in [("field_f1", "Field F1"), ("normalized_ted", "Normalized TED"),
                         ("json_validity", "JSON validity"), ("exact_match", "Exact match")]:
        lines += [f"    - type: {mtype}", f"      value: {m[mtype]:.4f}", f"      name: {mname}"]
    lines.append("---")
    return "\n".join(lines)


def build_card(key):
    """Render the full README.md text for one published artifact."""
    spec, m = CARD_SPECS[key], CARD_METRICS[key]

    glance = "\n".join(f"| {label} | {value} |" for label, value in spec["glance"])
    quant = f"\n## Quantization\n\n{spec['quant']}\n" if spec["quant"] else ""

    body = _BODY
    # Composite blocks first: they carry tokens of their own (@@BASE_MODEL@@ inside _TRAINING,
    # @@REPO_NAME@@ / @@PROMPT@@ inside the usage snippets), which the leaf pass below fills in.
    for token, value in [
        ("@@TITLE@@", spec["title"]),
        ("@@SUMMARY@@", spec["summary"]),
        ("@@GLANCE_ROWS@@", glance),
        ("@@USAGE@@", spec["usage"]),
        ("@@TRAINING@@", _TRAINING),
        ("@@QUANT_SECTION@@", quant),
        ("@@LIMITATIONS@@", _LIMITATIONS),
        ("@@OUTPUT_EXAMPLE@@", _OUTPUT_EXAMPLE),
        # leaves
        ("@@BASE_MODEL@@", spec["base_model"]),
        ("@@DATASET@@", DATASET_ID),
        ("@@REPO_NAME@@", spec["repo_name"]),
        ("@@PROMPT@@", FIXED_PROMPT.strip()),
        ("@@F1@@", f"{m['field_f1']:.4f}"),
        ("@@NTED@@", f"{m['normalized_ted']:.4f}"),
        ("@@JV@@", f"{m['json_validity']:.4f}"),
        ("@@EM@@", f"{m['exact_match']:.4f}"),
        ("@@SOURCE@@", m["source"]),
        ("@@LICENSE@@", CARD_LICENSE),
        ("@@CODE_REPO@@", CARD_CODE_REPO),
        ("@@CODE_URL@@", CARD_CODE_URL),
    ]:
        body = body.replace(token, value)

    assert "@@" not in body, f"unreplaced token in {key} card"

    return f"{_frontmatter(key)}\n\n{body}"
