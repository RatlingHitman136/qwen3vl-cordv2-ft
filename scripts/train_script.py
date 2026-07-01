#!/usr/bin/env python3
"""
Training script for fine-tuning model on recipe PDFs.
"""

import json
from pyexpat import model

from datasets import load_dataset
from peft import LoraConfig
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils import preprocess_cord, FIXED_PROMPT, DATASET_ID, BASE_MODEL_SAVE_DIR, CHECKPOINT_MODEL_SAVE_DIR, ADAPTER_OUTPUT_DIR
from huggingface_hub import snapshot_download
from trl import SFTConfig, SFTTrainer

import argparse
import logging


def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def training_pipeline(args, logger):
    logger.info("Starting training script...")
    logger.info(f"Using dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)

    train_dataset, validation_dataset, test_dataset = preprocess_cord(dataset, verbose=False)

    logger.info(f"Loaded dataset. TRAIN_COUNT: {len(train_dataset)}, VALIDATION_COUNT: {len(validation_dataset)}, TEST_COUNT: {len(test_dataset)}")

    model_id =  snapshot_download(
        repo_id=args.model_id,
        cache_dir=BASE_MODEL_SAVE_DIR + args.model_name
    )
    logger.info(f"Downloaded base model from Hugging Face Hub: {model_id}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=args.model_max_px*28*28)
    processor.tokenizer.padding_side = "right"

    lt_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    vt_targets = ["qkv", "proj", "linear_fc1", "linear_fc2"]

    training_targets = []
    if args.model_lt_tune:
        training_targets.extend(lt_targets)
    if args.model_vt_tune:
        training_targets.extend(vt_targets)


    peft_config = LoraConfig(
        r=args.model_lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=training_targets
    )

    def collate_fn(examples):
        # each example: {"image": <PIL.Image>, "label": "<output text>"}
        full_texts, prompt_texts, images = [], [], []
        for ex in examples:
            user_turn = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": FIXED_PROMPT},
            ]}]
            full_msg = user_turn + [{"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(ex["label"], ensure_ascii=False)},
            ]}]
            full_texts.append(
                processor.apply_chat_template(full_msg, tokenize=False, add_generation_prompt=False))
            prompt_texts.append(
                processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True))
            images.append(ex["image"])

        batch = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()

        # 1) mask the fixed prompt + image tokens per sample (right padding => prompt at the front).
        #    Re-run processor on prompt-only WITH the image so image-token expansion is counted.
        for i, (p_text, img) in enumerate(zip(prompt_texts, images)):
            plen = processor(text=[p_text], images=[img], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :plen] = -100

        # 2) mask padding
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # 3) defensively mask any image placeholder tokens that survive in the completion region

        image_token_id = getattr(model.config, "image_token_id", None) 
        if image_token_id is not None:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    effective_batch_size = args.model_batch_size * args.model_grad_accum_steps
    batch_count = (len(train_dataset) + effective_batch_size - 1) // effective_batch_size

    ft_targets = ""
    if args.model_lt_tune:
        ft_targets += "LT"
    if args.model_vt_tune:
        ft_targets += "VT"

    full_model_name = f"{args.model_name}-r{args.model_lora_rank}-{ft_targets}-px{args.model_max_px}"

    training_args = SFTConfig(
        output_dir=CHECKPOINT_MODEL_SAVE_DIR + full_model_name,
        per_device_train_batch_size=args.model_batch_size,
        gradient_accumulation_steps=args.model_grad_accum_steps,
        learning_rate=args.model_learning_rate,
        num_train_epochs=args.model_epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=batch_count,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="steps",
        eval_steps=batch_count,
        per_device_eval_batch_size=args.model_batch_size,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,    
        processing_class=processor, 
    )

    
    logger.info(f"Fine-tuning model: {args.model_name} with epochs: {args.model_epochs}, learning rate: {args.model_learning_rate}, lora rank: {args.model_lora_rank}, LT tune: {args.model_lt_tune}, VT tune: {args.model_vt_tune}, max pixels: {args.model_max_px}")
    logger.info(f"Saving checkpoints to {CHECKPOINT_MODEL_SAVE_DIR + full_model_name}")

    trainer.train()

    logger.info(f"Training completed. Saving final model (adapter) to {ADAPTER_OUTPUT_DIR + full_model_name}")

    trainer.save_model(ADAPTER_OUTPUT_DIR + full_model_name) 


def main(args):
    """Main entry point for the training script."""
    logger = logging.getLogger(__name__)    
    training_pipeline(args, logger)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune model on recipe PDFs"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model ID"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3vl-2b",
        help="Model Name"
    )
    parser.add_argument(
        "--model-max-px",
        type=int,
        default=1600,
        help="Maximum 28x28 pixels block count for the model's visual input"
    )
    parser.add_argument(
        "--model-lt-tune",
        type=bool,
        default=True,
        help="Whether to fine-tune the language transformer component"
    )
    parser.add_argument(
        "--model-vt-tune",
        type=bool,
        default=False,
        help="Whether to fine-tune the visual transformer component"
    )
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=2,
        help="Device batch size for the model (Both for training and validation)"
    )
    parser.add_argument(
        "--model-grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps for the model"
    )
    parser.add_argument(
        "--model-epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--model-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--model-lora-rank",
        type=int,
        default=128,
        help="Lora rank"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    main(args)
