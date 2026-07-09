#!/usr/bin/env python3
"""
Evaluation script for picking the best checkpoint/adapter by field F1 and nTED.

Two modes:
  full-model    Rebuilds the same `full_model_name` train_script.py used
                (from --name/--lora-rank/--lt-tune/--vt-tune/--max-px) and
                evaluates the bare base
                model, every saved checkpoint except the last (it's redundant
                with the final adapter), and the final adapter.
  adapter-list  Evaluates an explicit list of adapter/checkpoint paths against each other and
                the base model.
"""

import argparse
import glob
import json
import logging
import os
import re

from datasets import load_dataset
from huggingface_hub import snapshot_download
from utils import (
    preprocess_cord,
    DATASET_ID,
    BASE_MODEL_SAVE_DIR,
    CHECKPOINT_MODEL_SAVE_DIR,
    ADAPTER_OUTPUT_DIR,
    select_best,
)


def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def resolve_base_path(args):
    """Local snapshot dir for the base model, matching train_script's
    `cache_dir=BASE_MODEL_SAVE_DIR + model_name` convention. Cached, so this
    is a no-op download once the base model is already on disk."""
    return snapshot_download(repo_id=args.id, cache_dir=BASE_MODEL_SAVE_DIR + args.name)


def full_model_name_from_args(args):
    """Same construction as train_script.py's `full_model_name`."""
    ft_targets = ""
    if args.lt_tune:
        ft_targets += "LT"
    if args.vt_tune:
        ft_targets += "VT"
    return f"{args.name}-r{args.lora_rank}-{ft_targets}-px{args.max_px}"


def checkpoint_sort_key(path):
    match = re.search(r"checkpoint-(\d+)$", os.path.normpath(path))
    return int(match.group(1)) if match else -1


def load_eval_dataset(split, logger):
    logger.info(f"Using dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)
    _, validation_dataset, test_dataset = preprocess_cord(dataset, verbose=False)
    eval_dataset = validation_dataset if split == "validation" else test_dataset
    logger.info(f"Loaded '{split}' split for evaluation. COUNT: {len(eval_dataset)}")
    return eval_dataset


def eval_full_model(args, logger):
    full_model_name = full_model_name_from_args(args)
    logger.info(f"Reconstructed full model name: {full_model_name}")

    base_path = resolve_base_path(args)
    logger.info(f"Base model resolved to: {base_path}")

    checkpoint_dir = os.path.join(CHECKPOINT_MODEL_SAVE_DIR, full_model_name)
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=checkpoint_sort_key,
    )
    if checkpoints:
        logger.info(f"Skipping last checkpoint (redundant with final adapter): {checkpoints[-1]}")
        checkpoints = checkpoints[:-1]
    else:
        logger.warning(f"No checkpoints found under {checkpoint_dir}")

    adapter_path = os.path.join(ADAPTER_OUTPUT_DIR, full_model_name)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Final adapter not found at {adapter_path}")

    adapter_paths = [None] + checkpoints + [adapter_path]
    logger.info(f"Evaluating base model + {len(checkpoints)} checkpoint(s) + final adapter")

    eval_dataset = load_eval_dataset(args.split, logger)

    return select_best(
        base_path=base_path,
        adapter_paths=adapter_paths,
        select_ds=eval_dataset,
        decode=args.decode,
        max_pixels=args.max_px * 28 * 28,
    )


def eval_adapter_list(args, logger):
    base_path = resolve_base_path(args)
    logger.info(f"Base model resolved to: {base_path}")

    for path in args.adapter_paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Adapter/checkpoint path not found: {path}")

    adapter_paths = list(args.adapter_paths)
    if args.include_base:
        adapter_paths = [None] + adapter_paths
    logger.info(f"Evaluating {len(adapter_paths)} candidate(s): {adapter_paths}")

    eval_dataset = load_eval_dataset(args.split, logger)

    return select_best(
        base_path=base_path,
        adapter_paths=adapter_paths,
        select_ds=eval_dataset,
        decode=args.decode,
        max_pixels=args.max_px * 28 * 28,
    )


def main(args):
    """Main entry point for the eval script."""
    logger = logging.getLogger(__name__)
    logger.info("Starting eval script...")

    if args.mode == "full-model":
        results = eval_full_model(args, logger)
    else:
        results = eval_adapter_list(args, logger)

    # select_best() already prints the BEST F1 / BEST nTED summary; don't repeat it here.
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved full ranking to {args.output_json}")


def add_shared_arguments(parser):
    parser.add_argument(
        "--id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model ID (Hugging Face Hub repo)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="qwen3vl-2b",
        help="Model name (used to resolve the base model's cache dir)"
    )
    parser.add_argument(
        "--max-px",
        type=int,
        default=1600,
        help="Maximum 28x28 pixels block count for the model's visual input"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--decode",
        type=str,
        choices=["plain", "constrained"],
        default="plain",
        help="Decoding strategy: greedy ('plain') or grammar-constrained to the Receipt schema"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump the full ranking (all candidates + metrics) as JSON"
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints/adapters on recipe PDFs and pick the best by field F1"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    full_model = subparsers.add_parser(
        "full-model",
        help="Reconstruct the full model name from training parameters, then eval base model + checkpoints + final adapter"
    )
    add_shared_arguments(full_model)
    full_model.add_argument(
        "--lt-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the language transformer component was fine-tuned"
    )
    full_model.add_argument(
        "--vt-tune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the visual transformer component was fine-tuned"
    )
    full_model.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Lora rank"
    )

    adapter_list = subparsers.add_parser(
        "adapter-list",
        help="Evaluate an explicit list of adapter/checkpoint paths"
    )
    add_shared_arguments(adapter_list)
    adapter_list.add_argument(
        "--adapter-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to adapters/checkpoints to evaluate"
    )
    adapter_list.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate the bare base model (no adapter) alongside the given paths"
    )

    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    main(args)
