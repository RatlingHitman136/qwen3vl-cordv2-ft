#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 2B
python eval_script.py adapter-list --id "Qwen/Qwen3-VL-2B-Instruct" --name "qwen3vl-2b" --adapter-paths "../models/checkpoints/qwen3vl-2b-r16-LT-px1600/checkpoint-1200" --split test --decode plain --output-json "../results/2b-finetune-results.json"
python eval_script.py adapter-list --id "Qwen/Qwen3-VL-2B-Instruct" --name "qwen3vl-2b" --adapter-paths "../models/checkpoints/qwen3vl-2b-r16-LT-px1600/checkpoint-1200" --split test --decode constrained --output-json "../results/2b-finetune-cons-results.json"

# 4B
python eval_script.py adapter-list --id "Qwen/Qwen3-VL-4B-Instruct" --name "qwen3vl-4b" --adapter-paths "../models/checkpoints/qwen3vl-4b-r16-LT-px1600/checkpoint-900" --split test --decode plain --output-json "../results/4b-finetune-results.json"
python eval_script.py adapter-list --id "Qwen/Qwen3-VL-4B-Instruct" --name "qwen3vl-4b" --adapter-paths "../models/checkpoints/qwen3vl-4b-r16-LT-px1600/checkpoint-900" --split test --decode constrained --output-json "../results/4b-finetune-cons-results.json"

# 8B
python eval_script.py adapter-list --id "Qwen/Qwen3-VL-8B-Instruct" --name "qwen3vl-8b" --adapter-paths "../models/checkpoints/qwen3vl-8b-r16-LT-px1600/checkpoint-1100" --split test --decode plain --output-json "../results/8b-finetune-results.json"
python eval_script.py adapter-list --id "Qwen/Qwen3-VL-8B-Instruct" --name "qwen3vl-8b" --adapter-paths "../models/checkpoints/qwen3vl-8b-r16-LT-px1600/checkpoint-1100" --split test --decode constrained --output-json "../results/8b-finetune-cons-results.json"
