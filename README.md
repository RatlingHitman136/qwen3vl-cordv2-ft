# Fine-tuning Qwen3-VL-2B on Receipt PDFs

Fine-tune the [Qwen3-VL-2B](https://huggingface.co/Qwen) vision-language model to extract
structured information from receipt PDFs/images
## Project layout

```
.
├── scripts/
│   ├── train_script.py                # CLI: LoRA fine-tune
│   ├── eval_script.py                 # CLI: rank checkpoints/adapters by field F1 + nTED
│   ├── finetune_qwen3vl.ipynb         # main training notebook (local hardware)
│   ├── openvino_export.ipynb          # standalone OpenVINO/Intel export
│   ├── grammar_creator.ipynb          # derives the Receipt pydantic schema from CORD-v2
│   ├── spot_check.ipynb               # interactive single-receipt inspection tool
│   ├── test_diff_size_models.sh       # eval sweep across model sizes
│   ├── logs/                          # runtime-generated artifacts (gitignored)
│   └── utils/                         # shared package (dataset prep, schema, inference, metrics)
├── models/                            # trained/quantized/exported models (gitignored, local only)
├── results/                           # eval outputs (tracked)
├── hf_credentials.example.json        # copy to hf_credentials.json with your HF token
├── requirements.txt                   # Python dependencies
└── README.md
```

## Dataset

Loaded automatically from the Hugging Face Hub (`naver-clova-ix/cord-v2`) via
`datasets.load_dataset()` — no local dataset directory needed.

