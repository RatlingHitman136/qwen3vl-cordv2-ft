# Fine-tuning Qwen3-VL-2B on Receipt PDFs

Fine-tune the [Qwen3-VL-2B](https://huggingface.co/Qwen) vision-language model to extract
structured information from receipt PDFs/images
## Project layout

```
.
├── notebooks/
│   └── finetune_qwen3vl_colab.ipynb   # main training notebook (run on Colab)
├── data/                              # dataset (images + annotations) — not committed
├── requirements.txt                   # Python dependencies
└── README.md
```

## Dataset

Place receipt samples under `data/` as image/label pairs. Expected format is documented
in the notebook (image + target JSON of extracted fields).

