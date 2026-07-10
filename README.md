# Qwen3-VL fine-tuned for receipt → JSON

Turns a photo of a receipt into structured JSON. It is a LoRA fine-tune of
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) on
[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2).

The model this project ships is a 4B fine-tune quantized to 4 bits with GPTQ. It scores
**0.884 field_f1** on the held-out test split and its weights take **3.5 GB**, which is less
than the 2B model at bf16. Grab it here:
[`ratlinghitman/qwen3vl-4b-receipt-extraction-gptq`](https://huggingface.co/ratlinghitman/qwen3vl-4b-receipt-extraction-gptq).

The repo covers the whole path: fine-tuning, sweeping model sizes and LoRA ranks, comparing
plain against schema-constrained decoding, quantizing three different ways, and exporting to
safetensors and GGUF.

## What it does

One image plus one fixed prompt goes in. One JSON object comes out, with three top-level keys:
`menu` (the line items), `sub_total` and `total`. Every value is a string, copied off the
receipt.

```json
{"menu": [{"nm": "Es Teh Manis", "cnt": "2", "price": "8,000"}],
 "sub_total": {"subtotal_price": "8,000"},
 "total": {"total_price": "8,000", "cashprice": "10,000", "changeprice": "2,000"}}
```

## Quickstart

### Use the published model

Needs [GPTQModel](https://github.com/modelcloud/gptqmodel).

```python
import torch
from gptqmodel import GPTQModel
from transformers import AutoProcessor

REPO = "ratlinghitman/qwen3vl-4b-receipt-extraction-gptq"
model = GPTQModel.load(REPO, device_map="auto").model   # .model is the underlying HF model
processor = AutoProcessor.from_pretrained(REPO, max_pixels=1600 * 28 * 28)

messages = [{"role": "user", "content": [
    {"type": "image", "image": "receipt.jpg"},
    {"type": "text", "text": "Extract the receipt from the image into a structured JSON. Your output should contain ONLY correct JSON!"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Run it yourself

Python 3.12 and a CUDA GPU (flash-attn and the Marlin kernel both want one).

```bash
# --no-build-isolation is required: flash-attn needs torch present at build time
pip install -r requirements.txt --no-build-isolation
```

If your CPU has no AVX-512 (any AMD Zen 2, for instance), rebuild `pypcre` from source. The
published wheel is AVX-512 only and will crash on load:

```bash
pip cache remove "pypcre*"
PYPCRE_BUILD_FROM_SOURCE=1 pip install --no-cache-dir --force-reinstall \
    --no-deps --no-binary=:all: "pypcre==0.3.2"
```

Then add your Hugging Face token and open the notebook:

```bash
cp hf_credentials.example.json hf_credentials.json   # paste your token into it
jupyter lab scripts/finetune_qwen3vl.ipynb
```

Section 0 of the notebook does the same install, reads the token into `HF_TOKEN`, and loads the
GPTQModel Marlin extension used for 4-bit inference later on. The dataset downloads itself from
the Hub, so there is no local dataset folder to set up.

## Results

Everything below is the CORD-v2 **test** split: 100 receipts, greedy decoding.

Two metrics, because neither one covers the other:

- **field_f1** flattens the predicted and gold JSON into (field-path, value) leaf pairs and
  scores precision and recall over them. It asks whether the right values landed at the right
  paths.
- **nTED** (normalized tree-edit distance, from the CORD paper) compares the two JSON trees
  directly. It catches structural mistakes, like wrong nesting or a missing key, that
  leaf-flattening misses.

> The test split is only 100 receipts, so a gap of a point or two is noise, not a ranking. The
> 2B model was scored three separate times and gave 0.8788, 0.8856 and 0.8857 field_f1. That
> spread is the noise floor. Each table below sticks to one measurement run.

### Model size

All at LoRA rank 16, plain decoding.

| model | field_f1 | nTED |
|---|---:|---:|
| 2B | 0.8788 | 0.8976 |
| 4B | 0.8862 | 0.9105 |
| 8B | 0.8956 | 0.9118 |

Going 4B to 8B doubles the weights and buys 0.001 nTED. That is why the 4B is the one that gets
quantized and shipped.

### LoRA rank

2B only, plain decoding.

| rank | field_f1 | nTED |
|---|---:|---:|
| 16 | 0.8856 | 0.8976 |
| 32 | 0.8769 | 0.8891 |
| 64 | 0.8737 | 0.8934 |
| 128 | 0.8632 | 0.8962 |
| 256 | 0.8746 | 0.9021 |

The whole column spans about two points, which is inside the noise. Rank 16 is the cheapest to
train and to serve, so it wins by default.

### Plain vs schema-constrained decoding

Same models, rank 16. Constrained decoding masks any token that would break the receipt schema.

| model | field_f1 plain | field_f1 constrained |
|---|---:|---:|
| 2B | 0.8788 | 0.5231 |
| 4B | 0.8862 | 0.5219 |
| 8B | 0.8956 | 0.5453 |

Not a typo. Forcing the schema costs about 35 points.

### Quantizing the 4B

| build | field_f1 | nTED |
|---|---:|---:|
| bf16 | 0.8862 | 0.9105 |
| GPTQ 4-bit (calibrated) | 0.8840 | 0.9143 |
| bitsandbytes NF4 (data-free) | 0.8814 | 0.9006 |
| GGUF Q4_K_M (data-free) | 0.8737 | 0.9051 |

GPTQ sees 256 real receipts while it quantizes. The other two never look at the data.

### The 4-bit 4B against the plain 2B

| model | field_f1 | nTED | VRAM (weights) |
|---|---:|---:|---:|
| 4B GPTQ 4-bit | 0.8840 | 0.9143 | 3.51 GB |
| 2B bf16 | 0.8857 | 0.8990 | 4.33 GB |

The quantized 4B is the smaller model of the two, and it structures the JSON better. It is also
slower: roughly 7 minutes through the test split against 5 for the 2B.

## Fine-tuning outcomes

**Capacity saturates at 4B.** Most of the achievable quality is reached by 4B. Going to 8B
doubles the parameter count and returns 0.001 nTED, well inside the noise band of a 100-example
split. The 4B is therefore the best quality-per-parameter point, and it is what gets quantized
and shipped.

**LoRA rank is not a useful lever here.** Ranks 16 through 256 span roughly two points of
field_f1 with no monotonic trend, which puts the entire sweep inside noise. Rank 16 is the
cheapest to train and to serve, so it is the sensible default.

**Schema-constrained decoding degrades accuracy rather than improving it.** field_f1 falls by
about 35 points at every model size. The fine-tune already emits schema-valid JSON on its own,
so the grammar rarely rescues a malformed output. What it does instead is override the model's
preferred token at purely structural branch points, and because decoding is greedy, one forced
token conditions everything generated after it. Guaranteeing syntax does nothing for content
accuracy and measurably costs it.

**Validation loss is the wrong checkpoint-selection signal.** Loss begins rising around epoch 3,
the usual overfitting cue, while the best field_f1 and nTED land at epoch 9 or later. The two
signals disagree by a wide margin. Early-stopping on loss would have selected a clearly worse
model, so checkpoints here are chosen on validation field_f1 and only then scored once on the
held-out test split.

**Calibration is what makes 4-bit quantization safe.** GPTQ, calibrated on 256 real receipts,
gives up 0.0022 field_f1 against bf16 and gains slightly on nTED, which is a wash. The two
data-free methods, bitsandbytes NF4 and GGUF Q4_K_M, both lose more on both metrics. The cost of
quantization tracks whether the method sees the data distribution, not the bit width.

**Quantizing a larger model beats selecting a smaller one.** At 4 bits the 4B holds less weight
memory than the 2B at bf16 (3.51 GB vs 4.33 GB), matches it on field_f1, and leads it on nTED.
The trade is latency: roughly 7 minutes over the test split against 5 for the 2B. Where memory
and output structure matter more than throughput, the quantized 4B is the better deployment
target.

## Published models

All public, under [`ratlinghitman`](https://huggingface.co/ratlinghitman).

| repo | model | format |
|---|---|---|
| [`qwen3vl-4b-receipt-extraction`](https://huggingface.co/ratlinghitman/qwen3vl-4b-receipt-extraction) | 4B | bf16 safetensors, the master copy |
| [`qwen3vl-4b-receipt-extraction-gptq`](https://huggingface.co/ratlinghitman/qwen3vl-4b-receipt-extraction-gptq) | 4B | GPTQ 4-bit, calibrated |
| [`qwen3vl-4b-receipt-extraction-gguf`](https://huggingface.co/ratlinghitman/qwen3vl-4b-receipt-extraction-gguf) | 4B | Q4_K_M + f16 `mmproj`, for llama.cpp and Ollama |
| [`qwen3vl-2b-receipt-extraction`](https://huggingface.co/ratlinghitman/qwen3vl-2b-receipt-extraction) | 2B | bf16 safetensors |

The model cards are generated by `scripts/utils/_model_card.py` and uploaded with the weights.

## Repo layout

```
.
├── scripts/
│   ├── finetune_qwen3vl.ipynb         # main notebook: train -> eval -> merge -> quantize -> export
│   ├── openvino_export.ipynb          # standalone OpenVINO/Intel export (not published)
│   ├── grammar_creator.ipynb          # where the receipt schema was worked out from CORD-v2
│   ├── spot_check.ipynb               # run one receipt through a model and eyeball it
│   ├── train_script.py                # CLI: LoRA fine-tune
│   ├── eval_script.py                 # CLI: rank checkpoints/adapters by field_f1 + nTED
│   ├── test_diff_size_models.sh       # eval sweep across model sizes
│   ├── logs/                          # runtime-generated artifacts (gitignored)
│   └── utils/                         # dataset prep, schema, inference, metrics, model cards
├── models/                            # trained/quantized/exported models (gitignored, local only)
├── results/                           # eval outputs (tracked)
├── hf_credentials.example.json        # copy to hf_credentials.json with your HF token
├── requirements.txt                   # Python dependencies
└── README.md
```

## Reproduce

`scripts/finetune_qwen3vl.ipynb` runs top to bottom:

0. Install, authenticate, compile the Marlin kernel
1. Load CORD-v2 and normalize the labels
2. Fine-tune with LoRA
3. Rank the checkpoints
4. Read the evaluation results
5. Merge the chosen adapter into the base model
6. Quantize: GPTQ against bitsandbytes NF4
7. Compare the 4B GPTQ against the 2B fine-tune
8. Export to GGUF and publish everything to the Hub

The two CLIs do the same work outside the notebook, which is how the wider sweeps in `results/`
were run:

```bash
python scripts/train_script.py --id Qwen/Qwen3-VL-4B-Instruct --name qwen3vl-4b --lora-rank 16

python scripts/eval_script.py adapter-list \
  --id Qwen/Qwen3-VL-4B-Instruct --name qwen3vl-4b \
  --adapter-paths ../models/checkpoints/qwen3vl-4b-r16-LT-px1600/checkpoint-900 \
  --split test --decode plain --output-json ../results/my-results.json
```

Swap `--decode plain` for `--decode constrained` to force the schema.
`scripts/test_diff_size_models.sh` is an eval sweep across the three model sizes.

The OpenVINO export sits in its own notebook because optimum-intel pins `transformers<5.1`,
which cannot share an environment with gptqmodel. Nothing from it is published yet.

## Notes and limitations

- 100 receipts in the test split. Treat one- and two-point gaps as ties.
- CORD-v2 is photographed Indonesian retail and restaurant receipts. Other layouts, languages
  and document types will do worse.
- Field names come straight from CORD (`nm`, `cnt`, `unitprice`), not friendly ones.
- The output is transcribed text. Nothing checks that the line items add up to the total.
- Every timing here is from one machine. Measure on your own hardware before planning around it.
- `models/` is gitignored. Nothing large is tracked.

Code: [github.com/RatlingHitman136/qwen3vl-cordv2-ft](https://github.com/RatlingHitman136/qwen3vl-cordv2-ft)
