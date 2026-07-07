# constants shared between training (collate_fn) and eval (run_inference / run_constrained_inference)

FIXED_PROMPT = """Extract the receipt from the image into a structured JSON. Your output should contain ONLY correct JSON!
"""

DATASET_ID = "naver-clova-ix/cord-v2"

BASE_MODEL_SAVE_DIR = "../models/base/"
CHECKPOINT_MODEL_SAVE_DIR = "../models/checkpoints/"
ADAPTER_OUTPUT_DIR = "../models/adapters/"
MERGED_MODEL_SAVE_DIR = "../models/merged/"
QUANTIZED_MODEL_SAVE_DIR = "../models/quantized/"
EXPORT_MODEL_SAVE_DIR = "../models/export/"
