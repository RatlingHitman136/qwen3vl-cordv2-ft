# cord_preprocess.py
import json
from collections import Counter, defaultdict
from ._receipt_schema import assert_schema_valid

# Containers that flip dict<->list in CORD (single-item -> bare dict). Wrap to list.
# This is the ONLY structural transform — it changes representation, not content.
LIST_CONTAINERS = {"menu", "sub"}

def _gt_parse(row):
    return json.loads(row["ground_truth"])["gt_parse"]

# ---- frozen mixed-field set: leaves that appear as both str and list ----
def _walk_types(obj, tag_counts, type_obs, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            tag_counts[path] += 1
            type_obs[path][type(v).__name__] += 1
            _walk_types(v, tag_counts, type_obs, path)
    elif isinstance(obj, list):
        for item in obj:
            _walk_types(item, tag_counts, type_obs, prefix)

def _compute_mixed_fields(gt_strings):
    tag_counts, type_obs = Counter(), defaultdict(Counter)
    for s in gt_strings:
        _walk_types(json.loads(s)["gt_parse"], tag_counts, type_obs)
    return frozenset(
        tag for tag, t in type_obs.items()
        if t.get("str", 0) > 0 and t.get("list", 0) > 0
    )

# ---- lossless normalization: wrap containers + mixed-leaf -> list-of-str ----
def _normalize_value(v, path, mixed):
    if isinstance(v, str):
        return [v] if path in mixed else v
    if isinstance(v, dict):
        return _normalize_dict(v, path, mixed)
    if isinstance(v, list):
        if all(isinstance(x, str) for x in v):
            return v
        return [_normalize_value(x, path, mixed) for x in v]
    return v

def _normalize_dict(d, path, mixed):
    out = {}
    for k, v in d.items():
        child = f"{path}.{k}" if path else k
        if k in LIST_CONTAINERS and isinstance(v, dict):
            v = [v]                              # wrap single-item container
        out[k] = _normalize_value(v, child, mixed)
    return out

class LazySplit:
    """Sequence of {"image": PIL, "label": dict} rows. Labels live in RAM (small);
    images stay Arrow-backed in the HF dataset and decode per access, so a split
    never holds a thousand decoded PIL images at once (OOM-kills the kernel in
    memory-capped containers)."""

    def __init__(self, hf_split, labels):
        self._split = hf_split
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"image": self._split[int(idx)]["image"], "label": self.labels[idx]}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def preprocess_cord(dataset, mixed_fields_path="mixed_fields.json", verbose=True):
    """
    Returns (train, val, test). Each is a LazySplit of {"image": PIL, "label": dict}
    rows — same row interface as a list, but images decode on access instead of
    being materialized up front.

    LOSSLESS: labels are normalized (mixed leaves -> list-of-str) and have their
    dict<->list containers wrapped. NO keys are removed and NO content is dropped.

    Because nothing is stripped, labels containing keys outside the schema will
    FAIL validation — these are reported, not silenced. Decide per field whether
    to add it to the schema or exclude the receipt downstream.

    Mixed-field rule is derived from train+val only (test held out), frozen to disk.
    """
    # single-column access: reading only ground_truth skips decoding every image,
    # which would otherwise cost ~900 PIL decodes of RAM for a JSON-only pass
    rule_gts = list(dataset["train"]["ground_truth"]) + list(dataset["validation"]["ground_truth"])
    mixed = _compute_mixed_fields(rule_gts)
    with open(mixed_fields_path, "w") as f:
        json.dump(sorted(mixed), f, indent=2)

    def build(split):
        labels = [_normalize_dict(json.loads(s)["gt_parse"], "", mixed)
                  for s in split["ground_truth"]]
        return LazySplit(split, labels)

    train = build(dataset["train"])
    val   = build(dataset["validation"])
    test  = build(dataset["test"])

    if verbose:
        print(f"mixed fields ({len(mixed)}) frozen to {mixed_fields_path}")
        print(f"sizes: train={len(train)} val={len(val)} test={len(test)}")
        assert_schema_valid(train, "train")
        assert_schema_valid(val, "validation")
        assert_schema_valid(test, "test")

    return train, val, test

def test(dataset, row):
    mixed = _compute_mixed_fields(
        list(dataset["train"]["ground_truth"]) + list(dataset["validation"]["ground_truth"]))
    print(f"mixed fields ({len(mixed)}): {sorted(mixed)}")
    return _normalize_dict(_gt_parse(row), "", mixed)