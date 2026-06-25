# cord_preprocess.py
import json
from collections import Counter, defaultdict
from _receipt_schema import assert_schema_valid

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

def _compute_mixed_fields(rows):
    tag_counts, type_obs = Counter(), defaultdict(Counter)
    for row in rows:
        _walk_types(_gt_parse(row), tag_counts, type_obs)
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

def preprocess_cord(dataset, mixed_fields_path="mixed_fields.json", verbose=True):
    """
    Returns (train, val, test). Each is a list of {"image": PIL, "label": dict}.

    LOSSLESS: labels are normalized (mixed leaves -> list-of-str) and have their
    dict<->list containers wrapped. NO keys are removed and NO content is dropped.

    Because nothing is stripped, labels containing keys outside the schema will
    FAIL validation — these are reported, not silenced. Decide per field whether
    to add it to the schema or exclude the receipt downstream.

    Mixed-field rule is derived from train+val only (test held out), frozen to disk.
    """
    rule_rows = list(dataset["train"]) + list(dataset["validation"])
    mixed = _compute_mixed_fields(rule_rows)
    with open(mixed_fields_path, "w") as f:
        json.dump(sorted(mixed), f, indent=2)

    def build(rows):
        return [
            {"image": row["image"],
             "label": _normalize_dict(_gt_parse(row), "", mixed)}
            for row in rows
        ]

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
    mixed = _compute_mixed_fields(list(dataset["train"]) + list(dataset["validation"]))
    print(f"mixed fields ({len(mixed)}): {sorted(mixed)}")
    return _normalize_dict(_gt_parse(row), "", mixed)