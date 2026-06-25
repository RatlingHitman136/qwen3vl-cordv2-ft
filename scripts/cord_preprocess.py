"""
CORD-v2 preprocessing: selective list-of-str normalization.

Single entry point: preprocess_cord(ds) -> (train, val, test)

Each split is a list of {"image": PIL.Image, "label": dict}, where `label`
is the normalized ground-truth parse. Images pass through unchanged.

A field is normalized to list-of-str ONLY if it appears as BOTH str and list
in the corpus (mixed). Pure-str fields stay str; pure-list fields stay list.
The mixed-field set is derived from train+val (NOT test), frozen, and written
to disk so eval can apply the identical rule to model predictions.
"""

import json
from collections import Counter, defaultdict

# Container fields that flip dict<->list (single-item receipts emit a dict).
# These are coerced to a 1-element list, separate from str<->list normalization.
LIST_CONTAINERS = {"menu", "void_menu"}


def _gt_parse(row):
    return json.loads(row["ground_truth"])["gt_parse"]


def _walk_types(obj, tag_counts, type_obs, prefix=""):
    """Record, per dotted tag, how often each value type (str/list/dict) occurs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            tag_counts[path] += 1
            type_obs[path][type(v).__name__] += 1
            _walk_types(v, tag_counts, type_obs, path)
    elif isinstance(obj, list):
        for item in obj:                      # list items share the parent tag
            _walk_types(item, tag_counts, type_obs, prefix)


def _compute_mixed_fields(rows):
    """Tags that appear as BOTH str and list -> these get forced to list-of-str."""
    tag_counts, type_obs = Counter(), defaultdict(Counter)
    for row in rows:
        _walk_types(_gt_parse(row), tag_counts, type_obs)
    mixed = {
        tag for tag, t in type_obs.items()
        if t.get("str", 0) > 0 and t.get("list", 0) > 0
    }
    return frozenset(mixed)


def _normalize_value(v, path, mixed_fields):
    if isinstance(v, str):
        return [v] if path in mixed_fields else v        # wrap only if mixed
    if isinstance(v, dict):
        return _normalize_dict(v, path, mixed_fields)
    if isinstance(v, list):
        if all(isinstance(x, str) for x in v):
            return v                                       # leaf list: leave alone
        return [_normalize_value(x, path, mixed_fields) for x in v]  # array of dicts
    return v                                               # int/float/None passthrough


def _normalize_dict(d, path, mixed_fields):
    out = {}
    for k, v in d.items():
        child = f"{path}.{k}" if path else k
        if k in LIST_CONTAINERS and isinstance(v, dict):
            v = [v]                                         # single-item container -> list
        out[k] = _normalize_value(v, child, mixed_fields)
    return out


def _normalize_receipt(gt_parse, mixed_fields):
    return _normalize_dict(gt_parse, "", mixed_fields)


def _build_split(rows, mixed_fields):
    return [
        {"image": row["image"], "label": _normalize_receipt(_gt_parse(row), mixed_fields)}
        for row in rows
    ]


def preprocess_cord(ds, mixed_fields_path="mixed_fields.json", verbose=True):
    """
    Full CORD-v2 preprocessing.

    Args:
        ds: a DatasetDict with "train", "validation", "test" splits, each row
            having "image" (PIL) and "ground_truth" (JSON string).
        mixed_fields_path: where to persist the frozen mixed-field set so eval
            can load and apply the SAME rule to predictions.
        verbose: print the frozen set and split sizes.

    Returns:
        (train, val, test): each a list of {"image": PIL, "label": dict}.
    """
    # 1. derive the mixed-field rule from train+val ONLY (test is held out)
    rule_rows = list(ds["train"]) + list(ds["validation"])
    mixed_fields = _compute_mixed_fields(rule_rows)

    # 2. freeze it to disk for eval-time reuse (this is the artifact the 3-tuple omits)
    with open(mixed_fields_path, "w") as f:
        json.dump(sorted(mixed_fields), f, indent=2)

    # 3. apply the frozen rule to all three splits
    train = _build_split(ds["train"], mixed_fields)
    val   = _build_split(ds["validation"], mixed_fields)
    test  = _build_split(ds["test"], mixed_fields)

    if verbose:
        print(f"mixed fields (-> list-of-str), frozen to {mixed_fields_path}:")
        for t in sorted(mixed_fields):
            print("  ", t)
        print(f"sizes: train={len(train)} val={len(val)} test={len(test)}")

    return train, val, test


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("naver-clova-ix/cord-v2")
    train, val, test = preprocess_cord(ds)
    # spot-check one normalized label
    print(json.dumps(train[0]["label"], indent=2, ensure_ascii=False))