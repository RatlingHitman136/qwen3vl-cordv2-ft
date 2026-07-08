import json
import re
import zss
from collections import Counter
from pydantic import ValidationError
from ._receipt_schema import Receipt


def parse_json_lenient(s):
    """Parse model output text -> dict, or None.

    The model is trained on real JSON (json.dumps(label)), so this only
    strips markdown code fences / stray surrounding text -- it doesn't need
    to handle any other serialization (e.g. Python dict repr).
    """
    if isinstance(s, dict):
        return s
    if not s:
        return None
    s = re.sub(r"```(?:json)?", "", s).strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None


def canonical(obj):
    """Order-independent string form of a dict, for exact-match comparison."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _norm(v):
    """Collapse whitespace and stringify a leaf value."""
    return re.sub(r"\s+", " ", str(v)).strip()


def flatten_fields(obj, prefix=""):
    """Flatten nested JSON into a list of (leaf_path, value) pairs.

    List indices are dropped from the path on purpose, so menu-item ordering
    doesn't count as an error (field-F1 is a set/multiset comparison,
    matching the CORD/Donut convention).
    """
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten_fields(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for item in obj:
            out += flatten_fields(item, prefix)
    else:
        out.append((prefix, _norm(obj)))
    return out


def field_counts(pred_obj, label_obj):
    """Per-sample (true_positive, pred_total, label_total) over flattened leaves."""
    p = Counter(flatten_fields(pred_obj)) if pred_obj is not None else Counter()
    g = Counter(flatten_fields(label_obj))
    tp = sum((p & g).values())
    return tp, sum(p.values()), sum(g.values())


def _zss_tree(obj):
    """Build a zss.Node tree from nested JSON, for tree-edit-distance."""
    def build(o, label):
        n = zss.Node(label)
        if isinstance(o, dict):
            for k, v in o.items():
                n.addkid(build(v, k))
        elif isinstance(o, list):
            for it in o:
                n.addkid(build(it, "[]"))
        else:
            n.addkid(zss.Node(str(o)))
        return n

    return build(obj, "root")


def _count_nodes(o):
    """Count nodes in a nested JSON structure."""
    if isinstance(o, dict):
        return 1 + sum(_count_nodes(v) for v in o.values())
    if isinstance(o, list):
        return 1 + sum(_count_nodes(v) for v in o)
    return 1


def normalized_ted(pred_obj, label_obj):
    """1 - TED/maxnodes: structural similarity in [0,1], closest to the CORD
    paper's nTED accuracy. 0.0 if the prediction failed to parse."""
    if pred_obj is None:
        return 0.0
    dist = zss.simple_distance(_zss_tree(pred_obj), _zss_tree(label_obj))
    denom = max(_count_nodes(pred_obj), _count_nodes(label_obj)) or 1
    return max(0.0, 1.0 - dist / denom)


def aggregate_generation_metrics(preds, labels):
    """Score model text outputs against label dicts.

    preds: model output strings. labels: ground-truth receipts, already
    dicts (the model was trained on json.dumps(label), so the label side
    never needs a json.loads). Returns dataset-level metrics.
    """
    n = len(labels)
    valid = exact = 0
    tp_sum = pred_sum = label_sum = 0
    nted_vals = []

    for pred_text, label_obj in zip(preds, labels):
        pred_obj = parse_json_lenient(pred_text)

        if pred_obj is not None:
            try:
                Receipt.model_validate(pred_obj)
                valid += 1
            except ValidationError:
                pass

        if pred_obj is not None and canonical(pred_obj) == canonical(label_obj):
            exact += 1

        tp, pt, lt = field_counts(pred_obj, label_obj)
        tp_sum += tp; pred_sum += pt; label_sum += lt

        nted_vals.append(normalized_ted(pred_obj, label_obj))

    precision = tp_sum / pred_sum if pred_sum else 0.0
    recall    = tp_sum / label_sum if label_sum else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n":               n,
        "json_validity":   valid / n,
        "exact_match":     exact / n,
        "field_precision": precision,
        "field_recall":    recall,
        "field_f1":        f1,
        "normalized_ted":  (sum(nted_vals) / len(nted_vals)) if nted_vals else None,
    }
