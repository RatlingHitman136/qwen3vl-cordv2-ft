import json
import re
import zss
from collections import Counter
from pydantic import ValidationError
from ._receipt_schema import Receipt


def _parse_json_lenient(s):
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


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _norm(v):
    return re.sub(r"\s+", " ", str(v)).strip()


def _flatten_fields(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten_fields(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for item in obj:
            out += _flatten_fields(item, prefix)
    else:
        out.append((prefix, _norm(obj)))
    return out


def _field_counts(pred_obj, label_obj):
    p = Counter(_flatten_fields(pred_obj)) if pred_obj is not None else Counter()
    g = Counter(_flatten_fields(label_obj))
    tp = sum((p & g).values())
    return tp, sum(p.values()), sum(g.values())


def _zss_tree(obj):
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
    if isinstance(o, dict):
        return 1 + sum(_count_nodes(v) for v in o.values())
    if isinstance(o, list):
        return 1 + sum(_count_nodes(v) for v in o)
    return 1


def _normalized_ted(pred_obj, label_obj):
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
        pred_obj = _parse_json_lenient(pred_text)

        if pred_obj is not None:
            try:
                Receipt.model_validate(pred_obj)
                valid += 1
            except ValidationError:
                pass

        if pred_obj is not None and _canonical(pred_obj) == _canonical(label_obj):
            exact += 1

        tp, pt, lt = _field_counts(pred_obj, label_obj)
        tp_sum += tp; pred_sum += pt; label_sum += lt

        nted_vals.append(_normalized_ted(pred_obj, label_obj))

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
