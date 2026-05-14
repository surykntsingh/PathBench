# src/compute_metrics.py

import json
import re
from typing import Any, Dict, Tuple

from crqs.crqs_tcga.src.config import (
    CLINICAL_FIELDS,
    CRQS_WEIGHTS,
    KEY_FIELDS,
    NUMERIC_FIELDS,
    STRICT_FIELDS,
)


def is_present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def normalize_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        value = ", ".join(str(v) for v in sorted(value))

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def values_match(target_value: Any, pred_value: Any, field: str) -> bool:
    if not is_present(target_value) or not is_present(pred_value):
        return False

    t = normalize_value(target_value)
    p = normalize_value(pred_value)

    if field in STRICT_FIELDS:
        return t == p

    if field in NUMERIC_FIELDS:
        return numeric_match(t, p)

    return relaxed_text_match(t, p)


def relaxed_text_match(target: str, pred: str) -> bool:
    if target == pred:
        return True

    if target in pred or pred in target:
        return True

    target_tokens = set(target.split())
    pred_tokens = set(pred.split())

    if not target_tokens or not pred_tokens:
        return False

    overlap = len(target_tokens & pred_tokens) / len(target_tokens | pred_tokens)
    return overlap >= 0.6


def numeric_match(target: str, pred: str, tolerance: float = 0.15) -> bool:
    target_num = extract_first_number(target)
    pred_num = extract_first_number(pred)

    if target_num is None or pred_num is None:
        return relaxed_text_match(target, pred)

    if target_num == 0:
        return pred_num == 0

    return abs(target_num - pred_num) / abs(target_num) <= tolerance


def extract_first_number(text: str):
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    return float(m.group(0))


def compute_cfc(target_fields: Dict[str, Any], pred_fields: Dict[str, Any]) -> float:
    target_present = [
        field for field in CLINICAL_FIELDS
        if is_present(target_fields.get(field))
    ]

    if not target_present:
        return 1.0

    correct = sum(
        values_match(target_fields.get(field), pred_fields.get(field), field)
        for field in target_present
    )

    return correct / len(target_present)


def compute_kir(target_fields: Dict[str, Any], pred_fields: Dict[str, Any]) -> float:
    target_key_present = [
        field for field in KEY_FIELDS
        if is_present(target_fields.get(field))
    ]

    if not target_key_present:
        return 1.0

    correct = sum(
        values_match(target_fields.get(field), pred_fields.get(field), field)
        for field in target_key_present
    )

    return correct / len(target_key_present)


def compute_hr(target_fields: Dict[str, Any], pred_fields: Dict[str, Any]) -> float:
    pred_present = [
        field for field in CLINICAL_FIELDS
        if is_present(pred_fields.get(field))
    ]

    if not pred_present:
        return 0.0

    unsupported = 0

    for field in pred_present:
        if not is_present(target_fields.get(field)):
            unsupported += 1
        elif not values_match(target_fields.get(field), pred_fields.get(field), field):
            unsupported += 1

    return unsupported / len(pred_present)


def compute_cds(target_fields: Dict[str, Any], pred_fields: Dict[str, Any]) -> float:
    comparable = [
        field for field in CLINICAL_FIELDS
        if is_present(target_fields.get(field)) and is_present(pred_fields.get(field))
    ]

    if not comparable:
        return 0.0

    discordant = sum(
        not values_match(target_fields.get(field), pred_fields.get(field), field)
        for field in comparable
    )

    return discordant / len(comparable)


def compute_crqs(
    target_fields: Dict[str, Any],
    pred_fields: Dict[str, Any],
) -> Dict[str, Any]:
    cfc = compute_cfc(target_fields, pred_fields)
    kir = compute_kir(target_fields, pred_fields)
    hr = compute_hr(target_fields, pred_fields)
    cds = compute_cds(target_fields, pred_fields)

    crqs_raw = (
        CRQS_WEIGHTS["CFC"] * cfc
        + CRQS_WEIGHTS["KIR"] * kir
        + CRQS_WEIGHTS["HR"] * hr
        + CRQS_WEIGHTS["CDS"] * cds
    )

    crqs_norm = crqs_raw / 0.7

    return {
        "CFC": cfc,
        "KIR": kir,
        "HR": hr,
        "CDS": cds,
        "CRQS_raw": crqs_raw,
        "CRQS_norm": crqs_norm,
        "n_target_fields": sum(is_present(target_fields.get(f)) for f in CLINICAL_FIELDS),
        "n_pred_fields": sum(is_present(pred_fields.get(f)) for f in CLINICAL_FIELDS),
        "n_target_key_fields": sum(is_present(target_fields.get(f)) for f in KEY_FIELDS),
    }


def smoke_test():
    target = {
        "benign_vs_malignant": "malignant",
        "diagnosis": "papillary serous carcinoma",
        "histologic_type": "papillary, serous",
        "differentiation": "poorly differentiated",
        "metastatic_involvement": "present",
        "necrosis": "present",
    }

    pred_good = {
        "benign_vs_malignant": "malignant",
        "diagnosis": "papillary serous carcinoma",
        "histologic_type": "serous papillary",
        "differentiation": "poorly differentiated",
        "metastatic_involvement": "present",
        "necrosis": "present",
    }

    pred_bad = {
        "benign_vs_malignant": "benign_or_negative",
        "diagnosis": "seminoma",
        "histologic_type": "classic",
        "differentiation": "well differentiated",
        "metastatic_involvement": "absent",
        "necrosis": "absent",
    }

    print("\nGOOD PREDICTION")
    print(json.dumps(compute_crqs(target, pred_good), indent=2))

    print("\nBAD PREDICTION")
    print(json.dumps(compute_crqs(target, pred_bad), indent=2))


if __name__ == "__main__":
    smoke_test()
