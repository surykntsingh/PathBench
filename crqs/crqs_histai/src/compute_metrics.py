# src/compute_metrics.py

"""
Compute CRQS metrics from extracted clinical fields.

Metrics:
    CFC = correct target clinical fields / target clinical fields present
    KIR = correct key fields / target key fields present
    HR  = predicted unsupported fields / predicted fields
    CDS = discordant comparable fields / comparable fields

CRQS_raw  = 0.3*CFC + 0.4*KIR - 0.2*HR - 0.4*CDS
CRQS_norm = CRQS_raw / 0.7
"""

from typing import Dict, Any, Tuple

from crqs.crqs_histai.src.config import CRQS_MAX_RAW, CRQS_WEIGHTS, KEY_FIELDS, STRICT_FIELDS


def normalize_value(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).lower().strip()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = " ".join(value.split())

    synonyms = {
        "positive": "present",
        "detected": "present",
        "yes": "present",
        "negative": "absent",
        "not detected": "absent",
        "no": "absent",
        "none": "absent",
        "low grade": "low",
        "high grade": "high",
        "grade 1": "1",
        "grade 2": "2",
        "grade 3": "3",
        "g1": "1",
        "g2": "2",
        "g3": "3",
    }

    return synonyms.get(value, value)


def values_match(field: str, target_value: Any, pred_value: Any) -> bool:
    target = normalize_value(target_value)
    pred = normalize_value(pred_value)

    if target == "" or pred == "":
        return False

    if target == pred:
        return True

    # For non-strict text fields, allow partial containment.
    # Example:
    # target histologic_type = "invasive ductal carcinoma"
    # pred histologic_type = "ductal carcinoma"
    if field not in STRICT_FIELDS:
        if target in pred or pred in target:
            return True

    # Histologic type often differs in specificity.
    # Accept partial overlap for this field if one contains the other.
    if field == "histologic_type":
        if target in pred or pred in target:
            return True

    return False


def is_discordant(field: str, target_value: Any, pred_value: Any) -> bool:
    target = normalize_value(target_value)
    pred = normalize_value(pred_value)

    if target == "" or pred == "":
        return False

    if values_match(field, target, pred):
        return False

    opposite_pairs = {
        ("benign", "malignant"),
        ("malignant", "benign"),
        ("present", "absent"),
        ("absent", "present"),
        ("positive", "negative"),
        ("negative", "positive"),
        ("invasive", "in situ"),
        ("in situ", "invasive"),
        ("low", "high"),
        ("high", "low"),
    }

    if (target, pred) in opposite_pairs:
        return True

    # Malignancy category conflict.
    if field == "benign_vs_malignant":
        return target != pred

    # Exact clinical grade conflicts.
    if field in {
        "gleason_score",
        "grade_group",
        "tumor_grade",
        "dysplasia_grade",
        "hpylori_status",
        "lymphovascular_invasion",
        "perineural_invasion",
        "extraprostatic_extension",
        "in_situ_vs_invasive",
        "invasion_status",
        "intestinal_metaplasia",
        "inflammation_status",
    }:
        return target != pred

    # Diagnosis / histology conflicts when both are present but do not match.
    if field in {"diagnosis", "histologic_type", "lineage"}:
        return target != pred

    return target != pred


def present_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in fields.items()
        if value is not None and str(value).strip() != ""
    }


def compute_cfc(target: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[float, int, int]:
    target_present = present_fields(target)

    if not target_present:
        return 0.0, 0, 0

    correct = 0

    for field, target_value in target_present.items():
        if field in pred and values_match(field, target_value, pred[field]):
            correct += 1

    total = len(target_present)
    return correct / total, correct, total


def compute_kir(target: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[float, int, int]:
    target_key_present = {
        field: target[field]
        for field in KEY_FIELDS
        if field in target and target[field] is not None and str(target[field]).strip() != ""
    }

    if not target_key_present:
        return 0.0, 0, 0

    correct = 0

    for field, target_value in target_key_present.items():
        if field in pred and values_match(field, target_value, pred[field]):
            correct += 1

    total = len(target_key_present)
    return correct / total, correct, total


def compute_hr(target: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[float, int, int]:
    pred_present = present_fields(pred)

    if not pred_present:
        return 0.0, 0, 0

    unsupported = 0

    for field, pred_value in pred_present.items():
        if field not in target:
            unsupported += 1
        elif not values_match(field, target[field], pred_value):
            unsupported += 1

    total = len(pred_present)
    return unsupported / total, unsupported, total


def compute_cds(target: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[float, int, int]:
    comparable_fields = [
        field for field in target.keys()
        if field in pred
        and target[field] is not None
        and pred[field] is not None
        and str(target[field]).strip() != ""
        and str(pred[field]).strip() != ""
    ]

    if not comparable_fields:
        return 0.0, 0, 0

    discordant = 0

    for field in comparable_fields:
        if is_discordant(field, target[field], pred[field]):
            discordant += 1

    total = len(comparable_fields)
    return discordant / total, discordant, total


def compute_crqs(cfc: float, kir: float, hr: float, cds: float) -> Tuple[float, float]:
    raw = (
        CRQS_WEIGHTS["CFC"] * cfc
        + CRQS_WEIGHTS["KIR"] * kir
        + CRQS_WEIGHTS["HR"] * hr
        + CRQS_WEIGHTS["CDS"] * cds
    )

    norm = raw / CRQS_MAX_RAW

    return raw, norm


def compute_metrics(target: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    cfc, cfc_correct, cfc_total = compute_cfc(target, pred)
    kir, kir_correct, kir_total = compute_kir(target, pred)
    hr, hr_unsupported, hr_total = compute_hr(target, pred)
    cds, cds_discordant, cds_total = compute_cds(target, pred)

    crqs_raw, crqs_norm = compute_crqs(cfc, kir, hr, cds)

    return {
        "CFC": cfc,
        "KIR": kir,
        "HR": hr,
        "CDS": cds,
        "CRQS_raw": crqs_raw,
        "CRQS_norm": crqs_norm,

        "CFC_correct": cfc_correct,
        "CFC_total": cfc_total,
        "KIR_correct": kir_correct,
        "KIR_total": kir_total,
        "HR_unsupported": hr_unsupported,
        "HR_total": hr_total,
        "CDS_discordant": cds_discordant,
        "CDS_total": cds_total,
    }


def demo():
    target = {
        "benign_vs_malignant": "malignant",
        "diagnosis": "adenocarcinoma",
        "lineage": "carcinoma",
        "histologic_type": "acinar adenocarcinoma",
        "gleason_score": "7 (3+4)",
        "grade_group": "2",
        "perineural_invasion": "present",
    }

    pred_good = {
        "benign_vs_malignant": "malignant",
        "diagnosis": "adenocarcinoma",
        "lineage": "carcinoma",
        "histologic_type": "acinar adenocarcinoma",
        "gleason_score": "7 (3+4)",
        "grade_group": "2",
        "perineural_invasion": "present",
    }

    pred_missing = {
        "benign_vs_malignant": "malignant",
        "diagnosis": "adenocarcinoma",
        "histologic_type": "acinar adenocarcinoma",
    }

    pred_bad = {
        "benign_vs_malignant": "benign",
        "diagnosis": "gastritis",
        "lineage": "inflammatory",
        "histologic_type": "chronic gastritis",
    }

    examples = {
        "Perfect match": pred_good,
        "Missing key facts": pred_missing,
        "Discordant prediction": pred_bad,
    }

    for name, pred in examples.items():
        print("\n" + "=" * 80)
        print(name)
        print("TARGET:", target)
        print("PRED:", pred)
        print("METRICS:", compute_metrics(target, pred))


if __name__ == "__main__":
    demo()
