"""
Metric computation for CRQS evaluation on REG pathology reports.

Computes:
- CFC: Clinical Fact Coverage
- KIR: Key Information Recall
- HR: Hallucination Rate
- CDS: Clinical Discordance Score
- CRQS: Final weighted score
"""

from crqs.crqs_reg.src.config import (
    CLINICAL_FIELDS,
    CRQS_WEIGHTS,
    KEY_FIELDS,
    NUMERIC_FIELDS,
    STRICT_FIELDS,
)


def normalize_value(value):
    """
    Normalize extracted values before comparison.
    """
    if value is None:
        return None

    value = str(value).lower().strip()
    value = value.replace("grade ", "")
    value = value.replace("group ", "")
    value = value.replace("%", "")
    value = " ".join(value.split())

    return value


def values_match(field, target_value, pred_value):
    """
    Decide whether two field values match.

    Most clinical fields use exact normalized matching.
    Numeric fields can allow small tolerance if needed.
    """

    if target_value is None or pred_value is None:
        return False

    target_norm = normalize_value(target_value)
    pred_norm = normalize_value(pred_value)

    if target_norm == pred_norm:
        return True

    # Optional numeric tolerance for tumor volume-like fields
    if field in NUMERIC_FIELDS:
        try:
            target_num = float(target_norm)
            pred_num = float(pred_norm)

            # Allow small tolerance for approximate visual estimates
            return abs(target_num - pred_num) <= 10
        except ValueError:
            return False

    return False


def get_present_fields(facts, candidate_fields):
    """
    Return fields that are both in candidate_fields and present in facts.
    """
    return [
        field
        for field in candidate_fields
        if field in facts and facts[field] not in [None, ""]
    ]


def compute_cfc(target_facts, pred_facts):
    """
    Clinical Fact Coverage.

    Measures how many target clinical facts are correctly recovered.
    """

    target_fields = get_present_fields(target_facts, CLINICAL_FIELDS)

    if not target_fields:
        return 0.0, {
            "correct": [],
            "missed": [],
            "total": 0,
        }

    correct = []
    missed = []

    for field in target_fields:
        if field in pred_facts and values_match(field, target_facts[field], pred_facts[field]):
            correct.append(field)
        else:
            missed.append(field)

    score = len(correct) / len(target_fields)

    details = {
        "correct": correct,
        "missed": missed,
        "total": len(target_fields),
    }

    return score, details


def compute_kir(target_facts, pred_facts):
    """
    Key Information Recall.

    Measures recall over clinically important key fields only.
    """

    target_key_fields = get_present_fields(target_facts, KEY_FIELDS)

    if not target_key_fields:
        return 0.0, {
            "correct": [],
            "missed": [],
            "total": 0,
        }

    correct = []
    missed = []

    for field in target_key_fields:
        if field in pred_facts and values_match(field, target_facts[field], pred_facts[field]):
            correct.append(field)
        else:
            missed.append(field)

    score = len(correct) / len(target_key_fields)

    details = {
        "correct": correct,
        "missed": missed,
        "total": len(target_key_fields),
    }

    return score, details


def compute_hr(target_facts, pred_facts):
    """
    Hallucination Rate.

    Measures predicted fields that are unsupported by target facts.

    A field is hallucinated if:
    - it appears in prediction
    - it does not appear in target
    """

    pred_fields = get_present_fields(pred_facts, CLINICAL_FIELDS)

    if not pred_fields:
        return 0.0, {
            "hallucinated": [],
            "total_predicted": 0,
        }

    hallucinated = []

    for field in pred_fields:
        if field not in target_facts or target_facts[field] in [None, ""]:
            hallucinated.append(field)

    score = len(hallucinated) / len(pred_fields)

    details = {
        "hallucinated": hallucinated,
        "total_predicted": len(pred_fields),
    }

    return score, details


def compute_cds(target_facts, pred_facts):
    """
    Clinical Discordance Score.

    Measures disagreement for comparable fields present in both target and prediction.

    A field is discordant if:
    - present in target
    - present in prediction
    - values do not match
    """

    comparable_fields = []

    for field in CLINICAL_FIELDS:
        if (
            field in target_facts
            and field in pred_facts
            and target_facts[field] not in [None, ""]
            and pred_facts[field] not in [None, ""]
        ):
            comparable_fields.append(field)

    if not comparable_fields:
        return 0.0, {
            "discordant": [],
            "total_comparable": 0,
        }

    discordant = []

    for field in comparable_fields:
        if not values_match(field, target_facts[field], pred_facts[field]):
            discordant.append(field)

    score = len(discordant) / len(comparable_fields)

    details = {
        "discordant": discordant,
        "total_comparable": len(comparable_fields),
    }

    return score, details


def compute_crqs(cfc, kir, hr, cds):
    """
    Final weighted CRQS score.
    """

    crqs = (
        CRQS_WEIGHTS["CFC"] * cfc
        + CRQS_WEIGHTS["KIR"] * kir
        + CRQS_WEIGHTS["HR"] * hr
        + CRQS_WEIGHTS["CDS"] * cds
    )

    return crqs


def compute_all_metrics(target_facts, pred_facts):
    """
    Compute all CRQS metrics for one target/prediction pair.
    """

    cfc, cfc_details = compute_cfc(target_facts, pred_facts)
    kir, kir_details = compute_kir(target_facts, pred_facts)
    hr, hr_details = compute_hr(target_facts, pred_facts)
    cds, cds_details = compute_cds(target_facts, pred_facts)
    crqs = compute_crqs(cfc, kir, hr, cds)

    return {
        "CFC": cfc,
        "KIR": kir,
        "HR": hr,
        "CDS": cds,
        "CRQS": crqs,
        "details": {
            "CFC": cfc_details,
            "KIR": kir_details,
            "HR": hr_details,
            "CDS": cds_details,
        },
    }


if __name__ == "__main__":
    examples = [
        {
            "name": "Perfect match",
            "target": {
                "diagnosis": "acinar adenocarcinoma",
                "benign_vs_malignant": "malignant",
                "lineage": "carcinoma",
                "gleason_score": "7 (4+3)",
            },
            "pred": {
                "diagnosis": "acinar adenocarcinoma",
                "benign_vs_malignant": "malignant",
                "lineage": "carcinoma",
                "gleason_score": "7 (4+3)",
            },
        },
        {
            "name": "Wrong Gleason",
            "target": {
                "diagnosis": "acinar adenocarcinoma",
                "benign_vs_malignant": "malignant",
                "lineage": "carcinoma",
                "gleason_score": "7 (4+3)",
            },
            "pred": {
                "diagnosis": "acinar adenocarcinoma",
                "benign_vs_malignant": "malignant",
                "lineage": "carcinoma",
                "gleason_score": "6 (3+3)",
            },
        },
        {
            "name": "Hallucinated necrosis",
            "target": {
                "diagnosis": "ductal carcinoma in situ",
                "nuclear_grade": "high",
            },
            "pred": {
                "diagnosis": "ductal carcinoma in situ",
                "nuclear_grade": "high",
                "necrosis": "present",
            },
        },
    ]

    for example in examples:
        print("\n==============================")
        print(example["name"])
        print("TARGET:", example["target"])
        print("PRED:", example["pred"])

        result = compute_all_metrics(example["target"], example["pred"])

        print("RESULT:")
        print(result)
