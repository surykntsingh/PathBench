# src/test_extract_fields.py

import json

from crqs.crqs_tcga.src.extract_fields import (
    DIAGNOSIS_TERMS,
    HISTOLOGY_MODIFIERS,
    LINEAGE_TERMS,
    extract_fields,
)


TEST_VOCAB = {
    "diagnosis_vocab": DIAGNOSIS_TERMS,
    "histology_vocab": HISTOLOGY_MODIFIERS,
    "lineage_vocab": LINEAGE_TERMS,
}


TEST_CASES = [
    {
        "name": "ovarian papillary serous carcinoma",
        "report": "Right ovary: poorly differentiated papillary serous carcinoma with omental metastasis and necrosis.",
        "expected": {
            "benign_vs_malignant": "malignant",
            "diagnosis": "papillary serous carcinoma",
            "differentiation": "poorly differentiated",
            "metastatic_involvement": "present",
            "necrosis": "present",
        },
    },
    {
        "name": "classic seminoma without LVI",
        "report": "Left testis orchiectomy shows classic seminoma, 1.7 cm, confined to testis, no lymphovascular invasion.",
        "expected": {
            "benign_vs_malignant": "malignant",
            "diagnosis": "classic seminoma",
            "lymphovascular_invasion": "absent",
            "tumor_size": "1.7 cm",
        },
    },
    {
        "name": "papillary thyroid carcinoma with capsular invasion",
        "report": "Total thyroidectomy reveals multifocal papillary thyroid carcinoma, follicular variant, with capsular invasion.",
        "expected": {
            "diagnosis": "papillary thyroid carcinoma",
            "tumor_focality": "multifocal",
            "capsular_invasion": "present",
            "multifocal_involvement": "present",
        },
    },
    {
        "name": "endometrial adenocarcinoma with myometrial invasion",
        "report": "Endometrial adenocarcinoma with myometrial invasion. Margins negative.",
        "expected": {
            "diagnosis": "endometrial adenocarcinoma",
            "invasion_status": "present",
            "organ_specific_invasion": "present",
            "margin_status": "negative",
        },
    },
    {
        "name": "incomplete non-informative report",
        "report": "The pathology report is incomplete and does not provide tumor diagnosis.",
        "expected": {
            "benign_vs_malignant": "neoplasm_unspecified",
        },
    },
]


def check_case(case):
    extracted = extract_fields(case["report"], TEST_VOCAB)
    expected = case["expected"]

    failures = []
    for field, expected_value in expected.items():
        actual_value = extracted.get(field)
        if actual_value != expected_value:
            failures.append(
                {
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )

    return extracted, failures


def main():
    total = len(TEST_CASES)
    passed = 0

    print("\nRunning extract_fields smoke tests...\n")

    for case in TEST_CASES:
        extracted, failures = check_case(case)

        if not failures:
            passed += 1
            print(f"PASS: {case['name']}")
        else:
            print(f"FAIL: {case['name']}")
            print("Expected mismatches:")
            print(json.dumps(failures, indent=2))
            print("Extracted:")
            print(json.dumps(extracted, indent=2))

        print("-" * 80)

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")


if __name__ == "__main__":
    main()
