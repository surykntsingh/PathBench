# src/test_extract_fields.py

"""
Smoke tests for HistAI clinical fact extraction.

Run:
    python src/test_extract_fields.py

Optional:
    python src/test_extract_fields.py --vocab outputs/histai_vocab.json
"""

import argparse
from pathlib import Path

from crqs.crqs_histai.src.extract_fields import extract_fields, load_vocabulary


TEST_CASES = [
    {
        "name": "Prostate adenocarcinoma with Gleason and PNI",
        "text": (
            "Acinar adenocarcinoma of the prostate gland, "
            "Gleason score 3+4=7, WHO/ISUP Grade Group 2, "
            "with perineural invasion."
        ),
        "expected": {
            "benign_vs_malignant": "malignant",
            "diagnosis": "adenocarcinoma",
            "lineage": "carcinoma",
            "histologic_type": "acinar adenocarcinoma",
            "gleason_score": "7 (3+4)",
            "grade_group": "2",
            "perineural_invasion": "present",
        },
    },
    {
        "name": "Chronic inactive gastritis with metaplasia and negative H pylori",
        "text": (
            "Chronic inactive gastritis with intestinal metaplasia. "
            "H. pylori not detected. OLGIM Stage I."
        ),
        "expected": {
            "benign_vs_malignant": "benign",
            "diagnosis": "gastritis",
            "lineage": "inflammatory",
            "histologic_type": "gastritis",
            "inflammation_status": "present",
            "inflammation_activity": "inactive",
            "gastritis_type": "chronic inactive",
            "intestinal_metaplasia": "present",
            "hpylori_status": "negative",
            "olgim_stage": "1",
        },
    },
    {
        "name": "Tubular adenoma with low-grade dysplasia",
        "text": "Tubular adenoma of the colon with low-grade dysplasia.",
        "expected": {
            "benign_vs_malignant": "premalignant/benign",
            "diagnosis": "adenoma",
            "lineage": "epithelial",
            "histologic_type": "tubular adenoma",
            "adenoma_type": "tubular adenoma",
            "polyp_type": "adenoma",
            "dysplasia_grade": "low",
        },
    },
    {
        "name": "Invasive ductal carcinoma with DCIS",
        "text": (
            "Invasive ductal carcinoma of the breast, G2. "
            "Ductal carcinoma in situ is also present."
        ),
        "expected": {
            "benign_vs_malignant": "malignant",
            "diagnosis": "carcinoma",
            "lineage": "carcinoma",
            "histologic_type": "invasive ductal carcinoma",
            "in_situ_vs_invasive": "invasive",
            "in_situ_component": "dcis",
            "tumor_grade": "2",
            "invasion_status": "present",
        },
    },
    {
        "name": "Sessile serrated lesion without dysplasia",
        "text": "Sessile serrated lesion of the ascending colon without dysplasia.",
        "expected": {
            "benign_vs_malignant": "premalignant/benign",
            "diagnosis": "serrated lesion",
            "lineage": "epithelial",
            "histologic_type": "sessile serrated lesion",
            "serrated_lesion": "present",
            "polyp_type": "sessile serrated lesion",
            "dysplasia_grade": "absent",
        },
    },
    {
        "name": "Colorectal adenocarcinoma with LVI and PNI",
        "text": (
            "Colorectal adenocarcinoma, Grade 2, with lymphovascular "
            "and perineural invasion. pT3 pN1 R0."
        ),
        "expected": {
            "benign_vs_malignant": "malignant",
            "diagnosis": "adenocarcinoma",
            "lineage": "carcinoma",
            "histologic_type": "colorectal adenocarcinoma",
            "tumor_grade": "2",
            "lymphovascular_invasion": "present",
            "perineural_invasion": "present",
        },
    },
    {
        "name": "No inflammatory bowel disease",
        "text": (
            "The colonic mucosa exhibits typical histological features, "
            "without evidence of inflammatory bowel disease."
        ),
        "expected": {
            "benign_vs_malignant": "benign",
            "diagnosis": "normal mucosa",
            "inflammation_status": "absent",
        },
    },
]


def value_matches(expected_value, extracted_value):
    """
    Allows exact matches, and allows expected histologic type to be a substring
    of a more specific extracted histologic type.
    """
    if extracted_value is None:
        return False

    if expected_value == extracted_value:
        return True

    if isinstance(expected_value, str) and isinstance(extracted_value, str):
        return expected_value in extracted_value

    return False


def run_tests(vocab=None):
    passed_count = 0

    for i, case in enumerate(TEST_CASES, start=1):
        extracted = extract_fields(case["text"], vocab=vocab)
        expected = case["expected"]

        missing_or_wrong = {}

        for key, expected_value in expected.items():
            extracted_value = extracted.get(key)

            if not value_matches(expected_value, extracted_value):
                missing_or_wrong[key] = {
                    "expected": expected_value,
                    "got": extracted_value,
                }

        passed = len(missing_or_wrong) == 0

        if passed:
            passed_count += 1

        print("=" * 80)
        print(f"Test {i}: {case['name']}")
        print("Status:", "PASS" if passed else "FAIL")
        print("Text:", case["text"])
        print("Extracted:", extracted)

        if not passed:
            print("Problems:", missing_or_wrong)

    print("=" * 80)
    print(f"Passed {passed_count}/{len(TEST_CASES)} tests")

    if passed_count == len(TEST_CASES):
        print("Overall: PASS")
        return True

    print("Overall: FAIL")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run HistAI extraction smoke tests")
    parser.add_argument(
        "--vocab",
        type=str,
        default="outputs/histai_vocab.json",
        help="Optional learned vocabulary path",
    )

    args = parser.parse_args()

    vocab = None

    if args.vocab and Path(args.vocab).exists():
        vocab = load_vocabulary(args.vocab)
        print(f"Loaded vocabulary from {args.vocab}")
    else:
        print("No vocabulary found; running tests with rule-based extractor only.")

    ok = run_tests(vocab=vocab)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
