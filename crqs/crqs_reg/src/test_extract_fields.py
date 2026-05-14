"""
Smoke test for extract_fields.py

Purpose:
- Ensure extractor runs without crashing
- Ensure key fields are extracted correctly
- Catch regressions after code edits
"""

from crqs.crqs_reg.src.extract_fields import extract_fields, load_vocabulary

VOCAB_PATH = "outputs/reg_vocabulary.json"


def check(report, expected_pairs):
    vocab = load_vocabulary(VOCAB_PATH)
    result = extract_fields(report, vocab=vocab)

    print("\nREPORT:")
    print(report)
    print("EXTRACTED:")
    print(result)

    failed = []

    for key, expected_value in expected_pairs.items():
        actual = result.get(key)

        if actual != expected_value:
            failed.append((key, expected_value, actual))

    if failed:
        print("❌ FAIL")
        for f in failed:
            print(f"  {f[0]} | expected={f[1]} | got={f[2]}")
        return False

    print("✅ PASS")
    return True


def main():
    tests = [

        # -----------------------------------------
        # Breast carcinoma
        # -----------------------------------------
        (
            "Breast, biopsy; Invasive carcinoma of no special type, grade II "
            "(Tubule formation: 3, Nuclear grade: 2, Mitoses: 1)",
            {
                "diagnosis": "invasive carcinoma of no special type",
                "tumor_grade": "ii",
                "nuclear_grade": "2",
                "mitotic_score": "1",
            },
        ),

        # -----------------------------------------
        # Prostate
        # -----------------------------------------
        (
            "Prostate, biopsy; Acinar adenocarcinoma, "
            "Gleason's score 7 (4+3), grade group 3",
            {
                "diagnosis": "acinar adenocarcinoma",
                "gleason_score": "7 (4+3)",
                "grade_group": "3",
            },
        ),

        # -----------------------------------------
        # Bladder invasion
        # -----------------------------------------
        (
            "Urinary bladder, TURBT; Invasive urothelial carcinoma, "
            "with involvement of muscle proper",
            {
                "diagnosis": "invasive urothelial carcinoma",
                "invasion_depth": "muscle proper",
            },
        ),

        # -----------------------------------------
        # GI adenoma
        # -----------------------------------------
        (
            "Stomach, biopsy; Tubular adenoma with low grade dysplasia",
            {
                "diagnosis": "tubular adenoma",
                "dysplasia_grade": "low",
            },
        ),

        # -----------------------------------------
        # Cervix HSIL
        # -----------------------------------------
        (
            "Uterine cervix, biopsy; High-grade squamous intraepithelial lesion (HSIL; CIN 3)",
            {
                "diagnosis": "high-grade squamous intraepithelial lesion",
                "dysplasia_grade": "high",
                "cin_grade": "3",
            },
        ),

        # -----------------------------------------
        # Negative benign case
        # -----------------------------------------
        (
            "Lung, biopsy; No evidence of malignancy",
            {
                "benign_vs_malignant": "benign",
            },
        ),
    ]

    total = len(tests)
    passed = 0

    for report, expected in tests:
        ok = check(report, expected)
        if ok:
            passed += 1

    print("\n==============================")
    print(f"PASSED {passed}/{total} TESTS")

    if passed == total:
        print("🎉 extract_fields.py looks healthy.")
    else:
        print("⚠️ Some tests failed.")


if __name__ == "__main__":
    main()
