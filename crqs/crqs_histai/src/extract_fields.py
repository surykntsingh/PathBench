# src/extract_fields.py
# HistAI dataset-specific clinical fact extractor

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path


ROMAN_TO_INT = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
}


MALIGNANT_TERMS = [
    "adenocarcinoma",
    "carcinoma",
    "sarcoma",
    "lymphoma",
    "melanoma",
    "malignant",
    "metastasis",
    "metastases",
    "invasive carcinoma",
    "neuroendocrine tumor",
]


BENIGN_PREMALIGNANT_TERMS = [
    "gastritis",
    "colitis",
    "duodenitis",
    "ileitis",
    "proctitis",
    "esophagitis",
    "adenoma",
    "polyp",
    "hyperplasia",
    "metaplasia",
    "dysplasia",
    "fibroadenoma",
    "leiomyoma",
]


HISTOLOGY_PATTERNS = [
    r"acinar adenocarcinoma",
    r"ductal adenocarcinoma",
    r"pancreatic ductal adenocarcinoma",
    r"colorectal adenocarcinoma",
    r"endometrioid adenocarcinoma",
    r"tubular adenocarcinoma",
    r"mucinous adenocarcinoma",
    r"invasive ductal carcinoma",
    r"invasive lobular carcinoma",
    r"invasive carcinoma",
    r"squamous cell carcinoma",
    r"urothelial carcinoma",
    r"serous carcinoma",
    r"papillary thyroid carcinoma",
    r"diffuse large b-cell lymphoma",
    r"neuroendocrine tumor",
    r"tubular adenoma",
    r"tubulovillous adenoma",
    r"sessile serrated lesion",
    r"hyperplastic polyp",
    r"fundic gland polyp",
]


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = text.lower()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[^a-z0-9\+\=/().,% ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset_reports(path: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reports = []

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict) and row.get("report"):
                        reports.append(row["report"])
            elif isinstance(value, dict) and value.get("report"):
                reports.append(value["report"])

    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and row.get("report"):
                reports.append(row["report"])

    return reports


def learn_vocabulary(reports, min_freq: int = 2):
    phrase_counter = Counter()

    anchor_terms = [
        "adenocarcinoma",
        "carcinoma",
        "lymphoma",
        "melanoma",
        "sarcoma",
        "gastritis",
        "colitis",
        "adenoma",
        "polyp",
        "serrated lesion",
        "dysplasia",
        "hyperplasia",
        "metaplasia",
        "neuroendocrine tumor",
    ]

    for report in reports:
        text = normalize_text(report)

        for anchor in anchor_terms:
            if anchor not in text:
                continue

            pattern = rf"((?:[a-z0-9\-]+ ){{0,4}}{re.escape(anchor)}(?: [a-z0-9\-]+){{0,3}})"
            for match in re.finditer(pattern, text):
                phrase = match.group(1).strip(" .,;:")
                if 3 <= len(phrase) <= 80:
                    phrase_counter[phrase] += 1

    vocabulary = {
        "histology_terms": [
            phrase
            for phrase, count in phrase_counter.most_common()
            if count >= min_freq
        ]
    }

    return vocabulary


def save_vocabulary(vocab, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def load_vocabulary(path: str):
    path = Path(path)

    if not path.exists():
        return {"histology_terms": []}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def roman_to_int(value: str):
    value = value.lower().strip()
    return ROMAN_TO_INT.get(value, value)


def has_negation_near(text: str, term: str, window: int = 45) -> bool:
    for match in re.finditer(re.escape(term), text):
        start = max(0, match.start() - window)
        context = text[start:match.start()]
        if re.search(r"\b(no|not|without|absent|free of|negative for)\b", context):
            return True
    return False


def extract_benign_vs_malignant(text: str):
    malignant = any(term in text for term in MALIGNANT_TERMS)

    benign_or_premalignant = any(term in text for term in BENIGN_PREMALIGNANT_TERMS)

    if malignant:
        return "malignant"

    if benign_or_premalignant:
        if "dysplasia" in text or "adenoma" in text or "serrated lesion" in text:
            return "premalignant/benign"
        return "benign"

    if (
        "normal histological structure" in text
        or "typical histological structure" in text
        or "typical histological features" in text
        or "normal morphology" in text
    ):
        return "benign"

    return None


def extract_lineage(text: str):
    if "lymphoma" in text:
        return "hematolymphoid"
    if "melanoma" in text:
        return "melanocytic"
    if "sarcoma" in text:
        return "mesenchymal"
    if "neuroendocrine" in text:
        return "neuroendocrine"
    if "carcinoma" in text or "adenocarcinoma" in text:
        return "carcinoma"
    if "adenoma" in text or "polyp" in text or "serrated lesion" in text:
        return "epithelial"
    if any(x in text for x in ["gastritis", "colitis", "ileitis", "duodenitis", "proctitis"]):
        return "inflammatory"
    return None


def extract_histologic_type(text: str, vocab=None):
    GENERIC_HISTOLOGY_BLOCKLIST = {
    "metaplasia",
    "intestinal metaplasia",
    "atrophy",
    "dysplasia",
    "hyperplasia",
    "inflammation",
}
    candidates = []

    # Prefer curated pathology patterns first.
    candidates.extend(HISTOLOGY_PATTERNS)

    # Then use learned dataset vocabulary as fallback.
    if vocab:
        candidates.extend(vocab.get("histology_terms", []))

    # Preserve priority order instead of sorting by length.
    seen = set()
    ordered_candidates = []

    for term in candidates:
        term_norm = normalize_text(term)
        if term_norm and term_norm not in seen:
            seen.add(term_norm)
            ordered_candidates.append(term_norm)

    for term_norm in ordered_candidates:
        if term_norm in text and term_norm not in GENERIC_HISTOLOGY_BLOCKLIST:
            return term_norm

    fallback = re.search(
        r"\b(?:acinar|ductal|lobular|mucinous|serous|endometrioid|squamous|urothelial|papillary|tubular|neuroendocrine|clear cell|high-grade serous)"
        r"(?: [a-z]+){0,4} "
        r"(?:carcinoma|adenocarcinoma|tumor|neoplasm)\b",
        text,
    )
    if fallback:
        return fallback.group(0)

    return None

def extract_diagnosis(text: str, histologic_type=None):
    if histologic_type:
        if "adenocarcinoma" in histologic_type:
            return "adenocarcinoma"
        if "carcinoma" in histologic_type:
            return "carcinoma"
        if "lymphoma" in histologic_type:
            return "lymphoma"
        if "melanoma" in histologic_type:
            return "melanoma"
        if "sarcoma" in histologic_type:
            return "sarcoma"
        if "adenoma" in histologic_type:
            return "adenoma"
        if "polyp" in histologic_type:
            return "polyp"

    diagnosis_terms = [
        "adenocarcinoma",
        "carcinoma",
        "lymphoma",
        "melanoma",
        "sarcoma",
        "neuroendocrine tumor",
        "gastritis",
        "colitis",
        "duodenitis",
        "ileitis",
        "proctitis",
        "esophagitis",
        "adenoma",
        "polyp",
        "serrated lesion",
        "hyperplasia",
        "metaplasia",
        "dysplasia",
    ]

    for term in diagnosis_terms:
        if term in text:
            return term

    if (
        "normal histological structure" in text
        or "typical histological structure" in text
        or "typical histological features" in text
        or "normal morphology" in text
    ):
        return "normal mucosa"

    return None

def extract_primary_vs_metastatic(text: str):
    if re.search(r"\bmetastasis|metastases|metastatic\b", text):
        return "metastatic"
    if re.search(r"\bprimary\b", text):
        return "primary"
    return None


def extract_tumor_behavior(text: str):
    fields = {}

    if re.search(r"\binvasive\b|infiltrat|invasion into|with invasion", text):
        fields["in_situ_vs_invasive"] = "invasive"

    if re.search(r"\bcarcinoma in situ\b|\bdcis\b|\bcis\b", text):
        fields["in_situ_component"] = "present"
        if "in_situ_vs_invasive" not in fields:
            fields["in_situ_vs_invasive"] = "in_situ"

    if "ductal carcinoma in situ" in text:
        fields["in_situ_component"] = "dcis"

    return fields


def extract_grade(text: str):
    fields = {}

    gleason = re.search(
        r"gleason(?:\s+score)?(?:\s+of)?\s*"
        r"(?:(\d)\s*\+\s*(\d)\s*=\s*(\d)|"
        r"(\d)\s*\(\s*(\d)\s*\+\s*(\d)\s*\))",
        text,
    )
    if gleason:
        if gleason.group(1):
            primary, secondary, total = gleason.group(1), gleason.group(2), gleason.group(3)
        else:
            total, primary, secondary = gleason.group(4), gleason.group(5), gleason.group(6)

        fields["gleason_score"] = f"{total} ({primary}+{secondary})"

    grade_group = re.search(r"(?:grade group|grading group|isup grade)\s*([ivx]+|\d+)", text)
    if grade_group:
        fields["grade_group"] = roman_to_int(grade_group.group(1))

    tumor_grade = re.search(
        r"\b(?:figo grade|nottingham grade|nottingham histologic score grade|grade|g)\s*([1-3])\b",
        text,
    )
    if tumor_grade:
        fields["tumor_grade"] = tumor_grade.group(1)
    elif "low-grade" in text or "low grade" in text:
        fields["tumor_grade"] = "low"
    elif "high-grade" in text or "high grade" in text:
        fields["tumor_grade"] = "high"

    if "well differentiated" in text or "well-differentiated" in text:
        fields["differentiation"] = "well"
    elif "moderately differentiated" in text or "moderately-differentiated" in text:
        fields["differentiation"] = "moderate"
    elif "poorly differentiated" in text or "poorly-differentiated" in text:
        fields["differentiation"] = "poor"

    return fields


def extract_invasion(text: str):
    fields = {}

    invasion_present = re.search(
        r"\binvasive\b|invasion into|with invasion|infiltrat|invading|invasive growth",
        text,
    )
    invasion_absent = re.search(
        r"without (?:evidence of )?invasion|no (?:definitive )?(?:evidence of )?invasive growth|no signs of invasion",
        text,
    )

    if invasion_present:
        fields["invasion_status"] = "present"
    if invasion_absent:
        fields["invasion_status"] = "absent"

    if re.search(
    r"\blymphovascular invasion\b|with lymphovascular\b|\blymphovascular and perineural invasion\b|\blvi\+|\blvi1\b|\blv1\b|\bl\+\b",
    text,):
        fields["lymphovascular_invasion"] = "present"
    elif re.search(r"without lymphovascular invasion|\blvi\-|\blvi0\b|\blv0\b", text):
        fields["lymphovascular_invasion"] = "absent"

    if re.search(r"\bperineural invasion\b|\bpni\+|\bpni1\b|\bpn1\b|\bpn\+\b", text):
        if not re.search(r"without perineural invasion|no perineural invasion", text):
            fields["perineural_invasion"] = "present"
    elif re.search(r"without perineural invasion|no perineural invasion|\bpni\-|\bpni0\b|\bpn0\b", text):
        fields["perineural_invasion"] = "absent"

    if re.search(r"extraprostatic extension|extracapsular extension", text):
        if re.search(r"without extraprostatic extension|no extraprostatic extension|without extracapsular extension", text):
            fields["extraprostatic_extension"] = "absent"
        else:
            fields["extraprostatic_extension"] = "present"

    depth = re.search(r"invasion depth\s*([0-9.]+)\s*mm", text)
    if depth:
        fields["invasion_depth"] = depth.group(1) + " mm"

    return fields


def extract_dysplasia(text: str):
    fields = {}

    if re.search(r"low-grade dysplasia|low grade dysplasia|cin i\b|lsil", text):
        fields["dysplasia_grade"] = "low"
    elif re.search(r"high-grade dysplasia|high grade dysplasia|cin iii\b|cin 3\b|hsil", text):
        fields["dysplasia_grade"] = "high"
    elif "dysplasia" in text and re.search(r"without dysplasia|no dysplasia", text):
        fields["dysplasia_grade"] = "absent"
    elif "dysplasia" in text:
        fields["dysplasia_grade"] = "present"

    if "atypia" in text or "atypical" in text:
        fields["epithelial_atypia"] = "present"

    return fields


def extract_morphology(text: str):
    fields = {}

    if "necrosis" in text or "comedo-necrosis" in text:
        fields["necrosis"] = "present"

    if "calcification" in text or "microcalcification" in text:
        fields["calcification"] = "present"

    if "multifocal" in text or "multiple foci" in text:
        fields["multifocality"] = "present"

    if "bilateral" in text or "both ovaries" in text or "both breasts" in text:
        fields["bilateral_involvement"] = "present"

    size = re.search(r"(?:largest dimension|maximum size|greatest dimension|tumor size)\s*(?:of|is|:)?\s*([0-9.]+)\s*(cm|mm)", text)
    if size:
        fields["tumor_volume"] = f"{size.group(1)} {size.group(2)}"

    return fields


def extract_inflammation(text: str):
    fields = {}

    inflammatory_terms = [
        "gastritis",
        "colitis",
        "duodenitis",
        "ileitis",
        "proctitis",
        "esophagitis",
        "endocervicitis",
        "inflammation",
    ]

    if re.search(
    r"without inflammatory changes|without evidence of inflammatory bowel disease|no evidence of inflammatory bowel disease|no signs of inflammation|without inflammation",
    text,
):
        fields["inflammation_status"] = "absent"
    elif any(term in text for term in inflammatory_terms):
        fields["inflammation_status"] = "present"

    if "active" in text and "inactive" not in text:
        fields["inflammation_activity"] = "active"
    elif "inactive" in text:
        fields["inflammation_activity"] = "inactive"

    if "gastritis" in text:
        if "chronic" in text and "active" in text and "inactive" not in text:
            fields["gastritis_type"] = "chronic active"
        elif "chronic" in text and "inactive" in text:
            fields["gastritis_type"] = "chronic inactive"
        elif "atrophic gastritis" in text:
            fields["gastritis_type"] = "atrophic"
        else:
            fields["gastritis_type"] = "gastritis"

    return fields


def extract_gi_fields(text: str):
    fields = {}

    if re.search(r"without (?:signs of )?atrophy|without atrophy", text):
        fields["atrophy"] = "absent"
    elif "atrophy" in text or "atrophic" in text:
        fields["atrophy"] = "present"

    if re.search(r"without intestinal metaplasia|no intestinal metaplasia", text):
        fields["intestinal_metaplasia"] = "absent"
    elif "intestinal metaplasia" in text:
        fields["intestinal_metaplasia"] = "present"

    if "complete intestinal metaplasia" in text:
        fields["metaplasia_type"] = "complete"
    elif "incomplete intestinal metaplasia" in text:
        fields["metaplasia_type"] = "incomplete"

    if re.search(r"h\. pylori not detected|hp negative|hp\(-\)|h pylori negative|helicobacter pylori not detected", text):
        fields["hpylori_status"] = "negative"
    elif re.search(r"h\. pylori detected|hp positive|hp\(\+\)|h pylori positive|helicobacter pylori detected", text):
        fields["hpylori_status"] = "positive"

    olga = re.search(r"olga(?:/olgim)?(?:\s*-\s*)?\s*stage\s*([ivx]+|\d+)", text)
    if olga:
        fields["olga_stage"] = roman_to_int(olga.group(1))

    olgim = re.search(r"olgim(?:\s*-\s*)?\s*stage\s*([ivx]+|\d+)", text)
    if olgim:
        fields["olgim_stage"] = roman_to_int(olgim.group(1))

    return fields


def extract_polyp_fields(text: str):
    fields = {}

    if "sessile serrated lesion" in text or "serrated lesion" in text:
        fields["serrated_lesion"] = "present"
        fields["polyp_type"] = "sessile serrated lesion"

    if "tubulovillous adenoma" in text:
        fields["adenoma_type"] = "tubulovillous adenoma"
        fields["polyp_type"] = "adenoma"
    elif "tubular adenoma" in text:
        fields["adenoma_type"] = "tubular adenoma"
        fields["polyp_type"] = "adenoma"

    if "hyperplastic polyp" in text:
        fields["polyp_type"] = "hyperplastic polyp"

    if "fundic gland polyp" in text:
        fields["polyp_type"] = "fundic gland polyp"

    return fields


def extract_fields(report: str, vocab=None):
    text = normalize_text(report)

    fields = {}

    histologic_type = extract_histologic_type(text, vocab=vocab)
    diagnosis = extract_diagnosis(text, histologic_type=histologic_type)

    fields["benign_vs_malignant"] = extract_benign_vs_malignant(text)
    fields["diagnosis"] = diagnosis
    fields["primary_vs_metastatic"] = extract_primary_vs_metastatic(text)
    fields["lineage"] = extract_lineage(text)
    fields["histologic_type"] = histologic_type

    fields.update(extract_tumor_behavior(text))
    fields.update(extract_grade(text))
    fields.update(extract_invasion(text))
    fields.update(extract_dysplasia(text))
    fields.update(extract_morphology(text))
    fields.update(extract_inflammation(text))
    fields.update(extract_gi_fields(text))
    fields.update(extract_polyp_fields(text))

    return {k: v for k, v in fields.items() if v is not None}


def run_smoke_tests(vocab=None):
    examples = [
        {
            "text": "Acinar adenocarcinoma of the prostate gland, Gleason score 3+4=7, WHO/ISUP Grade Group 2, with perineural invasion.",
            "expected": {
                "benign_vs_malignant": "malignant",
                "gleason_score": "7 (3+4)",
                "grade_group": "2",
                "perineural_invasion": "present",
            },
        },
        {
            "text": "Chronic inactive gastritis with intestinal metaplasia. H. pylori not detected. OLGIM Stage I.",
            "expected": {
                "diagnosis": "gastritis",
                "inflammation_status": "present",
                "intestinal_metaplasia": "present",
                "hpylori_status": "negative",
                "olgim_stage": "1",
            },
        },
        {
            "text": "Tubular adenoma of the colon with low-grade dysplasia.",
            "expected": {
                "diagnosis": "adenoma",
                "adenoma_type": "tubular adenoma",
                "dysplasia_grade": "low",
            },
        },
        {
            "text": "Invasive ductal carcinoma of the breast, G2. Ductal carcinoma in situ is also present.",
            "expected": {
                "benign_vs_malignant": "malignant",
                "histologic_type": "invasive ductal carcinoma",
                "in_situ_vs_invasive": "invasive",
                "in_situ_component": "dcis",
                "tumor_grade": "2",
            },
        },
    ]

    all_passed = True

    for i, example in enumerate(examples, start=1):
        extracted = extract_fields(example["text"], vocab=vocab)
        expected = example["expected"]

        passed = True

        for k, v in expected.items():
            got = extracted.get(k)

            if k == "histologic_type":
                if got is None or v not in got:
                    passed = False
                    break
            else:
                if got != v:
                    passed = False
                    break

        all_passed = all_passed and passed

        print(f"\nSmoke test {i}: {'PASS' if passed else 'FAIL'}")
        print("Text:", example["text"])
        print("Extracted:", extracted)
        print("Expected subset:", expected)

    print("\nOverall:", "PASS" if all_passed else "FAIL")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="HistAI clinical field extractor")
    parser.add_argument("--input", type=str, default=None, help="Path to HistAI JSON dataset")
    parser.add_argument("--vocab", type=str, default="outputs/histai_vocab.json", help="Path to save/load vocabulary")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum phrase frequency for learned vocabulary")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests")
    args = parser.parse_args()

    vocab = {"histology_terms": []}

    if args.input:
        reports = load_dataset_reports(args.input)
        print(f"Loaded {len(reports)} reports from {args.input}")

        vocab = learn_vocabulary(reports, min_freq=args.min_freq)
        save_vocabulary(vocab, args.vocab)

        print(f"Saved vocabulary to {args.vocab}")
        print(f"Learned {len(vocab.get('histology_terms', []))} histology terms")

    elif Path(args.vocab).exists():
        vocab = load_vocabulary(args.vocab)
        print(f"Loaded vocabulary from {args.vocab}")

    if args.smoke_test or not args.input:
        run_smoke_tests(vocab=vocab)


if __name__ == "__main__":
    main()