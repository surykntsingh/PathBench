import json
import re
from collections import Counter
from pathlib import Path


# --------------------------------------------------
# Basic text utilities
# --------------------------------------------------

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def search(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.I)
    return m.group(1).strip() if m else None


def keyword_present(keywords, text: str) -> bool:
    return any(k in text for k in keywords)


def normalize_phrase(text: str) -> str:
    text = clean_text(text)
    text = text.strip(" .,:;-")
    text = re.sub(r"\s+", " ", text)
    return text


# --------------------------------------------------
# Dataset-driven vocabulary learning
# --------------------------------------------------

REMOVE_DIAGNOSIS_MODIFIERS = [
    r"\([^)]*\)",                         # remove all parenthetical fragments
    r"\bgleason'?s score\s+\d+.*",         # remove Gleason and everything after
    r"\bgrade group\s+\d+.*",
    r"\bgrade\s+[ivx\d]+.*",
    r"\bnuclear grade:\s*[a-z\d]+.*",
    r"\bmitoses:\s*\d+.*",
    r"\btubule formation:\s*\d+.*",
    r"\bnecrosis:\s*(present|absent).*",
    r"\btype:\s*[a-z\s\-]+.*",

    r"\bwith low grade dysplasia\b",
    r"\bwith high grade dysplasia\b",
    r"\blow grade dysplasia\b",
    r"\bhigh grade dysplasia\b",
    r"\bwell differentiated\b",
    r"\bmoderately differentiated\b",
    r"\bpoorly differentiated\b",
    r"\bwith involvement of muscle proper\b",
    r"\bwith involvement of subepithelial connective tissue\b",
    r"\bnote\).*",
]


def get_diagnosis_section(report_text: str) -> str:
    """
    REG reports usually look like:
        Organ, specimen; diagnosis text

    Split at the first semicolon that appears before the diagnosis section.
    Avoid semicolons inside parentheses such as (HSIL; CIN 3).
    """
    if not report_text:
        return ""

    text = str(report_text)

    # Prefer splitting after the specimen phrase before the diagnosis
    m = re.search(r"^[^;]+;\s*(.*)$", text, flags=re.S)
    if m:
        return m.group(1)

    return text


def remove_numbered_prefixes(text: str) -> str:
    """
    Removes list numbering such as:
        1. Invasive carcinoma
        2. DCIS
    """
    text = re.sub(r"\b\d+\.\s*", " ", text)
    text = re.sub(r"\b\d+\)\s*", " ", text)
    return text


def strip_diagnosis_modifiers(text: str) -> str:
    text = clean_text(text)
    text = remove_numbered_prefixes(text)

    # Remove all parenthetical fragments like (HSIL; CIN 3)
    text = re.sub(r"\([^)]*\)", " ", text)

    # Remove parenthetical grading fragments, but keep meaningful phrases elsewhere.
    text = re.sub(r"\([^)]*tubule formation[^)]*\)", " ", text)
    text = re.sub(r"\([^)]*gleason pattern[^)]*\)", " ", text)

    for pattern in REMOVE_DIAGNOSIS_MODIFIERS:
        text = re.sub(pattern, " ", text, flags=re.I)

    # Split on line/list separators but preserve meaningful diagnosis chunks.
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;-")

    return text


def split_candidate_phrases(text: str) -> list[str]:
    """
    Converts the cleaned diagnosis section into candidate phrases.

    Example:
        'Invasive carcinoma of no special type, grade II ... 2. DCIS'
    becomes candidates like:
        ['invasive carcinoma of no special type', 'ductal carcinoma in situ']
    """
    text = strip_diagnosis_modifiers(text)

    separators = [
        r"\s+-\s+",
        r"\s+with\s+1\)",
        r"\s+with\s+2\)",
        r"\s{2,}",
    ]

    for sep in separators:
        text = re.sub(sep, " | ", text)

    raw_parts = re.split(r"\||\n", text)

    candidates = []
    for part in raw_parts:
        part = normalize_phrase(part)

        if not part:
            continue

        # Remove weak note fragments.
        if part.startswith("note"):
            continue
        if part in {"present", "absent", "included", "not included"}:
            continue

        # Remove very short non-diagnostic fragments.
        if len(part) < 4:
            continue

        candidates.append(part)

    return candidates


def learn_vocabulary_from_records(records: dict, min_count: int = 1) -> dict:
    """
    Learns diagnosis/histology vocabulary from both target and prediction reports.

    Input:
        records = {
            "case_id": {
                "pred": "...",
                "target": "..."
            }
        }

    Output:
        {
            "diagnosis_terms": [...],
            "histologic_terms": [...]
        }
    """
    counter = Counter()

    for item in records.values():
        for key in ["target", "pred"]:
            report = item.get(key, "")
            dx_section = get_diagnosis_section(report)
            candidates = split_candidate_phrases(dx_section)

            for phrase in candidates:
                counter[phrase] += 1

    terms = [
        term
        for term, count in counter.items()
        if count >= min_count
    ]

    # Prefer longer / more specific terms before shorter terms.
    terms = sorted(terms, key=lambda x: (-len(x), x))

    return {
        "diagnosis_terms": terms,
        "histologic_terms": terms,
    }


def learn_vocabulary_from_json(json_path: str | Path, min_count: int = 1) -> dict:
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        records = json.load(f)

    return learn_vocabulary_from_records(records, min_count=min_count)


def save_vocabulary(vocab: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocabulary(vocab_path: str | Path) -> dict:
    vocab_path = Path(vocab_path)

    with open(vocab_path, "r") as f:
        return json.load(f)


def first_vocab_match(terms: list[str], text: str):
    """
    Returns first vocabulary term found in text.
    Terms should already be sorted from most specific to least specific.
    """
    for term in terms:
        if term and term in text:
            return term
    return None


# --------------------------------------------------
# Higher-level clinical classification helpers
# --------------------------------------------------

MALIGNANT_KEYWORDS = [
    "carcinoma",
    "adenocarcinoma",
    "lymphoma",
    "melanoma",
    "sarcoma",
    "small cell carcinoma",
    "neuroendocrine tumor",
    "malignant",
    "micro-invasive carcinoma",
    "invasive carcinoma",
]

PREMALIGNANT_KEYWORDS = [
    "ductal carcinoma in situ",
    "lobular carcinoma in situ",
    "urothelial carcinoma in situ",
    "carcinoma in situ",
    "hsil",
    "lsil",
    "cin 1",
    "cin 2",
    "cin 3",
    "adenoma",
    "dysplasia",
    "atypical ductal hyperplasia",
    "flat epithelial atypia",
]

BENIGN_OR_NEGATIVE_KEYWORDS = [
    "no tumor present",
    "no evidence of tumor",
    "no evidence of malignancy",
    "benign",
    "fibroadenoma",
    "fibrocystic change",
    "sclerosing adenosis",
    "hyperplastic polyp",
    "chronic gastritis",
    "chronic colitis",
    "inflammation",
    "papilloma",
]


def classify_benign_vs_malignant(text: str):
    if keyword_present(MALIGNANT_KEYWORDS, text):
        return "malignant"

    if keyword_present(PREMALIGNANT_KEYWORDS, text):
        return "premalignant"

    if keyword_present(BENIGN_OR_NEGATIVE_KEYWORDS, text):
        return "benign"

    return None


def classify_primary_vs_metastatic(text: str):
    if "metastatic" in text or "from colon primary" in text or "favor colorectal primary" in text:
        return "metastatic"
    if keyword_present(MALIGNANT_KEYWORDS, text):
        return "primary_or_unspecified"
    return None


def classify_lineage(text: str):
    if "lymphoma" in text:
        return "lymphoma"
    if "melanoma" in text:
        return "melanoma"
    if "sarcoma" in text:
        return "sarcoma"
    if "urothelial carcinoma" in text:
        return "urothelial carcinoma"
    if "carcinoma" in text or "adenocarcinoma" in text:
        return "carcinoma"
    if "neuroendocrine tumor" in text:
        return "neuroendocrine"
    if "adenoma" in text or "polyp" in text:
        return "epithelial polyp/adenoma"
    if "inflammation" in text or "gastritis" in text or "colitis" in text:
        return "inflammatory"
    return None


# --------------------------------------------------
# Main extraction function
# --------------------------------------------------

def extract_fields(report_text: str, vocab: dict | None = None) -> dict:
    """
    Dataset-driven clinical field extractor for REG pathology reports.

    Uses:
        1. learned vocabulary for diagnosis / histologic_type
        2. stable regex rules for structured clinical fields

    Input:
        report_text: raw pathology report
        vocab: optional learned vocabulary dictionary

    Output:
        dict of extracted clinical facts
    """

    raw = report_text if report_text else ""
    text = clean_text(raw)
    facts = {}

    # --------------------------------------------------
    # 1. Contextual metadata: organ site + specimen type
    # Not key visual fields, but useful for analysis.
    # --------------------------------------------------
    first_part = raw.split(";")[0].strip()

    if "," in first_part:
        organ, specimen = first_part.split(",", 1)
        facts["organ_site"] = clean_text(organ)
        facts["specimen_type"] = clean_text(specimen)
    elif first_part:
        facts["organ_site"] = clean_text(first_part)

    # --------------------------------------------------
    # 2. Diagnosis hierarchy
    # --------------------------------------------------
    bvm = classify_benign_vs_malignant(text)
    if bvm:
        facts["benign_vs_malignant"] = bvm

    pvm = classify_primary_vs_metastatic(text)
    if pvm:
        facts["primary_vs_metastatic"] = pvm

    lineage = classify_lineage(text)
    if lineage:
        facts["lineage"] = lineage

    if vocab:
        diagnosis_terms = vocab.get("diagnosis_terms", [])
        histologic_terms = vocab.get("histologic_terms", [])

        diagnosis = first_vocab_match(diagnosis_terms, text)
        histologic_type = first_vocab_match(histologic_terms, text)

        if diagnosis:
            facts["diagnosis"] = diagnosis

        if histologic_type:
            facts["histologic_type"] = histologic_type
            # Normalize cervical squamous intraepithelial lesion terms
            if "high-grade squamous intraepithelial lesion" in text:
                facts["diagnosis"] = "high-grade squamous intraepithelial lesion"
                facts["histologic_type"] = "high-grade squamous intraepithelial lesion"

            elif "low-grade squamous intraepithelial lesion" in text:
                facts["diagnosis"] = "low-grade squamous intraepithelial lesion"
                facts["histologic_type"] = "low-grade squamous intraepithelial lesion"

    else:
        # Fallback if no vocabulary is passed.
        dx_section = get_diagnosis_section(raw)
        candidates = split_candidate_phrases(dx_section)
        if candidates:
            facts["diagnosis"] = candidates[0]
            facts["histologic_type"] = candidates[0]

    # --------------------------------------------------
    # 3. Tumor behavior: in situ vs invasive
    # --------------------------------------------------
    if keyword_present(
        [
            "ductal carcinoma in situ",
            "lobular carcinoma in situ",
            "urothelial carcinoma in situ",
            "carcinoma in situ",
            "endocervical adenocarcinoma in situ",
            "hsil",
            "lsil",
            "cin 1",
            "cin 2",
            "cin 3",
        ],
        text,
    ):
        facts["in_situ_component"] = "present"

    if "non-invasive" in text:
        facts["in_situ_vs_invasive"] = "non-invasive"
    elif "in situ" in text or "carcinoma in situ" in text or "hsil" in text or "lsil" in text:
        facts["in_situ_vs_invasive"] = "in situ"
    elif "invasive" in text or "micro-invasive" in text:
        facts["in_situ_vs_invasive"] = "invasive"

    # --------------------------------------------------
    # 4. Grade / aggressiveness
    # --------------------------------------------------
    grade = search(r"\bgrade\s+([ivx]+|\d+)\b", text)
    if grade:
        facts["tumor_grade"] = grade.lower()

    for diff in [
        "well differentiated",
        "moderately differentiated",
        "poorly differentiated",
    ]:
        if diff in text:
            facts["differentiation"] = diff
            break

    gleason = search(r"gleason'?s score\s+(\d+\s*\(\s*\d\s*\+\s*\d\s*\))", text)
    if gleason:
        gleason = re.sub(r"\s+", "", gleason)
        gleason = gleason.replace("(", " (")
        facts["gleason_score"] = gleason

        total = search(r"gleason'?s score\s+(\d+)", text)
        primary = search(r"gleason'?s score\s+\d+\s*\(\s*(\d)\s*\+", text)
        secondary = search(r"gleason'?s score\s+\d+\s*\(\s*\d\s*\+\s*(\d)\s*\)", text)

        if total:
            facts["gleason_total"] = total
        if primary:
            facts["gleason_primary"] = primary
        if secondary:
            facts["gleason_secondary"] = secondary

    grade_group = search(r"\bgrade group\s+(\d+)\b", text)
    if grade_group:
        facts["grade_group"] = grade_group

    gp4 = search(r"gleason pattern 4:\s*(\d+%)", text)
    if gp4:
        facts["gleason_pattern_4_percent"] = gp4

    nuclear_grade = search(r"nuclear grade:\s*([a-z]+|\d+)", text)
    if nuclear_grade:
        facts["nuclear_grade"] = nuclear_grade

    mitotic_score = search(r"mitoses:\s*(\d+)", text)
    if mitotic_score:
        facts["mitotic_score"] = mitotic_score

    tubule_score = search(r"tubule formation:\s*(\d+)", text)
    if tubule_score:
        facts["tubule_score"] = tubule_score

    # --------------------------------------------------
    # 5. Invasion hierarchy
    # --------------------------------------------------
    if "involvement of muscle proper" in text:
        facts["invasion_status"] = "invasive"
        facts["invasion_depth"] = "muscle proper"
        facts["bladder_invasion_depth"] = "muscle proper"
    elif "involvement of subepithelial connective tissue" in text:
        facts["invasion_status"] = "invasive"
        facts["invasion_depth"] = "subepithelial connective tissue"
        facts["bladder_invasion_depth"] = "subepithelial connective tissue"
    elif "non-invasive" in text:
        facts["invasion_status"] = "non-invasive"
        facts["invasion_depth"] = "non-invasive"
    elif "invasive" in text:
        facts["invasion_status"] = "invasive"

    if "lymphovascular invasion" in text or "lymphatic invasion" in text:
        if "no lymphovascular invasion" in text or "without lymphovascular invasion" in text:
            facts["lymphovascular_invasion"] = "absent"
        else:
            facts["lymphovascular_invasion"] = "present"

    if "perineural invasion" in text:
        if "no perineural invasion" in text or "without perineural invasion" in text:
            facts["perineural_invasion"] = "absent"
        else:
            facts["perineural_invasion"] = "present"

    # --------------------------------------------------
    # 6. Premalignant / epithelial changes
    # --------------------------------------------------
    if "high grade dysplasia" in text:
        facts["dysplasia_grade"] = "high"
    elif "low grade dysplasia" in text:
        facts["dysplasia_grade"] = "low"
    elif "hsil" in text or "high-grade squamous intraepithelial lesion" in text:
        facts["dysplasia_grade"] = "high"
    elif "lsil" in text or "low-grade squamous intraepithelial lesion" in text:
        facts["dysplasia_grade"] = "low"

    cin = search(r"\bcin\s*(\d)\b", text)
    if cin:
        facts["cin_grade"] = cin

    if "atypical ductal hyperplasia" in text:
        facts["epithelial_atypia"] = "atypical ductal hyperplasia"
    elif "flat epithelial atypia" in text:
        facts["epithelial_atypia"] = "flat epithelial atypia"

    # --------------------------------------------------
    # 7. Associated morphology
    # --------------------------------------------------
    if "necrosis: present" in text:
        facts["necrosis"] = "present"
    elif "necrosis: absent" in text:
        facts["necrosis"] = "absent"
    elif "without necrosis" in text:
        facts["necrosis"] = "absent"
    elif "with necrosis" in text or "comedo-type" in text:
        facts["necrosis"] = "present"

    if "microcalcification" in text or "calcification" in text:
        facts["calcification"] = "present"

    tumor_volume = search(r"tumor volume:\s*(\d+%)", text)
    if tumor_volume:
        facts["tumor_volume"] = tumor_volume

    if "chronic active gastritis" in text:
        facts["inflammation_status"] = "chronic active gastritis"
    elif "chronic gastritis" in text:
        facts["inflammation_status"] = "chronic gastritis"
    elif "chronic active colitis" in text:
        facts["inflammation_status"] = "chronic active colitis"
    elif "chronic nonspecific inflammation" in text:
        facts["inflammation_status"] = "chronic nonspecific inflammation"
    elif "chronic granulomatous inflammation" in text:
        facts["inflammation_status"] = "chronic granulomatous inflammation"

    if "intestinal metaplasia" in text or "interstinal metaplasia" in text or "nterstinal metaplasia" in text:
        facts["intestinal_metaplasia"] = "present"

    if "erosion" in text:
        facts["erosion"] = "present"

    if "lymphoid aggregates" in text:
        facts["lymphoid_aggregates"] = "present"

    if "foveolar epithelial hyperplasia" in text:
        facts["foveolar_hyperplasia"] = "present"

    # --------------------------------------------------
    # 8. Bladder muscle proper adequacy
    # Contextual/sampling field, not a key visual field.
    # Important: check negative phrase first.
    # --------------------------------------------------
    if "does not include muscle proper" in text:
        facts["muscle_proper_status"] = "not included"
    elif "includes muscle proper" in text:
        facts["muscle_proper_status"] = "included"

    return facts


# --------------------------------------------------
# CLI / testing
# --------------------------------------------------

if __name__ == "__main__":
    data_path = Path("data/predictions_reg.json")
    vocab_path = Path("outputs/reg_vocabulary.json")

    if data_path.exists():
        print(f"Learning vocabulary from {data_path}...")
        vocab = learn_vocabulary_from_json(data_path, min_count=1)
        save_vocabulary(vocab, vocab_path)
        print(f"Saved vocabulary to {vocab_path}")
        print(f"Learned {len(vocab['diagnosis_terms'])} diagnosis terms.")
    else:
        print("No dataset found. Running examples without learned vocabulary.")
        vocab = None

    examples = [
        "Breast, biopsy; Invasive carcinoma of no special type, grade II (Tubule formation: 3, Nuclear grade: 2, Mitoses: 1)",
        "Prostate, biopsy; Acinar adenocarcinoma, Gleason's score 7 (4+3), grade group 3 (Gleason pattern 4: 70%), tumor volume: 80%",
        "Urinary bladder, transurethral resection; Invasive urothelial carcinoma, with involvement of muscle proper Note) The specimen includes muscle proper.",
        "Stomach, endoscopic biopsy; Tubular adenoma with low grade dysplasia",
        "Uterine cervix, colposcopic biopsy; High-grade squamous intraepithelial lesion (HSIL; CIN 3)",
    ]

    for e in examples:
        print("\nREPORT:")
        print(e)
        print("EXTRACTED:")
        print(extract_fields(e, vocab=vocab))