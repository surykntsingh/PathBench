# src/extract_fields.py

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


from crqs.crqs_tcga.src.config import CLINICAL_FIELDS, KEY_FIELDS, MISSING_VALUES


VOCAB_VERSION = "tcga_rule_based_v1"


DIAGNOSIS_TERMS = [
    "carcinoma",
    "adenocarcinoma",
    "serous carcinoma",
    "papillary carcinoma",
    "papillary serous carcinoma",
    "endometrioid adenocarcinoma",
    "papillary thyroid carcinoma",
    "classic seminoma",
    "endometrial adenocarcinoma",
    "clear cell carcinoma",
    "seminoma",
    "germ cell tumor",
    "pheochromocytoma",
    "paraganglioma",
    "papillary thyroid carcinoma",
    "follicular variant",
    "medullary thyroid carcinoma",
    "glioma",
    "astrocytoma",
    "oligodendroglioma",
    "oligoastrocytoma",
    "glioblastoma",
    "sarcoma",
    "melanoma",
    "lymphoma",
    "benign",
    "no malignancy",
    "no tumor",
]

HISTOLOGY_MODIFIERS = [
    "serous",
    "papillary",
    "endometrioid",
    "clear cell",
    "mucinous",
    "follicular",
    "tall cell",
    "classic",
    "anaplastic",
    "diffuse",
    "fibrillary",
    "gemistocytic",
    "mixed",
    "medullary",
    "seminoma",
]

LINEAGE_TERMS = [
    "epithelial",
    "carcinoma",
    "adenocarcinoma",
    "germ cell",
    "seminoma",
    "glial",
    "glioma",
    "astrocytic",
    "oligodendroglial",
    "thyroid",
    "neuroendocrine",
    "pheochromocytoma",
    "paraganglioma",
]


NEGATION_RE = re.compile(
    r"\b(no|not|without|negative for|free of|absence of|not identified|not seen|not present|uninvolved by)\b",
    re.I,
)


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES
    if isinstance(value, list):
        return len(value) == 0
    return False


def window_has_negation(text: str, start: int, window: int = 45) -> bool:
    prefix = text[max(0, start - window):start]
    return bool(NEGATION_RE.search(prefix))


def find_terms(text: str, terms: List[str]) -> List[str]:
    text_l = text.lower()
    hits = []
    for term in sorted(set(terms), key=len, reverse=True):
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        for m in re.finditer(pattern, text_l):
            if not window_has_negation(text_l, m.start()):
                hits.append(term.lower())
                break
    return sorted(set(hits))


def extract_presence(text: str, positive_terms: List[str], negative_terms: Optional[List[str]] = None) -> Optional[str]:
    text_l = text.lower()

    if negative_terms:
        for term in negative_terms:
            if re.search(r"\b" + re.escape(term.lower()) + r"\b", text_l):
                return "absent"

    for term in positive_terms:
        for m in re.finditer(r"\b" + re.escape(term.lower()) + r"\b", text_l):
            if window_has_negation(text_l, m.start()):
                return "absent"
            return "present"

    return None


def learn_vocabulary(dataset: Dict[str, Any], min_count: int = 2) -> Dict[str, Any]:
    reports = []
    for split, rows in dataset.items():
        if isinstance(rows, list):
            reports.extend([r.get("report", "") for r in rows if isinstance(r, dict)])

    text = " ".join(normalize_text(r).lower() for r in reports)

    candidate_terms = DIAGNOSIS_TERMS + HISTOLOGY_MODIFIERS + LINEAGE_TERMS
    counts = Counter()

    for term in candidate_terms:
        counts[term] = len(re.findall(r"\b" + re.escape(term.lower()) + r"\b", text))

    diagnosis_vocab = sorted([k for k, v in counts.items() if v >= min_count and k in DIAGNOSIS_TERMS])
    histology_vocab = sorted([k for k, v in counts.items() if v >= min_count and k in HISTOLOGY_MODIFIERS])
    lineage_vocab = sorted([k for k, v in counts.items() if v >= min_count and k in LINEAGE_TERMS])

    return {
        "version": VOCAB_VERSION,
        "min_count": min_count,
        "diagnosis_vocab": diagnosis_vocab,
        "histology_vocab": histology_vocab,
        "lineage_vocab": lineage_vocab,
        "term_counts": dict(counts),
    }


def extract_benign_vs_malignant(text: str) -> Optional[str]:
    text_l = text.lower()

    malignant_terms = [
        "malignant",
        "carcinoma",
        "adenocarcinoma",
        "sarcoma",
        "seminoma",
        "metastatic",
        "glioblastoma",
        "anaplastic",
        "high-grade",
        "high grade",
    ]
    benign_terms = [
        "benign",
        "no malignancy",
        "no evidence of malignancy",
        "negative for malignancy",
        "no tumor",
        "no neoplasm",
    ]

    for term in benign_terms:
        if term in text_l:
            return "benign_or_negative"

    for term in malignant_terms:
        if term in text_l:
            return "malignant"

    if "neoplasm" in text_l or "tumor" in text_l:
        return "neoplasm_unspecified"

    return None


def extract_primary_vs_metastatic(text: str) -> Optional[str]:
    text_l = text.lower()
    if re.search(r"\bmetastatic\b|\bmetastasis\b|\bmetastases\b", text_l):
        return "metastatic"
    if re.search(r"\bprimary\b", text_l):
        return "primary"
    return None


def extract_grade(text: str) -> Optional[str]:
    text_l = text.lower()

    patterns = [
        r"\bwho\s*grade\s*(i{1,3}|iv|v?i{0,3}|[1-4])\b",
        r"\bgrade\s*(i{1,3}|iv|[1-4])\b",
        r"\bgrade\s*([1-4])\s*(?:of|/)\s*4\b",
        r"\b(high[- ]grade|low[- ]grade|intermediate[- ]grade)\b",
    ]

    for pat in patterns:
        m = re.search(pat, text_l, re.I)
        if m:
            value = m.group(1).lower()
            value = value.replace(" ", "-")
            return value

    return None


def extract_differentiation(text: str) -> Optional[str]:
    text_l = text.lower()
    for value in ["poorly differentiated", "moderately differentiated", "well differentiated", "undifferentiated"]:
        if value in text_l:
            return value
    return None


def extract_tumor_size(text: str) -> Optional[str]:
    text_l = text.lower()
    m = re.search(
        r"\b(?:tumor|mass|nodule|lesion)?\s*(?:measures|measuring|size|diameter)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?\s*){0,2}(cm|mm)\b",
        text_l,
    )
    if m:
        return m.group(0).strip()
    return None


def extract_focality(text: str) -> Optional[str]:
    text_l = text.lower()
    if "multifocal" in text_l:
        return "multifocal"
    if "unifocal" in text_l or "solitary" in text_l:
        return "unifocal"
    return None


def extract_invasion_status(text: str) -> Optional[str]:
    text_l = text.lower()

    invasion_terms = [
        "invasion",
        "invades",
        "invasive",
        "infiltrating",
        "infiltrative",
        "extends into",
        "extension into",
        "involvement of",
    ]

    absence_terms = [
        "no invasion",
        "without invasion",
        "no evidence of invasion",
        "not invasive",
    ]

    return extract_presence(text_l, invasion_terms, absence_terms)


def extract_specific_invasion(text: str, target: str) -> Optional[str]:
    patterns = {
        "lymphovascular_invasion": ["lymphovascular invasion", "lymphatic invasion", "vascular invasion", "venous invasion"],
        "perineural_invasion": ["perineural invasion"],
        "capsular_invasion": ["capsular invasion", "capsule invasion", "tumor capsule"],
        "organ_specific_invasion": [
            "myometrial invasion",
            "serosal involvement",
            "extrathyroidal extension",
            "rete testis invasion",
            "tunica albuginea",
            "tunica vaginalis",
            "cervical stromal invasion",
        ],
    }
    return extract_presence(text, patterns[target])


def extract_metastatic_involvement(text: str) -> Optional[str]:
    return extract_presence(
        text,
        ["metastatic", "metastasis", "metastases", "metastatic carcinoma", "involved by carcinoma"],
        ["no metastasis", "no metastatic", "negative for metastatic"],
    )


def extract_lymph_node_involvement(text: str) -> Optional[str]:
    text_l = text.lower()
    if re.search(r"\b\d+\s*(?:of|/)\s*\d+\s*lymph nodes?\s*(?:positive|involved)", text_l):
        return "present"
    if re.search(r"lymph nodes?.{0,40}(negative|benign|uninvolved|no tumor|no carcinoma)", text_l):
        return "absent"
    if re.search(r"lymph nodes?.{0,40}(metastatic|involved|positive)", text_l):
        return "present"
    return None


def extract_binary_morphology(text: str, field: str) -> Optional[str]:
    terms = {
        "necrosis": ["necrosis", "tumor necrosis", "coagulative necrosis"],
        "inflammation_status": ["inflammation", "inflammatory", "thyroiditis", "salpingitis", "cervicitis"],
        "calcification": ["calcification", "calcifications", "microcalcification", "microcalcifications"],
        "psammoma_bodies": ["psammoma body", "psammoma bodies"],
        "hemorrhage": ["hemorrhage", "hemorrhagic", "bleeding"],
        "papillary_features": ["papillary"],
        "cystic_change": ["cystic", "microcystic", "cystic change"],
        "carcinoma_in_situ": ["carcinoma in situ", "in situ carcinoma"],
        "intratubular_germ_cell_neoplasia": ["intratubular germ cell neoplasia", "igcnu"],
        "epithelial_atypia": ["epithelial atypia", "atypia", "atypical"],
       
    }
    if field not in terms:
        return None
    return extract_presence(text, terms[field])


def extract_fields(report: str, vocab: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = normalize_text(report)
    vocab = vocab or {}

    diagnosis_vocab = vocab.get("diagnosis_vocab", DIAGNOSIS_TERMS)
    histology_vocab = vocab.get("histology_vocab", HISTOLOGY_MODIFIERS)
    lineage_vocab = vocab.get("lineage_vocab", LINEAGE_TERMS)

    facts = {field: None for field in CLINICAL_FIELDS}

    diagnoses = find_terms(text, diagnosis_vocab)
    histology = find_terms(text, histology_vocab)
    lineage = find_terms(text, lineage_vocab)

    facts["benign_vs_malignant"] = extract_benign_vs_malignant(text)
    facts["diagnosis"] = max(diagnoses, key=len) if diagnoses else None
    facts["primary_vs_metastatic"] = extract_primary_vs_metastatic(text)
    facts["lineage"] = lineage[0] if lineage else None
    facts["histologic_type"] = ", ".join(histology) if histology else None

    facts["in_situ_vs_invasive"] = extract_presence(
        text,
        ["invasive", "infiltrating", "infiltrative", "invades"],
        ["in situ", "non-invasive", "noninvasive"],
    )
    facts["in_situ_component"] = extract_binary_morphology(text, "carcinoma_in_situ")
    facts["tumor_focality"] = extract_focality(text)

    facts["tumor_grade"] = extract_grade(text)
    facts["differentiation"] = extract_differentiation(text)

    facts["invasion_status"] = extract_invasion_status(text)
    facts["lymphovascular_invasion"] = extract_specific_invasion(text, "lymphovascular_invasion")
    facts["perineural_invasion"] = extract_specific_invasion(text, "perineural_invasion")
    facts["capsular_invasion"] = extract_specific_invasion(text, "capsular_invasion")
    facts["organ_specific_invasion"] = extract_specific_invasion(text, "organ_specific_invasion")

    facts["metastatic_involvement"] = extract_metastatic_involvement(text)
    facts["lymph_node_involvement"] = extract_lymph_node_involvement(text)

    if facts["tumor_focality"] == "multifocal":
        facts["multifocal_involvement"] = "present"

    facts["dysplasia_grade"] = None
    facts["epithelial_atypia"] = extract_binary_morphology(text, "epithelial_atypia")
    facts["carcinoma_in_situ"] = extract_binary_morphology(text, "carcinoma_in_situ")
    facts["intratubular_germ_cell_neoplasia"] = extract_binary_morphology(text, "intratubular_germ_cell_neoplasia")

    for field in [
        "necrosis",
        "inflammation_status",
        "calcification",
        "psammoma_bodies",
        "hemorrhage",
        "papillary_features",
        "cystic_change",
    ]:
        facts[field] = extract_binary_morphology(text, field)

    facts["tumor_size"] = extract_tumor_size(text)

    facts["organ_site"] = extract_organ_site(text)
    facts["specimen_type"] = extract_specimen_type(text)
    facts["margin_status"] = extract_margin_status(text)

    return {k: v for k, v in facts.items() if not is_missing(v)}


def extract_organ_site(text: str) -> Optional[str]:
    sites = [
        "ovary", "ovarian", "uterus", "endometrium", "cervix",
        "thyroid", "testis", "testicle", "adrenal",
        "brain", "omentum", "bowel", "fallopian tube",
    ]
    hits = find_terms(text, sites)
    return ", ".join(hits) if hits else None


def extract_specimen_type(text: str) -> Optional[str]:
    specimens = [
        "hysterectomy",
        "thyroidectomy",
        "orchiectomy",
        "adrenalectomy",
        "biopsy",
        "resection",
        "excision",
        "oophorectomy",
        "salpingo-oophorectomy",
    ]
    hits = find_terms(text, specimens)
    return ", ".join(hits) if hits else None


def extract_margin_status(text: str) -> Optional[str]:
    text_l = text.lower()
    if re.search(r"margins?.{0,40}(uninvolved|negative|free of tumor|free of carcinoma)", text_l):
        return "negative"
    if re.search(r"margins?.{0,40}(involved|positive)", text_l):
        return "positive"
    return None


def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vocabulary(vocab: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def load_vocabulary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def smoke_test() -> None:
    reports = [
        "Right ovary: poorly differentiated papillary serous carcinoma with omental metastasis and necrosis.",
        "Left testis orchiectomy shows classic seminoma, 1.7 cm, confined to testis, no lymphovascular invasion.",
        "Total thyroidectomy reveals multifocal papillary thyroid carcinoma, follicular variant, with capsular invasion.",
        "Endometrial adenocarcinoma with myometrial invasion. Margins negative.",
        "The pathology report is incomplete and does not provide tumor diagnosis.",
    ]

    vocab = {
        "diagnosis_vocab": DIAGNOSIS_TERMS,
        "histology_vocab": HISTOLOGY_MODIFIERS,
        "lineage_vocab": LINEAGE_TERMS,
    }

    for i, report in enumerate(reports, 1):
        print(f"\n--- Smoke test {i} ---")
        print(report)
        print(json.dumps(extract_fields(report, vocab), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to TCGA reports JSON")
    parser.add_argument("--vocab_out", type=str, default="outputs/tcga_vocab.json")
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test or not args.input:
        smoke_test()
        return

    dataset = load_dataset(args.input)
    vocab = learn_vocabulary(dataset, min_count=args.min_count)
    save_vocabulary(vocab, args.vocab_out)

    print(f"Saved vocabulary to: {args.vocab_out}")
    print(f"Diagnosis terms: {len(vocab['diagnosis_vocab'])}")
    print(f"Histology terms: {len(vocab['histology_vocab'])}")
    print(f"Lineage terms: {len(vocab['lineage_vocab'])}")


if __name__ == "__main__":
    main()
