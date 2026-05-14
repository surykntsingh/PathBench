# src/config.py

"""
Dataset-specific CRQS configuration for HistAI pathology reports.

This schema prioritizes clinically important, histology-inferable facts.
It intentionally avoids over-weighting metadata-heavy fields such as
organ site, specimen type, laterality, margins, lymph nodes, and stage.
"""

# -------------------------
# Clinical fact schema
# -------------------------

CLINICAL_FIELDS = [

    # 1. Diagnosis hierarchy
    "benign_vs_malignant",
    "diagnosis",
    "primary_vs_metastatic",
    "lineage",
    "histologic_type",

    # 2. Tumor behavior
    "in_situ_vs_invasive",
    "in_situ_component",

    # 3. Grade / aggressiveness
    "tumor_grade",
    "differentiation",
    "gleason_score",
    "grade_group",
    "nuclear_grade",
    "mitotic_score",

    # 4. Invasion hierarchy
    "invasion_status",
    "invasion_depth",
    "lymphovascular_invasion",
    "perineural_invasion",
    "extraprostatic_extension",

    # 5. Premalignant / epithelial change
    "dysplasia_grade",
    "epithelial_atypia",

    # 6. Associated morphology
    "necrosis",
    "inflammation_status",
    "inflammation_activity",
    "calcification",
    "tumor_extent",
    "tumor_volume",

    # 7. GI-specific mucosal pathology
    "atrophy",
    "intestinal_metaplasia",
    "metaplasia_type",
    "hpylori_status",
    "gastritis_type",
    "olga_stage",
    "olgim_stage",

    # 8. Polyp / adenoma pathology
    "polyp_type",
    "serrated_lesion",
    "adenoma_type",

    # 9. Multifocal / bilateral disease
    "multifocality",
    "bilateral_involvement",
]


# -------------------------
# Key fields for KIR
# -------------------------

KEY_FIELDS = [
    "benign_vs_malignant",
    "diagnosis",
    "histologic_type",
    "in_situ_vs_invasive",
    "tumor_grade",
    "gleason_score",
    "dysplasia_grade",
    "invasion_status",
    "intestinal_metaplasia",
    "inflammation_status",
]


# -------------------------
# CRQS weights
# -------------------------
# Raw formula:
# CRQS_raw = 0.3*CFC + 0.4*KIR - 0.2*HR - 0.4*CDS
#
# Maximum possible raw score is 0.7 when:
# CFC = 1, KIR = 1, HR = 0, CDS = 0
#
# Normalized formula:
# CRQS_norm = CRQS_raw / 0.7

CRQS_WEIGHTS = {
    "CFC": 0.3,
    "KIR": 0.4,
    "HR": -0.2,
    "CDS": -0.4,
}

CRQS_MAX_RAW = 0.7


# -------------------------
# Strict fields
# -------------------------
# These should be compared strictly because small differences are clinically meaningful.

STRICT_FIELDS = [
    "benign_vs_malignant",
    "diagnosis",
    "histologic_type",
    "in_situ_vs_invasive",
    "gleason_score",
    "grade_group",
    "tumor_grade",
    "dysplasia_grade",
    "lymphovascular_invasion",
    "perineural_invasion",
    "extraprostatic_extension",
    "hpylori_status",
]


# -------------------------
# Numeric / semi-numeric fields
# -------------------------
# These require normalization before comparison.

NUMERIC_FIELDS = [
    "gleason_score",
    "grade_group",
    "tumor_grade",
    "nuclear_grade",
    "mitotic_score",
    "olga_stage",
    "olgim_stage",
    "tumor_volume",
    "tumor_extent",
]