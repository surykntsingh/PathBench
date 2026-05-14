"""
Configuration for CRQS evaluation on the REG pathology report dataset.
"""

CLINICAL_FIELDS = [
    "organ_site",
    "specimen_type",
    "diagnosis",
    "histologic_type",
    "tumor_grade",
    "gleason_score",
    "grade_group",
    "tumor_volume",
    "differentiation",
    "invasion_depth",
    "in_situ_component",
    "nuclear_grade",
    "mitotic_score",
    "tubule_score",
    "necrosis",
    "calcification",
    "dysplasia_grade",
    "muscle_proper_status",
]

KEY_FIELDS = [
    "diagnosis",
    "histologic_type",
    "tumor_grade",
    "gleason_score",
    "invasion_depth",
]

CRQS_WEIGHTS = {
    "CFC": 0.3,
    "KIR": 0.4,
    "HR": -0.2,
    "CDS": -0.4,
}

# Fields where exact numeric/severity mismatch should be treated as discordance
STRICT_FIELDS = [
    "diagnosis",
    "histologic_type",
    "tumor_grade",
    "gleason_score",
    "grade_group",
    "differentiation",
    "invasion_depth",
    "dysplasia_grade",
    "muscle_proper_status",
]

# Fields allowed to use small numeric tolerance later if needed
NUMERIC_FIELDS = [
    "tumor_volume",
]