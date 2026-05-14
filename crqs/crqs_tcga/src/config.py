# src/config.py

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
    "tumor_focality",

    # 3. Grade / aggressiveness
    "tumor_grade",
    "differentiation",

    # 4. Invasion hierarchy
    "invasion_status",
    "lymphovascular_invasion",
    "perineural_invasion",
    "capsular_invasion",
    "organ_specific_invasion",

    # 5. Spread / involvement
    "metastatic_involvement",
    "lymph_node_involvement",
    "multifocal_involvement",

    # 6. Premalignant / epithelial change
    "dysplasia_grade",
    "epithelial_atypia",
    "carcinoma_in_situ",
    "intratubular_germ_cell_neoplasia",

    # 7. Associated morphology
    "necrosis",
    "inflammation_status",
    "calcification",
    "psammoma_bodies",
    "hemorrhage",
    "papillary_features",
    "cystic_change",

    # 8. Tumor extent / burden
    "tumor_size",

    # 9. Contextual / lower-priority metadata
    "organ_site",
    "specimen_type",
    "margin_status",
]


KEY_FIELDS = [
    "benign_vs_malignant",
    "diagnosis",
    "histologic_type",
    "tumor_grade",
    "differentiation",
    "invasion_status",
    "lymphovascular_invasion",
    "metastatic_involvement",
    "necrosis",
]


STRICT_FIELDS = [
    "benign_vs_malignant",
    "diagnosis",
    "histologic_type",
    "tumor_grade",
    "differentiation",
    "invasion_status",
    "lymphovascular_invasion",
    "metastatic_involvement",
    "necrosis",
]


NUMERIC_FIELDS = [
    "tumor_size",
]


CRQS_WEIGHTS = {
    "CFC": 0.3,
    "KIR": 0.4,
    "HR": -0.2,
    "CDS": -0.4,
}


CRQS_RAW_FORMULA = "0.3*CFC + 0.4*KIR - 0.2*HR - 0.4*CDS"
CRQS_NORM_FORMULA = "CRQS_raw / 0.7"


FIELD_ALIASES = {
    "diagnosis": [
        "final diagnosis",
        "pathologic diagnosis",
        "primary diagnosis",
        "tumor diagnosis",
    ],
    "histologic_type": [
        "histology",
        "histologic subtype",
        "tumor type",
        "subtype",
    ],
    "tumor_grade": [
        "grade",
        "who grade",
        "histologic grade",
        "high grade",
        "low grade",
    ],
    "differentiation": [
        "well differentiated",
        "moderately differentiated",
        "poorly differentiated",
        "undifferentiated",
    ],
    "lymphovascular_invasion": [
        "lymphovascular invasion",
        "lymphatic invasion",
        "vascular invasion",
        "venous invasion",
    ],
    "metastatic_involvement": [
        "metastatic",
        "metastasis",
        "metastatic carcinoma",
        "involved by carcinoma",
    ],
    "necrosis": [
        "necrosis",
        "tumor necrosis",
        "coagulative necrosis",
    ],
}


MISSING_VALUES = {
    None,
    "",
    "unknown",
    "not specified",
    "not reported",
    "not mentioned",
    "cannot determine",
    "indeterminate",
    "n/a",
}