import csv


IDS_MAPPING_PATH = "data/reference/IDS_mapping.csv"
ICD9_MAPPING_PATH = "data/reference/V26 I-9 Diagnosis.txt"
ICD9_CATEGORY_PATH = "data/reference/Dtab09.txt"


MEDICATION_FEATURES = {
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
}


def load_uci_mappings():

    mappings = {
        "admission_type_id": {},
        "discharge_disposition_id": {},
        "admission_source_id": {}
    }

    current_category = None

    with open(IDS_MAPPING_PATH, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:

            if not row:
                continue

            # Detect beginning of each mapping section
            if row[0] in mappings:
                current_category = row[0]
                continue

            # Skip blank separator rows
            if not row[0].strip():
                continue

            # Store ID → description
            if current_category and row[0].strip().isdigit():
                mappings[current_category][row[0].strip()] = row[1].strip()

    return mappings


def load_icd9_mappings():

    mappings = {}

    with open(ICD9_MAPPING_PATH, encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Split once between ICD-9 code and description
            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            code, description = parts

            # Store code without decimal so it matches normalized UCI codes
            normalized_code = code.replace(".", "")

            mappings[normalized_code] = description.strip()

    return mappings


def load_icd9_categories():

    mappings = {}

    with open(ICD9_CATEGORY_PATH, encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            code, description = parts

            # Store only broad 3-digit ICD-9 categories
            if len(code) == 3 and code.isdigit():
                mappings[code] = description.strip()

    return mappings


UCI_MAPPINGS = load_uci_mappings()
ICD9_MAPPINGS = load_icd9_mappings()
ICD9_CATEGORIES = load_icd9_categories()


def translate_icd9(code):

    normalized_code = code.replace(".", "")

    # Use exact diagnosis when UCI provides sufficient specificity
    description = ICD9_MAPPINGS.get(normalized_code)

    if description:
        return description

    # Otherwise use the broad 3-digit diagnosis category
    category_code = code.split(".")[0][:3]
    description = ICD9_CATEGORIES.get(category_code)

    if description:
        return description

    return "Unknown diagnosis"


def clean_description(description):

    if description.upper() == "NULL":
        return "Unknown / not recorded"

    return description


def format_feature_name(feature_name):

    # Diagnosis features
    if feature_name.startswith("diag_1_"):
        code = feature_name.replace("diag_1_", "")
        description = translate_icd9(code)
        return f"Primary diagnosis: {description} (ICD-9 {code})"

    if feature_name.startswith("diag_2_"):
        code = feature_name.replace("diag_2_", "")
        description = translate_icd9(code)
        return f"Secondary diagnosis: {description} (ICD-9 {code})"

    if feature_name.startswith("diag_3_"):
        code = feature_name.replace("diag_3_", "")
        description = translate_icd9(code)
        return f"Additional diagnosis: {description} (ICD-9 {code})"

    # Age
    if feature_name.startswith("age_"):
        age_range = feature_name.replace("age_", "")
        return f"Age: {age_range}"

    # UCI coded categorical features
    if feature_name.startswith("admission_type_id_"):
        value = feature_name.replace("admission_type_id_", "")
        description = UCI_MAPPINGS["admission_type_id"].get(value, value)
        description = clean_description(description)
        return f"Admission type: {description}"

    if feature_name.startswith("discharge_disposition_id_"):
        value = feature_name.replace("discharge_disposition_id_", "")
        description = UCI_MAPPINGS["discharge_disposition_id"].get(value, value)
        description = clean_description(description)
        return f"Discharge disposition: {description}"

    if feature_name.startswith("admission_source_id_"):
        value = feature_name.replace("admission_source_id_", "")
        description = UCI_MAPPINGS["admission_source_id"].get(value, value)
        description = clean_description(description)
        return f"Admission source: {description}"

    # Common numerical features
    feature_labels = {
        "number_inpatient": "Prior inpatient visits",
        "number_emergency": "Prior emergency visits",
        "number_outpatient": "Prior outpatient visits",
        "num_medications": "Number of medications",
        "num_lab_procedures": "Number of lab procedures",
        "num_procedures": "Number of procedures",
        "number_diagnoses": "Number of diagnoses",
        "time_in_hospital": "Length of hospital stay"
    }

    if feature_name in feature_labels:
        return feature_labels[feature_name]

    # Translate medication status into clinician-readable language
    medication_statuses = {
        "No": "Not prescribed",
        "Steady": "Dose unchanged",
        "Up": "Dose increased",
        "Down": "Dose decreased"
    }

    for medication in MEDICATION_FEATURES:

        prefix = f"{medication}_"

        if feature_name.startswith(prefix):

            status = feature_name[len(prefix):]

            readable_status = medication_statuses.get(
                status,
                status
            )

            readable_medication = (
                medication
                .replace("-", " ")
                .title()
            )

            return f"{readable_medication}: {readable_status}"

    # Generic fallback for remaining categorical features
    return feature_name.replace("_", ": ", 1).replace("_", " ")