import csv


IDS_MAPPING_PATH = "data/reference/IDS_mapping.csv"


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


UCI_MAPPINGS = load_uci_mappings()


def format_feature_name(feature_name):

    # Diagnosis features
    if feature_name.startswith("diag_1_"):
        code = feature_name.replace("diag_1_", "")
        return f"Primary diagnosis: ICD-9 {code}"

    if feature_name.startswith("diag_2_"):
        code = feature_name.replace("diag_2_", "")
        return f"Secondary diagnosis: ICD-9 {code}"

    if feature_name.startswith("diag_3_"):
        code = feature_name.replace("diag_3_", "")
        return f"Additional diagnosis: ICD-9 {code}"

    # Age
    if feature_name.startswith("age_"):
        age_range = feature_name.replace("age_", "")
        return f"Age: {age_range}"

    # UCI coded categorical features
    if feature_name.startswith("admission_type_id_"):
        value = feature_name.replace("admission_type_id_", "")
        description = UCI_MAPPINGS["admission_type_id"].get(value, value)
        return f"Admission type: {description}"

    if feature_name.startswith("discharge_disposition_id_"):
        value = feature_name.replace("discharge_disposition_id_", "")
        description = UCI_MAPPINGS["discharge_disposition_id"].get(value, value)
        return f"Discharge disposition: {description}"

    if feature_name.startswith("admission_source_id_"):
        value = feature_name.replace("admission_source_id_", "")
        description = UCI_MAPPINGS["admission_source_id"].get(value, value)
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

    # Generic fallback, including medication/status features
    return feature_name.replace("_", ": ", 1).replace("_", " ")