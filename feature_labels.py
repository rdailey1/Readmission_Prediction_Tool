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

    # Coded categorical features
    if feature_name.startswith("admission_type_id_"):
        value = feature_name.replace("admission_type_id_", "")
        return f"Admission type: {value}"

    if feature_name.startswith("discharge_disposition_id_"):
        value = feature_name.replace("discharge_disposition_id_", "")
        return f"Discharge disposition: {value}"

    if feature_name.startswith("admission_source_id_"):
        value = feature_name.replace("admission_source_id_", "")
        return f"Admission source: {value}"

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