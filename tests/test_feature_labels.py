from feature_labels import (
    format_feature_name,
    translate_icd9,
    clean_description,
    UCI_MAPPINGS
)


def test_primary_diagnosis_translation():
    result = format_feature_name("diag_1_440")

    assert "Primary diagnosis" in result
    assert "Atherosclerosis" in result
    assert "ICD-9 440" in result


def test_secondary_diagnosis_translation():
    result = format_feature_name("diag_2_996")

    assert "Secondary diagnosis" in result
    assert "Complications peculiar to certain specified procedures" in result
    assert "ICD-9 996" in result


def test_additional_diagnosis_translation():
    result = format_feature_name("diag_3_433")

    assert "Additional diagnosis" in result
    assert "Occlusion and stenosis of precerebral arteries" in result
    assert "ICD-9 433" in result


def test_unknown_icd9_code():
    result = translate_icd9("000999")

    assert result == "Unknown diagnosis"


def test_age_translation():
    result = format_feature_name("age_[50-60)")

    assert result == "Age: [50-60)"


def test_admission_type_translation():
    result = format_feature_name("admission_type_id_1")

    assert result == "Admission type: Emergency"


def test_discharge_disposition_translation():
    result = format_feature_name("discharge_disposition_id_1")

    assert result == "Discharge disposition: Discharged to home"


def test_admission_source_translation():
    result = format_feature_name("admission_source_id_7")

    assert result == "Admission source: Emergency Room"


def test_null_description_translation():
    result = clean_description("NULL")

    assert result == "Unknown / not recorded"


def test_null_description_case_insensitive():
    result = clean_description("null")

    assert result == "Unknown / not recorded"


def test_normal_description_unchanged():
    result = clean_description("Emergency")

    assert result == "Emergency"


def test_numerical_feature_translation():
    assert format_feature_name(
        "number_inpatient"
    ) == "Prior inpatient visits"

    assert format_feature_name(
        "num_medications"
    ) == "Number of medications"

    assert format_feature_name(
        "number_diagnoses"
    ) == "Number of diagnoses"


def test_medication_not_prescribed():
    result = format_feature_name("rosiglitazone_No")

    assert result == "Rosiglitazone: Not prescribed"


def test_medication_dose_unchanged():
    result = format_feature_name("metformin_Steady")

    assert result == "Metformin: Dose unchanged"


def test_medication_dose_increased():
    result = format_feature_name("insulin_Up")

    assert result == "Insulin: Dose increased"


def test_medication_dose_decreased():
    result = format_feature_name("insulin_Down")

    assert result == "Insulin: Dose decreased"


def test_combination_medication_translation():
    result = format_feature_name("glyburide-metformin_No")

    assert result == "Glyburide Metformin: Not prescribed"


def test_non_medication_no_not_mistranslated():
    result = format_feature_name("change_No")

    assert result != "Change: Not prescribed"


def test_uci_mapping_loaded():
    assert len(UCI_MAPPINGS["admission_type_id"]) > 0
    assert len(UCI_MAPPINGS["discharge_disposition_id"]) > 0
    assert len(UCI_MAPPINGS["admission_source_id"]) > 0