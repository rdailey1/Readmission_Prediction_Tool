import joblib
import numpy as np

from app import (
    ARTIFACT_PATH,
    TEST_DATA_PATH,
    get_prediction,
    get_contributions
)


def load_test_objects():

    artifacts = joblib.load(ARTIFACT_PATH)
    test_data = joblib.load(TEST_DATA_PATH)

    model = artifacts["logistic_model"]
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    return model, X_test, y_test


def test_prediction_returns_binary_class():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    prediction, _ = get_prediction(
        model,
        patient
    )

    assert prediction in [0, 1]


def test_probability_between_zero_and_one():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    _, probability = get_prediction(
        model,
        patient
    )

    assert 0.0 <= probability <= 1.0


def test_contributions_match_model_features():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    risk_increasing, risk_decreasing = get_contributions(
        model,
        patient
    )

    total_contributors = (
        len(risk_increasing)
        + len(risk_decreasing)
    )

    nonzero_features = np.count_nonzero(
        patient.iloc[0].values * model.coef_[0]
    )

    assert total_contributors == nonzero_features


def test_risk_increasing_contributions_are_positive():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    risk_increasing, _ = get_contributions(
        model,
        patient
    )

    assert (risk_increasing > 0).all()


def test_risk_decreasing_contributions_are_negative():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    _, risk_decreasing = get_contributions(
        model,
        patient
    )

    assert (risk_decreasing < 0).all()


def test_risk_increasing_sorted_descending():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    risk_increasing, _ = get_contributions(
        model,
        patient
    )

    values = risk_increasing.tolist()

    assert values == sorted(
        values,
        reverse=True
    )


def test_risk_decreasing_sorted_by_magnitude_direction():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    _, risk_decreasing = get_contributions(
        model,
        patient
    )

    values = risk_decreasing.tolist()

    assert values == sorted(values)


def test_prediction_matches_predict_proba_threshold():

    model, X_test, _ = load_test_objects()

    patient = X_test.iloc[[0]]

    prediction, probability = get_prediction(
        model,
        patient
    )

    expected_prediction = (
        1 if probability >= 0.5 else 0
    )

    assert prediction == expected_prediction