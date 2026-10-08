import joblib

from app import (
    ARTIFACT_PATH,
    TEST_DATA_PATH
)


def load_artifacts():

    artifacts = joblib.load(ARTIFACT_PATH)
    test_data = joblib.load(TEST_DATA_PATH)

    return artifacts, test_data


def test_required_models_exist():

    artifacts, _ = load_artifacts()

    assert "logistic_model" in artifacts
    assert "random_forest_model" in artifacts


def test_test_data_exists():

    _, test_data = load_artifacts()

    assert "X_test" in test_data
    assert "y_test" in test_data


def test_test_features_and_labels_same_length():

    _, test_data = load_artifacts()

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    assert len(X_test) == len(y_test)


def test_held_out_data_not_empty():

    _, test_data = load_artifacts()

    assert len(test_data["X_test"]) > 0


def test_model_feature_count_matches_test_data():

    artifacts, test_data = load_artifacts()

    model = artifacts["logistic_model"]
    X_test = test_data["X_test"]

    assert model.coef_.shape[1] == X_test.shape[1]


def test_no_duplicate_feature_names():

    _, test_data = load_artifacts()

    columns = test_data["X_test"].columns

    assert not columns.duplicated().any()


def test_target_is_binary():

    _, test_data = load_artifacts()

    values = set(
        test_data["y_test"].unique()
    )

    assert values.issubset({0, 1})