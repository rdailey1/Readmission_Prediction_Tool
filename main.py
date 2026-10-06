from preprocessing import preprocess_data
from model_training import (
    train_models,
    evaluate_models,
    perform_cross_validation,
    tune_models,
    save_artifacts,
    save_test_encounters
)
import joblib


ARTIFACT_PATH = "artifacts/readmission_model_bundle.joblib"
TEST_DATA_PATH = "data/test/held_out_encounters.joblib"


def train_and_save():
    print("Preprocessing data...")
    processed_df, scaler, feature_columns = preprocess_data()

    print("Training baseline models...")
    logistic_model, random_forest_model, X_train, X_test, y_train, y_test = train_models(
        processed_df
    )

    print("Cross-validating...")
    cross_validation_results = perform_cross_validation(
        logistic_model,
        random_forest_model,
        X_train,
        y_train
    )

    print("Tuning models...")
    tuned_logistic_model, tuned_random_forest_model = tune_models(
        X_train,
        y_train
    )

    metrics = evaluate_models(
        tuned_logistic_model,
        tuned_random_forest_model,
        X_test,
        y_test 
    )

    save_artifacts(
        tuned_logistic_model,
        tuned_random_forest_model,
        scaler,
        feature_columns,
        metrics,
        cross_validation_results
    )

    save_test_encounters(
        X_test,
        y_test
    )

    print("Training complete.")


def load_artifacts():
    return joblib.load(ARTIFACT_PATH)

def load_test_encounters():
    return joblib.load(TEST_DATA_PATH)

def main():
    while True:
        print("\nReadmission Prediction Tool")
        print("1. Use saved model")
        print("2. Retrain model")
        print("3. Quit")
        print()

        choice = input("Select option: ")


        if choice == "1":
            artifacts = load_artifacts()
            test_data = load_test_encounters()

            model = artifacts["logistic_model"]
            X_test = test_data["X_test"]
            y_test = test_data["y_test"]

            print(f"Held-out patients available: 0-{len(X_test) - 1}")

            while True:
                try:
                    patient_index = int(input("Select patient: "))

                    if 0 <= patient_index < len(X_test):
                        break

                    print("Patient number out of range.")

                except ValueError:
                    print("Enter a valid patient number.")

            patient = X_test.iloc[[patient_index]]

            prediction = model.predict(patient)[0]
            actual = y_test.iloc[patient_index]

            print("Predicted readmission:", "Yes" if prediction == 1 else "No")
            print("Actual readmission:", "Yes" if actual == 1 else "No")

        elif choice == "2":
            train_and_save()

        elif choice == "3":
            break

        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()