from preprocessing import preprocess_data
from model_training import (
    train_models,
    evaluate_models,
    perform_cross_validation,
    tune_models,
    save_artifacts
)
import joblib


ARTIFACT_PATH = "artifacts/readmission_model_bundle.joblib"


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

    print("Training complete.")


def load_artifacts():
    return joblib.load(ARTIFACT_PATH)


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
            print("Saved model loaded.")
            # patient prediction goes here next

        elif choice == "2":
            train_and_save()

        elif choice == "3":
            break

        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()