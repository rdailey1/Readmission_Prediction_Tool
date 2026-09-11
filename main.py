from preprocessing import preprocess_data
from model_training import train_models, evaluate_models, perform_cross_validation, tune_models

processed_df = preprocess_data()

logistic_model, random_forest_model, X_train, X_test, y_train, y_test = train_models(
    processed_df
)

# all metrics on a scale of 0-1 where 1 is perfect
evaluate_models(
    logistic_model,
    random_forest_model,
    X_test,
    y_test
)

perform_cross_validation(
    logistic_model,
    random_forest_model,
    X_train,
    y_train
)

tuned_logistic_model, tuned_random_forest_model = tune_models(
    X_train,
    y_train
)

print("\nTuned Model Evaluation")
evaluate_models(
    tuned_logistic_model,
    tuned_random_forest_model,
    X_test,
    y_test
)

print("Preprocessing, training, pre-tuning evaluation, and cross-validation, " \
"post-tuning evaluation complete from main.")