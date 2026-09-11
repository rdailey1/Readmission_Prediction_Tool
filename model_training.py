from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

# initialize models
def build_models():

    logistic_model = LogisticRegression(
        max_iter=50 
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=20,
        random_state=42
    )

    return logistic_model, random_forest_model


def train_models(processed_df):

    # extract target from features for training
    X = processed_df.drop("readmitted", axis=1)
    y = processed_df["readmitted"]

    # split data into test and train 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logistic_model, random_forest_model = build_models()

    # train the models. model learns relationships but does not yet make predictions
    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    return logistic_model, random_forest_model, X_train, X_test, y_train, y_test


def evaluate_models(logistic_model, random_forest_model, X_test, y_test):

    models = {
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model
    }

    metrics = {}

    for model_name, model in models.items():

        # make predictions
        predictions = model.predict(X_test)

        # compares predictions to actual
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        metrics[model_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics


# evaluate model stability across different training/validation splits
def perform_cross_validation(logistic_model, random_forest_model, X_train, y_train):

    models = {
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model
    }

    cross_validation_results = {}

    for model_name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1"
        )

        cross_validation_results[model_name] = {
            "f1_scores": scores.tolist(),
            "mean_f1": scores.mean()
        }
        
    # return results to store for non-user reference
    return cross_validation_results
  

def tune_models(X_train, y_train):

    print("Starting Logistic Regression hyperparameter tuning")

    # LR 3 hyperparameters, 2 tuning options each
    logistic_param_grid = {
        "max_iter": [100, 500],
        "C": [0.1, 1.0],
        "class_weight": [None, "balanced"]
    }

    # comparison framework for LR hyperparameter tunings
    logistic_grid = GridSearchCV(
        LogisticRegression(),
        logistic_param_grid,
        # 3-fold cross-validation for each hyperparameter combo
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    # runs full grid search and cross validation
    logistic_grid.fit(X_train, y_train)

    # obtains best F1 after fit()
    print("Best Logistic Regression F1 score:", logistic_grid.best_score_)
    print("Starting Random Forest hyperparameter tuning")

    # RF 3 hyperparameters, 2 tuning options each
    random_forest_param_grid = {
        "n_estimators": [20, 50],
        "max_depth": [None, 10],
        "class_weight": [None, "balanced"]
    }

    # comparison framework for RF hyperparameter tunings
    random_forest_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        random_forest_param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    # compares RF hyperparameter tunings based on framework and yields F1 from best combo
    random_forest_grid.fit(X_train, y_train)

    print("Best Random Forest F1 score:", random_forest_grid.best_score_)

    return logistic_grid.best_estimator_, random_forest_grid.best_estimator_        


def save_artifacts(
    logistic_model,
    random_forest_model,
    scaler,
    feature_columns,
    metrics,
    cross_validation_results
):
    # ensure artifact directory exists
    os.makedirs("artifacts", exist_ok=True)
    # defines/populates dictionary with args acquired from other processing/training functions
    artifacts = {
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "logistic_hyperparameters": logistic_model.get_params(),
        "random_forest_hyperparameters": random_forest_model.get_params(),
        "cross_validation_results": cross_validation_results
    }

    # serializes/saves dictionary to disk. Overwrites any existing. 
    joblib.dump(
        artifacts,
        "artifacts/readmission_model_bundle.joblib"
    )

    print("Training artifacts saved.")