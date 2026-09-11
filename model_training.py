from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

def build_models():

    logistic_model = LogisticRegression(
        max_iter=50 #limit optimization iterations to 50
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=20, # start with 20 decision trees to average
        random_state=42
    )

    return logistic_model, random_forest_model


def train_models(processed_df):

    # remove target from features model to train on
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

    # Build models
    logistic_model, random_forest_model = build_models()

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

        predictions = model.predict(X_test)

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        metrics[model_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics


def perform_cross_validation(logistic_model, random_forest_model, X_train, y_train):

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }

    for model_name, model in models.items():

        print(f"\nPerforming cross-validation for {model_name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5, # for k-fold cross-validation, data is chunked into 5 (4 folds)
            scoring="f1"
        )
  


def tune_models(X_train, y_train):

    print("\nStarting Logistic Regression hyperparameter tuning")

    # 3 hyperparameter tunings with binary options = 8 combinations of hyperparameter tunings
    logistic_param_grid = {
        # two separate trials are run at 100 and 500 iterations to test if further iterations
        # improves outcomes
        "max_iter": [100, 500],
        # C is a an inverse regularization coefficient. If a feature is found to be
        # extremely predictive, it is penalized because it assumes risk of decreased 
        # generalizability where that feature is falsely emphasized, overfitting to this
        # particular dataset. 'C' can only impose penalties and ranges from 0 to any + num
        # it is inverse in that a high feature coefficient yields a small C but high penalty
        "C": [0.1, 1.0],
        # a class is a categorical outcome 'yes/no'. in a healthcare model where readmission
        # is fairly low, but consequences are high, in this 'needle in a haystack' scenario
        # we test emphasizing the importance of the needle although this may decrease accuracy 
        # and increase false positives. 'Balanced' emphasizes the needle
        "class_weight": [None, "balanced"]
    }

    # define the grid, including the passed hyperparameter grid
    logistic_grid = GridSearchCV(
        LogisticRegression(),
        logistic_param_grid,
        # tests across 3 fold variations, 1 test and 2 train each var then avg
        cv=3,
        # used as the measure of which hyperparameter tuning variation is best
        # is better for this particular datset than other metrics
        scoring="f1",
        # allows jobs to run in parallel, -1 indicating to use as many available cores as 
        # possible for faster processing
        n_jobs=-1
    )

    # train the data on the grid
    logistic_grid.fit(X_train, y_train)

    print("Best Logistic Regression F1 score:", logistic_grid.best_score_)

    print("\nStarting Random Forest hyperparameter tuning")

    random_forest_param_grid = {
        # number of decision trees in forest 
        "n_estimators": [20, 50],
        # depth of tree decisions
        "max_depth": [None, 10],
        "class_weight": [None, "balanced"]
    }

    random_forest_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        random_forest_param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    random_forest_grid.fit(X_train, y_train)

    print("Best Random Forest F1 score:", random_forest_grid.best_score_)

    return logistic_grid.best_estimator_, random_forest_grid.best_estimator_        

def save_artifacts(
    logistic_model,
    random_forest_model,
    scaler,
    feature_columns,
    metrics
):
    os.makedirs("artifacts", exist_ok=True)

    artifacts = {
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "logistic_hyperparameters": logistic_model.get_params(),
        "random_forest_hyperparameters": random_forest_model.get_params()
    }

    joblib.dump(
        artifacts,
        "artifacts/readmission_model_bundle.joblib"
    )

    print("Training artifacts saved.")