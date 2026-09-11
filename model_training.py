from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }

    for model_name, model in models.items():

        predictions = model.predict(X_test)

        # % of correct predictions
        accuracy = accuracy_score(y_test, predictions)
        # how many of predicted + were + (true positives)
        precision = precision_score(y_test, predictions)
        # of all +, how many were found
        recall = recall_score(y_test, predictions)
        # balance of precision and recall
        f1 = f1_score(y_test, predictions)

        print(f"\n{model_name} Evaluation")


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

    return logistic_grid.best_estimator_, random_forest_grid.best_estimator_        