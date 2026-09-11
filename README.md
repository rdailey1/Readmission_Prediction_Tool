# Hospital Readmission Risk Prediction

## Overview

Hospital readmissions are an important clinical and operational problem. Preventable readmissions can indicate gaps in discharge planning or follow-up care and can affect hospital reimbursement and publicly reported performance. Identifying high-risk patients before discharge gives clinicians an opportunity to prioritize interventions and resources toward patients most likely to return to the hospital.

This project predicts whether a diabetic patient will be readmitted within 30 days of discharge using the UCI Diabetes 130-US Hospitals dataset. Logistic Regression and Random Forest classification models are trained and evaluated using approximately 100,000 historical patient encounters.

The project is being developed beyond basic prediction into a clinician-facing decision-support tool. Rather than providing only a binary prediction, the goal is to provide patient-specific readmission risk and identify the most important actionable factors contributing to that risk, allowing clinicians to better prioritize interventions and discharge planning.

---

# Project Objectives

* Preprocess a large healthcare dataset
* Prepare categorical and numerical healthcare data for machine learning
* Train and evaluate Logistic Regression and Random Forest classifiers
* Compare baseline and tuned model performance
* Perform hyperparameter tuning and cross-validation
* Persist trained models and preprocessing artifacts for reuse
* Separate expensive model training from patient-specific prediction
* Develop interpretable, clinician-focused readmission risk output

---

# Dataset

Dataset: Diabetes 130-US Hospitals for Years 1999–2008

Source:

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Dataset characteristics:

* Approximately 100,000 patient encounters
* Multiple hospitals across the United States
* Includes demographic, medication, and encounter information
* Binary classification target: readmitted within 30 days or not

---

# Models Used

## Logistic Regression

Logistic Regression was selected primarily for interpretability. The model learns a coefficient for each feature representing how that feature contributes to the prediction. This provides a foundation for patient-specific explanations showing which features most strongly increase or decrease predicted readmission risk.

Logistic Regression is planned as the primary clinician-facing model because its feature contributions can be presented more concisely and transparently.

## Random Forest Classifier

Random Forest was selected because it can capture nonlinear relationships, thresholds, and interactions between features that Logistic Regression may not represent well.

Random Forest remains useful as a comparison model during development. Because its predictions are more difficult to reduce into concise patient-specific explanations, it is currently intended to remain an internal comparison model rather than contribute directly to clinician-facing output.

---

# Data Preprocessing

The preprocessing process performs the following operations:

* Removes sparse and low-information columns
* Replaces missing-value placeholders with a consistent categorical value
* Converts the target variable into binary format
* Performs one-hot encoding on categorical variables
* Standardizes feature values using StandardScaler
* Preserves the fitted scaler and feature-column schema for later patient inference

Removed columns:

* encounter_id
* patient_nbr
* weight
* payer_code
* medical_specialty
* max_glu_serum
* A1Cresult

---

# Model Evaluation

The project currently evaluates trained models using:

* Precision
* Recall
* F1 Score
* 5-fold cross-validation F1 scores
* Mean cross-validation F1 score

The dataset is highly imbalanced, meaning substantially fewer patients were readmitted within 30 days than were not readmitted. F1 score is therefore used as the primary model-selection metric because it incorporates both recall and precision.

These metrics are intended for model development and validation rather than clinician-facing output. They are stored with the trained model for future engineering reference.

---

# Cross-Validation

Five-fold cross-validation is performed on the baseline models to evaluate performance across different training and validation splits.

The individual F1 scores and mean F1 score are stored as model metadata rather than displayed to the eventual clinical user.

---

# Hyperparameter Tuning

GridSearchCV tests multiple hyperparameter combinations for each model using cross-validation. Parameter combinations are compared using F1 score, and the best-performing estimator is retained.

## Logistic Regression Parameters

* max_iter
* C
* class_weight

## Random Forest Parameters

* n_estimators
* max_depth
* class_weight

Hyperparameter tuning substantially improved F1 performance compared with the initial baseline models by reducing the models' tendency to favor the majority non-readmission class.

---

# Model Persistence

Training and clinical inference are separated so the complete dataset does not need to be preprocessed and the models retrained every time a patient prediction is requested.

After training, the application stores a model artifact containing:

* Tuned Logistic Regression model
* Tuned Random Forest model
* Fitted StandardScaler
* Feature-column schema
* Precision, recall, and F1 metrics
* Selected model hyperparameters
* Cross-validation results

The artifact is serialized using `joblib` and stored at:

```text
artifacts/readmission_model_bundle.joblib
```

Normal use can load this artifact directly rather than repeating the full training process. Retraining replaces the currently stored model artifact.

---

# Current Program Flow

Running `main.py` presents a terminal menu:

```text
Readmission Prediction Tool
1. Use saved model
2. Retrain model
3. Quit
```

Selecting the saved-model option loads the existing trained model artifact without rerunning preprocessing, cross-validation, or hyperparameter tuning.

Selecting retraining performs:

1. Dataset preprocessing
2. Baseline model training
3. Cross-validation
4. Hyperparameter tuning
5. Tuned-model evaluation
6. Model and metadata persistence

Patient-specific inference using the loaded model is the next feature under development.

---

# Clinician-Facing Development

The next development milestone is converting the stored Logistic Regression model into a patient-specific decision-support tool.

Planned functionality:

1. Accept a single patient's clinical features
2. Process those features using the saved preprocessing information
3. Generate estimated 30-day readmission probability using `predict_proba()`
4. Calculate patient-specific Logistic Regression feature contributions
5. Rank features by their impact on the patient's prediction
6. Filter results to a small number of clinically actionable factors
7. Translate encoded model features into clinician-readable labels
8. Present concise output containing:
   * Estimated 30-day readmission risk
   * Top actionable contributing factors
   * Relative impact of those factors

The goal is not simply to tell a clinician that a patient is high risk, but to provide enough explanation to help the clinician understand what factors are contributing to that prediction.

---

# Future Improvements

Future development could include:

* Integration with an EMR to automatically retrieve patient features
* Random Forest patient-level explainability if it provides useful information beyond Logistic Regression
* Improved training data containing clinically important variables currently unavailable or too sparse in the dataset
* Real-time monitoring of changing patient risk factors
* Patient and clinician alerts
* Cloud deployment and security controls

Real-time monitoring and alerting are intentionally outside the scope of the current static decision-support tool.

---

# Project Structure

```text
ReadmissionPredictionTool/
│
├── artifacts/
│   └── readmission_model_bundle.joblib
│
├── data/
│   └── raw/
│       └── diabetic_data.csv
│
├── preprocessing.py
├── model_training.py
├── main.py
├── README.md
└── .gitignore
```

---

# Requirements

Python 3

Required libraries:

* pandas
* scikit-learn
* joblib

Install dependencies:

```bash
pip install pandas scikit-learn joblib
```

---

# Running the Project

Run from the project root directory:

```bash
python3 main.py
```

The program allows the user to load the existing trained model, retrain and replace the model artifact, or quit.

---

# Author

Ryan Dailey