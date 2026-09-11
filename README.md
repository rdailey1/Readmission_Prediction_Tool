# Hospital Readmission Risk Prediction

## Overview
The purpose of this project is to predict whether a diabetitc patient will be readmitted to the hospital within 30 days post discharge. This is an important metric in real hospital sytems for determining a hospital's reimbursement by Medicare and it impacts the hospital's public rating. The hospiatal would then be able to propose further interventions for patient found to be at a higher risk. The project uses the UCI Diabetes 130-US hospitals dataset and implements Logistic Regression and Random Forest classification models to make learna and predictions.   

---

# Project Objectives

* Preprocess a large healthcare dataset
* Prepare categorical and numerical healthcare data for machine learning
* Train and evaluate Logistic Regressor and Random Forest classification algorithms
* Compare baseline vs tuned model performance
* Implement hyperparameter tuning and cross-validation

---

# Dataset

Dataset: Diabetes 130-US hospitals for years 1999-2008

Source:
[https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Dataset characteristics:

* Approximately 100,000 patient encounters
* Multiple hospitals across the United States
* Includes demographic, medication, and encounter information
* Binary classification target: readmitted within 30 days or not

---

# Models Used

## Logistic Regression

Logistic Regression was selected for its ease of interpretability. The clinical importance of a feature is represented by a measureable coefficient which represents strength linearly and these features are then used in producing a binary prediction. 

## Random Forest Classifier

Random Forest Classifier was selected because it can capture nonlinear relationships of a feature and between features, such that the impact of a feature may increase or decrease variably despite linear change in the feature. Multiple features at low risk thresholds may also compound to produce high risk scenarios. Although less directly interpretable than Logistic Regression, Random Forest can also provide clinically usable feature importance information.

---

# Data Preprocessing

The preprocessing pipeline performs the following operations:

* Removes sparse and low-information columns
* Inserts defaults for missing values in kept columns
* Converts the target variable into binary format
* Performs one-hot encoding on categorical variables
* Scales numerical features using StandardScaler
* Saves a sample of the processed dataset

Removed columns:

* encounter_id
* patient_nbr
* weight
* payer_code
* medical_specialty
* max_glu_serum
* A1Cresult

---

# Model Evaluation Metrics

The project evaluates model performance using:

* Accuracy
* Precision
* Recall
* F1 Score
* Cross-validation F1 Score

The dataset is highly imbalanced, meaning far fewer patients were readmitted within 30 days than not readmitted. Therefore, F1 score is particularly important as it is computed from recall, which tests for the model's ability to capture all true positives, and precision, which assesses the models ability to catch those positive instances with minimal false positives.

---

# Hyperparameter Tuning

GridSearchCV was used to test multiple hyperparameter combinations for each model using cross-validation. The best-performing parameter combination was selected based on the most optimal F1 score.

## Logistic Regression Parameters

* max_iter
* C
* class_weight

## Random Forest Parameters

* n_estimators
* max_depth
* class_weight

Hyperparameter tuning significantly F1 improved performance compared to the initial baseline models. Although overall accuracy decreased after tuning, this was an acceptable tradeoff for the more clinically valuable F1 metric.

---

# Results Summary

Initial baseline models achieved high accuracy but poor F1 performance due to class imbalance.

After hyperparameter tuning:

* F1 score improved substantially
* Models became more effective at identifying high-risk patients
* Overall accuracy decreased as the models became less biased toward predicting the majority class

---

# Future Improvements

The project would most easily be improved by improving its usabiltiy from the clinician perspective. A simple yes/no on rehospitalization risk is minimally useful where the clinician has to make assumptions where perhaps increased interventions such as medications could lead to more or less hospitalization risk but the clinician has no knowledge other than increased risk being present. To prevent black-box ambiguity and suspicion, providing information on specific feature importance, feature thresholds, or the impact of feature combinations would greatly improve a clinician's ability to plan skilled interventions to decrease rehospitalizataion risk. Furthermore, indicating a degree of risk instead of a binary risk assessment would be useful for improved patient prioritization even among at risk patients. The app could also be improved through use of a more detailed or intact data set, as some of the dropped columns such as weight and A1C are generally significant health indicators. This could also be further expanded into a real time monitoring system where if certain features such as bloodsugar level or weight readings were found to be of high consequence, then patients and clinicians could be alerted that feature was outside of the acceptable range.  

---

# Project Structure

```
readmission-risk/
│
├── data/
│   ├── raw/
│   └── processed/
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
* numpy
* scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

# Running the Project

Run the project from the project root directory:

```bash
python3 main.py
```

The program will:

1. Preprocess the dataset
2. Train baseline models
3. Evaluate model performance
4. Perform cross-validation
5. Perform hyperparameter tuning
6. Evaluate tuned models

---

# Author

Ryan Dailey
