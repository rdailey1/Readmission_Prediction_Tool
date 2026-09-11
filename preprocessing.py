import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    # Load raw dataset and convert csv to dataframe to use inbuilt operations
    df = pd.read_csv("data/raw/diabetic_data.csv")

    # Drop ID cols and cols with majority missing content
    columns_to_drop = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
        "max_glu_serum",
        "A1Cresult"
    ]
    df = df.drop(columns=columns_to_drop)

    # NaN valid option (effectively serves as unknown regardless of data type) but is
    # not usable for ML purposes and may crash in the training process
    # using 'unknown' should avoid this
    df = df.replace("?", "unknown")

    # Convert target variable 'readmitted' to binary. 1 if yes <30 days, else 0 for no
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Separate target/features before encoding/scaling
    target = df["readmitted"]
    features = df.drop("readmitted", axis=1)

    # one-hot encoding takes categorical values such as race or non-binary vaules such as
    # age and internally produces columns to numerically represent these features, one 
    # col per category. required since AI/ML models cannot use strings, etc for math ops
    features = pd.get_dummies(features, drop_first=True)

    # Scale numeric features to reduce bias large vals have over small vals in computation
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # create a new dataframe obj and populate it with scaled features and reassign col titles
    processed_df = pd.DataFrame(features_scaled, columns=features.columns)

    # add target col into the new scaled dataframe obj
    processed_df["readmitted"] = target.values

    # Save fragment of preprocessed dataset. currently stalling at save process of whole set
    # not actually required for use, but saved to view the format of processed data
    processed_df.head(1000).to_csv("data/processed/preprocessed_diabetes_sample.csv", index=False)

    # returns the full 100k records to be used for training
    return processed_df, scaler, features.columns.tolist()