import pandas as pd
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

    # Replace missing-value placeholders with a consistent categorical value
    df = df.replace("?", "unknown")

    # Convert target variable to binary
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Coded IDs represent categories rather than numerical quantities
    categorical_id_columns = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id"
    ]
    df[categorical_id_columns] = df[categorical_id_columns].astype(str)

    # Distinguish target vs features
    target = df["readmitted"]
    features = df.drop("readmitted", axis=1)

    # one-hot encoding
    features = pd.get_dummies(features, drop_first=True)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    processed_df = pd.DataFrame(features_scaled, columns=features.columns)
    processed_df["readmitted"] = target.values

    # returns the preprocessed records for training
    return processed_df, scaler, features.columns.tolist()