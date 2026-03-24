"""
Data processing pipeline for Churn Prediction System.
Handles loading, cleaning, feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load the Telco Churn CSV and perform initial type fixes."""
    df = pd.read_csv(path)

    # TotalCharges has spaces for missing values — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ──────────────────────────────────────────────
# 2. Feature Engineering
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that help the model."""
    df = df.copy()

    # Charges per month of tenure (spending intensity)
    df["charges_per_month"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Tenure groups (interview-friendly bucketing)
    bins = [0, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

    # Total number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = sum(
        df[col].apply(lambda x: 1 if x not in ["No", "No internet service", "No phone service"] else 0)
        for col in service_cols
    )

    # Has any support service (security, backup, protection, tech support)
    support_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["has_support"] = (
        sum(df[col].apply(lambda x: 1 if x == "Yes" else 0) for col in support_cols) > 0
    ).astype(int)

    return df


# ──────────────────────────────────────────────
# 3. Preprocessing Pipeline
# ──────────────────────────────────────────────
def get_feature_lists(df: pd.DataFrame):
    """Identify numeric and categorical feature columns."""
    # Drop non-feature columns
    exclude = ["customerID", "Churn"]
    feature_df = df.drop(columns=[c for c in exclude if c in df.columns])

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the full preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


# ──────────────────────────────────────────────
# 4. Full Preparation Pipeline
# ──────────────────────────────────────────────
def prepare_data(
    data_path: str,
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    End-to-end data preparation.
    Returns X_train, X_test, y_train, y_test, preprocessor, feature_names.
    """
    # Load
    df = load_data(data_path)

    # Engineer features
    df = engineer_features(df)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Identify features
    numeric_cols, categorical_cols = get_feature_lists(df)
    print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Split
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Churn rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # Build & fit preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get final feature names
    cat_feature_names = []
    if categorical_cols:
        cat_feature_names = list(
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_cols)
        )
    feature_names = numeric_cols + cat_feature_names

    # Save preprocessor and feature names
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.joblib"))
    joblib.dump(numeric_cols, os.path.join(models_dir, "numeric_cols.joblib"))
    joblib.dump(categorical_cols, os.path.join(models_dir, "categorical_cols.joblib"))
    print(f"\nPreprocessor and feature info saved to {models_dir}/")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "telco_churn.csv")
    models_dir = os.path.join(project_root, "models")

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(
        data_path, models_dir
    )
    print(f"\nProcessed feature count: {len(feature_names)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
