"""
FastAPI Prediction API for Churn Prediction System.
Loads trained model artifacts and serves churn predictions.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ──────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability using trained ML model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Load Model Artifacts
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

try:
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    numeric_cols = joblib.load(os.path.join(MODELS_DIR, "numeric_cols.joblib"))
    categorical_cols = joblib.load(os.path.join(MODELS_DIR, "categorical_cols.joblib"))
    model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.joblib"))
    print(f"Loaded model: {model_name}")
    print(f"Features: {len(feature_names)}")
except Exception as e:
    print(f"WARNING: Could not load model artifacts: {e}")
    print("Run src/train_model.py first to generate model artifacts.")
    model = None


# ──────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────
class CustomerData(BaseModel):
    """Input schema for customer data."""
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=72, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=844.20)


class PredictionResponse(BaseModel):
    """Output schema for predictions."""
    churn_probability: float
    churn_prediction: int
    risk_level: str
    model_used: str


# ──────────────────────────────────────────────
# Feature Engineering (mirrors data_processing.py)
# ──────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered features used during training."""
    df = df.copy()

    # charges_per_month
    df["charges_per_month"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # tenure_group
    bins = [0, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )

    # num_services
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = sum(
        df[col].apply(lambda x: 1 if x not in ["No", "No internet service", "No phone service"] else 0)
        for col in service_cols
    )

    # has_support
    support_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["has_support"] = (
        sum(df[col].apply(lambda x: 1 if x == "Yes" else 0) for col in support_cols) > 0
    ).astype(int)

    return df


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name if model else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """Predict churn probability for a customer."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first.",
        )

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([customer.model_dump()])

        # Apply feature engineering
        input_data = add_engineered_features(input_data)

        # Preprocess using saved preprocessor
        X = preprocessor.transform(input_data)

        # Predict
        probability = float(model.predict_proba(X)[0][1])
        prediction = int(probability >= 0.5)

        # Risk level
        if probability >= 0.75:
            risk_level = "HIGH"
        elif probability >= 0.45:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            model_used=model_name,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/feature-importance")
def get_feature_importance():
    """Return feature importance data."""
    importance_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if not os.path.exists(importance_path):
        raise HTTPException(status_code=404, detail="Feature importance not available.")

    df = pd.read_csv(importance_path)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
