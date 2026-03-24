"""
Model training pipeline for Churn Prediction System.
Trains Logistic Regression, Random Forest, and LightGBM.
Evaluates with ROC-AUC, F1, Precision, Recall — saves best model.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from lightgbm import LGBMClassifier

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_processing import prepare_data

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Define Models
# ──────────────────────────────────────────────
def get_models() -> dict:
    """Return dictionary of models to train."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
        ),
    }


# ──────────────────────────────────────────────
# 2. Evaluate Model
# ──────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }


# ──────────────────────────────────────────────
# 3. Train All Models
# ──────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train all models, print comparison, return results."""
    models = get_models()
    results = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics}

        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print(f"  F1:        {metrics['F1']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")

    # Comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    comparison = pd.DataFrame({
        name: result["metrics"] for name, result in results.items()
    }).T
    comparison = comparison.sort_values("ROC-AUC", ascending=False)
    print(comparison.to_string())

    # Best model
    best_name = comparison.index[0]
    best_model = results[best_name]["model"]
    print(f"\n{'*' * 50}")
    print(f"  BEST MODEL: {best_name}")
    print(f"  ROC-AUC:    {comparison.loc[best_name, 'ROC-AUC']:.4f}")
    print(f"{'*' * 50}")

    # Classification report for best model
    y_pred = best_model.predict(X_test)
    print(f"\nClassification Report ({best_name}):")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    return results, best_name, best_model


# ──────────────────────────────────────────────
# 4. Feature Importance
# ──────────────────────────────────────────────
def print_feature_importance(model, feature_names, model_name, top_n=15):
    """Print and return top feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not expose feature importances.")
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\n{'=' * 70}")
    print(f"TOP {top_n} FEATURE IMPORTANCES ({model_name})")
    print("=" * 70)
    for i, row in importance_df.head(top_n).iterrows():
        bar = "#" * int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"  {row['feature']:40s} {row['importance']:.4f}  {bar}")

    return importance_df


# ──────────────────────────────────────────────
# 5. Save Model Artifacts
# ──────────────────────────────────────────────
def save_model(model, model_name, feature_importance_df, models_dir):
    """Save the best model and its feature importances."""
    joblib.dump(model, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(model_name, os.path.join(models_dir, "best_model_name.joblib"))

    if feature_importance_df is not None:
        feature_importance_df.to_csv(
            os.path.join(models_dir, "feature_importance.csv"), index=False
        )

    print(f"\nModel artifacts saved to {models_dir}/")
    print(f"  - best_model.joblib")
    print(f"  - best_model_name.joblib")
    print(f"  - feature_importance.csv")
    print(f"  - preprocessor.joblib  (saved during data processing)")
    print(f"  - feature_names.joblib (saved during data processing)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    data_path = os.path.join(project_root, "data", "telco_churn.csv")
    models_dir = os.path.join(project_root, "models")

    # Prepare data
    print("STEP 1: Data Preparation")
    print("-" * 40)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(
        data_path, models_dir
    )

    # Train models
    print("\nSTEP 2: Model Training")
    print("-" * 40)
    results, best_name, best_model = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # Feature importance
    print("\nSTEP 3: Feature Importance Analysis")
    print("-" * 40)
    importance_df = print_feature_importance(best_model, feature_names, best_name)

    # Save
    print("\nSTEP 4: Saving Artifacts")
    print("-" * 40)
    save_model(best_model, best_name, importance_df, models_dir)

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
