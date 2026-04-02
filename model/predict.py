"""
Load trained diabetes model and score a single patient row.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "diabetes_model.pkl"

ZERO_TO_MEDIAN_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def _clean_single_row(
    row: pd.DataFrame, medians: dict[str, float]
) -> pd.DataFrame:
    out = row.copy()
    for col in ZERO_TO_MEDIAN_COLS:
        if col not in out.columns:
            continue
        val = out[col].iloc[0]
        if pd.isna(val) or val == 0:
            out[col] = medians.get(col, out[col].iloc[0])
    return out


def predict_diabetes_risk(
    features: pd.DataFrame,
    medians: dict[str, float] | None = None,
) -> dict:
    """
    Predict diabetes risk from a single-row DataFrame with the 8 feature columns.

    Parameters
    ----------
    features :
        One row, columns: Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age.
    medians :
        Optional overrides for imputing 0/NaN in Glucose, BloodPressure, SkinThickness,
        Insulin, BMI. Defaults to medians stored with the trained model.
    """
    if len(features) != 1:
        raise ValueError("features must be exactly one row")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    train_medians: dict[str, float] = bundle.get("training_medians", {})

    use_medians = {**train_medians, **(medians or {})}
    row = _clean_single_row(features[feature_cols], use_medians)

    X = row[feature_cols]
    risk_score = float(model.predict_proba(X)[0, 1])

    if risk_score < 0.3:
        risk_label = "Low"
    elif risk_score <= 0.6:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    return {"risk_score": risk_score, "risk_label": risk_label}


def load_model_bundle():
    """Return the saved joblib bundle (model, feature_cols, metadata)."""
    return joblib.load(MODEL_PATH)
