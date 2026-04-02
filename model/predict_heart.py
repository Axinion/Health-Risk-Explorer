"""
Load trained heart disease model and score a single patient row.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "heart_model.pkl"


def _impute_row(
    row: pd.DataFrame, medians: dict[str, float], feature_cols: list[str]
) -> pd.DataFrame:
    out = row.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        val = out[col].iloc[0]
        if pd.isna(val):
            out[col] = medians.get(col, 0.0)
    return out


def predict_heart_risk(
    features: pd.DataFrame,
    medians: dict[str, float] | None = None,
) -> dict:
    """
    Predict heart disease risk from a single-row DataFrame with the 13 feature columns.

    Parameters
    ----------
    features :
        One row: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal.
    medians :
        Optional overrides for imputing missing values. Defaults to medians from
        the trained model bundle.
    """
    if len(features) != 1:
        raise ValueError("features must be exactly one row")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    train_medians: dict[str, float] = bundle.get("training_medians", {})

    use_medians = {**train_medians, **(medians or {})}
    row = _impute_row(features[feature_cols], use_medians, feature_cols)

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
