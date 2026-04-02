"""
Train XGBoost classifier on PIMA Indians Diabetes dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Project root (parent of model/)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pima_diabetes.csv"
MODEL_PATH = ROOT / "model" / "diabetes_model.pkl"
TRAINING_FEATURES_PATH = ROOT / "data" / "training_features.npy"
FEATURE_NAMES_JSON_PATH = ROOT / "data" / "feature_names.json"

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_TO_MEDIAN_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def ensure_dataset(path: Path) -> None:
    """Download PIMA Indians Diabetes data (UCI schema) into data/pima_diabetes.csv."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import requests

    # UCI’s old direct URL often 404s; this CSV matches UCI/Kaggle columns.
    urls = (
        "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "pima-indians-diabetes/pima-indians-diabetes.data"
        ),
    )
    last_err: Exception | None = None
    for url in urls:
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            if url.endswith(".csv"):
                path.write_bytes(r.content)
                return
            lines = r.text.strip().splitlines()
            rows = [line.split(",") for line in lines]
            df = pd.DataFrame(rows, dtype=float)
            df.columns = FEATURE_COLS + ["Outcome"]
            df = df.astype(
                {
                    "Pregnancies": int,
                    "Glucose": float,
                    "BloodPressure": float,
                    "SkinThickness": float,
                    "Insulin": float,
                    "BMI": float,
                    "DiabetesPedigreeFunction": float,
                    "Age": int,
                    "Outcome": int,
                }
            )
            df.to_csv(path, index=False)
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not download PIMA dataset: {last_err!r}. "
        f"Place a CSV with columns {FEATURE_COLS + ['Outcome']} at {path}."
    ) from last_err


def clean_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    medians: dict[str, float] = {}
    for col in ZERO_TO_MEDIAN_COLS:
        col_series = out[col].replace(0, np.nan)
        median = float(col_series.median())
        medians[col] = median
        out[col] = col_series.fillna(median)
    return out, medians


def main() -> None:
    ensure_dataset(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df, training_medians = clean_zeros(df)

    X = df[FEATURE_COLS]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TRAINING_FEATURES_PATH, X_train.to_numpy(dtype=np.float64))
    with open(FEATURE_NAMES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "zero_median_cols": ZERO_TO_MEDIAN_COLS,
            "training_medians": training_medians,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model bundle to {MODEL_PATH}")
    print(f"Saved training features to {TRAINING_FEATURES_PATH}")
    print(f"Saved feature names to {FEATURE_NAMES_JSON_PATH}")


if __name__ == "__main__":
    main()
    sys.exit(0)
