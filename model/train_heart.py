"""
Train XGBoost classifier on UCI-style Heart Disease data (heart.csv).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_PATH = ROOT / "model" / "heart_model.pkl"
TRAINING_FEATURES_PATH = ROOT / "data" / "heart_training_features.npy"
FEATURE_NAMES_JSON_PATH = ROOT / "data" / "heart_feature_names.json"
TRAINING_MEDIANS_JSON_PATH = ROOT / "data" / "heart_medians.json"
VALID_RANGES_JSON_PATH = ROOT / "data" / "heart_valid_ranges.json"
METRICS_JSON_PATH = ROOT / "data" / "heart_metrics.json"
TEST_PROBS_PATH = ROOT / "data" / "heart_test_probs.npy"
TEST_LABELS_PATH = ROOT / "data" / "heart_test_labels.npy"

# Primary URL from spec; mirror if unavailable (same column schema).
HEART_URL_PRIMARY = (
    "https://raw.githubusercontent.com/rohankokkula/Heart-disease-prediction/master/heart.csv"
)
HEART_URL_FALLBACK = (
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
)

VALID_RANGES: dict[str, list[int | float]] = {
    "age": [20, 80],
    "trestbps": [80, 200],
    "chol": [100, 600],
    "thalach": [60, 220],
    "oldpeak": [0, 7],
    "sex": [0, 1],
    "cp": [0, 3],
    "fbs": [0, 1],
    "restecg": [0, 2],
    "exang": [0, 1],
    "slope": [0, 2],
    "ca": [0, 4],
    "thal": [0, 3],
}

FEATURE_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COL = "target"


def ensure_dataset(path: Path) -> None:
    """Download heart.csv into data/ (primary URL, then fallback)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import requests

    last_err: Exception | None = None
    for url in (HEART_URL_PRIMARY, HEART_URL_FALLBACK):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            path.write_bytes(r.content)
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not download Heart Disease CSV: {last_err!r}. "
        f"Save a file with columns {FEATURE_COLS + [TARGET_COL]} at {path}."
    ) from last_err


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?", "? "])
    for c in FEATURE_COLS + [TARGET_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} in {path}")
    df = df.dropna(subset=["ca", "thal"])
    for c in FEATURE_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    # Binary / multiclass target → 0/1
    y = df[TARGET_COL].astype(int)
    if y.max() > 1:
        y = (y > 0).astype(int)
    df = df.copy()
    df[TARGET_COL] = y
    return df


def main() -> None:
    ensure_dataset(DATA_PATH)
    df = load_and_clean(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

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

    train_feature_medians = {col: float(X_train[col].median()) for col in FEATURE_COLS}
    with open(TRAINING_MEDIANS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(train_feature_medians, f, indent=2)
    with open(VALID_RANGES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(VALID_RANGES, f, indent=2)

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
    np.save(TEST_PROBS_PATH, y_proba.astype(np.float64))
    np.save(TEST_LABELS_PATH, y_test.to_numpy(dtype=np.int64))

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    p1 = rep.get("1", {})
    precision = float(p1.get("precision", 0.0))
    recall = float(p1.get("recall", 0.0))
    f1 = float(p1.get("f1-score", 0.0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_list = [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    def _threshold_json_float(t: float) -> float:
        t = float(t)
        if math.isfinite(t):
            return t
        return 1e308 if t > 0 else -1e308

    metrics: dict = {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm_list,
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": [_threshold_json_float(float(t)) for t in thresholds],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_features": 13,
        "model_name": "XGBoost Classifier",
        "dataset": "UCI Heart Disease Dataset",
    }
    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_JSON_PATH}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "training_medians": train_feature_medians,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model bundle to {MODEL_PATH}")
    print(f"Saved training features to {TRAINING_FEATURES_PATH}")
    print(f"Saved feature names to {FEATURE_NAMES_JSON_PATH}")
    print(f"Saved training medians to {TRAINING_MEDIANS_JSON_PATH}")
    print(f"Saved valid ranges to {VALID_RANGES_JSON_PATH}")
    print(f"Saved test probabilities to {TEST_PROBS_PATH}")
    print(f"Saved test labels to {TEST_LABELS_PATH}")


if __name__ == "__main__":
    main()
    sys.exit(0)
