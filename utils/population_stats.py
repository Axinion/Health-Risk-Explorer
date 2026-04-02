"""
Population reference statistics from saved training features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import percentileofscore

_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_NPY = _ROOT / "data" / "training_features.npy"
_FEATURE_JSON = _ROOT / "data" / "feature_names.json"
_HEART_TRAINING_NPY = _ROOT / "data" / "heart_training_features.npy"
_HEART_FEATURE_JSON = _ROOT / "data" / "heart_feature_names.json"

DISEASE_DIABETES = "Diabetes"
DISEASE_HEART = "Heart Disease"


def get_percentiles(input_dict: dict, disease: str = DISEASE_DIABETES) -> dict:
    """
    For each feature in ``input_dict``, compute the percentile rank of the value
    vs. the saved training feature matrix using ``scipy.stats.percentileofscore``
    with ``kind='strict'`` (percentage of reference values strictly below the user's).

    ``disease`` selects diabetes vs. heart training arrays.
    """
    is_heart = disease == DISEASE_HEART
    npy_path = _HEART_TRAINING_NPY if is_heart else _TRAINING_NPY
    json_path = _HEART_FEATURE_JSON if is_heart else _FEATURE_JSON
    train_hint = "model/train_heart.py" if is_heart else "model/train.py"

    if not npy_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(
            f"Missing {npy_path.name} or {json_path.name}. Run {train_hint}."
        )

    arr = np.load(npy_path)
    with open(json_path, encoding="utf-8") as f:
        names: list[str] = json.load(f)

    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError("Training features array shape does not match feature_names.json.")

    name_to_col = {n: i for i, n in enumerate(names)}
    out: dict[str, int] = {}

    for feat, raw_val in input_dict.items():
        if feat not in name_to_col:
            continue
        try:
            score = float(raw_val)
        except (TypeError, ValueError):
            continue
        col = arr[:, name_to_col[feat]]
        pct = percentileofscore(col, score, kind="strict")
        out[feat] = int(round(float(pct)))

    return out
