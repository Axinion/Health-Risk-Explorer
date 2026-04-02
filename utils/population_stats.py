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


def get_percentiles(input_dict: dict) -> dict:
    """
    For each feature in ``input_dict``, compute the percentile rank of the value
    vs. the saved training feature matrix using ``scipy.stats.percentileofscore``
    with ``kind='strict'`` (percentage of reference values strictly below the user's).

    Returns integer percentiles 0–100, e.g. ``{"Glucose": 78, "BMI": 62, ...}``.
    """
    if not _TRAINING_NPY.is_file() or not _FEATURE_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {_TRAINING_NPY.name} or {_FEATURE_JSON.name}. Run model/train.py."
        )

    arr = np.load(_TRAINING_NPY)
    with open(_FEATURE_JSON, encoding="utf-8") as f:
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
