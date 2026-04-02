"""
Shared input validation and cleaning (training medians + valid ranges).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from model.train import FEATURE_COLS as DIABETES_FEATURE_COLS
from model.train_heart import FEATURE_COLS as HEART_FEATURE_COLS

_ROOT = Path(__file__).resolve().parent.parent
_MEDIANS_PATH = _ROOT / "data" / "training_medians.json"
_RANGES_PATH = _ROOT / "data" / "valid_ranges.json"
_MEDIANS_HEART_PATH = _ROOT / "data" / "heart_medians.json"
_RANGES_HEART_PATH = _ROOT / "data" / "heart_valid_ranges.json"

DISEASE_DIABETES = "Diabetes"
DISEASE_HEART = "Heart Disease"

_ZERO_BIOLOGY_FEATURES = frozenset({"Glucose", "BloodPressure", "BMI"})


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_and_clean_input(
    input_dict: dict, disease: str = DISEASE_DIABETES
) -> tuple[dict[str, float], list[str]]:
    """
    Clean feature dict: impute missing values with training medians, flag out-of-range
    and (for diabetes) biologically implausible zeros. Non-blocking.

    ``disease`` must be ``DISEASE_DIABETES`` or ``DISEASE_HEART``.
    """
    is_heart = disease == DISEASE_HEART
    feature_cols = HEART_FEATURE_COLS if is_heart else DIABETES_FEATURE_COLS
    medians_path = _MEDIANS_HEART_PATH if is_heart else _MEDIANS_PATH
    ranges_path = _RANGES_HEART_PATH if is_heart else _RANGES_PATH
    train_cmd = "`python model/train_heart.py`" if is_heart else "`python model/train.py`"

    warnings: list[str] = []
    medians: dict[str, float] = {}
    ranges: dict[str, list] = {}

    try:
        medians = {k: float(v) for k, v in _load_json(medians_path).items()}
    except FileNotFoundError:
        warnings.append(
            f"⚠️ {medians_path.name} not found. Run {train_cmd}. "
            "Missing values cannot be imputed from training medians."
        )
    try:
        ranges = _load_json(ranges_path)
        ranges = {k: [float(v[0]), float(v[1])] for k, v in ranges.items()}
    except FileNotFoundError:
        warnings.append(
            f"⚠️ {ranges_path.name} not found. Run {train_cmd}. "
            "Range checks are skipped."
        )

    cleaned: dict[str, float] = {}

    for feat in feature_cols:
        raw = input_dict.get(feat)
        missing = False
        val: float

        if raw is None or (isinstance(raw, str) and str(raw).strip() == ""):
            missing = True
        else:
            try:
                if pd.isna(raw):
                    missing = True
                else:
                    val = float(raw)
                    if math.isnan(val):
                        missing = True
            except (TypeError, ValueError):
                missing = True

        if missing:
            if feat in medians:
                val = float(medians[feat])
                warnings.append(
                    f"⚠️ {feat} was missing — replaced with population median ({val:g})"
                )
            else:
                val = 0.0
                warnings.append(
                    f"⚠️ {feat} was missing and no training median is available; using 0. "
                    f"Run {train_cmd}."
                )
        cleaned[feat] = float(val)

        if ranges and feat in ranges:
            lo, hi = ranges[feat]
            v = cleaned[feat]
            if v < lo or v > hi:
                warnings.append(
                    f"⚠️ {feat} value {v:g} seems outside the normal range ({lo:g}–{hi:g}). "
                    "Please verify your data."
                )

        if (
            not is_heart
            and feat in _ZERO_BIOLOGY_FEATURES
            and cleaned[feat] == 0.0
        ):
            warnings.append(
                f"⚠️ {feat} is 0, which is usually not biologically plausible for this measure. "
                "Please verify your input."
            )

    return cleaned, warnings
