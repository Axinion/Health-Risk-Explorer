"""
Shared input validation and cleaning (training medians + valid ranges).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from model.train import FEATURE_COLS

_ROOT = Path(__file__).resolve().parent.parent
_MEDIANS_PATH = _ROOT / "data" / "training_medians.json"
_RANGES_PATH = _ROOT / "data" / "valid_ranges.json"

_ZERO_BIOLOGY_FEATURES = frozenset({"Glucose", "BloodPressure", "BMI"})


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_and_clean_input(input_dict: dict) -> tuple[dict[str, float], list[str]]:
    """
    Clean feature dict: impute missing values with training medians, flag out-of-range
    and biologically implausible zeros. Non-blocking — returns values suitable for prediction.

    Returns
    -------
    cleaned_dict
        All keys in ``FEATURE_COLS`` as floats.
    warnings_list
        Human-readable strings for ``st.warning`` (already prefixed with ⚠️ where needed).
    """
    warnings: list[str] = []
    medians: dict[str, float] = {}
    ranges: dict[str, list] = {}

    try:
        medians = {k: float(v) for k, v in _load_json(_MEDIANS_PATH).items()}
    except FileNotFoundError:
        warnings.append(
            "⚠️ training_medians.json not found. Run `python model/train.py`. "
            "Missing values cannot be imputed from training medians."
        )
    try:
        ranges = _load_json(_RANGES_PATH)
        ranges = {k: [float(v[0]), float(v[1])] for k, v in ranges.items()}
    except FileNotFoundError:
        warnings.append(
            "⚠️ valid_ranges.json not found. Run `python model/train.py`. "
            "Range checks are skipped."
        )

    cleaned: dict[str, float] = {}

    for feat in FEATURE_COLS:
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
                    "Run `python model/train.py`."
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

        if feat in _ZERO_BIOLOGY_FEATURES and cleaned[feat] == 0.0:
            warnings.append(
                f"⚠️ {feat} is 0, which is usually not biologically plausible for this measure. "
                "Please verify your input."
            )

    return cleaned, warnings
