"""
Clinical-style banding for PIMA features (demo thresholds — not diagnostic guidance).
"""

from __future__ import annotations

THRESHOLDS = {
    "Glucose": {"normal": (0, 99), "borderline": (100, 125), "high": (126, 9999)},
    "BloodPressure": {"normal": (0, 79), "borderline": (80, 89), "high": (90, 9999)},
    "BMI": {"normal": (0, 24.9), "borderline": (25, 29.9), "high": (30, 9999)},
    "Insulin": {"normal": (0, 140), "borderline": (141, 200), "high": (201, 9999)},
    "Age": {"normal": (0, 35), "borderline": (36, 55), "high": (56, 9999)},
    "Pregnancies": {"normal": (0, 5), "borderline": (6, 9), "high": (10, 9999)},
    "SkinThickness": {"normal": (0, 30), "borderline": (31, 45), "high": (46, 9999)},
    "DiabetesPedigreeFunction": {
        "normal": (0, 0.5),
        "borderline": (0.51, 1.0),
        "high": (1.01, 9999),
    },
}

_UNKNOWN = ("⚪", "Unknown", "#95a5a6")


def get_badge(feature: str, value) -> tuple[str, str, str]:
    """
    Returns a tuple: (emoji, label, color_hex)

    - 🟢 Normal → ("#2ecc71")
    - 🟡 Borderline → ("#f39c12")
    - 🔴 High → ("#e74c3c")
    - ⚪ Unknown → if feature not in THRESHOLDS or value is not classifiable
    """
    if feature not in THRESHOLDS:
        return _UNKNOWN

    try:
        v = float(value)
    except (TypeError, ValueError):
        return _UNKNOWN

    bands = THRESHOLDS[feature]
    n_lo, n_hi = bands["normal"]
    b_lo, b_hi = bands["borderline"]
    h_lo, h_hi = bands["high"]

    if n_lo <= v <= n_hi:
        return ("🟢", "Normal", "#2ecc71")
    if b_lo <= v <= b_hi:
        return ("🟡", "Borderline", "#f39c12")
    if h_lo <= v <= h_hi:
        return ("🔴", "High", "#e74c3c")
    if v > h_hi and v >= h_lo:
        return ("🔴", "High", "#e74c3c")
    if v < n_lo:
        return _UNKNOWN
    return _UNKNOWN
