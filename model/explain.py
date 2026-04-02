"""
SHAP explainability for diabetes and heart disease XGBoost models.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from model.predict import MODEL_PATH as DIABETES_MODEL_PATH, _clean_single_row
from model.predict_heart import MODEL_PATH as HEART_MODEL_PATH, _impute_row
from model.train import DATA_PATH as DIABETES_DATA_PATH, FEATURE_COLS as DIABETES_FEATURE_COLS, clean_zeros
from model.train_heart import (
    DATA_PATH as HEART_DATA_PATH,
    FEATURE_COLS as HEART_FEATURE_COLS,
    load_and_clean,
)

_COLOR_INCREASE = "#dc2626"
_COLOR_DECREASE = "#16a34a"

DISEASE_DIABETES = "Diabetes"
DISEASE_HEART = "Heart Disease"


def _normalize_disease(disease: str) -> str:
    return DISEASE_HEART if disease == DISEASE_HEART else DISEASE_DIABETES


def _load_bundle(disease: str) -> dict:
    d = _normalize_disease(disease)
    path = HEART_MODEL_PATH if d == DISEASE_HEART else DIABETES_MODEL_PATH
    return joblib.load(path)


def _training_frame_for_background(
    disease: str, n: int, random_state: int
) -> pd.DataFrame:
    d = _normalize_disease(disease)
    from sklearn.model_selection import train_test_split

    if d == DISEASE_HEART:
        df = load_and_clean(HEART_DATA_PATH)
        X = df[HEART_FEATURE_COLS]
        y = df["target"]
        X_train, _, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train.sample(n=min(n, len(X_train)), random_state=random_state)

    df = pd.read_csv(DIABETES_DATA_PATH)
    df, _ = clean_zeros(df)
    X = df[DIABETES_FEATURE_COLS]
    y = df["Outcome"]
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train.sample(n=min(n, len(X_train)), random_state=random_state)


_explainers: dict[str, shap.TreeExplainer] = {}


def _get_explainer(disease: str) -> shap.TreeExplainer:
    d = _normalize_disease(disease)
    if d not in _explainers:
        model = _load_bundle(d)["model"]
        bg = _training_frame_for_background(d, 200, random_state=42)
        _explainers[d] = shap.TreeExplainer(
            model,
            data=bg,
            feature_perturbation="interventional",
            model_output="probability",
        )
    return _explainers[d]


def _shap_values_for_positive_class(
    explainer: shap.TreeExplainer, X: pd.DataFrame
) -> tuple[np.ndarray, float]:
    raw = explainer.shap_values(X)
    ev = explainer.expected_value

    if isinstance(raw, list):
        sv = np.asarray(raw[1], dtype=float)
        base = float(np.asarray(ev, dtype=float).ravel()[1])
    else:
        sv = np.asarray(raw, dtype=float)
        base_arr = np.asarray(ev, dtype=float).ravel()
        base = float(base_arr[-1] if base_arr.size > 1 else base_arr[0])

    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv, base


def _preprocess_for_model(
    input_df: pd.DataFrame, feature_cols: list[str], disease: str
) -> pd.DataFrame:
    d = _normalize_disease(disease)
    bundle = _load_bundle(d)
    train_medians = bundle.get("training_medians", {})
    sub = input_df[feature_cols].copy()
    if d == DISEASE_HEART:
        row = _impute_row(sub, train_medians, feature_cols)
    else:
        row = _clean_single_row(sub, train_medians)
    return row[feature_cols]


def get_shap_values(
    input_df: pd.DataFrame, disease: str = DISEASE_DIABETES
) -> tuple[np.ndarray, float, list[str]]:
    """
    SHAP decomposition for one patient row (positive-class probability).

    ``disease`` must be ``DISEASE_DIABETES`` or ``DISEASE_HEART``.
    """
    if len(input_df) != 1:
        raise ValueError("input_df must be exactly one row")

    d = _normalize_disease(disease)
    bundle = _load_bundle(d)
    feature_names = list(bundle["feature_cols"])
    X = _preprocess_for_model(input_df, feature_names, d)

    explainer = _get_explainer(d)
    sv, base = _shap_values_for_positive_class(explainer, X)
    return sv[0].copy(), float(base), feature_names


def build_shap_chart(
    input_df: pd.DataFrame, disease: str = DISEASE_DIABETES
) -> go.Figure:
    """Horizontal bar chart of signed SHAP contributions for one row."""
    d = _normalize_disease(disease)
    label = "heart disease" if d == DISEASE_HEART else "diabetes"
    shap_vals, _, names = get_shap_values(input_df, d)
    names = list(names)
    order = np.argsort(np.abs(shap_vals))
    names_ord = [names[i] for i in order]
    vals_ord = shap_vals[order].astype(float)
    colors = [
        _COLOR_INCREASE if v > 0 else _COLOR_DECREASE if v < 0 else "#737373"
        for v in vals_ord
    ]

    fig = go.Figure(
        go.Bar(
            x=vals_ord,
            y=names_ord,
            orientation="h",
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>SHAP contributions</b> · "
                "<span style='color:" + _COLOR_INCREASE + "'>Factors Increasing Risk</span> · "
                "<span style='color:" + _COLOR_DECREASE + "'>Factors Decreasing Risk</span>"
            ),
            font=dict(size=15),
        ),
        xaxis_title=f"SHAP value (impact on {label} probability)",
        yaxis=dict(
            title=None,
            automargin=True,
            categoryorder="array",
            categoryarray=names_ord,
        ),
        height=max(360, 48 * len(names_ord)),
        margin=dict(l=120, r=24, t=72, b=56),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,246,241,0.6)",
    )
    fig.add_vline(x=0, line_width=1, line_color="rgba(20,21,26,0.35)")
    return fig


def build_shap_waterfall_chart(
    input_df: pd.DataFrame, disease: str = DISEASE_DIABETES
) -> go.Figure:
    """SHAP waterfall for baseline → features → predicted positive-class probability."""
    d = _normalize_disease(disease)
    cond = "heart disease" if d == DISEASE_HEART else "diabetes"
    short = "P(heart disease)" if d == DISEASE_HEART else "P(diabetes)"

    shap_vals, base, names = get_shap_values(input_df, d)
    names = list(names)
    pred = float(np.sum(shap_vals) + base)
    order = np.argsort(-np.abs(shap_vals))
    ordered_names = [names[i] for i in order]
    ordered_shap = shap_vals[order].astype(float)

    measure = ["absolute"] + ["relative"] * len(ordered_names) + ["total"]
    x_labels = ["Baseline<br>E[f(x)]"] + ordered_names + [f"Predicted<br>{short}"]
    y_vals = [base] + list(ordered_shap) + [pred]

    connector_line = dict(color="rgba(148, 163, 184, 0.5)")
    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measure,
            x=x_labels,
            y=y_vals,
            text=[f"{base:.3f}"]
            + [f"+{v:.3f}" if v >= 0 else f"{v:.3f}" for v in ordered_shap]
            + [f"{pred:.3f}"],
            textposition="outside",
            connector=dict(line=connector_line),
            increasing=dict(marker=dict(color="#f87171")),
            decreasing=dict(marker=dict(color="#4ade80")),
            totals=dict(marker=dict(color="#38bdf8")),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"<b>SHAP waterfall</b> · how each feature moves {cond} risk from baseline",
            font=dict(size=15),
        ),
        yaxis=dict(title="Probability contribution", showgrid=True, zeroline=True),
        xaxis=dict(title=None, tickangle=-35),
        showlegend=False,
        height=520,
        margin=dict(l=56, r=24, t=72, b=120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,246,241,0.6)",
    )
    return fig


def build_global_feature_importance_chart(
    n_samples: int = 200,
    random_state: int = 42,
    disease: str = DISEASE_DIABETES,
) -> go.Figure:
    """Mean |SHAP| over ``n_samples`` random training rows (same split as training)."""
    d = _normalize_disease(disease)
    label = "heart disease" if d == DISEASE_HEART else "diabetes"
    bg = _training_frame_for_background(d, n_samples, random_state)
    explainer = _get_explainer(d)
    sv, _ = _shap_values_for_positive_class(explainer, bg)
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)
    fnames = list(_load_bundle(d)["feature_cols"])
    names = [fnames[i] for i in order]
    vals = mean_abs[order]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color="#0d9488",
            showlegend=False,
            hovertemplate="%{y}<br>mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Global feature influence</b> · mean |SHAP| over "
                f"{len(bg)} training rows"
            ),
            font=dict(size=15),
        ),
        xaxis_title=f"Mean |SHAP| ({label} probability)",
        yaxis=dict(title=None, automargin=True),
        height=max(360, 44 * len(names)),
        margin=dict(l=120, r=24, t=72, b=56),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,246,241,0.6)",
    )
    return fig
