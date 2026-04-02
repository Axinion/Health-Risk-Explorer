"""
Personal Health Risk Explorer — Streamlit UI (sample sliders or CSV upload).
"""
# MOBILE TEST CHECKLIST:
# [ ] Landing screen buttons stack vertically on iPhone
# [ ] Sliders are wide enough to tap
# [ ] Gauge charts not clipped
# [ ] Chat input visible above keyboard
# [ ] PDF download button accessible
# [ ] Sidebar collapsed by default
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from model.explain import (
    DISEASE_DIABETES,
    DISEASE_HEART,
    build_global_feature_importance_chart,
    build_shap_waterfall_chart,
    get_shap_values,
)
from model.llm_explain import RISK_SCORE_PCT_KEY, generate_explanation, run_followup_chat
from model.predict import predict_diabetes_risk
from model.predict_heart import predict_heart_risk
from model.train import FEATURE_COLS
from model.train_heart import FEATURE_COLS as HEART_FEATURE_COLS
from scripts.setup import setup_models
from utils.health_thresholds import get_badge, get_heart_badge
from utils.pdf_report import generate_pdf_report
from utils.population_stats import get_percentiles
from utils.validation import validate_and_clean_input
from utils.feature_glossary import DIABETES_GLOSSARY, HEART_GLOSSARY

HEART_SIMULATOR_FEATURES: tuple[str, ...] = ("trestbps", "chol", "thalach", "oldpeak")
HEART_SIM_LABELS: dict[str, str] = {
    "trestbps": "💓 Resting blood pressure (mm Hg)",
    "chol": "🧈 Cholesterol (mg/dL)",
    "thalach": "🏃 Max heart rate (thalach)",
    "oldpeak": "📉 ST depression (oldpeak)",
}

_CP_HELP = (
    "Chest pain type: 0 = Typical angina, 1 = Atypical angina, "
    "2 = Non-anginal pain, 3 = Asymptomatic"
)
_THAL_HELP = (
    "Thalassemia: 0 = (unused in some cohorts); 1 = Normal; "
    "2 = Fixed defect; 3 = Reversible defect"
)

SIMULATOR_FEATURES: tuple[str, ...] = ("BMI", "Glucose", "BloodPressure", "Insulin")
SIMULATOR_LABELS: dict[str, str] = {
    "BMI": "⚖️ BMI (losing 1 BMI point ≈ 3kg for avg adult)",
    "Glucose": "🍬 Fasting Glucose (mg/dL)",
    "BloodPressure": "💓 Blood Pressure (mm Hg)",
    "Insulin": "💉 Insulin Level (μU/mL)",
}

SLIDER_CONFIG: dict[str, dict] = {
    "Pregnancies": {"label": "Pregnancies", "min": 0, "max": 17, "default": 3, "step": 1, "int": True},
    "Glucose": {"label": "Glucose (mg/dL)", "min": 50, "max": 200, "default": 110, "step": 1, "int": True},
    "BloodPressure": {
        "label": "Blood pressure (mm Hg)",
        "min": 40,
        "max": 130,
        "default": 72,
        "step": 1,
        "int": True,
    },
    "SkinThickness": {
        "label": "Skin thickness (mm)",
        "min": 0,
        "max": 100,
        "default": 23,
        "step": 1,
        "int": True,
    },
    "Insulin": {"label": "Insulin (μU/mL)", "min": 0, "max": 846, "default": 79, "step": 1, "int": True},
    "BMI": {"label": "BMI", "min": 10.0, "max": 70.0, "default": 28.0, "step": 0.1, "int": False},
    "DiabetesPedigreeFunction": {
        "label": "Diabetes pedigree function",
        "min": 0.05,
        "max": 2.5,
        "default": 0.47,
        "step": 0.01,
        "int": False,
    },
    "Age": {"label": "Age (years)", "min": 18, "max": 90, "default": 33, "step": 1, "int": True},
}

_APP_ROOT = Path(__file__).resolve().parent
_METRICS_JSON_PATHS: dict[str, Path] = {
    DISEASE_DIABETES: _APP_ROOT / "data" / "diabetes_metrics.json",
    DISEASE_HEART: _APP_ROOT / "data" / "heart_metrics.json",
}
_TEST_PROBS_PATHS: dict[str, Path] = {
    DISEASE_DIABETES: _APP_ROOT / "data" / "diabetes_test_probs.npy",
    DISEASE_HEART: _APP_ROOT / "data" / "heart_test_probs.npy",
}
_TEST_LABELS_PATHS: dict[str, Path] = {
    DISEASE_DIABETES: _APP_ROOT / "data" / "diabetes_test_labels.npy",
    DISEASE_HEART: _APP_ROOT / "data" / "heart_test_labels.npy",
}

_DATASET_LINKS: dict[str, str] = {
    DISEASE_DIABETES: "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
    DISEASE_HEART: "https://archive.ics.uci.edu/dataset/45/heart+disease",
}

_SAMPLE_CSV_PATH = _APP_ROOT / "data" / "sample_patients.csv"
_PROGRESS_LABELS = (
    "Enter Data",
    "View Risk Score",
    "Get AI Explanation",
    "Download Report",
)


def _style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 25, 34, 0.55)",
        font=dict(color="#e8eaef", family="DM Sans, sans-serif"),
        title_font=dict(color="#f4f4f5"),
    )
    return fig


def _render_progress_stepper(progress_step: int) -> None:
    """Top-of-screen visual progress tracker (steps 1..4)."""
    step = max(1, min(4, int(progress_step)))
    if st.session_state.get("is_mobile"):
        st.markdown(
            f"<div style='text-align:center; margin:8px 0 16px 0; color:#cbd5e1; "
            f"font-weight:600;'>Step {step} of 4 — {_PROGRESS_LABELS[step - 1]}</div>",
            unsafe_allow_html=True,
        )
        return
    circles: list[str] = []
    for i, label in enumerate(_PROGRESS_LABELS, start=1):
        is_complete = i < step
        is_current = i == step
        if is_complete:
            bg = "#2ecc71"
            border = "#2ecc71"
            txt = "✓"
            label_color = "#2ecc71"
            label_weight = "700"
        elif is_current:
            bg = "#3498db"
            border = "#3498db"
            txt = str(i)
            label_color = "#e2e8f0"
            label_weight = "700"
        else:
            bg = "transparent"
            border = "#555"
            txt = str(i)
            label_color = "#94a3b8"
            label_weight = "500"

        circles.append(
            (
                "<div style='display:flex; flex-direction:column; align-items:center; "
                "min-width:100px;'>"
                f"<div style='width:34px; height:34px; border-radius:50%; "
                f"border:2px solid {border}; background:{bg}; color:#fff; "
                "display:flex; align-items:center; justify-content:center; "
                "font-weight:700;'>"
                f"{txt}</div>"
                f"<div style='margin-top:8px; font-size:0.86rem; color:{label_color}; "
                f"font-weight:{label_weight}; text-align:center;'>{label}</div>"
                "</div>"
            )
        )

    connectors: list[str] = []
    for i in range(1, 4):
        clr = "#2ecc71" if step > (i + 1) else "#555"
        connectors.append(
            f"<div style='width:80px; height:2px; background:{clr}; margin:0 6px 26px 6px;'></div>"
        )

    html = (
        "<div style='display:flex; align-items:center; justify-content:center; "
        "padding:16px 0; margin-bottom:20px;'>"
        f"{circles[0]}{connectors[0]}{circles[1]}{connectors[1]}{circles[2]}{connectors[2]}{circles[3]}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _load_test_metrics(disease: str) -> dict | None:
    path = _METRICS_JSON_PATHS.get(disease, _METRICS_JSON_PATHS[DISEASE_DIABETES])
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _disease_display_name(disease: str) -> str:
    return "Heart Disease" if disease == DISEASE_HEART else "Diabetes"


def _threshold_mode(threshold: float) -> tuple[str, str]:
    t = float(threshold)
    if t < 0.35:
        return (
            "High Sensitivity Mode",
            "🔴 High Sensitivity Mode\n"
            "The model flags more people as at-risk.\n"
            "Good for: screening where missing a case is costly.\n"
            "Tradeoff: more false alarms.",
        )
    if t > 0.65:
        return (
            "High Specificity Mode",
            "🔵 High Specificity Mode\n"
            "The model only flags clear high-risk cases.\n"
            "Good for: reducing unnecessary follow-ups.\n"
            "Tradeoff: may miss borderline cases.",
        )
    return (
        "Balanced Mode",
        "🟡 Balanced Mode\n"
        "Standard clinical threshold.\n"
        "Good for: general population screening.",
    )


def _metric_color(metric: str, value: float) -> str:
    cuts = {
        "Accuracy": 0.75,
        "Precision": 0.70,
        "Recall (Sensitivity)": 0.70,
        "Specificity": 0.70,
        "F1 Score": 0.70,
    }
    return "#2ecc71" if value >= cuts.get(metric, 0.7) else "#e74c3c"


def _render_model_performance_tab(disease: str) -> None:
    """Tab 2: metrics cards, ROC, confusion matrix, dataset expander."""
    m = _load_test_metrics(disease)
    _pk = "h" if disease == DISEASE_HEART else "d"
    dname = _disease_display_name(disease)

    st.markdown("### About This Model")

    if m is None:
        train_cmd = (
            "`python model/train_heart.py`"
            if disease == DISEASE_HEART
            else "`python model/train.py`"
        )
        st.warning(
            f"Performance metrics file not found. Run {train_cmd} to generate "
            f"`{_METRICS_JSON_PATHS[disease].name}`."
        )
        return

    st.caption(
        f"Held-out test metrics for the **{_disease_display_name(disease)}** XGBoost model "
        "(same split as in training scripts)."
    )

    acc = float(m["accuracy"])
    roc_auc = float(m["roc_auc"])
    prec = float(m["precision"])
    rec = float(m["recall"])
    fpr = [float(x) for x in m["fpr"]]
    tpr = [float(x) for x in m["tpr"]]
    cm = m["confusion_matrix"]
    test_size = int(m["test_size"])
    train_size = int(m["train_size"])
    n_feat = int(m["n_features"])
    dataset_label = str(m.get("dataset", ""))

    st.markdown("#### Headline metrics (held-out test set)")
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric("Accuracy", f"{acc * 100:.1f}%")
    with metric_cols[1]:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")
    metric_cols2 = st.columns(2)
    with metric_cols2[0]:
        st.metric("Precision", f"{prec * 100:.1f}%")
    with metric_cols2[1]:
        st.metric("Recall", f"{rec * 100:.1f}%")

    st.markdown("#### ROC curve")
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"XGBoost (AUC = {roc_auc:.3f})",
            line=dict(color="#2563eb", width=2.5),
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="#94a3b8", width=2, dash="dash"),
        )
    )
    fig_roc.update_layout(
        title=dict(
            text=f"ROC Curve — {dname}",
            font=dict(size=16, color="#f4f4f5"),
        ),
        xaxis=dict(
            title="False Positive Rate",
            range=[0, 1],
            gridcolor="rgba(148,163,184,0.25)",
            zeroline=False,
        ),
        yaxis=dict(
            title="True Positive Rate",
            range=[0, 1],
            gridcolor="rgba(148,163,184,0.25)",
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 25, 34, 0.55)",
        font=dict(color="#e8eaef", family="DM Sans, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=56, b=48, l=56, r=24),
        height=420,
    )
    fig_roc = _style_plotly(fig_roc)
    st.plotly_chart(fig_roc, use_container_width=True, theme=None, key=f"roc_curve_{_pk}")

    st.markdown("#### Confusion matrix")
    TN, FP = int(cm[0][0]), int(cm[0][1])
    FN, TP = int(cm[1][0]), int(cm[1][1])
    total = max(test_size, TN + FP + FN + TP)
    z_text = [
        [
            f"{TN}<br>({100.0 * TN / total:.1f}%)",
            f"{FP}<br>({100.0 * FP / total:.1f}%)",
        ],
        [
            f"{FN}<br>({100.0 * FN / total:.1f}%)",
            f"{TP}<br>({100.0 * TP / total:.1f}%)",
        ],
    ]
    z_color = [[1, 0], [0, 1]]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=z_color,
            x=["Negative", "Positive"],
            y=["Negative", "Positive"],
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "#f8fafc"},
            colorscale=[[0, "#dc2626"], [1, "#16a34a"]],
            showscale=False,
            hoverinfo="skip",
            xgap=2,
            ygap=2,
        )
    )
    fig_cm.update_layout(
        title=dict(
            text=f"Confusion Matrix (Test Set, n={test_size})",
            font=dict(size=16, color="#f4f4f5"),
        ),
        xaxis=dict(title="Predicted", side="bottom", tickfont=dict(color="#94a3b8")),
        yaxis=dict(title="Actual", autorange="reversed", tickfont=dict(color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 25, 34, 0.35)",
        font=dict(color="#e8eaef"),
        margin=dict(t=56, b=72, l=72, r=48),
        height=380,
    )
    fig_cm.update_xaxes(title_standoff=8)
    fig_cm.update_yaxes(title_standoff=8)
    st.plotly_chart(fig_cm, use_container_width=True, theme=None, key=f"cm_heatmap_{_pk}")

    st.markdown("### ⚖️ Classification Threshold Tuner")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.get("custom_threshold", 0.50)),
        step=0.01,
        key=f"threshold_slider_{_pk}",
        help=(
            "Adjusting this threshold changes the tradeoff between catching more "
            "cases (sensitivity) vs. avoiding false alarms (specificity)."
        ),
    )
    st.session_state["custom_threshold"] = float(threshold)

    probs_path = _TEST_PROBS_PATHS[disease]
    labels_path = _TEST_LABELS_PATHS[disease]
    if probs_path.exists() and labels_path.exists():
        y_pred_proba = np.load(probs_path)
        y_true = np.load(labels_path).astype(int)
        y_pred_thr = (y_pred_proba >= threshold).astype(int)

        acc_t = float(accuracy_score(y_true, y_pred_thr))
        prec_t = float(precision_score(y_true, y_pred_thr, zero_division=0))
        rec_t = float(recall_score(y_true, y_pred_thr, zero_division=0))
        f1_t = float(f1_score(y_true, y_pred_thr, zero_division=0))
        cm_t = confusion_matrix(y_true, y_pred_thr, labels=[0, 1])
        tn, fp = int(cm_t[0, 0]), int(cm_t[0, 1])
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        c_left, c_mid, c_right = st.columns(3)

        with c_left:
            m_names = [
                "Accuracy",
                "Precision",
                "Recall (Sensitivity)",
                "Specificity",
                "F1 Score",
            ]
            m_vals = [acc_t, prec_t, rec_t, specificity, f1_t]
            m_txt = [f"{v:.3f}" for v in m_vals]
            val_colors = [_metric_color(n, v) for n, v in zip(m_names, m_vals)]
            fig_tbl = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=["Metric", "Value"], fill_color="#1f2937", font=dict(color="white")),
                        cells=dict(
                            values=[m_names, m_txt],
                            fill_color="#111827",
                            font=dict(color=["#e5e7eb", val_colors]),
                        ),
                    )
                ]
            )
            fig_tbl.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_tbl, use_container_width=True, theme=None, key=f"thr_metrics_{_pk}")

        with c_mid:
            thr_grid = np.round(np.arange(0.10, 0.901, 0.01), 2)
            p_curve = []
            r_curve = []
            for t in thr_grid:
                yp = (y_pred_proba >= t).astype(int)
                p_curve.append(float(precision_score(y_true, yp, zero_division=0)))
                r_curve.append(float(recall_score(y_true, yp, zero_division=0)))
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=thr_grid, y=p_curve, mode="lines", name="Precision", line=dict(color="#3498db")))
            fig_pr.add_trace(go.Scatter(x=thr_grid, y=r_curve, mode="lines", name="Recall", line=dict(color="#f39c12")))
            fig_pr.add_shape(
                type="line",
                x0=threshold,
                x1=threshold,
                y0=0,
                y1=1,
                line=dict(color="#cbd5e1", width=2, dash="dash"),
            )
            fig_pr.update_layout(
                title="Precision vs. Recall Tradeoff",
                xaxis_title="Threshold (0.1 to 0.9)",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=320,
                margin=dict(l=20, r=20, t=40, b=30),
            )
            st.plotly_chart(_style_plotly(fig_pr), use_container_width=True, theme=None, key=f"thr_tradeoff_{_pk}")

        with c_right:
            mode_name, mode_text = _threshold_mode(threshold)
            st.markdown(
                f"""
<div style='background:#1a2a3a; padding:14px; border-radius:10px; border-left:4px solid #3498db; min-height:290px;'>
  <div style='font-weight:700; margin-bottom:8px;'>{mode_name}</div>
  <div style='white-space:pre-line; color:#dbeafe;'>{mode_text}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption(
            "Threshold tuner data files are missing. Re-run `python model/train.py` and "
            "`python model/train_heart.py` to generate `*_test_probs.npy` and `*_test_labels.npy`."
        )

    st.info(
        "💡 The default threshold of 0.5 is not always optimal.\n"
        "In medical screening, doctors often lower the threshold to catch more\n"
        "cases, accepting more false positives to avoid missing true ones.\n"
        "This tradeoff is called the Sensitivity-Specificity tradeoff."
    )

    with st.expander("📋 Dataset & Training Details", expanded=False):
        st.markdown(
            f"**Dataset:** {dataset_label}  \n"
            f"**Training samples:** {train_size} · **Test samples:** {test_size} · "
            f"**Features:** {n_feat}"
        )
        st.markdown(
            "**Model architecture:** XGBoost classifier — `n_estimators=200`, "
            "`max_depth=4`, `learning_rate=0.05` (same for both disease models in this app)."
        )
        if disease == DISEASE_HEART:
            st.markdown(
                "This model estimates the **probability of heart disease (positive class)** "
                "from tabular clinical features. It was trained on a **small, historical UCI-style "
                "cohort**, which may not represent your population, demographics, or modern care "
                "pathways. **Performance on the test split is only a rough guide** — it can "
                "overfit idiosyncrasies of the data and should **not** be used for diagnosis, "
                "triage, or treatment decisions. Always rely on licensed clinicians and "
                "validated clinical tools."
            )
            st.markdown(
                f"**Source:** [UCI Machine Learning Repository — Heart Disease]"
                f"({_DATASET_LINKS[DISEASE_HEART]})"
            )
        else:
            st.markdown(
                "This model estimates the **probability of diabetes (positive class)** "
                "using PIMA-style measurements. The PIMA dataset is **well-known but dated and "
                "limited in scope** (e.g., specific population, few features). **Metrics on a "
                "held-out test split do not guarantee real-world accuracy.** The app is for "
                "**education and transparency** (SHAP, reporting), not clinical screening or "
                "self-diagnosis."
            )
            st.markdown(
                f"**Source (dataset mirror / common host):** "
                f"[Kaggle — Pima Indians Diabetes Database]"
                f"({_DATASET_LINKS[DISEASE_DIABETES]})"
            )


_FOLLOWUP_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    DISEASE_DIABETES: (
        "What does high glucose mean for me long term?",
        "How can I lower my BMI effectively?",
        "Should I get tested for pre-diabetes?",
    ),
    DISEASE_HEART: (
        "What does my cholesterol level mean?",
        "Is my blood pressure dangerous?",
        "What lifestyle changes help most for heart health?",
    ),
}


def _followup_disease_label(disease: str) -> str:
    return "heart disease" if disease == DISEASE_HEART else "diabetes"


def _format_top_shap_summary(row: pd.DataFrame, disease: str) -> str:
    shap_vals, _, fnames = get_shap_values(row, disease=disease)
    sv = [float(x) for x in shap_vals]
    indexed = list(enumerate(sv))
    indexed.sort(key=lambda t: abs(t[1]), reverse=True)
    parts: list[str] = []
    for feat_idx, val in indexed[:3]:
        feat = fnames[int(feat_idx)]
        if val > 0:
            parts.append(f"{feat} (increasing risk)")
        elif val < 0:
            parts.append(f"{feat} (decreasing risk)")
        else:
            parts.append(f"{feat} (neutral)")
    return "; ".join(parts) if parts else "(none)"


def _build_followup_system_prompt(
    disease: str,
    risk_label: str,
    risk_pct: float,
    metrics_line: str,
    top_shap_line: str,
    initial_explanation: str,
) -> str:
    d = _followup_disease_label(disease)
    return (
        "You are a knowledgeable, empathetic health advisor. A patient has just "
        f"received their {d} risk assessment. Here is their context:\n\n"
        f"Risk Level: {risk_label}\n"
        f"Risk Score: {risk_pct:.1f}%\n"
        f"Their Metrics: {metrics_line}\n"
        f"Top Risk Factors: {top_shap_line}\n"
        f"Initial Explanation: {initial_explanation}\n\n"
        "Answer the patient's follow-up questions in plain English.\n"
        "Be warm, clear, and specific to their numbers.\n"
        "Always end with: 'Remember to consult your doctor for personalized medical advice.'\n"
        "Keep answers to 2–3 sentences unless more detail is clearly needed."
    )


def _process_followup_user_message(
    user_message: str,
    *,
    row: pd.DataFrame,
    disease: str,
    risk_label: str,
    risk_pct: float,
    metrics_line: str,
    initial_explanation: str,
) -> None:
    prior = [dict(m) for m in st.session_state["chat_history"]]
    st.session_state["chat_history"].append({"role": "user", "content": user_message})
    try:
        top_shap = _format_top_shap_summary(row, disease)
    except Exception:  # pragma: no cover
        top_shap = "(SHAP unavailable)"
    system = _build_followup_system_prompt(
        disease,
        risk_label,
        risk_pct,
        metrics_line,
        top_shap,
        initial_explanation,
    )
    reply = run_followup_chat(system, prior, user_message)
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
    st.rerun()


def _render_followup_chat_section(
    row: pd.DataFrame,
    disease: str,
    risk_label: str,
    risk_pct: float,
    feat_order: list[str],
    input_d: dict[str, float],
    initial_explanation: str,
    _pk: str,
) -> None:
    st.session_state.setdefault("chat_history", [])
    metrics_line = ", ".join(f"{k}: {input_d[k]}" for k in feat_order)

    title_col, btn_col = st.columns([4, 1])
    with title_col:
        st.markdown("##### 💬 Ask a Follow-Up Question")
    with btn_col:
        if st.button("🗑️ Clear Chat", key=f"btn_clear_chat_{_pk}"):
            st.session_state["chat_history"] = []
            st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    hist = st.session_state["chat_history"]
    if len(hist) == 0:
        st.caption("Try one of these to get started:")
        sug = _FOLLOWUP_SUGGESTIONS.get(disease, _FOLLOWUP_SUGGESTIONS[DISEASE_DIABETES])
        s1, s2, s3 = st.columns(3)
        for i, q in enumerate(sug):
            col = (s1, s2, s3)[i]
            with col:
                if st.button(q, key=f"fq_suggest_{_pk}_{i}"):
                    _process_followup_user_message(
                        q,
                        row=row,
                        disease=disease,
                        risk_label=risk_label,
                        risk_pct=risk_pct,
                        metrics_line=metrics_line,
                        initial_explanation=initial_explanation,
                    )

    user_inp = st.chat_input("Ask about your results...", key=f"followup_chat_input_{_pk}")
    if user_inp and user_inp.strip():
        _process_followup_user_message(
            user_inp.strip(),
            row=row,
            disease=disease,
            risk_label=risk_label,
            risk_pct=risk_pct,
            metrics_line=metrics_line,
            initial_explanation=initial_explanation,
        )


def _clear_disease_switch_state() -> None:
    """Reset explanation and simulator session keys when disease model changes."""
    for k in ("llm_explanation", "_llm_explanation_row_sig", "_risk_sim_sig", "chat_history"):
        st.session_state.pop(k, None)
    for f in SIMULATOR_FEATURES:
        st.session_state.pop(f"sim_{f}", None)
    for f in HEART_SIMULATOR_FEATURES:
        st.session_state.pop(f"sim_{f}", None)


def _render_landing_screen() -> None:
    st.markdown(
        """
<div style="text-align:center; padding-top: 1rem;">
  <h1 style="margin-bottom: 0.35rem;">🏥 Personal Health Risk Explorer</h1>
  <p style="font-size:1.05rem; color:#cbd5e1; max-width:900px; margin:0 auto;">
    Assess your risk for diabetes and heart disease using AI-powered analysis.
    Get personalized insights, explainable predictions, and a downloadable health report.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div style='background:#1e2a3a; border-radius:12px; padding:24px;
text-align:center; border:1px solid #2e4a6a;'>
  <div style='font-size:2rem;'>📋</div>
  <div style='font-size:1.1rem; font-weight:700; margin-top:0.4rem;'>① Enter Data</div>
  <div style='margin-top:0.45rem; color:#cbd5e1;'>Input your health metrics using sliders or upload a CSV file</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div style='background:#1e2a3a; border-radius:12px; padding:24px;
text-align:center; border:1px solid #2e4a6a;'>
  <div style='font-size:2rem;'>🔬</div>
  <div style='font-size:1.1rem; font-weight:700; margin-top:0.4rem;'>② Analyze Risk</div>
  <div style='margin-top:0.45rem; color:#cbd5e1;'>Our XGBoost model calculates your risk score with SHAP explainability</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div style='background:#1e2a3a; border-radius:12px; padding:24px;
text-align:center; border:1px solid #2e4a6a;'>
  <div style='font-size:2rem;'>💡</div>
  <div style='font-size:1.1rem; font-weight:700; margin-top:0.4rem;'>③ Understand & Act</div>
  <div style='margin-top:0.45rem; color:#cbd5e1;'>Get an AI-written plain-English explanation and download your report</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<style>
div.stButton > button {
  width: 100%;
  height: 60px;
  font-size: 1.1em;
  font-weight: bold;
  border-radius: 10px;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("is_mobile"):
        b1 = st.container()
        b2 = st.container()
    else:
        b1, b2 = st.columns(2)
    with b1:
        if st.button("🎯 Try with Sample Data", key="landing_try_sample"):
            sample_row: dict[str, float] = {}
            if _SAMPLE_CSV_PATH.exists():
                try:
                    sdf = pd.read_csv(_SAMPLE_CSV_PATH)
                    if len(sdf) > 0:
                        sample_row = sdf.iloc[0].to_dict()
                except Exception:  # pragma: no cover
                    sample_row = {}
            st.session_state["landing_sample_row"] = sample_row
            st.session_state["input_mode"] = "sample"
            st.session_state["input_mode_ui"] = "Try a Sample"
            st.session_state["app_stage"] = "input"
            st.rerun()
    with b2:
        if st.button("📂 Upload My Own CSV", key="landing_upload_csv"):
            st.session_state["input_mode"] = "upload"
            st.session_state["input_mode_ui"] = "Upload Your Data"
            st.session_state["app_stage"] = "input"
            st.rerun()

    st.caption(
        "⚠️ This tool is for educational purposes only.\n"
        "Not a substitute for professional medical advice."
    )


BADGE_STYLES = {
    "Low": {
        "text": "Low Risk",
        "fg": "#bbf7d0",
        "bg": "rgba(22, 163, 74, 0.25)",
        "border": "rgba(34, 197, 94, 0.55)",
    },
    "Moderate": {
        "text": "Moderate Risk",
        "fg": "#fef08a",
        "bg": "rgba(234, 179, 8, 0.2)",
        "border": "rgba(250, 204, 21, 0.45)",
    },
    "High": {
        "text": "High Risk",
        "fg": "#fecaca",
        "bg": "rgba(239, 68, 68, 0.22)",
        "border": "rgba(248, 113, 113, 0.5)",
    },
}


def _inject_product_css() -> None:
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Outfit:wght@500;600;700&display=swap');

  html, body, [class*="stApp"] {
    font-family: "DM Sans", system-ui, sans-serif;
  }
  h1, h2, h3, .product-title {
    font-family: "Outfit", "DM Sans", sans-serif !important;
    letter-spacing: -0.02em;
  }
  .product-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 0 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.25rem;
  }
  .product-header .emoji {
    font-size: 2.25rem;
    line-height: 1;
  }
  .product-header .titles h1 {
    font-size: clamp(1.55rem, 2.4vw, 2rem);
    margin: 0;
    font-weight: 700;
    color: #f4f4f5 !important;
    border: none !important;
  }
  .product-header .titles p {
    margin: 0.2rem 0 0 0;
    color: #94a3b8;
    font-size: 0.98rem;
  }
  div[data-testid="stSidebarContent"] {
    background: linear-gradient(195deg, #0f1419 0%, #0a0c10 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  section.main > div {
    padding-top: 1.25rem;
  }
  .stMetric {
    background: transparent;
  }
  div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    background: rgba(255,255,255,0.02);
  }
  textarea:disabled {
    opacity: 0.85;
    -webkit-text-fill-color: #94a3b8;
  }
  @media (max-width: 768px) {
    section[data-testid="stSidebar"] {
      width: 0px !important;
      min-width: 0px !important;
    }
    .main .block-container {
      padding-left: 1rem;
      padding-right: 1rem;
      max-width: 100%;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    .stButton > button { font-size: 0.9em !important; }
    .stSlider { padding: 0 !important; }
    [data-testid="stChatMessageContainer"] {
      padding-bottom: 120px !important;
    }
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _clinical_badge_markdown(feature: str, value, disease: str) -> str:
    if disease == DISEASE_HEART:
        emoji, label, color_hex = get_heart_badge(feature, value)
    else:
        emoji, label, color_hex = get_badge(feature, value)
    return (
        f"{emoji} <span style='background:{color_hex}; padding:2px 8px; "
        f"border-radius:10px; color:white; font-size:0.75em'>{label}</span>"
    )


def _glossary_for_disease(disease: str) -> dict[str, dict]:
    return HEART_GLOSSARY if disease == DISEASE_HEART else DIABETES_GLOSSARY


def _render_feature_info_panel(feature: str, disease: str) -> None:
    entry = _glossary_for_disease(disease).get(feature)
    if not entry:
        return
    st.markdown(
        f"""
<div style='background:#1a2a3a; padding:12px; border-radius:8px;
border-left:3px solid #3498db;'>
<b>{entry['label']}</b><br>
{entry['plain']}<br><br>
<span style='color:#aaa'>📏 Normal range: {entry['normal_range']}</span><br>
<i style='color:#7fb3d3'>💡 {entry['why_it_matters']}</i>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_info_button(feature: str, disease: str, key_suffix: str) -> None:
    tkey = f"glossary_open_{disease}_{feature}_{key_suffix}"
    if st.button("ℹ️", key=f"btn_{tkey}", help="Feature explanation"):
        st.session_state[tkey] = not bool(st.session_state.get(tkey, False))
    if st.session_state.get(tkey, False):
        with st.expander("Feature details", expanded=True):
            _render_feature_info_panel(feature, disease)


def _coerce_feature_value(key: str, raw: float) -> float:
    spec = SLIDER_CONFIG[key]
    return float(int(round(raw))) if spec["int"] else float(raw)


def _risk_label_from_score(score_0_1: float) -> str:
    if score_0_1 > 0.6:
        return "High"
    if score_0_1 > 0.3:
        return "Moderate"
    return "Low"


def _population_percentile_figure(
    pct_map: dict[str, int], feature_order: list[str]
) -> go.Figure:
    names = [f for f in feature_order if f in pct_map]
    vals = [pct_map[f] for f in names]
    colors = [
        "#22c55e" if v < 50 else "#eab308" if v <= 75 else "#ef4444"
        for v in vals
    ]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            customdata=[[v] for v in vals],
            hovertemplate=(
                "Your %{y} is higher than %{customdata[0]}% of the population<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis=dict(
            title="Population Percentile",
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(148,163,184,0.2)",
            zeroline=False,
        ),
        yaxis=dict(title=None, automargin=True),
        height=max(320, 40 * len(names)),
        margin=dict(l=140, r=48, t=40, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 25, 34, 0.55)",
        font=dict(color="#e8eaef"),
        title=dict(
            text="<b>📊 How You Compare to the Population</b>",
            font=dict(size=16, color="#f4f4f5"),
            x=0,
            xanchor="left",
        ),
    )
    fig.add_shape(
        type="line",
        x0=50,
        x1=50,
        y0=0,
        y1=1,
        yref="paper",
        xref="x",
        line=dict(color="rgba(248, 250, 252, 0.55)", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=50,
        y=1.02,
        yref="paper",
        xref="x",
        text="Average",
        showarrow=False,
        font=dict(color="#94a3b8", size=12),
        xanchor="center",
    )
    return fig


def build_animated_gauge(
    final_score: float, risk_label: str, title: str = "Risk Score", height: int = 300
) -> go.Figure:
    final_score = float(min(1.0, max(0.0, final_score)))
    bar_color = (
        "#e74c3c"
        if risk_label == "High"
        else "#f39c12"
        if risk_label == "Moderate"
        else "#2ecc71"
    )
    frames: list[go.Frame] = []
    steps = 30
    for i in range(steps + 1):
        current = (final_score * i) / steps
        frames.append(
            go.Frame(
                data=[
                    go.Indicator(
                        mode="gauge+number",
                        value=current * 100.0,
                        number={"suffix": "%", "font": {"size": 36}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": bar_color},
                            "steps": [
                                {"range": [0, 30], "color": "#1a3a1a"},
                                {"range": [30, 60], "color": "#3a2a00"},
                                {"range": [60, 100], "color": "#3a0000"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.75,
                                "value": final_score * 100.0,
                            },
                        },
                    )
                ],
                name=str(i),
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title={"text": title, "x": 0.5},
            height=height,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 30, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        }
                    ],
                    "visible": False,
                }
            ],
        ),
    )
    return fig


def _sim_row_signature(row_df: pd.DataFrame, cols: list[str]) -> tuple[float, ...]:
    return tuple(round(float(row_df[c].iloc[0]), 6) for c in cols)


def _sync_simulator_to_row(row_df: pd.DataFrame, sim_feats: tuple[str, ...]) -> None:
    cols = list(row_df.columns)
    sig = _sim_row_signature(row_df, cols)
    if st.session_state.get("_risk_sim_sig") != sig:
        st.session_state["_risk_sim_sig"] = sig
        for f in sim_feats:
            st.session_state[f"sim_{f}"] = float(row_df[f].iloc[0])


def _heart_sim_coerce(feat: str, raw: float) -> float:
    if feat == "oldpeak":
        return float(round(raw, 1))
    return float(int(round(raw)))


def _render_risk_trajectory_simulator(row_df: pd.DataFrame, disease: str) -> None:
    st.markdown("### 🎯 Risk Trajectory Simulator — What If You Made Changes?")
    st.subheader("Adjust the sliders below to simulate lifestyle changes")

    if disease == DISEASE_HEART:
        _sync_simulator_to_row(row_df, HEART_SIMULATOR_FEATURES)
        heart_specs: dict[str, dict] = {
            "trestbps": {"min": 80, "max": 200, "step": 1},
            "chol": {"min": 100, "max": 600, "step": 1},
            "thalach": {"min": 60, "max": 220, "step": 1},
            "oldpeak": {"min": 0.0, "max": 7.0, "step": 0.1},
        }
        pair_rows = (("trestbps", "chol"), ("thalach", "oldpeak"))
        for f1, f2 in pair_rows:
            c_a, c_b = st.columns(2)
            for feat, col in ((f1, c_a), (f2, c_b)):
                sp = heart_specs[feat]
                with col:
                    st.slider(
                        HEART_SIM_LABELS[feat],
                        min_value=float(sp["min"]),
                        max_value=float(sp["max"]),
                        step=float(sp["step"]),
                        key=f"sim_{feat}",
                    )

        baseline = {c: float(row_df[c].iloc[0]) for c in HEART_FEATURE_COLS}
        modified = dict(baseline)
        for f in HEART_SIMULATOR_FEATURES:
            modified[f] = _heart_sim_coerce(f, float(st.session_state[f"sim_{f}"]))

        orig_row = pd.DataFrame([baseline])[HEART_FEATURE_COLS]
        sim_row = pd.DataFrame([modified])[HEART_FEATURE_COLS]

        try:
            orig_score = float(predict_heart_risk(orig_row)["risk_score"])
            sim_score = float(predict_heart_risk(sim_row)["risk_score"])
        except Exception:  # pragma: no cover
            return
    else:
        _sync_simulator_to_row(row_df, SIMULATOR_FEATURES)
        pair_rows = (("BMI", "Glucose"), ("BloodPressure", "Insulin"))
        for f1, f2 in pair_rows:
            c_a, c_b = st.columns(2)
            for feat, col in ((f1, c_a), (f2, c_b)):
                spec = SLIDER_CONFIG[feat]
                with col:
                    st.slider(
                        SIMULATOR_LABELS[feat],
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        step=float(spec["step"]),
                        key=f"sim_{feat}",
                    )

        baseline = {c: float(row_df[c].iloc[0]) for c in FEATURE_COLS}
        modified = dict(baseline)
        for f in SIMULATOR_FEATURES:
            raw = float(st.session_state[f"sim_{f}"])
            modified[f] = _coerce_feature_value(f, raw)

        orig_row = pd.DataFrame([baseline])[FEATURE_COLS]
        sim_row = pd.DataFrame([modified])[FEATURE_COLS]

        try:
            orig_score = float(predict_diabetes_risk(orig_row)["risk_score"])
            sim_score = float(predict_diabetes_risk(sim_row)["risk_score"])
        except Exception:  # pragma: no cover
            return

    o_pct = orig_score * 100.0
    s_pct = sim_score * 100.0

    _sk = "h" if disease == DISEASE_HEART else "d"
    g_left, g_right = st.columns(2)
    sim_gauge_h = 250 if st.session_state.get("is_mobile") else 240
    with g_left:
        curr_label = _risk_label_from_score(orig_score)
        st.plotly_chart(
            build_animated_gauge(orig_score, curr_label, "Current Risk", height=sim_gauge_h),
            use_container_width=True,
            theme=None,
            key=f"sim_gauge_current_{_sk}",
            config={"displayModeBar": False},
        )
    with g_right:
        sim_label = _risk_label_from_score(sim_score)
        st.plotly_chart(
            build_animated_gauge(sim_score, sim_label, "Simulated Risk", height=sim_gauge_h),
            use_container_width=True,
            theme=None,
            key=f"sim_gauge_simulated_{_sk}",
            config={"displayModeBar": False},
        )

    delta_pp = s_pct - o_pct
    if abs(delta_pp) < 0.05:
        st.markdown(
            "<p style='text-align:center;color:#94a3b8;font-size:1.05rem;'>→ No change in risk score</p>",
            unsafe_allow_html=True,
        )
    elif delta_pp < 0:
        st.markdown(
            f"<p style='text-align:center;color:#4ade80;font-size:1.05rem;font-weight:600;'>"
            f"✅ Your changes could reduce your risk by {-delta_pp:.1f} percentage points</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='text-align:center;color:#f87171;font-size:1.05rem;font-weight:600;'>"
            f"⚠️ These changes increase your risk by {delta_pp:.1f} percentage points</p>",
            unsafe_allow_html=True,
        )


def _render_inputs_sample(disease: str) -> pd.DataFrame | None:
    if disease == DISEASE_HEART:
        st.subheader("Patient parameters")
        st.caption(
            "Adjust inputs to explore how the model responds to UCI-style heart disease features."
        )
        c1, c2 = st.columns(2)
        values: dict[str, float] = {}

        def _pair(i: int) -> object:
            return c1 if i % 2 == 0 else c2

        i = 0
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["age"] = float(
                    st.slider("Age (years)", 20, 80, 45, key="heart_sample_age")
                )
            with row_i:
                _render_info_button("age", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('age', values['age'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                sex_ix = st.selectbox(
                    "Sex",
                    options=[0, 1],
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    index=1,
                    key="heart_sample_sex",
                )
                values["sex"] = float(sex_ix)
            with row_i:
                _render_info_button("sex", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('sex', values['sex'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["cp"] = float(
                    st.slider(
                        "Chest pain type (cp)",
                        0,
                        3,
                        0,
                        help=_CP_HELP,
                        key="heart_sample_cp",
                    )
                )
            with row_i:
                _render_info_button("cp", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('cp', values['cp'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["trestbps"] = float(
                    st.slider(
                        "Resting blood pressure (trestbps, mm Hg)",
                        80,
                        200,
                        120,
                        key="heart_sample_trestbps",
                    )
                )
            with row_i:
                _render_info_button("trestbps", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('trestbps', values['trestbps'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["chol"] = float(
                    st.slider(
                        "Cholesterol (chol, mg/dL)",
                        100,
                        600,
                        200,
                        key="heart_sample_chol",
                    )
                )
            with row_i:
                _render_info_button("chol", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('chol', values['chol'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                fbs_ix = st.selectbox(
                    "Fasting blood sugar > 120 mg/dL (fbs)",
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key="heart_sample_fbs",
                )
                values["fbs"] = float(fbs_ix)
            with row_i:
                _render_info_button("fbs", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('fbs', values['fbs'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["restecg"] = float(
                    st.slider("Resting ECG (restecg)", 0, 2, 0, key="heart_sample_restecg")
                )
            with row_i:
                _render_info_button("restecg", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('restecg', values['restecg'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["thalach"] = float(
                    st.slider(
                        "Max heart rate (thalach)",
                        60,
                        220,
                        150,
                        key="heart_sample_thalach",
                    )
                )
            with row_i:
                _render_info_button("thalach", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('thalach', values['thalach'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                ex_ix = st.selectbox(
                    "Exercise induced angina (exang)",
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key="heart_sample_exang",
                )
                values["exang"] = float(ex_ix)
            with row_i:
                _render_info_button("exang", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('exang', values['exang'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["oldpeak"] = float(
                    st.slider(
                        "ST depression (oldpeak)",
                        0.0,
                        7.0,
                        1.0,
                        0.1,
                        key="heart_sample_oldpeak",
                    )
                )
            with row_i:
                _render_info_button("oldpeak", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('oldpeak', values['oldpeak'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["slope"] = float(
                    st.slider("Slope of ST segment (slope)", 0, 2, 1, key="heart_sample_slope")
                )
            with row_i:
                _render_info_button("slope", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('slope', values['slope'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["ca"] = float(
                    st.slider(
                        "Major vessels colored (ca)",
                        0,
                        4,
                        0,
                        key="heart_sample_ca",
                    )
                )
            with row_i:
                _render_info_button("ca", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('ca', values['ca'], disease)}</div>",
                    unsafe_allow_html=True,
                )
        i += 1
        with _pair(i):
            row_l, row_i, row_r = st.columns([3, 0.5, 1])
            with row_l:
                values["thal"] = float(
                    st.slider(
                        "Thalassemia (thal)",
                        0,
                        3,
                        2,
                        help=_THAL_HELP,
                        key="heart_sample_thal",
                    )
                )
            with row_i:
                _render_info_button("thal", disease, "sample")
            with row_r:
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown('thal', values['thal'], disease)}</div>",
                    unsafe_allow_html=True,
                )

        cleaned, val_warns = validate_and_clean_input(values, disease=disease)
        for w in val_warns:
            st.warning(w)
        return pd.DataFrame([cleaned])[HEART_FEATURE_COLS]

    st.subheader("Patient parameters")
    st.caption("Adjust sliders to explore how the model responds to realistic PIMA-style inputs.")
    keys = list(SLIDER_CONFIG.keys())
    c1, c2 = st.columns(2)
    values_d: dict[str, float] = {}
    for i, key in enumerate(keys):
        spec = SLIDER_CONFIG[key]
        col = c1 if i % 2 == 0 else c2
        with col:
            row_l, row_i, row_r = st.columns([4, 0.5, 1])
            with row_l:
                v = st.slider(
                    spec["label"],
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    step=float(spec["step"]),
                    key=f"sample_{key}",
                )
            with row_i:
                _render_info_button(key, disease, "sample")
            with row_r:
                cv = _coerce_feature_value(key, v)
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>"
                    f"{_clinical_badge_markdown(key, cv, disease)}</div>",
                    unsafe_allow_html=True,
                )
            values_d[key] = cv
    cleaned, val_warns = validate_and_clean_input(values_d, disease=disease)
    for w in val_warns:
        st.warning(w)
    return pd.DataFrame([cleaned])[FEATURE_COLS]


_HEART_UPLOAD_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "sex": "Sex (0=F, 1=M)",
    "cp": "Chest pain type (cp)",
    "trestbps": "Resting BP (trestbps)",
    "chol": "Cholesterol (chol)",
    "fbs": "Fasting BS >120 (fbs)",
    "restecg": "Resting ECG (restecg)",
    "thalach": "Max HR (thalach)",
    "exang": "Exercise angina (exang)",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST slope",
    "ca": "Major vessels (ca)",
    "thal": "Thal",
}


def _render_inputs_upload(disease: str) -> pd.DataFrame | None:
    st.subheader("Your dataset")
    up = st.file_uploader("CSV file", type=["csv"], key="csv_upload")
    cols = HEART_FEATURE_COLS if disease == DISEASE_HEART else FEATURE_COLS
    if up is None:
        if disease == DISEASE_HEART:
            st.info(
                "Upload a `.csv` with the 13 heart disease feature columns "
                "(exact names: age, sex, cp, trestbps, chol, fbs, restecg, "
                "thalach, exang, oldpeak, slope, ca, thal). Optional: `target`."
            )
        else:
            st.info("Upload a `.csv` that includes the eight PIMA feature columns (exact names).")
        return None
    try:
        df = pd.read_csv(up)
    except Exception as e:  # pragma: no cover
        st.error(f"Could not read CSV: {e}")
        return None

    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(
            "**Some required columns are missing.**\n\n"
            f"Add these columns (spelling and case must match): `{', '.join(missing)}`.\n\n"
            f"**Required:** `{', '.join(cols)}`"
        )
        return None

    if len(df) == 0:
        st.warning("The file has no rows to analyze.")
        return None

    idx = st.selectbox(
        "Row to analyze",
        options=list(range(len(df))),
        format_func=lambda i: f"Row {i}",
        key="upload_row_idx",
    )
    row_coerced = df.iloc[[idx]][cols].apply(pd.to_numeric, errors="coerce")
    raw_dict = {c: row_coerced[c].iloc[0] for c in cols}
    cleaned, val_warns = validate_and_clean_input(raw_dict, disease=disease)
    for w in val_warns:
        st.warning(w)
    row = pd.DataFrame([cleaned])[cols]

    st.markdown("##### Row values & clinical badges")
    uc1, uc2 = st.columns(2)
    for i, feat in enumerate(cols):
        col = uc1 if i % 2 == 0 else uc2
        raw = float(cleaned[feat])
        if disease == DISEASE_HEART:
            disp = float(raw) if feat == "oldpeak" else float(int(round(raw)))
            lbl = _HEART_UPLOAD_LABELS.get(feat, feat)
        else:
            spec = SLIDER_CONFIG[feat]
            disp = float(int(round(raw))) if spec["int"] else float(raw)
            lbl = spec["label"]
        with col:
            row_l, row_r = st.columns([3, 1])
            with row_l:
                st.markdown(
                    f"<div style='padding-top:0.35rem'><strong>{lbl}</strong> · `{disp}`</div>",
                    unsafe_allow_html=True,
                )
            with row_r:
                st.markdown(
                    f"<div style='padding-top:0.35rem; text-align:right'>"
                    f"{_clinical_badge_markdown(feat, disp, disease)}</div>",
                    unsafe_allow_html=True,
                )
    return row


def _render_shared_output(row: pd.DataFrame, disease: str) -> None:
    _pk = "h" if disease == DISEASE_HEART else "d"
    try:
        if disease == DISEASE_HEART:
            out = predict_heart_risk(row)
        else:
            out = predict_diabetes_risk(row)
    except FileNotFoundError:
        # Deployment safety net: if model files are missing at runtime, bootstrap once.
        with st.spinner("🔄 Model files missing. Running setup now, please wait..."):
            try:
                setup_models()
                if disease == DISEASE_HEART:
                    out = predict_heart_risk(row)
                else:
                    out = predict_diabetes_risk(row)
            except Exception:
                train_cmd = (
                    "`python model/train_heart.py`"
                    if disease == DISEASE_HEART
                    else "`python model/train.py`"
                )
                st.warning(f"Train the model first: {train_cmd} from the project root.")
                return
    except Exception as e:  # pragma: no cover
        st.error(f"Prediction failed: {e}")
        return

    score = float(out["risk_score"])
    label_key = out["risk_label"]
    pct = score * 100.0
    custom_threshold = float(st.session_state.get("custom_threshold", 0.50))
    at_risk_flag = bool(score >= custom_threshold)
    mode_name, _ = _threshold_mode(custom_threshold)
    feat_order = HEART_FEATURE_COLS if disease == DISEASE_HEART else FEATURE_COLS
    st.session_state["progress_step"] = max(int(st.session_state.get("progress_step", 1)), 2)

    st.markdown("### Results")

    g1, g2 = st.columns([1.25, 1])
    with g1:
        gauge_title = "Heart Disease Risk Score" if disease == DISEASE_HEART else "Diabetes Risk Score"
        main_gauge_h = 250 if st.session_state.get("is_mobile") else 300
        st.plotly_chart(
            build_animated_gauge(score, label_key, gauge_title, height=main_gauge_h),
            use_container_width=True,
            theme=None,
            key=f"gauge_chart_{_pk}",
            config={"displayModeBar": False},
        )
    with g2:
        st.markdown("<br>", unsafe_allow_html=True)
        b = BADGE_STYLES[label_key]
        st.markdown(
            f"""
<div style="padding: 1.5rem 1.25rem; border-radius: 14px; border: 1px solid {b["border"]};
background: {b["bg"]}; margin-top: 0.5rem;">
  <div style="font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Risk tier</div>
  <div style="font-size: 1.75rem; font-weight: 700; color: {b["fg"]}; margin-top: 0.35rem; font-family: Outfit, sans-serif;">{b["text"]}</div>
  <div style="color: #94a3b8; margin-top: 0.75rem; font-size: 0.92rem; line-height: 1.45;">
    Bands: <strong style="color:#86efac;">&lt;30%</strong> low ·
    <strong style="color:#fde047;">30–60%</strong> moderate ·
    <strong style="color:#fca5a5;">&gt;60%</strong> high
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Using custom threshold: {custom_threshold:.2f} ({mode_name}) · "
            f"Screening decision: {'At risk' if at_risk_flag else 'Not flagged'}"
        )

    input_d = {c: float(row[c].iloc[0]) for c in feat_order}
    _row_sig = tuple(round(float(input_d[c]), 6) for c in feat_order)
    _prev_sig = st.session_state.get("_llm_explanation_row_sig")
    if _prev_sig is not None and _prev_sig != _row_sig:
        st.session_state.pop("llm_explanation", None)
        st.session_state["chat_history"] = []
    st.session_state["_llm_explanation_row_sig"] = _row_sig

    pct_map: dict[str, int] = {}
    _pop_load_err: str | None = None
    try:
        pct_map = get_percentiles(input_d, disease=disease)
    except FileNotFoundError as e:
        _pop_load_err = str(e)
    except Exception as e:  # pragma: no cover
        _pop_load_err = f"Could not load population reference: {e}"

    with st.expander("📊 Population Comparison", expanded=False):
        st.markdown("#### 📊 How You Compare to the Population")
        if _pop_load_err is not None:
            if "Missing" in _pop_load_err or "Run model" in _pop_load_err:
                st.caption(_pop_load_err)
            else:
                st.warning(_pop_load_err)
        elif not pct_map:
            st.caption("No percentiles computed for the current row.")
        else:
            pfig = _population_percentile_figure(pct_map, feat_order)
            st.plotly_chart(
                _style_plotly(pfig),
                use_container_width=True,
                theme=None,
                key=f"population_pct_chart_{_pk}",
            )

    shap_fig = None
    try:
        shap_fig = _style_plotly(build_shap_waterfall_chart(row, disease=disease))
    except Exception as e:  # pragma: no cover
        st.error(f"SHAP chart failed: {e}")
    else:
        st.plotly_chart(
            shap_fig,
            use_container_width=True,
            theme=None,
            key=f"shap_waterfall_{_pk}",
        )

    def _build_llm_payload() -> dict[str, float]:
        payload = {c: float(row[c].iloc[0]) for c in feat_order}
        payload[RISK_SCORE_PCT_KEY] = score * 100.0
        return payload

    def _run_llm_explanation() -> str:
        shap_vals, _, fnames = get_shap_values(row, disease=disease)
        return generate_explanation(
            label_key,
            shap_vals,
            list(fnames),
            _build_llm_payload(),
            disease=disease,
        )

    st.markdown("##### 🤖 AI explanation")
    b_gen, b_regen = st.columns(2)
    with b_gen:
        clicked_gen = st.button(
            "🤖 Generate AI Explanation",
            key=f"btn_gen_ai_explanation_{_pk}",
            use_container_width=True,
        )
    with b_regen:
        clicked_regen = st.button(
            "🔄 Regenerate Explanation",
            key=f"btn_regen_ai_explanation_{_pk}",
            use_container_width=True,
        )

    if clicked_gen:
        st.session_state["chat_history"] = []
        with st.spinner("Generating your personalized explanation..."):
            try:
                st.session_state["llm_explanation"] = _run_llm_explanation()
            except Exception:  # pragma: no cover
                st.session_state["llm_explanation"] = (
                    "Explanation unavailable. Please check your API connection and try again."
                )

    if clicked_regen:
        st.session_state.pop("llm_explanation", None)
        st.session_state["chat_history"] = []
        with st.spinner("Generating your personalized explanation..."):
            try:
                st.session_state["llm_explanation"] = _run_llm_explanation()
            except Exception:  # pragma: no cover
                st.session_state["llm_explanation"] = (
                    "Explanation unavailable. Please check your API connection and try again."
                )

    _llm_text = st.session_state.get("llm_explanation")
    if _llm_text:
        st.session_state["progress_step"] = max(int(st.session_state.get("progress_step", 1)), 3)
        st.info("🤖 **Your Personalized Health Insight**\n\n" + str(_llm_text))
        st.caption(
            "⚠️ This explanation is AI-generated and for informational purposes only. "
            "Always consult a qualified healthcare professional."
        )

    if _llm_text and str(_llm_text).strip():
        _render_followup_chat_section(
            row,
            disease,
            label_key,
            pct,
            feat_order,
            input_d,
            str(_llm_text).strip(),
            _pk,
        )

    _render_risk_trajectory_simulator(row, disease)

    with st.expander("Global model context — mean |SHAP| on 200 training rows", expanded=False):
        try:
            gfig = _style_plotly(
                build_global_feature_importance_chart(200, 42, disease=disease)
            )
        except Exception as e:  # pragma: no cover
            st.error(f"Global chart failed: {e}")
        else:
            st.plotly_chart(
                gfig,
                use_container_width=True,
                theme=None,
                key=f"global_shap_{_pk}",
            )

    try:
        pdf_buf = generate_pdf_report(
            user_inputs=input_d,
            risk_score=score,
            risk_label=label_key,
            shap_fig=shap_fig,
            percentiles=pct_map if pct_map else None,
            llm_explanation=str(st.session_state.get("llm_explanation") or ""),
            disease=disease,
        )
    except Exception as e:  # pragma: no cover
        st.caption(f"Could not prepare PDF: {e}")
    else:
        clicked_pdf = st.download_button(
            "📄 Download Health Report (PDF)",
            data=pdf_buf.getvalue(),
            file_name="health_risk_report.pdf",
            mime="application/pdf",
            key=f"download_health_pdf_{_pk}",
        )
        if clicked_pdf:
            st.session_state["progress_step"] = 4


def main() -> None:
    st.set_page_config(
        page_title="Personal Health Risk Explorer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_product_css()

    mobile_q = st.query_params.get("mobile", "false")
    if isinstance(mobile_q, list):
        mobile_q = mobile_q[0] if mobile_q else "false"
    st.session_state["is_mobile"] = str(mobile_q).lower() in ("1", "true", "yes")

    if "app_stage" not in st.session_state:
        st.session_state["app_stage"] = "landing"
    if "disease_model" not in st.session_state:
        st.session_state["disease_model"] = DISEASE_DIABETES
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("custom_threshold", 0.50)

    if st.session_state["app_stage"] == "landing":
        _render_landing_screen()
        return

    st.session_state.setdefault("progress_step", 1)

    with st.sidebar:
        if st.button("← Back to Home", key="btn_back_home"):
            st.session_state.clear()
            st.session_state["app_stage"] = "landing"
            st.rerun()
        st.markdown("---")
        st.radio(
            "🫀 Select Disease Model",
            [DISEASE_DIABETES, DISEASE_HEART],
            key="disease_model",
        )
        prev_dm = st.session_state.get("_prev_disease_model", st.session_state["disease_model"])
        if prev_dm != st.session_state["disease_model"]:
            _clear_disease_switch_state()
        st.session_state["_prev_disease_model"] = st.session_state["disease_model"]

        disease = st.session_state["disease_model"]

        st.markdown("---")
        st.markdown("### Mode")
        mode_default_idx = (
            1 if st.session_state.get("input_mode") == "upload" else 0
        )
        mode = st.radio(
            "How would you like to provide inputs?",
            ["Try a Sample", "Upload Your Data"],
            index=mode_default_idx,
            label_visibility="collapsed",
            key="input_mode_ui",
        )
        st.session_state["input_mode"] = "sample" if mode == "Try a Sample" else "upload"
        st.markdown("---")
        st.markdown("##### About")
        if disease == DISEASE_HEART:
            st.markdown(
                "- **Model:** XGBoost on UCI-style heart disease data (13 features).\n"
                "- **Score:** predicted probability of heart disease (class 1), shown as %.\n"
                "- **Explainability:** SHAP waterfall shows how each value shifts risk from the cohort baseline."
            )
        else:
            st.markdown(
                "- **Model:** XGBoost on PIMA Indians Diabetes (8 features).\n"
                "- **Score:** predicted probability of diabetes (class 1), shown as %.\n"
                "- **Explainability:** SHAP waterfall shows how each value shifts risk from the cohort baseline."
            )
        with st.expander("📖 Feature Guide", expanded=False):
            gdict = _glossary_for_disease(disease)
            for feat, entry in gdict.items():
                st.markdown(f"**{entry['label']}** (`{feat}`)")
                st.caption(entry["plain"])

    if disease == DISEASE_HEART:
        header_emoji = "🫀"
        header_title = "Personal Health Risk Explorer — Heart Disease"
        header_blurb = (
            "Clinical-style heart disease risk estimation with transparent SHAP explanations "
            "— demo only, not medical advice."
        )
    else:
        header_emoji = "🩺"
        header_title = "Personal Health Risk Explorer — Diabetes"
        header_blurb = (
            "Clinical-style diabetes risk estimation with transparent SHAP explanations "
            "— demo only, not medical advice."
        )

    st.markdown(
        f"""
<div class="product-header">
  <div class="emoji" aria-hidden="true">{header_emoji}</div>
  <div class="titles">
    <h1>{header_title}</h1>
    <p>{header_blurb}</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    _render_progress_stepper(int(st.session_state.get("progress_step", 1)))

    tab_risk, tab_perf = st.tabs(["🔍 Risk Analysis", "📈 Model Performance"])

    with tab_risk:
        if mode == "Try a Sample":
            row_df = _render_inputs_sample(disease)
        else:
            row_df = _render_inputs_upload(disease)

        if row_df is not None:
            st.markdown("---")
            _render_shared_output(row_df, disease)

    with tab_perf:
        _render_model_performance_tab(disease)

    st.markdown("")
    st.caption(
        "Educational prototype — not FDA-cleared and not for clinical or diagnostic use."
    )


if __name__ == "__main__":
    main()
