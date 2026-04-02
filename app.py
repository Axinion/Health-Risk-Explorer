"""
Personal Health Risk Explorer — Streamlit UI (sample sliders or CSV upload).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model.explain import (
    build_global_feature_importance_chart,
    build_shap_waterfall_chart,
    get_shap_values,
)
from model.llm_explain import RISK_SCORE_PCT_KEY, generate_explanation
from model.predict import predict_diabetes_risk
from model.train import FEATURE_COLS
from utils.health_thresholds import get_badge
from utils.pdf_report import generate_pdf_report
from utils.population_stats import get_percentiles
from utils.validation import validate_and_clean_input

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
</style>
        """,
        unsafe_allow_html=True,
    )


def _clinical_badge_markdown(feature: str, value) -> str:
    emoji, label, color_hex = get_badge(feature, value)
    return (
        f"{emoji} <span style='background:{color_hex}; padding:2px 8px; "
        f"border-radius:10px; color:white; font-size:0.75em'>{label}</span>"
    )


def _coerce_feature_value(key: str, raw: float) -> float:
    spec = SLIDER_CONFIG[key]
    return float(int(round(raw))) if spec["int"] else float(raw)


def _style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 25, 34, 0.55)",
        font=dict(color="#e8eaef", family="DM Sans, sans-serif"),
        title_font=dict(color="#f4f4f5"),
    )
    return fig


def _risk_gauge_compact(pct_0_100: float, title_html: str) -> go.Figure:
    v = float(min(100, max(0, pct_0_100)))
    needle = v
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix": "%", "font": {"size": 28, "color": "#f4f4f5"}},
            title={"text": title_html, "font": {"size": 14, "color": "#94a3b8"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#475569",
                    "tickfont": {"size": 10, "color": "#94a3b8"},
                },
                "bar": {"color": "rgba(45, 212, 191, 0.35)", "thickness": 0.2},
                "bgcolor": "#11151c",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.35)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(34, 197, 94, 0.45)"},
                    {"range": [30, 60], "color": "rgba(234, 179, 8, 0.42)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.42)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 2},
                    "thickness": 0.82,
                    "value": needle,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(t=56, b=12, l=28, r=28),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaef"),
    )
    return fig


def _population_percentile_figure(pct_map: dict[str, int]) -> go.Figure:
    names = [f for f in FEATURE_COLS if f in pct_map]
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


def _risk_gauge_figure(pct_0_100: float) -> go.Figure:
    v = float(min(100, max(0, pct_0_100)))
    needle = v
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix": "%", "font": {"size": 44, "color": "#f4f4f5"}},
            title={
                "text": "<b>Diabetes risk score</b><br><span style='font-size:0.75em;color:#94a3b8;font-weight:400'>Model probability × 100</span>",
                "font": {"size": 18, "color": "#e8eaef"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#475569",
                    "tickfont": {"color": "#94a3b8"},
                },
                "bar": {"color": "rgba(45, 212, 191, 0.35)", "thickness": 0.22},
                "bgcolor": "#11151c",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.35)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(34, 197, 94, 0.45)"},
                    {"range": [30, 60], "color": "rgba(234, 179, 8, 0.42)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.42)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.85,
                    "value": needle,
                },
            },
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(t=100, b=24, l=36, r=36),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaef"),
    )
    return fig


def _sim_row_signature(row_df: pd.DataFrame) -> tuple[float, ...]:
    return tuple(round(float(row_df[c].iloc[0]), 6) for c in FEATURE_COLS)


def _sync_simulator_to_row(row_df: pd.DataFrame) -> None:
    sig = _sim_row_signature(row_df)
    if st.session_state.get("_risk_sim_sig") != sig:
        st.session_state["_risk_sim_sig"] = sig
        for f in SIMULATOR_FEATURES:
            st.session_state[f"sim_{f}"] = float(row_df[f].iloc[0])


def _render_risk_trajectory_simulator(row_df: pd.DataFrame) -> None:
    st.markdown("### 🎯 Risk Trajectory Simulator — What If You Made Changes?")
    st.subheader("Adjust the sliders below to simulate lifestyle changes")

    _sync_simulator_to_row(row_df)

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

    g_left, g_right = st.columns(2)
    with g_left:
        st.plotly_chart(
            _risk_gauge_compact(o_pct, "<b>Current Risk</b>"),
            use_container_width=True,
            theme=None,
            key="sim_gauge_current",
        )
    with g_right:
        st.plotly_chart(
            _risk_gauge_compact(s_pct, "<b>Simulated Risk</b>"),
            use_container_width=True,
            theme=None,
            key="sim_gauge_simulated",
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


def _render_inputs_sample() -> pd.DataFrame | None:
    st.subheader("Patient parameters")
    st.caption("Adjust sliders to explore how the model responds to realistic PIMA-style inputs.")
    keys = list(SLIDER_CONFIG.keys())
    c1, c2 = st.columns(2)
    values: dict[str, float] = {}
    for i, key in enumerate(keys):
        spec = SLIDER_CONFIG[key]
        col = c1 if i % 2 == 0 else c2
        with col:
            row_l, row_r = st.columns([3, 1])
            with row_l:
                v = st.slider(
                    spec["label"],
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    step=float(spec["step"]),
                    key=f"sample_{key}",
                )
            with row_r:
                cv = _coerce_feature_value(key, v)
                st.markdown(
                    f"<div style='padding-top:1.15rem; text-align:right'>{_clinical_badge_markdown(key, cv)}</div>",
                    unsafe_allow_html=True,
                )
            values[key] = cv
    cleaned, val_warns = validate_and_clean_input(values)
    for w in val_warns:
        st.warning(w)
    return pd.DataFrame([cleaned])[FEATURE_COLS]


def _render_inputs_upload() -> pd.DataFrame | None:
    st.subheader("Your dataset")
    up = st.file_uploader("CSV file", type=["csv"], key="csv_upload")
    if up is None:
        st.info("Upload a `.csv` that includes the eight PIMA feature columns (exact names).")
        return None
    try:
        df = pd.read_csv(up)
    except Exception as e:  # pragma: no cover
        st.error(f"Could not read CSV: {e}")
        return None

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(
            "**Some required columns are missing.**\n\n"
            f"Add these columns (spelling and case must match): `{', '.join(missing)}`.\n\n"
            f"**Required:** `{', '.join(FEATURE_COLS)}`"
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
    row_coerced = df.iloc[[idx]][FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    raw_dict = {c: row_coerced[c].iloc[0] for c in FEATURE_COLS}
    cleaned, val_warns = validate_and_clean_input(raw_dict)
    for w in val_warns:
        st.warning(w)
    row = pd.DataFrame([cleaned])[FEATURE_COLS]

    st.markdown("##### Row values & clinical badges")
    uc1, uc2 = st.columns(2)
    for i, feat in enumerate(FEATURE_COLS):
        spec = SLIDER_CONFIG[feat]
        col = uc1 if i % 2 == 0 else uc2
        raw = float(cleaned[feat])
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
                    f"<div style='padding-top:0.35rem; text-align:right'>{_clinical_badge_markdown(feat, disp)}</div>",
                    unsafe_allow_html=True,
                )
    return row


def _render_shared_output(row: pd.DataFrame) -> None:
    try:
        out = predict_diabetes_risk(row)
    except FileNotFoundError:
        st.warning("Train the model first: `python model/train.py` from the project root.")
        return
    except Exception as e:  # pragma: no cover
        st.error(f"Prediction failed: {e}")
        return

    score = float(out["risk_score"])
    label_key = out["risk_label"]
    pct = score * 100.0

    st.markdown("### Results")

    g1, g2 = st.columns([1.25, 1])
    with g1:
        st.plotly_chart(
            _risk_gauge_figure(pct),
            use_container_width=True,
            theme=None,
            key="gauge_chart",
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

    input_d = {c: float(row[c].iloc[0]) for c in FEATURE_COLS}
    _row_sig = tuple(round(float(input_d[c]), 6) for c in FEATURE_COLS)
    _prev_sig = st.session_state.get("_llm_explanation_row_sig")
    if _prev_sig is not None and _prev_sig != _row_sig:
        st.session_state.pop("llm_explanation", None)
    st.session_state["_llm_explanation_row_sig"] = _row_sig

    pct_map: dict[str, int] = {}
    _pop_load_err: str | None = None
    try:
        pct_map = get_percentiles(input_d)
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
            pfig = _population_percentile_figure(pct_map)
            st.plotly_chart(
                _style_plotly(pfig),
                use_container_width=True,
                theme=None,
                key="population_pct_chart",
            )

    shap_fig = None
    try:
        shap_fig = _style_plotly(build_shap_waterfall_chart(row))
    except Exception as e:  # pragma: no cover
        st.error(f"SHAP chart failed: {e}")
    else:
        st.plotly_chart(shap_fig, use_container_width=True, theme=None, key="shap_waterfall")

    def _build_llm_payload() -> dict[str, float]:
        payload = {c: float(row[c].iloc[0]) for c in FEATURE_COLS}
        payload[RISK_SCORE_PCT_KEY] = score * 100.0
        return payload

    def _run_llm_explanation() -> str:
        shap_vals, _, fnames = get_shap_values(row)
        return generate_explanation(label_key, shap_vals, list(fnames), _build_llm_payload())

    st.markdown("##### 🤖 AI explanation")
    b_gen, b_regen = st.columns(2)
    with b_gen:
        clicked_gen = st.button("🤖 Generate AI Explanation", key="btn_gen_ai_explanation", use_container_width=True)
    with b_regen:
        clicked_regen = st.button(
            "🔄 Regenerate Explanation", key="btn_regen_ai_explanation", use_container_width=True
        )

    if clicked_gen:
        with st.spinner("Generating your personalized explanation..."):
            try:
                st.session_state["llm_explanation"] = _run_llm_explanation()
            except Exception:  # pragma: no cover
                st.session_state["llm_explanation"] = (
                    "Explanation unavailable. Please check your API connection and try again."
                )

    if clicked_regen:
        st.session_state.pop("llm_explanation", None)
        with st.spinner("Generating your personalized explanation..."):
            try:
                st.session_state["llm_explanation"] = _run_llm_explanation()
            except Exception:  # pragma: no cover
                st.session_state["llm_explanation"] = (
                    "Explanation unavailable. Please check your API connection and try again."
                )

    _llm_text = st.session_state.get("llm_explanation")
    if _llm_text:
        st.info("🤖 **Your Personalized Health Insight**\n\n" + str(_llm_text))
        st.caption(
            "⚠️ This explanation is AI-generated and for informational purposes only. "
            "Always consult a qualified healthcare professional."
        )

    _render_risk_trajectory_simulator(row)

    with st.expander("Global model context — mean |SHAP| on 200 training rows", expanded=False):
        try:
            gfig = _style_plotly(build_global_feature_importance_chart(200, 42))
        except Exception as e:  # pragma: no cover
            st.error(f"Global chart failed: {e}")
        else:
            st.plotly_chart(gfig, use_container_width=True, theme=None, key="global_shap")

    try:
        pdf_buf = generate_pdf_report(
            user_inputs=input_d,
            risk_score=score,
            risk_label=label_key,
            shap_fig=shap_fig,
            percentiles=pct_map if pct_map else None,
            llm_explanation=str(st.session_state.get("llm_explanation") or ""),
        )
    except Exception as e:  # pragma: no cover
        st.caption(f"Could not prepare PDF: {e}")
    else:
        st.download_button(
            "📄 Download Health Report (PDF)",
            data=pdf_buf.getvalue(),
            file_name="health_risk_report.pdf",
            mime="application/pdf",
            key="download_health_pdf",
        )


def main() -> None:
    st.set_page_config(
        page_title="Personal Health Risk Explorer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_product_css()

    st.markdown(
        """
<div class="product-header">
  <div class="emoji" aria-hidden="true">🏥</div>
  <div class="titles">
    <h1>Personal Health Risk Explorer</h1>
    <p>Clinical-style diabetes risk estimation with transparent SHAP explanations — demo only, not medical advice.</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Mode")
        mode = st.radio(
            "How would you like to provide inputs?",
            ["Try a Sample", "Upload Your Data"],
            index=0,
            label_visibility="collapsed",
            key="input_mode",
        )
        st.markdown("---")
        st.markdown("##### About")
        st.markdown(
            "- **Model:** XGBoost on PIMA Indians Diabetes (8 features).\n"
            "- **Score:** predicted probability of diabetes (class 1), shown as %.\n"
            "- **Explainability:** SHAP waterfall shows how each value shifts risk from the cohort baseline."
        )

    st.markdown("---")

    if mode == "Try a Sample":
        row_df = _render_inputs_sample()
    else:
        row_df = _render_inputs_upload()

    if row_df is not None:
        st.markdown("---")
        _render_shared_output(row_df)

    st.markdown("")
    st.caption(
        "Educational prototype — not FDA-cleared and not for clinical or diagnostic use."
    )


if __name__ == "__main__":
    main()
