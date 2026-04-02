"""
LLM-generated patient-facing explanations via Groq.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

GROQ_MODEL = "llama-3.1-8b-instant"

# Optional: merge this key into ``input_values`` with model probability × 100.
RISK_SCORE_PCT_KEY = "risk_score_pct"

_FALLBACK_MSG = "Explanation unavailable. Please check your API connection and try again."


def _client() -> Groq:
    import os

    key = os.getenv("GROQ_API_KEY")
    if not key or key.strip() == "your_key_here":
        raise ValueError("GROQ_API_KEY is missing or still set to the placeholder.")
    return Groq(api_key=key)


def generate_explanation(
    risk_label: str,
    shap_values: np.ndarray | Sequence[float],
    feature_names: Sequence[str],
    input_values: dict[str, float],
    disease: str = "Diabetes",
) -> str:
    """
    Produce a short empathetic explanation using Groq (``llama-3.1-8b-instant``).

    ``shap_values`` and ``feature_names`` must align (one value per feature).

    ``input_values`` should include all model features. Optionally add the key
    ``RISK_SCORE_PCT_KEY`` with model probability × 100 for the “Risk score: X%” line.

    ``disease``: ``\"Diabetes\"`` (default) or ``\"Heart Disease\"`` adjusts the prompt.
    """
    shap_arr = np.asarray(shap_values, dtype=float).ravel()
    names = list(feature_names)
    if shap_arr.shape[0] != len(names):
        raise ValueError("shap_values and feature_names must have the same length.")

    order = np.argsort(-np.abs(shap_arr))[:3]
    top_parts: list[str] = []
    for idx in order:
        feat = names[int(idx)]
        sv = float(shap_arr[int(idx)])
        if sv > 0:
            top_parts.append(f"{feat} is increasing risk")
        elif sv < 0:
            top_parts.append(f"{feat} is decreasing risk")
        else:
            top_parts.append(f"{feat} is neither increasing nor decreasing risk")

    top_risk_str = ", ".join(top_parts)

    metrics_bits: list[str] = []
    for feat in names:
        val = input_values.get(feat)
        if val is not None:
            metrics_bits.append(f"{feat}: {val}")
    metrics_str = ", ".join(metrics_bits) if metrics_bits else "(not provided)"

    rs_raw = input_values.get(RISK_SCORE_PCT_KEY)
    try:
        rs_f = float(rs_raw) if rs_raw is not None else None
    except (TypeError, ValueError):
        rs_f = None
    if rs_f is not None:
        score_line = f"- Risk score: {rs_f:.1f}%"
    else:
        score_line = f"- Risk score: (use the percentage from your assessment; level: {risk_label})"

    if disease == "Heart Disease":
        prompt = f"""You are a helpful health advisor explaining a heart disease risk assessment to a patient in plain, empathetic English. Keep medical jargon minimal.

A patient has been assessed for heart disease risk. Use the following context:

Patient assessment:
- Risk level: {risk_label}
{score_line}
- Key health metrics: {metrics_str}
- Top contributors (SHAP): {top_risk_str}

Write exactly 3 sentences:
1. Summarize the patient's overall heart disease risk level in plain English
2. Explain the single most important contributor and why it matters for cardiovascular health
3. Give one specific, actionable recommendation focused on heart-healthy habits (e.g. diet, exercise, blood pressure, follow-up care) based on their metrics

Do not use bullet points. Do not include disclaimers. Write in second person (You...)."""
    else:
        prompt = f"""You are a helpful health advisor explaining a diabetes risk assessment to a patient in plain, empathetic English. Keep medical jargon minimal.

Patient assessment:
- Risk level: {risk_label}
{score_line}
- Key health metrics: {metrics_str}
- Top risk factors: {top_risk_str}

Write exactly 3 sentences:
1. Summarize the patient's overall risk level in plain English
2. Explain the single most important risk factor and why it matters
3. Give one specific, actionable lifestyle recommendation based on their metrics

Do not use bullet points. Do not include disclaimers. Write in second person (You...)."""

    try:
        client = _client()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
        )
        text = resp.choices[0].message.content
        if not text:
            return _FALLBACK_MSG
        return text.strip()
    except Exception:
        return _FALLBACK_MSG


def run_followup_chat(
    system_prompt: str,
    prior_turns: list[dict[str, str]],
    user_message: str,
) -> str:
    """
    Multi-turn follow-up using Groq. ``prior_turns`` is chat history before this user
    message, each item ``{"role": "user"|"assistant", "content": str}``.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]
    for m in prior_turns:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        client = _client()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
        )
        text = resp.choices[0].message.content
        if not text:
            return (
                "I couldn't generate a reply just now. Please try again. "
                "Remember to consult your doctor for personalized medical advice."
            )
        return text.strip()
    except Exception:
        return (
            "I'm sorry, I couldn't reach the assistant. Please check your API connection "
            "and try again. Remember to consult your doctor for personalized medical advice."
        )
