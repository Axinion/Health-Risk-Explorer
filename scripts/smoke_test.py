#!/usr/bin/env python3
"""
End-to-end smoke test: model load, predict, SHAP, Groq explanation.

Run after: pip install -r requirements.txt && python model/train.py
Optional: set GROQ_API_KEY in .env for a live LLM response (otherwise may get fallback text).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path when run as: python scripts/smoke_test.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd


def main() -> int:
    try:
        from model.explain import get_shap_values
        from model.llm_explain import RISK_SCORE_PCT_KEY, generate_explanation
        from model.predict import MODEL_PATH, predict_diabetes_risk
        from model.train import FEATURE_COLS

        bundle = joblib.load(MODEL_PATH)
        print(f"Loaded model bundle (keys: {list(bundle.keys())})")

        sample = {
            "Pregnancies": 3,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 100,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 35,
        }
        row = pd.DataFrame([{k: sample[k] for k in FEATURE_COLS}])[FEATURE_COLS]

        out = predict_diabetes_risk(row)
        print(f"Risk score: {out['risk_score']:.4f}")
        print(f"Risk label: {out['risk_label']}")

        shap_vals, _base, fnames = get_shap_values(row)
        order = np.argsort(-np.abs(shap_vals))[:3]
        print("Top 3 |SHAP| features:")
        for idx in order:
            i = int(idx)
            print(f"  - {fnames[i]}: {shap_vals[i]:+.4f}")

        payload = {**{k: float(sample[k]) for k in FEATURE_COLS}, RISK_SCORE_PCT_KEY: float(out["risk_score"]) * 100.0}
        llm_text = generate_explanation(out["risk_label"], shap_vals, fnames, payload)
        print("LLM explanation:")
        print(llm_text)

        print("✅ All systems OK")
        return 0
    except Exception as e:
        print(f"❌ Smoke test failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
