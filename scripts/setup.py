"""First-start model bootstrap for Streamlit Cloud deployments."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def setup_models() -> None:
    diabetes_model = ROOT / "model" / "diabetes_model.pkl"
    heart_model = ROOT / "model" / "heart_model.pkl"

    if not diabetes_model.exists():
        _run([sys.executable, "model/train.py"])

    if not heart_model.exists():
        _run([sys.executable, "model/train_heart.py"])

    print("✅ Models ready")
