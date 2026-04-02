# Personal Health Risk Explorer

**[GitHub](https://github.com/Axinion/Health-Risk-Explorer.git)**

This project is a **Streamlit** web app that estimates **diabetes risk** from eight **PIMA Indians Diabetes**–style features using **XGBoost**, then explains results with **SHAP**, population percentiles, clinical banding, an optional **Groq (Llama 3.1 8B)** narrative, a what-if simulator, and a **ReportLab** PDF report. It is meant for learning and demos—not for clinical use.

## Setup

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Configure Groq:** Add `GROQ_API_KEY` to a `.env` file in the project root (see `.env.example`).
3. **Train the model and generate data artifacts:** `python model/train.py`
4. **Run the smoke test:** `python scripts/smoke_test.py`
5. **Launch the app:** `streamlit run app.py`

Optional: use a virtual environment (`python -m venv .venv` then activate it) before step 1.

## Features

- Interactive **sliders** or **CSV upload** for all eight PIMA features  
- **Risk score** (probability × 100) with Low / Moderate / High labels  
- **Clinical badges** (Normal / Borderline / High) via `utils/health_thresholds.py`  
- **Input validation** and median imputation for missing CSV values (`utils/validation.py`)  
- **Population percentiles** vs. training cohort (`utils/population_stats.py`)  
- **SHAP** waterfall and global importance views (`model/explain.py`)  
- **Risk trajectory simulator** for BMI, glucose, blood pressure, insulin  
- **AI explanation** via **Groq** (`model/llm_explain.py`)  
- **PDF health report** download (`utils/pdf_report.py`)  
- Dark Streamlit theme (`.streamlit/config.toml`)

## Tech stack

Python, **XGBoost**, **SHAP**, **Streamlit**, **Plotly**, **Groq (Llama 3.1 8B Instant)**, **ReportLab**, **Pandas**, NumPy, scikit-learn, SciPy, Joblib, Kaleido (Plotly PNG export), Requests (dataset fetch).

## Disclaimer

**For educational purposes only. Not medical advice.** Do not use for diagnosis or treatment. Consult a qualified healthcare professional for any health decisions.
