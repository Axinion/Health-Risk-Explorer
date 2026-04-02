# 🏥 Personal Health Risk Explorer

> AI-powered diabetes and heart disease risk assessment with explainable ML and conversational health insights.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://health-risk-explorer-s.streamlit.app/)

**Live App:** [health-risk-explorer-s.streamlit.app](https://health-risk-explorer-s.streamlit.app/)

<!-- INSERT DEMO GIF HERE -->

Personal Health Risk Explorer is a production-style ML demo that turns two clinical risk models (diabetes and heart disease) into a complete, explainable decision-support workflow. Instead of showing a raw probability only, it combines calibrated model output, SHAP-based feature attribution, population percentile context, threshold tuning, and conversational AI follow-up in one polished interface. The result is an end-to-end application that demonstrates model training, explainability, product thinking, and deployment readiness using Python, XGBoost, SHAP, Plotly, Groq, and Streamlit.

## 🚀 Why This Project Stands Out

- **Dual-model architecture:** One app, two healthcare risk domains, shared UX patterns, and disease-specific logic.
- **Trust-first ML UX:** Every prediction is paired with SHAP drivers, confusion matrix/ROC metrics, and plain-English interpretation.
- **Interactive decision policy:** Threshold tuning shows sensitivity vs specificity tradeoffs in real time.
- **Real product surface:** Landing flow, mobile responsiveness, chat follow-ups, and downloadable PDF report.
- **Deployment-ready engineering:** Streamlit Cloud deployment, CI smoke tests, Docker + Render support, and reproducible dependencies.

## ✨ Features

- 🩺 **Diabetes Risk Prediction** (PIMA dataset, XGBoost)
- 🫀 **Heart Disease Risk Prediction** (UCI dataset, XGBoost)
- 📊 **SHAP Explainability** — see exactly which factors drive your risk
- 🎯 **Risk Trajectory Simulator** — "what if" scenario modeling
- 💬 **Conversational AI Follow-Up** — ask questions about your results
- 📈 **Model Performance Dashboard** — ROC curve, confusion matrix, AUC
- 👥 **Population Percentile Comparison**
- 🏷️ **Clinical Health Badges** per metric
- 📄 **Downloadable PDF Health Report**

## 🎬 30-Second Demo Flow

1. Open the live app and choose **Diabetes** or **Heart Disease**.
2. Enter sample values (or upload CSV) and generate a risk score.
3. Show the top SHAP factors that explain *why* the score moved.
4. Switch to **Model Performance** and adjust the threshold slider.
5. Ask a follow-up question in chat and export the PDF report.

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| ML Model | XGBoost | Binary classifiers (diabetes & heart disease) with calibrated probability outputs |
| Explainability | SHAP (TreeExplainer) | Per-prediction attributions, waterfall charts, global importance |
| Frontend | Streamlit | Tabs, sidebar disease selector, sliders, CSV upload, chat UI |
| Visualization | Plotly | Gauges, SHAP waterfall, ROC curves, confusion-matrix heatmaps |
| LLM | Groq API — Llama 3.1 8B Instant | Initial explanations and multi-turn follow-up grounded in patient context |
| PDF | ReportLab (+ Kaleido for figures) | One-page downloadable health summary with metrics and SHAP snapshot |
| Data | Pandas, NumPy, SciPy | Loading, cleaning, training matrices, percentile statistics |

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or newer (3.11+ recommended)
- **pip** and a virtual environment (recommended)
- A **Groq API key** for AI explanations and chat ([Groq Console](https://console.groq.com/))

### 1. Clone the repository

```bash
git clone https://github.com/Axinion/Health-Risk-Explorer.git
cd Health-Risk-Explorer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Groq

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env` and set:

```env
GROQ_API_KEY=your_key_here
```

On Streamlit Cloud, add `GROQ_API_KEY` in the **Secrets** section of your app dashboard — never commit the real `secrets.toml`.

### 5. Train both models

Training downloads or uses bundled CSVs, fits XGBoost, and writes model bundles plus metrics under `data/` and `model/`.

```bash
python model/train.py
python model/train_heart.py
```

### 6. Run the smoke test (optional)

```bash
python scripts/smoke_test.py
```

### 7. Launch the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

**Deploy:** See **[DEPLOYMENT.md](DEPLOYMENT.md)** for Streamlit Community Cloud, Docker (Fly/Railway/Render, etc.), and required secrets.

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t health-risk-explorer .

# Run with your API key
docker run -p 8501:8501 -e GROQ_API_KEY=your_key_here health-risk-explorer

# Or using docker-compose (recommended)
echo "GROQ_API_KEY=your_key_here" > .env
docker-compose up --build

# Access at http://localhost:8501
```

### Deploy to Render (Free)
1. Push this repo to GitHub
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects the Dockerfile
5. Add environment variable: GROQ_API_KEY = your key
6. Click Deploy
7. Your app is live at https://your-app-name.onrender.com

Note: Free tier spins down after inactivity.
First load after inactivity may take ~30-60 seconds while the container wakes up.
Upgrade to paid tier for always-on availability.

## 📁 Project Structure

```
.
├── app.py                      # Streamlit UI: risk analysis tab, model performance tab, chat, PDF download
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # Project documentation (this file)
├── DEPLOYMENT.md               # Streamlit Cloud, Docker, and env vars for hosting
├── Dockerfile                  # Container image (trains models at build, runs Streamlit)
├── .dockerignore               # Keeps images small; excludes .env and venv
├── .env.example                # Example environment variables for Groq
├── .gitignore                  # Ignores venv, secrets, and generated model artifacts
├── .streamlit/
│   └── config.toml             # Streamlit theme and UI defaults
├── data/
│   ├── pima_diabetes.csv       # PIMA Indians Diabetes data (downloaded if missing)
│   ├── heart.csv               # Heart disease CSV (downloaded if missing)
│   ├── training_features.npy   # Diabetes training matrix (generated by train.py)
│   ├── heart_training_features.npy  # Heart training matrix (generated by train_heart.py)
│   ├── feature_names.json      # Diabetes feature order for population stats
│   ├── heart_feature_names.json     # Heart feature order
│   ├── training_medians.json   # Imputation / validation medians (diabetes)
│   ├── heart_medians.json      # Imputation medians (heart)
│   ├── valid_ranges.json       # Allowed value ranges for diabetes inputs
│   ├── heart_valid_ranges.json # Allowed ranges for heart inputs
│   ├── diabetes_metrics.json   # Test metrics & ROC arrays (diabetes, from train.py)
│   └── heart_metrics.json      # Test metrics & ROC arrays (heart, from train_heart.py)
├── model/
│   ├── __init__.py             # Package marker
│   ├── train.py                # PIMA data prep, XGBoost training, diabetes_metrics.json
│   ├── train_heart.py          # Heart data prep, XGBoost training, heart_metrics.json
│   ├── predict.py              # Load diabetes bundle, score one row, risk bands
│   ├── predict_heart.py        # Load heart bundle, score one row, risk bands
│   ├── explain.py              # SHAP charts and global importance (both diseases)
│   ├── llm_explain.py          # Groq prompts: initial explanation + follow-up chat
│   ├── diabetes_model.pkl      # Serialized diabetes model + metadata (gitignored by default)
│   └── heart_model.pkl         # Serialized heart model + metadata (gitignored by default)
├── scripts/
│   └── smoke_test.py           # End-to-end check: predict, SHAP, LLM, after training
└── utils/
    ├── __init__.py             # Package marker
    ├── health_thresholds.py    # Clinical-style badges for diabetes & heart features
    ├── population_stats.py     # Percentile ranks vs training cohort
    ├── validation.py           # Median imputation and range warnings for inputs
    └── pdf_report.py           # ReportLab PDF with risk summary and SHAP figure
```

## 🧠 How It Works

**Data preprocessing.** The PIMA pipeline treats implausible zeros in select labs as missing and imputes cohort medians before training; the heart pipeline drops rows with missing `ca` / `thal` and coerces numeric columns. Both use stratified train/test splits and persist medians, valid ranges, and training feature matrices so the app can validate uploads and compare users to the same reference population.

**XGBoost training.** Two separate `XGBClassifier` models share hyperparameters (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`). Each exports a joblib bundle (model + feature columns + medians), alongside JSON metrics including ROC FPR/TPR arrays and confusion matrices for the performance dashboard.

**SHAP explainability.** `TreeExplainer` runs in probability space with an interventional background sample from training data so each prediction gets signed contributions for a waterfall chart and a consistent ranking of top drivers—critical for trust and for feeding the LLM “top factors” text.

**Groq LLM integration.** The app builds structured prompts with risk label, probability, metrics, and SHAP-derived highlights. The same client powers the initial paragraph and the follow-up chat, which sends a fixed system preamble plus rolling user/assistant history so answers stay anchored to the patient’s numbers while preserving a clear medical-disclaimer posture.

## 📸 Screenshots

<!-- Risk Analysis Tab -->

<!-- Model Performance Tab -->

<!-- PDF Report -->

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It is **not** a medical device, diagnostic tool, or substitute for professional healthcare advice, examination, or treatment. Model outputs—including risk scores, SHAP explanations, AI text, and PDFs—are **demonstrations** trained on public datasets that may not reflect your health, population, or clinical context. **Always consult a qualified healthcare professional** for decisions about your health.
