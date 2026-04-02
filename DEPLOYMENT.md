# Deploying Personal Health Risk Explorer

This app is a **Streamlit** UI that expects **trained model files** under `model/` and **cached training matrices** under `data/` (for population percentiles). Groq features need a **`GROQ_API_KEY`**.

---

## Option A — Streamlit Community Cloud (fastest)

1. **Push the repo to GitHub** (already done if you use Cloud’s GitHub integration).

2. **Commit artifacts Cloud cannot build for you**  
   The free tier does not run your training scripts before launch. Either:
   - Run locally: `python model/train.py && python model/train_heart.py`, then **force-add** and commit:
     - `model/diabetes_model.pkl`, `model/heart_model.pkl`
     - `data/training_features.npy`, `data/feature_names.json`
     - `data/heart_training_features.npy`, `data/heart_feature_names.json`  
     (Temporarily remove those lines from `.gitignore` or use `git add -f …`.)
   - Or rely on **Docker** (Option B) / another host that trains on deploy.

3. In [Streamlit Community Cloud](https://streamlit.io/cloud), **New app** → pick repo, branch, **Main file path:** `app.py`.

4. **Secrets** (app → Settings → Secrets), add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
   The app reads this via `st.secrets` (see `model/llm_explain.py`).

5. **Redeploy** after changing secrets or data files.

---

## Option B — Docker (Fly.io, Railway, Render, Google Cloud Run, etc.)

From the project root:

```bash
docker build -t health-risk-explorer .
docker run -p 8501:8501 -e GROQ_API_KEY=gsk_your_key_here health-risk-explorer
```

The image **runs training during `docker build`** (needs network for dataset URLs), so you do not have to commit `.pkl` files to Git.

On platforms that inject a dynamic **PORT**:

```bash
docker run -e PORT=8080 -e GROQ_API_KEY=gsk_... -p 8080:8080 health-risk-explorer
```

The `CMD` in the Dockerfile uses `$PORT` when set.

### Render.com (example)

1. New **Web Service** → connect repo, **Docker** runtime, root `Dockerfile`.
2. **Environment** → add `GROQ_API_KEY`.
3. Render sets `PORT`; the container command already respects it.

### Railway / Fly.io

- **Build:** Dockerfile at repo root.
- **Variable:** `GROQ_API_KEY` (required for AI explanation and follow-up chat).

---

## Environment variables

| Variable | Required | Where |
|----------|----------|--------|
| `GROQ_API_KEY` | Yes, for AI features | `.env` locally, Docker `-e`, Streamlit Cloud Secrets |
| `PORT` | Optional | Set by PaaS; local Docker defaults to `8501` |

---

## After deploy

- Open the URL, select **Diabetes** or **Heart Disease**, run a sample analysis.
- If **population percentiles** or **PDF** fail, ensure `data/training_features.npy` and `data/feature_names.json` (and heart counterparts) exist on the host—they are produced by the training scripts.

---

## Security

- Never commit `.env` or real API keys.
- This app is **educational**; do not expose internal clinical data on a public URL without a security review.
