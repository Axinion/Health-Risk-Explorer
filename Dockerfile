FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for XGBoost and scientific Python
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — dependencies change less than code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create directories that may not exist in the repo
RUN mkdir -p data model

# Train models at build time so the container starts instantly
# (no cold-start delay for users)
RUN python model/train.py && python model/train_heart.py

# Expose Streamlit's default port
EXPOSE 8501

# Health check so Docker knows if the app is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
