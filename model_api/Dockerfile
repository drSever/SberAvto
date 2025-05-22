# ---- base image ----
FROM python:3.11-slim

# — система: gcc + libgomp нужны CatBoost / LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
COPY models/ /app/models/

# — зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        pandas numpy scipy scikit-learn joblib catboost lightgbm optuna \
        imbalanced-learn fastapi uvicorn[standard] python-multipart \
        gradio>=4 

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
