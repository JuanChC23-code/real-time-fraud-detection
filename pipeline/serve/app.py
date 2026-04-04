import json
import uuid
import joblib
import pandas as pd

from datetime import datetime
from fastapi import FastAPI, HTTPException

from pipeline.utils.config import (
    FEATURES_PATH,
    XGBOOST_MODEL_PATH,
    MODEL_VERSION,
    PREDICTION_THRESHOLD,
)

model = joblib.load(XGBOOST_MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Fraud Detection API running"}


@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict(transaction: dict):
    try:
        df = pd.DataFrame([transaction])

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_cols]

        prob = model.predict_proba(df)[0][1]
        prediction = int(prob >= PREDICTION_THRESHOLD)

        response = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "fraud_probability": float(prob),
            "model_version": MODEL_VERSION
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))