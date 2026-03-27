from fastapi import FastAPI, HTTPException
import joblib
import json
import numpy as np
import pandas as pd
import uuid
from datetime import datetime

# Load model and feature schema
model = joblib.load("model_registry/xgboost_v1.pkl")

with open("model_registry/feature_cols_v1.json", "r") as f:
    feature_cols = json.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Fraud Detection API running"}

@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "model_version": "xgboost_v1"
    }

@app.post("/predict")
def predict(transaction: dict):
    try:
        # Convert input to dataframe
        df = pd.DataFrame([transaction])

        # Ensure all required features exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Keep correct order
        df = df[feature_cols]

        # Predict probability
        prob = model.predict_proba(df)[0][1]

        prediction = int(prob > 0.5)

        response = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "fraud_probability": float(prob),
            "model_version": "xgboost_v1"
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))