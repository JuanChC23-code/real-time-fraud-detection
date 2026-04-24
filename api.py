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
    global active_model, active_info

    try:
        #  SECURITY VALIDATION
        from pipeline.security.input_validation import validate_input
        validate_input(transaction)

        request_id = str(uuid.uuid4())

        df = pd.DataFrame([transaction])

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_cols]

        bucket = assign_ab_bucket(request_id)

        model_used = active_model
        model_name_used = active_info["active_model_name"]
        model_version_used = active_info["active_model_version"]

        prob = model_used.predict_proba(df)[0][1]
        prediction = int(prob >= PREDICTION_THRESHOLD)

        timestamp = datetime.utcnow().isoformat()

        result = {
            "request_id": request_id,
            "timestamp": timestamp,
            "prediction": prediction,
            "fraud_probability": float(prob),
            "model_name": model_name_used,
            "model_version": model_version_used,
            "ab_bucket": bucket
        }

        log_prediction(result)

        trace = build_prediction_trace(
            request_id=request_id,
            data_snapshot_id="live_stream"
        )

        save_prediction_trace(trace)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))