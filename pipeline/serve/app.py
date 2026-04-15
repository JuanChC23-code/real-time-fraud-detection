import os
import json
import uuid
import joblib
import pandas as pd

from datetime import datetime
from fastapi import FastAPI, HTTPException

from pipeline.utils.config import (
    FEATURES_PATH,
    PREDICTION_THRESHOLD,
)

from pipeline.retrain.active_model import (
    get_active_model_info,
    set_active_model,
)

from pipeline.experiment.ab_router import (
    assign_ab_bucket,
)

from pipeline.provenance.trace import (
    build_prediction_trace,
    save_prediction_trace,
)

PREDICTION_LOG_PATH = "data/snapshots/predictions_live_ab.csv"

app = FastAPI()


def load_feature_columns() -> list:
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_active_model():
    info = get_active_model_info()
    model = load_model(info["active_model_path"])
    return model, info


def log_prediction(record: dict):
    os.makedirs("data/snapshots", exist_ok=True)

    expected_columns = [
        "request_id",
        "timestamp",
        "prediction",
        "fraud_probability",
        "model_name",
        "model_version",
        "ab_bucket",
    ]

    df = pd.DataFrame([record], columns=expected_columns)

    if not os.path.exists(PREDICTION_LOG_PATH):
        df.to_csv(PREDICTION_LOG_PATH, index=False)
    else:
        df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)


feature_cols = load_feature_columns()
active_model, active_info = load_active_model()


@app.get("/")
def root():
    return {"message": "Fraud Detection API running"}


@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "active_model_name": active_info["active_model_name"],
        "active_model_version": active_info["active_model_version"]
    }


@app.post("/predict")
def predict(transaction: dict):
    global active_model, active_info

    try:
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

        #  LOGGING NORMAL
        log_prediction(result)

        #  PROVENANCE TRACE
        trace = build_prediction_trace(
            request_id=request_id,
            data_snapshot_id="live_stream"
        )

        save_prediction_trace(trace)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model")
def reload_model():
    global active_model, active_info

    active_model, active_info = load_active_model()

    return {
        "status": "reloaded",
        "model_name": active_info["active_model_name"],
        "model_version": active_info["active_model_version"]
    }


@app.post("/switch_model")
def switch_model(model_name: str, model_version: str, model_path: str):
    global active_model, active_info

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Model path not found")

    set_active_model(model_name, model_version, model_path)

    active_model, active_info = load_active_model()

    return {
        "status": "switched",
        "model_name": active_info["active_model_name"],
        "model_version": active_info["active_model_version"]
    }