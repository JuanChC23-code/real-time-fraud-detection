import os
import sys
import math
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from pipeline.serve.app import app
from pipeline.eval.drift_check import compute_psi, classify_psi
from pipeline.eval.online_kpi import get_latest_file


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_endpoint():
    sample_transaction = {
        "V1": 0.1,
        "V2": -0.2,
        "V3": 0.3,
        "V4": 0.4,
        "V5": -0.5,
        "V6": 0.6,
        "V7": -0.7,
        "V8": 0.8,
        "V9": -0.9,
        "V10": 1.0,
        "V11": 0.11,
        "V12": -0.12,
        "V13": 0.13,
        "V14": -0.14,
        "V15": 0.15,
        "V16": -0.16,
        "V17": 0.17,
        "V18": -0.18,
        "V19": 0.19,
        "V20": -0.20,
        "V21": 0.21,
        "V22": -0.22,
        "V23": 0.23,
        "V24": -0.24,
        "V25": 0.25,
        "V26": -0.26,
        "V27": 0.27,
        "V28": -0.28,
        "Amount": 100.0
    }

    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200

    data = response.json()
    assert "request_id" in data
    assert "timestamp" in data
    assert "prediction" in data
    assert "fraud_probability" in data
    assert "model_version" in data


def test_classify_psi_no_significant():
    assert classify_psi(0.05) == "no_significant_drift"


def test_classify_psi_moderate():
    assert classify_psi(0.15) == "moderate_drift"


def test_classify_psi_significant():
    assert classify_psi(0.30) == "significant_drift"


def test_compute_psi_runs():
    expected = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    actual = pd.Series([1, 2, 2, 4, 5, 6, 7, 8, 9, 12])

    psi_value = compute_psi(expected, actual, bins=5)

    assert isinstance(psi_value, float)
    assert not math.isnan(psi_value)


def test_get_latest_file_finds_transactions_snapshot():
    latest_file = get_latest_file("transactions_*.csv")
    assert latest_file.endswith(".csv")
    assert "transactions_" in os.path.basename(latest_file)


def test_get_latest_file_finds_predictions_snapshot():
    latest_file = get_latest_file("predictions_*.csv")
    assert latest_file.endswith(".csv")
    assert "predictions_" in os.path.basename(latest_file)