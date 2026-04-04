import os
import sys
import math
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.utils.schemas import transaction_schema, prediction_schema
from pipeline.ingest.stream_ingestor import (
    validate_transaction_event,
    validate_prediction_event,
    flatten_transaction_event,
    flatten_prediction_event,
)
from pipeline.eval.drift_check import compute_psi
from pipeline.eval.online_kpi import get_latest_file
from pipeline.serve import app as serve_app


def test_transaction_schema_invalid_missing_label():
    df = pd.DataFrame([{
        "request_id": "abc123",
        "timestamp": "2026-04-04T21:00:00",
    }])

    failed = False
    try:
        transaction_schema.validate(df)
    except Exception:
        failed = True

    assert failed is True


def test_prediction_schema_invalid_missing_model_version():
    df = pd.DataFrame([{
        "request_id": "abc123",
        "timestamp": "2026-04-04T21:00:00",
        "fraud_probability": 0.91,
        "prediction": 1,
        "label": 1,
    }])

    failed = False
    try:
        prediction_schema.validate(df)
    except Exception:
        failed = True

    assert failed is True


def test_validate_transaction_event_false_when_missing_label():
    event = {
        "request_id": "req-1",
        "timestamp": "2026-04-04T21:00:00",
        "features": {"V1": 0.1, "V2": -0.2},
    }
    assert validate_transaction_event(event) is False


def test_validate_prediction_event_false_when_missing_prediction():
    event = {
        "request_id": "req-1",
        "timestamp": "2026-04-04T21:00:00",
        "model_version": "xgboost_v1",
        "fraud_probability": 0.42,
        "label": 0,
    }
    assert validate_prediction_event(event) is False


def test_flatten_transaction_event_without_features():
    event = {
        "request_id": "req-2",
        "timestamp": "2026-04-04T21:00:00",
        "label": 1,
    }
    flat = flatten_transaction_event(event)

    assert flat["request_id"] == "req-2"
    assert flat["timestamp"] == "2026-04-04T21:00:00"
    assert flat["label"] == 1
    assert len(flat.keys()) == 3


def test_flatten_prediction_event_values():
    event = {
        "request_id": "req-3",
        "timestamp": "2026-04-04T21:00:00",
        "model_version": "xgboost_v1",
        "fraud_probability": 0.88,
        "prediction": 1,
        "label": 1,
    }
    flat = flatten_prediction_event(event)

    assert flat["request_id"] == "req-3"
    assert flat["model_version"] == "xgboost_v1"
    assert math.isclose(flat["fraud_probability"], 0.88)
    assert flat["prediction"] == 1
    assert flat["label"] == 1


def test_compute_psi_with_identical_distributions():
    expected = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    actual = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    psi_value = compute_psi(expected, actual, bins=5)

    assert isinstance(psi_value, float)
    assert psi_value >= 0


def test_get_latest_file_raises_for_missing_pattern():
    failed = False
    try:
        get_latest_file("file_that_should_not_exist_*.csv")
    except FileNotFoundError:
        failed = True

    assert failed is True


def test_predict_error_branch(monkeypatch):
    class BrokenModel:
        def predict_proba(self, df):
            raise RuntimeError("forced test error")

    monkeypatch.setattr(serve_app, "model", BrokenModel())

    failed = False
    try:
        serve_app.predict({"V1": 0.1, "Amount": 100.0})
    except Exception as e:
        failed = True
        assert "forced test error" in str(e.detail)

    assert failed is True