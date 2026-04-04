import os
import sys
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


def test_transaction_schema_validates():
    df = pd.DataFrame([{
        "request_id": "abc123",
        "timestamp": "2026-04-04T21:00:00",
        "label": 1,
    }])
    validated = transaction_schema.validate(df)
    assert not validated.empty


def test_prediction_schema_validates():
    df = pd.DataFrame([{
        "request_id": "abc123",
        "timestamp": "2026-04-04T21:00:00",
        "model_version": "xgboost_v1",
        "fraud_probability": 0.91,
        "prediction": 1,
        "label": 1,
    }])
    validated = prediction_schema.validate(df)
    assert not validated.empty


def test_validate_transaction_event_true():
    event = {
        "request_id": "req-1",
        "timestamp": "2026-04-04T21:00:00",
        "features": {"V1": 0.1, "V2": -0.2},
        "label": 0,
    }
    assert validate_transaction_event(event) is True


def test_validate_prediction_event_true():
    event = {
        "request_id": "req-1",
        "timestamp": "2026-04-04T21:00:00",
        "model_version": "xgboost_v1",
        "fraud_probability": 0.42,
        "prediction": 0,
        "label": 0,
    }
    assert validate_prediction_event(event) is True


def test_flatten_transaction_event():
    event = {
        "request_id": "req-2",
        "timestamp": "2026-04-04T21:00:00",
        "features": {"V1": 1.5, "V2": -2.0},
        "label": 1,
    }
    flat = flatten_transaction_event(event)

    assert flat["request_id"] == "req-2"
    assert flat["timestamp"] == "2026-04-04T21:00:00"
    assert flat["label"] == 1
    assert flat["V1"] == 1.5
    assert flat["V2"] == -2.0


def test_flatten_prediction_event():
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
    assert flat["timestamp"] == "2026-04-04T21:00:00"
    assert flat["model_version"] == "xgboost_v1"
    assert flat["fraud_probability"] == 0.88
    assert flat["prediction"] == 1
    assert flat["label"] == 1

    import tempfile
from pipeline.ingest.stream_ingestor import ensure_csv_header, append_csv_row, delivery_report


def test_ensure_csv_header_creates_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_path = tmp.name

  
    os.remove(file_path)

    ensure_csv_header(file_path, ["col1", "col2"])

    assert os.path.exists(file_path)


def test_append_csv_row_writes_data():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_path = tmp.name

    os.remove(file_path)

    row = {"col1": 1, "col2": 2}

    append_csv_row(file_path, ["col1", "col2"], row)

    df = pd.read_csv(file_path)
    assert not df.empty
    assert df.iloc[0]["col1"] == 1


def test_delivery_report_no_error():
    # Simula callback sin error
    delivery_report(None, type("Msg", (), {"topic": lambda: "test", "partition": lambda: 0})())