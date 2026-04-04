import os
import json
import csv
import requests
import pandas as pd

from datetime import datetime
from confluent_kafka import Consumer, Producer

from pipeline.utils.config import (
    KAFKA_BOOTSTRAP,
    KAFKA_API_KEY,
    KAFKA_API_SECRET,
    TRANSACTIONS_TOPIC,
    PREDICTIONS_TOPIC,
    KAFKA_GROUP,
    API_URL,
    SNAPSHOT_DIR,
    REQUEST_TIMEOUT,
)

from pipeline.utils.schemas import (
    transaction_schema,
    prediction_schema,
)


os.makedirs(SNAPSHOT_DIR, exist_ok=True)

today_str = datetime.utcnow().strftime("%Y%m%d")
transactions_snapshot = os.path.join(SNAPSHOT_DIR, f"transactions_{today_str}.csv")
predictions_snapshot = os.path.join(SNAPSHOT_DIR, f"predictions_{today_str}.csv")


consumer = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "PLAIN",
    "sasl.username": KAFKA_API_KEY,
    "sasl.password": KAFKA_API_SECRET,
    "group.id": KAFKA_GROUP,
    "auto.offset.reset": "earliest"
})

producer = Producer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "PLAIN",
    "sasl.username": KAFKA_API_KEY,
    "sasl.password": KAFKA_API_SECRET
})


def ensure_csv_header(file_path: str, fieldnames: list[str]) -> None:
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_csv_row(file_path: str, fieldnames: list[str], row: dict) -> None:
    ensure_csv_header(file_path, fieldnames)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def validate_transaction_event(event: dict) -> bool:
    try:
        df = pd.DataFrame([{
            "request_id": event.get("request_id"),
            "timestamp": event.get("timestamp"),
            "label": event.get("label"),
        }])
        transaction_schema.validate(df)
        return True
    except Exception as e:
        print(f"Schema validation failed (transaction): {e}")
        return False


def validate_prediction_event(event: dict) -> bool:
    try:
        df = pd.DataFrame([event])
        prediction_schema.validate(df)
        return True
    except Exception as e:
        print(f"Schema validation failed (prediction): {e}")
        return False


def flatten_transaction_event(event: dict) -> dict:
    flattened = {
        "request_id": event.get("request_id"),
        "timestamp": event.get("timestamp"),
        "label": event.get("label")
    }

    features = event.get("features", {})
    for key, value in features.items():
        flattened[key] = value

    return flattened


def flatten_prediction_event(event: dict) -> dict:
    return {
        "request_id": event.get("request_id"),
        "timestamp": event.get("timestamp"),
        "model_version": event.get("model_version"),
        "fraud_probability": event.get("fraud_probability"),
        "prediction": event.get("prediction"),
        "label": event.get("label")
    }


def delivery_report(err, msg):
    if err is not None:
        print(f"Prediction delivery failed: {err}")
    else:
        print(f"Prediction sent to {msg.topic()} [{msg.partition()}]")


def run_consumer() -> None:
    consumer.subscribe([TRANSACTIONS_TOPIC])

    print("Consumer running... Waiting for events.")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode("utf-8"))

            # output validation
            if not validate_transaction_event(event):
                continue

            transaction_row = flatten_transaction_event(event)
            transaction_fields = list(transaction_row.keys())
            append_csv_row(transactions_snapshot, transaction_fields, transaction_row)

            request_id = event.get("request_id")
            features = event.get("features", {})
            label = event.get("label")

            try:
                response = requests.post(API_URL, json=features, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                prediction = response.json()

                result_event = {
                    "request_id": request_id,
                    "timestamp": prediction.get("timestamp"),
                    "model_version": prediction.get("model_version"),
                    "fraud_probability": prediction.get("fraud_probability"),
                    "prediction": prediction.get("prediction"),
                    "label": label
                }

                # output validation
                if not validate_prediction_event(result_event):
                    continue

                prediction_row = flatten_prediction_event(result_event)
                prediction_fields = list(prediction_row.keys())
                append_csv_row(predictions_snapshot, prediction_fields, prediction_row)

                producer.produce(
                    PREDICTIONS_TOPIC,
                    key=str(request_id),
                    value=json.dumps(result_event),
                    callback=delivery_report
                )
                producer.poll(0)

                print(f"Processed request_id={request_id}")
                print(f"Saved transaction snapshot -> {transactions_snapshot}")
                print(f"Saved prediction snapshot -> {predictions_snapshot}")

            except Exception as e:
                print(f"API error for request_id={request_id}: {e}")

    except KeyboardInterrupt:
        print("Stopping consumer...")

    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    run_consumer()