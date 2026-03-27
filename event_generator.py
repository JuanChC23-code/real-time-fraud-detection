import os
import json
import time
import uuid
import pandas as pd
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
KAFKA_API_KEY = os.getenv("KAFKA_API_KEY")
KAFKA_API_SECRET = os.getenv("KAFKA_API_SECRET")
TRANSACTIONS_TOPIC = os.getenv("TRANSACTIONS_TOPIC", "fraud.transactions.v1")

producer = Producer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "PLAIN",
    "sasl.username": KAFKA_API_KEY,
    "sasl.password": KAFKA_API_SECRET
})

df = pd.read_csv("data/creditcard.csv")

feature_cols = [col for col in df.columns if col != "Class"]

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")
    else:
        print(f"Sent to {msg.topic()} [{msg.partition()}]")

for _, row in df.head(20).iterrows():
    event = {
        "request_id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "features": {col: float(row[col]) for col in feature_cols if col != "Time"},
        "label": int(row["Class"])
    }

    producer.produce(
        TRANSACTIONS_TOPIC,
        key=event["request_id"],
        value=json.dumps(event),
        callback=delivery_report
    )

    producer.poll(0)
    time.sleep(0.5)

producer.flush()
print("Finished sending events.")