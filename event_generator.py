from confluent_kafka import Producer
import pandas as pd
import json
import time

print("Loading dataset for streaming...")

# Cargar dataset
df = pd.read_csv("data/creditcard.csv")

# Cargar esquema de columnas
with open("model_registry/feature_cols_v1.json", "r") as f:
    feature_cols = json.load(f)

producer = Producer({'bootstrap.servers': 'localhost:9092'})

topic = "fraud.transactions.v1"

print("Starting real data stream...")

for idx, row in df.iterrows():

    event = row[feature_cols].to_dict()

    producer.produce(
        topic,
        key=str(idx),
        value=json.dumps(event)
    )

    producer.poll(0)

    if idx % 1000 == 0:
        print(f"Sent {idx} events...")

    time.sleep(0.01)  # Simula streaming real

producer.flush()

print("Streaming completed.")
