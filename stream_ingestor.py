from confluent_kafka import Consumer, Producer
import json
import joblib
import time
import uuid

MODEL_VERSION = "xgboost_v1"
INPUT_TOPIC = "fraud.transactions.v1"
OUTPUT_TOPIC = "fraud.predictions.v1"
THRESHOLD = 0.8

print("Loading model and schema...")

model = joblib.load("model_registry/xgboost_v1.pkl")
with open("model_registry/feature_cols_v1.json", "r") as f:
    feature_cols = json.load(f)

consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "fraud-predictor-v1",
    "auto.offset.reset": "earliest",
})

producer = Producer({
    "bootstrap.servers": "localhost:9092",
})

consumer.subscribe([INPUT_TOPIC])

count = 0
alerts = 0
start_time = time.time()

print(f"Streaming: {INPUT_TOPIC} -> {OUTPUT_TOPIC} (Ctrl+C to stop)")

def delivery_report(err, msg):
    if err:
        print("Delivery failed:", err)

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            print("Consumer error:", msg.error())
            continue

        data = json.loads(msg.value().decode())

        # Vector en orden correcto
        features = [data[col] for col in feature_cols]
        prob = float(model.predict_proba([features])[0][1])

        request_id = str(uuid.uuid4())
        ts = time.time()

        out_event = {
            "request_id": request_id,
            "timestamp": ts,
            "model_version": MODEL_VERSION,
            "input_topic": INPUT_TOPIC,
            "output_topic": OUTPUT_TOPIC,
            "fraud_probability": prob,
            "is_alert": prob > THRESHOLD,
        }

        # (Opcional) incluir campos útiles para auditoría
        if "Time" in data:
            out_event["Time"] = data["Time"]
        if "Amount" in data:
            out_event["Amount"] = data["Amount"]

        producer.produce(
            OUTPUT_TOPIC,
            key=request_id,
            value=json.dumps(out_event),
            callback=delivery_report
        )
        producer.poll(0)

        count += 1
        if prob > THRESHOLD:
            alerts += 1

        if count % 200 == 0:
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"[STATS] processed={count} alerts={alerts} rate={rate:.1f} msg/s")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    consumer.close()
    producer.flush()