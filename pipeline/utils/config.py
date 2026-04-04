import os
from dotenv import load_dotenv

load_dotenv()

# Kafka
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
KAFKA_API_KEY = os.getenv("KAFKA_API_KEY")
KAFKA_API_SECRET = os.getenv("KAFKA_API_SECRET")
TRANSACTIONS_TOPIC = os.getenv("TRANSACTIONS_TOPIC", "fraud.transactions.v1")
PREDICTIONS_TOPIC = os.getenv("PREDICTIONS_TOPIC", "fraud.predictions.v1")
KAFKA_GROUP = os.getenv("KAFKA_GROUP", "fraud-consumer-group")

# API
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
MODEL_VERSION = os.getenv("MODEL_VERSION", "xgboost_v1")

# Paths
DATA_DIR = os.getenv("DATA_DIR", "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
MODEL_DIR = os.getenv("MODEL_DIR", "model_registry")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

# Files
DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_cols_v1.json")
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_v1.pkl")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_v1.pkl")

# Runtime
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
EVENT_SLEEP_SECONDS = float(os.getenv("EVENT_SLEEP_SECONDS", "0.5"))
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
