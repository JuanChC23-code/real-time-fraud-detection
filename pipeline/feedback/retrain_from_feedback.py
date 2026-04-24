import os
import json
import time
import joblib
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from pipeline.utils.config import MODEL_DIR, FEATURES_PATH
from pipeline.retrain.model_registry import build_model_metadata, save_model_metadata
from pipeline.retrain.active_model import set_active_model

FEEDBACK_DATASET_PATH = "data/snapshots/training_feedback_dataset.csv"


def retrain_from_feedback() -> None:
    print("Loading feedback training dataset...")

    if not os.path.exists(FEEDBACK_DATASET_PATH):
        raise FileNotFoundError(f"Feedback dataset not found: {FEEDBACK_DATASET_PATH}")

    df = pd.read_csv(FEEDBACK_DATASET_PATH)

    required_columns = [
        "request_id",
        "timestamp",
        "prediction",
        "fraud_probability",
        "model_name",
        "model_version",
        "ab_bucket",
        "true_label",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in feedback dataset: {missing}")

    model_name_encoded = (df["model_name"] == "xgboost").astype(int)
    bucket_encoded = (df["ab_bucket"] == "A").astype(int)

    X = pd.DataFrame({
        "prediction": df["prediction"].astype(float),
        "fraud_probability": df["fraud_probability"].astype(float),
        "model_name_encoded": model_name_encoded.astype(float),
        "bucket_encoded": bucket_encoded.astype(float),
    })

    y = df["true_label"].astype(int)

    if len(df) < 2:
        raise ValueError("Not enough feedback records to retrain")

    os.makedirs(MODEL_DIR, exist_ok=True)

    feature_cols = list(X.columns)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    class_counts = y.value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    if len(df) < 5 or not can_stratify:
        print("Dataset too small for safe stratified split. Training with full feedback dataset.")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    print("Training new feedback-based XGBoost version...")
    start = time.time()

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    train_time = time.time() - start

    y_probs = model.predict_proba(X_test)[:, 1]

    if len(set(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_probs)
        pr_auc = average_precision_score(y_test, y_probs)
    else:
        roc_auc = 0.5
        pr_auc = 0.5

    version_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_version = f"xgboost_feedback_{version_suffix}"
    model_filename = f"{model_version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(model, model_path)

    metadata = build_model_metadata(
        model_name="xgboost_feedback",
        model_version=model_version,
        model_path=model_path,
        data_snapshot_id=os.path.basename(FEEDBACK_DATASET_PATH),
        metrics={
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "training_time_sec": float(round(train_time, 6)),
            "feedback_records": int(len(df))
        }
    )

    metadata_path = save_model_metadata(model_version, metadata)

    set_active_model(
        model_name="xgboost_feedback",
        model_version=model_version,
        model_path=model_path
    )

    print("New feedback model trained and activated.")
    print("Model path:", model_path)
    print("Metadata path:", metadata_path)
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))


if __name__ == "__main__":
    retrain_from_feedback()