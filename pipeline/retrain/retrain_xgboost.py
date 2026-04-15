import os
import json
import time
import joblib
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from pipeline.utils.config import (
    DATA_PATH,
    MODEL_DIR,
    FEATURES_PATH,
)
from pipeline.retrain.model_registry import build_model_metadata, save_model_metadata
from pipeline.retrain.active_model import set_active_model


def retrain_xgboost() -> None:
    print("Loading dataset for retraining...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    feature_cols = list(X.columns)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training new XGBoost version...")
    start = time.time()

    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    train_time = time.time() - start

    y_probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    version_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_version = f"xgboost_{version_suffix}"
    model_filename = f"{model_version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(model, model_path)

    metadata = build_model_metadata(
        model_name="xgboost",
        model_version=model_version,
        model_path=model_path,
        data_snapshot_id=os.path.basename(DATA_PATH),
        metrics={
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "training_time_sec": float(round(train_time, 6))
        }
    )

    metadata_path = save_model_metadata(model_version, metadata)

    set_active_model(
        model_name="xgboost",
        model_version=model_version,
        model_path=model_path
    )

    print("New model trained and activated.")
    print("Model path:", model_path)
    print("Metadata path:", metadata_path)
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))


if __name__ == "__main__":
    retrain_xgboost()
    