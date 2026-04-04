import os
import json
import time
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from pipeline.utils.config import (
    DATA_PATH,
    MODEL_DIR,
    FEATURES_PATH,
    LOGISTIC_MODEL_PATH,
    XGBOOST_MODEL_PATH,
)


def train_and_save_models() -> None:
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    feature_cols = list(X.columns)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    print("Feature schema saved.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Logistic Regression...")
    start = time.time()

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    log_time = time.time() - start
    log_probs = log_model.predict_proba(X_test)[:, 1]

    log_auc = roc_auc_score(y_test, log_probs)
    log_pr = average_precision_score(y_test, log_probs)

    joblib.dump(log_model, LOGISTIC_MODEL_PATH)

    print("\nLogistic Regression Results:")
    print("Training time:", round(log_time, 2), "seconds")
    print("ROC-AUC:", round(log_auc, 4))
    print("PR-AUC:", round(log_pr, 4))

    print("\nTraining XGBoost...")
    start = time.time()

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)

    xgb_time = time.time() - start
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    xgb_auc = roc_auc_score(y_test, xgb_probs)
    xgb_pr = average_precision_score(y_test, xgb_probs)

    joblib.dump(xgb_model, XGBOOST_MODEL_PATH)

    print("\nXGBoost Results:")
    print("Training time:", round(xgb_time, 2), "seconds")
    print("ROC-AUC:", round(xgb_auc, 4))
    print("PR-AUC:", round(xgb_pr, 4))

    print("\nModels saved in model_registry/")


if __name__ == "__main__":
    train_and_save_models()