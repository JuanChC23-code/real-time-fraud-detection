import os
import json
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from pipeline.utils.config import (
    DATA_PATH,
    RESULTS_DIR,
    FEATURES_PATH,
    LOGISTIC_MODEL_PATH,
    XGBOOST_MODEL_PATH,
)


def compare_models() -> None:
    output_file = os.path.join(RESULTS_DIR, "model_comparison.csv")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df["Class"]

    # Split reproducible para evaluación offline
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Solo usamos el conjunto de prueba para evaluar
    X_eval = X_test
    y_eval = y_test

    models = {
        "logistic_v1": LOGISTIC_MODEL_PATH,
        "xgboost_v1": XGBOOST_MODEL_PATH
    }

    results = []

    for model_name, model_path in models.items():
        model = joblib.load(model_path)

        # Tamaño del modelo
        model_size_kb = os.path.getsize(model_path) / 1024

        # Benchmark de inferencia en test
        start_time = time.perf_counter()
        y_proba = model.predict_proba(X_eval)[:, 1]
        end_time = time.perf_counter()

        inference_time_sec = end_time - start_time
        avg_latency_ms = (inference_time_sec / len(X_eval)) * 1000
        throughput_rows_sec = len(X_eval) / inference_time_sec if inference_time_sec > 0 else 0

        # Predicción binaria
        y_pred = (y_proba >= 0.5).astype(int)

        # Métricas sobre el conjunto de evaluación
        roc_auc = roc_auc_score(y_eval, y_proba)
        pr_auc = average_precision_score(y_eval, y_proba)
        accuracy = accuracy_score(y_eval, y_pred)
        precision = precision_score(y_eval, y_pred, zero_division=0)
        recall = recall_score(y_eval, y_pred, zero_division=0)

        results.append({
            "model_name": model_name,
            "roc_auc": round(roc_auc, 6),
            "pr_auc": round(pr_auc, 6),
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "inference_time_sec": round(inference_time_sec, 6),
            "avg_latency_ms_per_row": round(avg_latency_ms, 6),
            "throughput_rows_per_sec": round(throughput_rows_sec, 2),
            "model_size_kb": round(model_size_kb, 2)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print("\nModel comparison results:\n")
    print(results_df)
    print(f"\nSaved comparison file to: {output_file}")


if __name__ == "__main__":
    compare_models()