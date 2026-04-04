import os
import glob
import pandas as pd

from pipeline.utils.config import SNAPSHOT_DIR


def get_latest_file(pattern: str) -> str:
    files = glob.glob(os.path.join(SNAPSHOT_DIR, pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def compute_online_kpi() -> None:
    transactions_file = get_latest_file("transactions_*.csv")
    predictions_file = get_latest_file("predictions_*.csv")

    transactions_df = pd.read_csv(transactions_file)
    predictions_df = pd.read_csv(predictions_file)

    merged_df = predictions_df.merge(
        transactions_df[["request_id", "label"]],
        on="request_id",
        how="left",
        suffixes=("_pred", "_tx")
    )

    if "label_pred" in merged_df.columns:
        merged_df["label"] = merged_df["label_pred"]
    elif "label_tx" in merged_df.columns:
        merged_df["label"] = merged_df["label_tx"]

    total_predictions = len(merged_df)
    predicted_fraud_rate = merged_df["prediction"].mean() if total_predictions > 0 else 0
    true_fraud_rate = merged_df["label"].mean() if total_predictions > 0 else 0
    correct_predictions = (merged_df["prediction"] == merged_df["label"]).sum() if total_predictions > 0 else 0
    online_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("\n=== Online KPI Summary ===")
    print(f"Transactions snapshot: {transactions_file}")
    print(f"Predictions snapshot: {predictions_file}")
    print(f"Total predictions: {total_predictions}")
    print(f"Predicted fraud rate: {predicted_fraud_rate:.4f}")
    print(f"True fraud rate: {true_fraud_rate:.4f}")
    print(f"Online accuracy: {online_accuracy:.4f}")

    output_path = os.path.join(SNAPSHOT_DIR, "online_kpi_summary.csv")
    summary_df = pd.DataFrame([{
        "transactions_file": transactions_file,
        "predictions_file": predictions_file,
        "total_predictions": total_predictions,
        "predicted_fraud_rate": round(predicted_fraud_rate, 6),
        "true_fraud_rate": round(true_fraud_rate, 6),
        "online_accuracy": round(online_accuracy, 6),
    }])
    summary_df.to_csv(output_path, index=False)

    print(f"\nSaved online KPI summary to: {output_path}")


if __name__ == "__main__":
    compute_online_kpi()
    