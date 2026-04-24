import os
import pandas as pd

from pipeline.fairness.fairness_rules import evaluate_fairness_requirements

SNAPSHOT_PATH = "data/snapshots/predictions_live_ab.csv"
OUTPUT_PATH = "data/snapshots/fairness_report.csv"


def load_predictions() -> pd.DataFrame:
    if not os.path.exists(SNAPSHOT_PATH):
        raise FileNotFoundError(f"Predictions file not found: {SNAPSHOT_PATH}")

    df = pd.read_csv(SNAPSHOT_PATH)

    required_columns = [
        "request_id",
        "timestamp",
        "prediction",
        "fraud_probability",
        "model_name",
        "model_version",
        "ab_bucket",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in predictions file: {missing}")

    return df


def compute_group_exposure(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("ab_bucket")
        .agg(
            total_predictions=("request_id", "count"),
            positive_predictions=("prediction", "sum"),
            avg_fraud_probability=("fraud_probability", "mean"),
        )
        .reset_index()
    )

    summary["positive_rate"] = (
        summary["positive_predictions"] / summary["total_predictions"]
    ).round(6)

    summary["avg_fraud_probability"] = summary["avg_fraud_probability"].round(6)

    return summary


def classify_fairness_difference(summary: pd.DataFrame) -> str:
    if len(summary) < 2:
        return "insufficient_groups"

    rates = summary["positive_rate"].tolist()
    gap = abs(rates[0] - rates[1])

    if gap < 0.05:
        return "balanced_exposure"
    if gap < 0.15:
        return "moderate_exposure_gap"
    return "high_exposure_gap"


def run_fairness_analysis() -> None:
    df = load_predictions()
    summary = compute_group_exposure(df)
    fairness_status = classify_fairness_difference(summary)

    rule_evaluation = evaluate_fairness_requirements(fairness_status)

    summary["fairness_status"] = fairness_status
    summary["fairness_rule_status"] = rule_evaluation["status"]
    summary["fairness_message"] = rule_evaluation["message"]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Fairness Analysis Summary ===")
    print(summary)
    print("\n=== Fairness Rule Evaluation ===")
    print(rule_evaluation)
    print(f"\nSaved fairness report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_fairness_analysis() 