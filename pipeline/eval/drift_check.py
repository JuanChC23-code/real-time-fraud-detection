import os
import json
import numpy as np
import pandas as pd

from pipeline.utils.config import DATA_PATH, FEATURES_PATH, SNAPSHOT_DIR


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = pd.qcut(expected, q=bins, duplicates="drop", retbins=True)[1]
    breakpoints[0] = float("-inf")
    breakpoints[-1] = float("inf")

    expected_counts = pd.cut(expected, bins=breakpoints).value_counts(normalize=True, sort=False)
    actual_counts = pd.cut(actual, bins=breakpoints).value_counts(normalize=True, sort=False)

    expected_counts = expected_counts.replace(0, 0.0001)
    actual_counts = actual_counts.replace(0, 0.0001)

    psi = ((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)).sum()
    return float(psi)


def classify_psi(psi_value: float) -> str:
    if psi_value < 0.1:
        return "no_significant_drift"
    if psi_value < 0.25:
        return "moderate_drift"
    return "significant_drift"


def run_drift_check() -> None:
    transactions_files = [
        f for f in os.listdir(SNAPSHOT_DIR)
        if f.startswith("transactions_") and f.endswith(".csv")
    ]

    if not transactions_files:
        raise FileNotFoundError("No transaction snapshot files found in data/snapshots/")

    latest_transactions = sorted(transactions_files)[-1]
    latest_transactions_path = os.path.join(SNAPSHOT_DIR, latest_transactions)

    baseline_df = pd.read_csv(DATA_PATH)
    current_df = pd.read_csv(latest_transactions_path)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    numeric_features = []
    for col in feature_cols:
        if col in baseline_df.columns and col in current_df.columns:
            if pd.api.types.is_numeric_dtype(baseline_df[col]) and pd.api.types.is_numeric_dtype(current_df[col]):
                numeric_features.append(col)

    results = []

    for col in numeric_features:
        try:
            psi_value = compute_psi(baseline_df[col], current_df[col], bins=10)
            results.append({
                "feature": col,
                "psi": round(psi_value, 6),
                "drift_status": classify_psi(psi_value)
            })
        except Exception as e:
            results.append({
                "feature": col,
                "psi": None,
                "drift_status": f"error: {e}"
            })

    results_df = pd.DataFrame(results)
    output_path = os.path.join(SNAPSHOT_DIR, "drift_report.csv")
    results_df.to_csv(output_path, index=False)

    print("\n=== Drift Check Summary ===")
    print(results_df.head(10))
    print(f"\nSaved drift report to: {output_path}")


if __name__ == "__main__":
    run_drift_check()