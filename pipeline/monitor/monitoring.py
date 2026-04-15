import os
import pandas as pd

from pipeline.eval.online_kpi import compute_online_kpi
from pipeline.eval.drift_check import run_drift_check

SNAPSHOT_DIR = "data/snapshots"


def check_online_kpi_thresholds(kpi_file: str) -> list:
    alerts = []

    df = pd.read_csv(kpi_file)

    accuracy = df["online_accuracy"].iloc[0]
    fraud_rate = df["predicted_fraud_rate"].iloc[0]

    if accuracy < 0.90:
        alerts.append(f"LOW_ACCURACY_ALERT: accuracy={accuracy}")

    if fraud_rate > 0.10:
        alerts.append(f"HIGH_FRAUD_RATE_ALERT: fraud_rate={fraud_rate}")

    return alerts


def check_drift_thresholds(drift_file: str) -> list:
    alerts = []

    df = pd.read_csv(drift_file)

    drifted_features = df[df["drift_status"] == "significant_drift"]

    if not drifted_features.empty:
        alerts.append(f"DRIFT_ALERT: {len(drifted_features)} features with significant drift")

    return alerts


def run_monitoring() -> None:
    print("\n=== Running Monitoring Pipeline ===")

    compute_online_kpi()
    run_drift_check()

    kpi_file = os.path.join(SNAPSHOT_DIR, "online_kpi_summary.csv")
    drift_file = os.path.join(SNAPSHOT_DIR, "drift_report.csv")

    alerts = []

    if os.path.exists(kpi_file):
        alerts.extend(check_online_kpi_thresholds(kpi_file))

    if os.path.exists(drift_file):
        alerts.extend(check_drift_thresholds(drift_file))

    print("\n=== ALERT SUMMARY ===")

    if not alerts:
        print("SYSTEM HEALTHY: No alerts triggered")
    else:
        for alert in alerts:
            print("ALERT:", alert)


if __name__ == "__main__":
    run_monitoring()