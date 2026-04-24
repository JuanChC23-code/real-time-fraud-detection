import os
import pandas as pd

PREDICTIONS_PATH = "data/snapshots/predictions_live_ab.csv"
FEEDBACK_PATH = "data/snapshots/feedback_labels.csv"
OUTPUT_PATH = "data/snapshots/training_feedback_dataset.csv"


def load_predictions() -> pd.DataFrame:
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH}")
    return pd.read_csv(PREDICTIONS_PATH)


def load_feedback() -> pd.DataFrame:
    if not os.path.exists(FEEDBACK_PATH):
        raise FileNotFoundError(f"Feedback file not found: {FEEDBACK_PATH}")
    return pd.read_csv(FEEDBACK_PATH)


def merge_feedback(predictions: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        feedback,
        on="request_id",
        how="inner"
    )
    return merged


def save_training_dataset(df: pd.DataFrame) -> str:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


def run_feedback_pipeline() -> None:
    print("\n=== Running Feedback Loop ===")

    predictions = load_predictions()
    feedback = load_feedback()

    merged = merge_feedback(predictions, feedback)

    print(f"Merged records: {len(merged)}")

    output_file = save_training_dataset(merged)

    print(f"Training dataset created at: {output_file}")


if __name__ == "__main__":
    run_feedback_pipeline()