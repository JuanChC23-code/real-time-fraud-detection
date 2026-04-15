import os
import json
import hashlib
from datetime import datetime

from pipeline.utils.config import FEATURES_PATH
from pipeline.retrain.active_model import get_active_model_info


def compute_file_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def build_prediction_trace(
    request_id: str,
    data_snapshot_id: str = "live_prediction",
    pipeline_git_sha: str = "unknown",
    container_image_digest: str = "unknown",
) -> dict:
    active_model = get_active_model_info()

    feature_schema_sha = compute_file_sha256(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else "missing"
    model_path = active_model.get("active_model_path", "")
    model_sha = compute_file_sha256(model_path) if model_path and os.path.exists(model_path) else "missing"

    return {
        "request_id": request_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_name": active_model.get("active_model_name"),
        "model_version": active_model.get("active_model_version"),
        "model_path": model_path,
        "model_sha256": model_sha,
        "data_snapshot_id": data_snapshot_id,
        "pipeline_git_sha": pipeline_git_sha,
        "container_image_digest": container_image_digest,
        "feature_schema_sha256": feature_schema_sha,
    }


def save_prediction_trace(trace: dict, output_path: str = "data/snapshots/prediction_traces.jsonl") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")

    return output_path