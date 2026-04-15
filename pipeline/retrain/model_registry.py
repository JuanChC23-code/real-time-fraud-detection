import os
import json
import hashlib
from datetime import datetime

from pipeline.utils.config import MODEL_DIR, FEATURES_PATH


def ensure_model_registry() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def compute_file_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def build_model_metadata(
    model_name: str,
    model_version: str,
    model_path: str,
    data_snapshot_id: str,
    metrics: dict,
    pipeline_git_sha: str = "unknown",
    container_image_digest: str = "unknown",
) -> dict:
    feature_schema_sha = compute_file_sha256(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else "missing"

    return {
        "model_name": model_name,
        "model_version": model_version,
        "model_path": model_path,
        "created_at_utc": datetime.utcnow().isoformat(),
        "data_snapshot_id": data_snapshot_id,
        "metrics": metrics,
        "pipeline_git_sha": pipeline_git_sha,
        "container_image_digest": container_image_digest,
        "feature_schema_sha256": feature_schema_sha,
    }


def save_model_metadata(model_version: str, metadata: dict) -> str:
    ensure_model_registry()

    metadata_path = os.path.join(MODEL_DIR, f"{model_version}_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def load_model_metadata(model_version: str) -> dict:
    metadata_path = os.path.join(MODEL_DIR, f"{model_version}_metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found for model version: {model_version}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)