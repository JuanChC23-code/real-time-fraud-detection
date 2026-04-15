import os
import json

from pipeline.utils.config import MODEL_DIR


ACTIVE_MODEL_FILE = os.path.join(MODEL_DIR, "active_model.json")


def ensure_active_model_file() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(ACTIVE_MODEL_FILE):
        default_data = {
            "active_model_name": "xgboost",
            "active_model_version": "xgboost_v1",
            "active_model_path": os.path.join(MODEL_DIR, "xgboost_v1.pkl")
        }

        with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=2)


def get_active_model_info() -> dict:
    ensure_active_model_file()

    with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def set_active_model(model_name: str, model_version: str, model_path: str) -> dict:
    ensure_active_model_file()

    data = {
        "active_model_name": model_name,
        "active_model_version": model_version,
        "active_model_path": model_path
    }

    with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data