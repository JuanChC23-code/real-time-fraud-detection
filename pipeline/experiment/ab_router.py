import hashlib


def assign_ab_bucket(entity_id: str) -> str:
    hashed = hashlib.md5(entity_id.encode("utf-8")).hexdigest()
    bucket_value = int(hashed, 16) % 2
    return "A" if bucket_value == 0 else "B"


def choose_model_by_bucket(entity_id: str) -> dict:
    bucket = assign_ab_bucket(entity_id)

    if bucket == "A":
        return {
            "bucket": "A",
            "model_name": "xgboost",
            "model_version": "xgboost_candidate"
        }

    return {
        "bucket": "B",
        "model_name": "logistic",
        "model_version": "logistic_baseline"
    }