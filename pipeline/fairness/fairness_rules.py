def evaluate_fairness_requirements(fairness_status: str) -> dict:
    """
    Define reglas de negocio para fairness.
    """

    if fairness_status == "balanced_exposure":
        return {
            "status": "PASS",
            "message": "Fairness acceptable: exposure balanced between groups."
        }

    if fairness_status == "moderate_exposure_gap":
        return {
            "status": "WARNING",
            "message": "Moderate fairness gap detected. Monitor closely."
        }

    if fairness_status == "high_exposure_gap":
        return {
            "status": "FAIL",
            "message": "High fairness gap detected. Retraining or model review required."
        }

    return {
        "status": "UNKNOWN",
        "message": "Insufficient data for fairness evaluation."
    }