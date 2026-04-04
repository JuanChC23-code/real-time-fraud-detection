import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.utils.config import DATA_PATH, FEATURES_PATH
from pipeline.train.train_models import train_and_save_models
from pipeline.eval.compare_models import compare_models
from pipeline.eval.online_kpi import compute_online_kpi
from pipeline.eval.drift_check import run_drift_check


def test_training_pipeline_runs():
    train_and_save_models()
    assert os.path.exists("model_registry/logistic_v1.pkl")
    assert os.path.exists("model_registry/xgboost_v1.pkl")


def test_model_comparison_runs():
    compare_models()
    assert os.path.exists("results/model_comparison.csv")


def test_online_kpi_runs():
    compute_online_kpi()
    assert os.path.exists("data/snapshots/online_kpi_summary.csv")


def test_drift_check_runs():
    run_drift_check()
    assert os.path.exists("data/snapshots/drift_report.csv")


def test_feature_schema_exists():
    assert os.path.exists(FEATURES_PATH)


def test_dataset_loads():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty


def test_feature_columns_match():
    df = pd.read_csv(DATA_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    for col in feature_cols:
        assert col in df.columns