"""Регулярные расчёты (паттерн hw9): ETL + проверка дрифта + переобучение.

DAG ``ecom_retrain``:
    extract  -> build_features -> drift_check -> retrain -> publish_s3

Запускается по расписанию (ежедневно). Так мы закрываем критерии защиты
«регулярные расчёты/процессы» и «ML-мониторинг (Data/Concept Drift)».
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "e.puzyreva",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

RAW_TRAIN = "/opt/airflow/data/raw/train.parquet.snappy"
RAW_TEST = "/opt/airflow/data/raw/test.parquet.snappy"
TREE = "/opt/airflow/data/raw/tree.csv"
PROCESSED = "/opt/airflow/data/processed"
ARTIFACTS = "/opt/airflow/artifacts/serving"


def _extract(**_):
    from pathlib import Path

    assert Path(RAW_TRAIN).exists(), "train data not mounted into Airflow"
    return {"train": RAW_TRAIN, "test": RAW_TEST, "tree": TREE}


def _build_features(ti=None, **_):
    from pathlib import Path

    from ecom_category_project.data.prepare_dataset import prepare_datasets

    prepare_datasets(Path(RAW_TRAIN), Path(RAW_TEST), Path(TREE), Path(PROCESSED))
    return f"{PROCESSED}/train_features.csv.gz"


def _drift_check(**_):
    from pathlib import Path

    from drift_report import run_drift  # см. PYTHONPATH в compose (monitoring/drift)

    summary = run_drift(
        Path(f"{PROCESSED}/train_features.csv.gz"),
        Path(f"{PROCESSED}/test_features.csv.gz"),
        Path("/opt/airflow/reports/drift"),
    )
    return summary


def _retrain(**_):
    import subprocess

    subprocess.run(
        [
            "python", "-m", "ecom_category_project.models.train_text_baseline",
            "--config", "/opt/airflow/configs/submission.toml",
            "--train", f"{PROCESSED}/train_features.csv.gz",
            "--output-dir", ARTIFACTS,
            "--use-mlflow",
        ],
        check=True,
    )


def _publish_s3(**_):
    import subprocess

    subprocess.run(
        [
            "python", "/opt/airflow/scripts/upload_artifacts_s3.py",
            "up", f"{ARTIFACTS}/models/text_baseline.joblib", "models/model.joblib",
        ],
        check=False,
    )


with DAG(
    dag_id="ecom_retrain",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["ecom", "retrain", "drift"],
) as dag:
    extract = PythonOperator(task_id="extract", python_callable=_extract)
    build_features = PythonOperator(task_id="build_features", python_callable=_build_features)
    drift_check = PythonOperator(task_id="drift_check", python_callable=_drift_check)
    retrain = PythonOperator(task_id="retrain", python_callable=_retrain)
    publish = PythonOperator(task_id="publish_s3", python_callable=_publish_s3)

    extract >> build_features >> drift_check >> retrain >> publish
