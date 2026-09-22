from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def log_run_if_enabled(
    *,
    enabled: bool,
    config: dict[str, Any],
    run_name: str,
    metrics: dict[str, float],
    model,
    metrics_path: Path,
    label_counts_path: Path,
) -> None:
    if not enabled:
        return

    import mlflow
    import mlflow.sklearn

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or "ecom-category-baselines"
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID") or None

    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name)

    flat_params: dict[str, Any] = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat_params[f"{section}.{key}"] = value

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flat_params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="artifacts")
        mlflow.log_artifact(str(label_counts_path), artifact_path="artifacts")
        mlflow.sklearn.log_model(model, artifact_path="model")
