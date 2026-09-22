#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
mkdir -p artifacts/mlflow_runs

for config in configs/experiments/*.toml; do
  run_name=$(basename "$config" .toml)
  echo "[MLflow sweep] running ${run_name}"
  python -m ecom_category_project.models.train_text_baseline \
    --config "$config" \
    --train data/processed/train_features.csv.gz \
    --output-dir "artifacts/mlflow_runs/${run_name}" \
    --use-mlflow
  echo "[MLflow sweep] finished ${run_name}"
done
