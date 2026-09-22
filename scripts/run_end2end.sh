#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

mkdir -p data/processed artifacts/local_run submissions docs/generated

python -m ecom_category_project.data.prepare_dataset \
  --train data/raw/train.parquet.snappy \
  --test data/raw/test.parquet.snappy \
  --tree data/raw/tree.csv \
  --output-dir data/processed

python -m ecom_category_project.data.eda \
  --train data/raw/train.parquet.snappy \
  --tree data/raw/tree.csv \
  --output-dir docs/generated

python -m ecom_category_project.models.train_text_baseline \
  --config configs/experiments/run03_full_text_domain.toml \
  --train data/processed/train_features.csv.gz \
  --output-dir artifacts/local_run

python -m ecom_category_project.models.make_submission \
  --config configs/submission.toml \
  --train data/processed/train_features.csv.gz \
  --test data/processed/test_features.csv.gz \
  --output submissions/text_baseline_submission.csv

echo "End-to-end pipeline finished successfully."
