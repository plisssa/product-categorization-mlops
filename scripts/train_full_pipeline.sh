#!/usr/bin/env bash
# Полный GPU-пайплайн: данные -> эмбеддинги (текст+картинки) -> модели -> ансамбль.
# Требует:  uv sync --extra embeddings   (torch/transformers/CLIP)
set -euo pipefail
export PYTHONPATH=src
mkdir -p data/processed data/images artifacts/emb artifacts/proba submissions docs/generated

echo "[1/6] Подготовка данных + EDA"
python -m ecom_category_project.data.prepare_dataset \
  --train data/raw/train.parquet.snappy --test data/raw/test.parquet.snappy \
  --tree data/raw/tree.csv --output-dir data/processed
python -m ecom_category_project.data.eda \
  --train data/raw/train.parquet.snappy --tree data/raw/tree.csv --output-dir docs/generated

echo "[2/6] Скачивание картинок (для CLIP)"
python -m ecom_category_project.data.download_images --input data/processed/train_features.csv.gz --out-dir data/images --workers 24 || true
python -m ecom_category_project.data.download_images --input data/processed/test_features.csv.gz  --out-dir data/images --workers 24 || true

echo "[3/6] TF-IDF baseline -> submission + proba"
python -m ecom_category_project.models.make_submission \
  --config configs/experiments/run06_full_text_word_char_sgd.toml \
  --train data/processed/train_features.csv.gz --test data/processed/test_features.csv.gz \
  --output submissions/tfidf.csv --proba-out artifacts/proba/tfidf.npz

echo "[4/6] E5 text head -> proba"
python -m ecom_category_project.models.train_embedding_head \
  --config configs/experiments/run10_e5_logreg.toml \
  --train data/processed/train_features.csv.gz --test data/processed/test_features.csv.gz \
  --output-dir artifacts/e5_logreg --proba-out artifacts/proba/e5_logreg.npz --use-mlflow

echo "[5/6] E5 + CLIP fusion head -> proba"
python -m ecom_category_project.models.train_embedding_head \
  --config configs/experiments/run11_e5_clip_fusion_mlp.toml \
  --train data/processed/train_features.csv.gz --test data/processed/test_features.csv.gz \
  --output-dir artifacts/e5_clip --proba-out artifacts/proba/e5_clip.npz --use-mlflow

echo "[6/6] Ансамбль -> финальный submission"
python -m ecom_category_project.models.blend_submission \
  --proba artifacts/proba/tfidf.npz artifacts/proba/e5_logreg.npz artifacts/proba/e5_clip.npz \
  --weights 1.0 1.5 2.0 --output submissions/ensemble_final.csv

echo "Full pipeline finished. -> submissions/ensemble_final.csv"
