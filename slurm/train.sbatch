#!/bin/bash
#SBATCH --job-name=ecom
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/ecom_%j.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export MLFLOW_TRACKING_URI="$SLURM_SUBMIT_DIR/mlruns"
export MLFLOW_EXPERIMENT_NAME="ecom-category"
mkdir -p data/processed artifacts/proba submissions docs/generated
python -c "import torch;print('CUDA:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo "== prepare =="; python -m ecom_category_project.data.prepare_dataset --train data/raw/train.parquet.snappy --test data/raw/test.parquet.snappy --tree data/raw/tree.csv --output-dir data/processed
echo "== eda =="; python -m ecom_category_project.data.eda --train data/raw/train.parquet.snappy --tree data/raw/tree.csv --output-dir docs/generated || true

echo "== tfidf =="; python -m ecom_category_project.models.train_text_baseline --config configs/experiments/run06_full_text_word_char_sgd.toml --train data/processed/train_features.csv.gz --output-dir artifacts/tfidf --use-mlflow || true
python -m ecom_category_project.models.make_submission --config configs/experiments/run06_full_text_word_char_sgd.toml --train data/processed/train_features.csv.gz --test data/processed/test_features.csv.gz --output submissions/tfidf.csv --proba-out artifacts/proba/tfidf.npz || true

echo "== e5-large logreg =="; python -m ecom_category_project.models.train_embedding_head --config configs/experiments/run10_e5_logreg.toml --train data/processed/train_features.csv.gz --test data/processed/test_features.csv.gz --output-dir artifacts/e5_logreg --proba-out artifacts/proba/e5_logreg.npz --use-mlflow || true

echo "== e5-large mlp =="; python -m ecom_category_project.models.train_embedding_head --config configs/experiments/run12_e5_mlp.toml --train data/processed/train_features.csv.gz --test data/processed/test_features.csv.gz --output-dir artifacts/e5_mlp --proba-out artifacts/proba/e5_mlp.npz --use-mlflow || true

echo "== blend =="; python -m ecom_category_project.models.blend_submission --proba artifacts/proba/tfidf.npz artifacts/proba/e5_logreg.npz artifacts/proba/e5_mlp.npz --weights 1.0 1.5 1.5 --output submissions/ensemble_final.csv || true

echo "== tests =="; python -m pytest -q || true
echo "== DONE =="
