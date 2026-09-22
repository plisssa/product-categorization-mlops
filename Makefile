.PHONY: prepare eda images train sweep submit full blend drift triton-onnx serve up down test lint typecheck

prepare:
	PYTHONPATH=src python -m ecom_category_project.data.prepare_dataset \
		--train data/raw/train.parquet.snappy --test data/raw/test.parquet.snappy \
		--tree data/raw/tree.csv --output-dir data/processed

eda:
	PYTHONPATH=src python -m ecom_category_project.data.eda \
		--train data/raw/train.parquet.snappy --tree data/raw/tree.csv --output-dir docs/generated

images:
	PYTHONPATH=src python -m ecom_category_project.data.download_images \
		--input data/processed/train_features.csv.gz --out-dir data/images --workers 24

train:
	PYTHONPATH=src python -m ecom_category_project.models.train_text_baseline \
		--config configs/experiments/run06_full_text_word_char_sgd.toml \
		--train data/processed/train_features.csv.gz --output-dir artifacts/local_run --use-mlflow

sweep:
	bash scripts/run_mlflow_sweep.sh

submit:
	PYTHONPATH=src python -m ecom_category_project.models.make_submission \
		--config configs/experiments/run06_full_text_word_char_sgd.toml \
		--train data/processed/train_features.csv.gz --test data/processed/test_features.csv.gz \
		--output submissions/tfidf.csv --proba-out artifacts/proba/tfidf.npz

full:
	bash scripts/train_full_pipeline.sh

blend:
	PYTHONPATH=src python -m ecom_category_project.models.blend_submission \
		--proba artifacts/proba/tfidf.npz artifacts/proba/e5_logreg.npz artifacts/proba/e5_clip.npz \
		--weights 1.0 1.5 2.0 --output submissions/ensemble_final.csv

drift:
	PYTHONPATH=src python monitoring/drift/drift_report.py \
		--reference data/processed/train_features.csv.gz --current data/processed/test_features.csv.gz \
		--out-dir docs/generated/drift

triton-onnx:
	PYTHONPATH=src python triton/prepare_models.py \
		--head artifacts/e5_logreg/models/embedding_head.joblib --dim 1024

serve:
	MODEL_BACKEND=local LOCAL_MODEL_PATH=artifacts/local_run/models/text_baseline.joblib \
		PYTHONPATH=src:. SERVICE_PORT=8080 python service/app.py

up:
	./run_service.sh

down:
	docker compose down

test:
	uv run pytest -q

lint:
	uv run ruff check src service tests triton monitoring

typecheck:
	uv run mypy src
