from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ecom_category_project.models.common import (
    build_pipeline,
    load_config,
    load_processed_train,
    save_json,
)
from ecom_category_project.tracking.mlflow_utils import log_run_if_enabled

_VERSION_RE = re.compile(r"_v(\d+)$")


def resolve_versioned_dir(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if _VERSION_RE.search(output_dir.name):
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    max_version = 0
    pattern = re.compile(rf"^{re.escape(output_dir.name)}_v(\d+)$")

    for existing in output_dir.parent.iterdir():
        if not existing.is_dir():
            continue
        match = pattern.match(existing.name)
        if match:
            max_version = max(max_version, int(match.group(1)))

    resolved_dir = output_dir.parent / f"{output_dir.name}_v{max_version + 1}"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir


def split_with_rare_classes(
    df: pd.DataFrame,
    label_column: str,
    test_size: float,
    random_state: int,
):
    counts = df[label_column].value_counts()
    rare_labels = counts[counts < 2].index
    common_df = df[~df[label_column].isin(rare_labels)]
    rare_df = df[df[label_column].isin(rare_labels)]

    train_df, valid_df = train_test_split(
        common_df,
        test_size=test_size,
        random_state=random_state,
        stratify=common_df[label_column],
    )
    train_df = pd.concat([train_df, rare_df], ignore_index=True)
    return train_df, valid_df


def compute_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def main(config_path: Path, train_path: Path, output_dir: Path, use_mlflow: bool) -> None:
    config = load_config(config_path)

    resolved_output_dir = resolve_versioned_dir(output_dir)
    models_dir = resolved_output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_train(train_path, config=config)
    label_column = config["dataset"].get("label_column", "category_ind")
    train_df, valid_df = split_with_rare_classes(
        df,
        label_column=label_column,
        test_size=config["split"].get("test_size", 0.15),
        random_state=config["split"].get("random_state", 42),
    )

    pipeline = build_pipeline(config)
    pipeline.fit(train_df["full_text"], train_df[label_column])

    train_pred = pipeline.predict(train_df["full_text"])
    valid_pred = pipeline.predict(valid_df["full_text"])

    metrics = {}
    metrics.update(compute_metrics(train_df[label_column], train_pred, prefix="train"))
    metrics.update(compute_metrics(valid_df[label_column], valid_pred, prefix="test"))

    label_counts = df[label_column].value_counts().sort_index()
    label_counts_json = {str(key): int(value) for key, value in label_counts.items()}

    metrics_path = resolved_output_dir / "classification_metrics.json"
    label_counts_path = resolved_output_dir / "dataset_label_counts.json"
    model_path = models_dir / "text_baseline.joblib"

    save_json(metrics, metrics_path)
    save_json(label_counts_json, label_counts_path)
    joblib.dump(pipeline, model_path)

    log_run_if_enabled(
        enabled=use_mlflow,
        config=config,
        run_name=config["run"].get("name", config_path.stem),
        metrics=metrics,
        model=pipeline,
        metrics_path=metrics_path,
        label_counts_path=label_counts_path,
    )

    print(f"Saved model to: {model_path}")
    print(f"Artifacts directory: {resolved_output_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--use-mlflow", action="store_true")
    args = parser.parse_args()
    main(args.config, args.train, args.output_dir, args.use_mlflow)
