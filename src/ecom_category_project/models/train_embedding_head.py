"""Обучение классификатора-головы поверх эмбеддингов (текст [+ картинки]).

Архитектура (как в лекции 1: E5 + ViT/CLIP + голова):
    full_text --E5-->  text_emb  \
                                   concat --> head (LogReg / MLP / LightGBM) --> category_ind
    image     --CLIP-> image_emb /

Голова обучается с учётом дисбаланса (class_weight). Тяжёлые импорты — ленивые.
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ecom_category_project.data.download_images import url_to_filename
from ecom_category_project.features.embeddings import (
    compute_image_embeddings,
    compute_text_embeddings,
    fuse,
)
from ecom_category_project.models.common import save_json
from ecom_category_project.models.proba_io import save_proba
from ecom_category_project.tracking.mlflow_utils import log_run_if_enabled


def load_config(path: Path) -> dict:
    with open(path, "rb") as fp:
        return tomllib.load(fp)


def _image_paths(df: pd.DataFrame, images_dir: Path) -> list[str | None]:
    paths: list[str | None] = []
    for url in df.get("image_url", pd.Series([""] * len(df))).fillna("").astype(str):
        if url.startswith("http"):
            paths.append(str(images_dir / url_to_filename(url)))
        else:
            paths.append(None)
    return paths


def _features(df: pd.DataFrame, emb_cfg: dict, split: str) -> np.ndarray:
    text_cache = emb_cfg.get(f"text_cache_{split}", emb_cfg.get("text_cache"))
    text_emb = compute_text_embeddings(
        df["full_text"].fillna("").astype(str).tolist(),
        model_name=emb_cfg.get("text_model", "intfloat/multilingual-e5-large"),
        batch_size=emb_cfg.get("batch_size", 128),
        cache_path=text_cache,
    )
    image_emb = None
    if emb_cfg.get("use_image", False):
        image_cache = emb_cfg.get(f"image_cache_{split}", emb_cfg.get("image_cache"))
        image_emb = compute_image_embeddings(
            _image_paths(df, Path(emb_cfg.get("images_dir", "data/images"))),
            model_name=emb_cfg.get("image_model", "openai/clip-vit-base-patch32"),
            batch_size=emb_cfg.get("image_batch_size", 64),
            cache_path=image_cache,
        )
    return fuse(text_emb, image_emb, image_weight=emb_cfg.get("image_weight", 1.0))


def _build_head(model_cfg: dict):
    kind = model_cfg.get("kind", "logreg")
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=model_cfg.get("C", 1.0),
            max_iter=model_cfg.get("max_iter", 1000),
            class_weight=model_cfg.get("class_weight", "balanced"),
            n_jobs=-1,
        )
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=tuple(model_cfg.get("hidden_layer_sizes", [512])),
            alpha=model_cfg.get("alpha", 1e-4),
            max_iter=model_cfg.get("max_iter", 60),
            early_stopping=True,
        )
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=model_cfg.get("n_estimators", 600),
            learning_rate=model_cfg.get("learning_rate", 0.05),
            num_leaves=model_cfg.get("num_leaves", 127),
            class_weight=model_cfg.get("class_weight", "balanced"),
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported head kind: {kind}")


def main(args: argparse.Namespace) -> None:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    config = load_config(args.config)
    emb_cfg = config.get("embeddings", {})
    model_cfg = config.get("model", {})
    label_column = config.get("dataset", {}).get("label_column", "category_ind")

    train_df = pd.read_csv(args.train)
    X = _features(train_df, emb_cfg, split="train")
    y = train_df[label_column].to_numpy()

    counts = pd.Series(y).value_counts()
    rare = counts[counts < 2].index
    stratify = None if len(rare) else y
    Xtr, Xval, ytr, yval = train_test_split(
        X, y,
        test_size=config.get("split", {}).get("test_size", 0.15),
        random_state=config.get("split", {}).get("random_state", 42),
        stratify=stratify,
    )

    head = _build_head(model_cfg)
    head.fit(Xtr, ytr)
    metrics = {
        "train_accuracy": float(accuracy_score(ytr, head.predict(Xtr))),
        "train_f1_macro": float(f1_score(ytr, head.predict(Xtr), average="macro")),
        "test_accuracy": float(accuracy_score(yval, head.predict(Xval))),
        "test_f1_macro": float(f1_score(yval, head.predict(Xval), average="macro")),
    }
    print("Metrics:", metrics)

    out_dir = Path(args.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "models" / "embedding_head.joblib"
    metrics_path = out_dir / "classification_metrics.json"
    joblib.dump({"head": head, "classes": head.classes_, "config": config}, model_path)
    save_json(metrics, metrics_path)

    label_counts_path = out_dir / "dataset_label_counts.json"
    save_json({str(k): int(v) for k, v in counts.sort_index().items()}, label_counts_path)
    log_run_if_enabled(
        enabled=args.use_mlflow,
        config=config,
        run_name=config.get("run", {}).get("name", args.config.stem),
        metrics=metrics,
        model=head,
        metrics_path=metrics_path,
        label_counts_path=label_counts_path,
    )

    # Полное дообучение на всём train + предсказание теста (вероятности / сабмишн).
    if args.test:
        head_full = _build_head(model_cfg)
        head_full.fit(X, y)
        test_df = pd.read_csv(args.test)
        Xte = _features(test_df, emb_cfg, split="test")
        proba = head_full.predict_proba(Xte)
        ids = test_df["ID"] if "ID" in test_df.columns else np.arange(len(test_df))
        if args.proba_out:
            save_proba(args.proba_out, np.asarray(ids), head_full.classes_, proba)
            print(f"Saved proba to: {args.proba_out}")
        if args.submission_out:
            preds = head_full.classes_[proba.argmax(axis=1)]
            Path(args.submission_out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ID": ids, "category_ind": preds}).to_csv(args.submission_out, index=False)
            print(f"Saved submission to: {args.submission_out}")

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--proba-out", type=Path, default=None)
    parser.add_argument("--submission-out", type=Path, default=None)
    parser.add_argument("--use-mlflow", action="store_true")
    main(parser.parse_args())
