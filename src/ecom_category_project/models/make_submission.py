"""Обучение TF-IDF-модели на полном train и создание submission (+ вероятности).

При наличии ``--proba-out`` сохраняет матрицу вероятностей по тесту в .npz —
её затем можно подмешать в ансамбль (см. blend_submission.py).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ecom_category_project.features.text import TextBuildConfig, build_text_column
from ecom_category_project.models.common import build_pipeline, load_config, load_processed_train
from ecom_category_project.models.proba_io import save_proba

_VERSION_RE = re.compile(r"_v(\d+)$")


def resolve_versioned_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = output_path.stem, output_path.suffix
    if _VERSION_RE.search(stem):
        return output_path
    max_version = 0
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+){re.escape(suffix)}$")
    for existing in output_path.parent.iterdir():
        if existing.is_file():
            match = pattern.match(existing.name)
            if match:
                max_version = max(max_version, int(match.group(1)))
    return output_path.parent / f"{stem}_v{max_version + 1}{suffix}"


def _ensure_full_text(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if "full_text" in df.columns:
        return df
    ds = config["dataset"]
    cfg = TextBuildConfig(
        text_fields=ds.get("text_fields", ["vendor", "name", "model", "type_prefix", "description"]),
        include_domain=ds.get("include_domain", True),
        include_url_tokens=ds.get("include_url_tokens", True),
        description_max_len=ds.get("description_max_len", 400),
        url_token_limit=ds.get("url_token_limit", 12),
    )
    df = df.copy()
    df["full_text"] = build_text_column(df, cfg)
    return df


def main(config_path: Path, train_path: Path, test_path: Path, output_path: Path, proba_out: Path | None) -> None:
    config = load_config(config_path)
    label_column = config["dataset"].get("label_column", "category_ind")

    train_df = load_processed_train(train_path, config=config)
    test_df = _ensure_full_text(pd.read_csv(test_path), config)

    pipeline = build_pipeline(config)
    pipeline.fit(train_df["full_text"], train_df[label_column])

    ids = test_df["ID"] if "ID" in test_df.columns else pd.Series(range(len(test_df)), name="ID")

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(test_df["full_text"])
        classes = pipeline.classes_
        preds = classes[proba.argmax(axis=1)]
    else:  # LinearSVC -> softmax over decision_function
        scores = pipeline.decision_function(test_df["full_text"])
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        classes = pipeline.classes_
        preds = classes[proba.argmax(axis=1)]

    resolved = resolve_versioned_output_path(output_path)
    pd.DataFrame({"ID": ids, "category_ind": preds}).to_csv(resolved, index=False)
    print(f"Submission saved to: {resolved}")

    if proba_out:
        save_proba(proba_out, np.asarray(ids), np.asarray(classes), np.asarray(proba))
        print(f"Proba saved to: {proba_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--proba-out", type=Path, default=None)
    args = parser.parse_args()
    main(args.config, args.train, args.test, args.output, args.proba_out)
