from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ecom_category_project.features.text import TextBuildConfig, build_text_column


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "rb") as fp:
        return tomllib.load(fp)


def _build_text_config(dataset_cfg: dict[str, Any]) -> TextBuildConfig:
    return TextBuildConfig(
        text_fields=dataset_cfg.get(
            "text_fields",
            ["vendor", "name", "model", "type_prefix", "description"],
        ),
        include_domain=dataset_cfg.get("include_domain", True),
        include_url_tokens=dataset_cfg.get("include_url_tokens", True),
        description_max_len=dataset_cfg.get("description_max_len", 400),
        url_token_limit=dataset_cfg.get("url_token_limit", 12),
    )


def load_processed_train(path: Path, config: dict[str, Any] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    text_source_columns = {"vendor", "name", "model", "type_prefix", "description", "url"}
    has_raw_text_columns = any(col in df.columns for col in text_source_columns)

    if has_raw_text_columns:
        dataset_cfg = (config or {}).get("dataset", {})
        text_cfg = _build_text_config(dataset_cfg)
        df["full_text"] = build_text_column(df, text_cfg)
        return df

    if "full_text" not in df.columns:
        text_cfg = TextBuildConfig(
            text_fields=["vendor", "name", "model", "type_prefix", "description"],
            include_domain=True,
            include_url_tokens=True,
        )
        df["full_text"] = build_text_column(df, text_cfg)

    return df


def _build_single_vectorizer(vec_cfg: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=vec_cfg.get("analyzer", "word"),
        ngram_range=(vec_cfg.get("ngram_min", 1), vec_cfg.get("ngram_max", 2)),
        min_df=vec_cfg.get("min_df", 2),
        max_features=vec_cfg.get("max_features", 100000),
        sublinear_tf=vec_cfg.get("sublinear_tf", True),
        lowercase=True,
    )


def _build_word_char_vectorizer(vec_cfg: dict[str, Any]) -> FeatureUnion:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(
            vec_cfg.get("word_ngram_min", vec_cfg.get("ngram_min", 1)),
            vec_cfg.get("word_ngram_max", vec_cfg.get("ngram_max", 2)),
        ),
        min_df=vec_cfg.get("word_min_df", vec_cfg.get("min_df", 2)),
        max_features=vec_cfg.get("word_max_features", vec_cfg.get("max_features", 100000)),
        sublinear_tf=vec_cfg.get("sublinear_tf", True),
        lowercase=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer=vec_cfg.get("char_analyzer", "char_wb"),
        ngram_range=(
            vec_cfg.get("char_ngram_min", 3),
            vec_cfg.get("char_ngram_max", 5),
        ),
        min_df=vec_cfg.get("char_min_df", vec_cfg.get("min_df", 2)),
        max_features=vec_cfg.get("char_max_features", vec_cfg.get("max_features", 100000)),
        sublinear_tf=vec_cfg.get("sublinear_tf", True),
        lowercase=True,
    )

    return FeatureUnion(
        [
            ("word", word_vectorizer),
            ("char", char_vectorizer),
        ]
    )


def _build_vectorizer(vec_cfg: dict[str, Any]):
    kind = vec_cfg.get("kind")
    if kind is None:
        return _build_single_vectorizer(vec_cfg)

    if kind == "word":
        return _build_single_vectorizer(
            {
                **vec_cfg,
                "analyzer": "word",
            }
        )

    if kind == "char":
        return _build_single_vectorizer(
            {
                **vec_cfg,
                "analyzer": vec_cfg.get("analyzer", "char_wb"),
                "ngram_min": vec_cfg.get("char_ngram_min", vec_cfg.get("ngram_min", 3)),
                "ngram_max": vec_cfg.get("char_ngram_max", vec_cfg.get("ngram_max", 5)),
                "max_features": vec_cfg.get("char_max_features", vec_cfg.get("max_features", 150000)),
            }
        )

    if kind == "word_char":
        return _build_word_char_vectorizer(vec_cfg)

    raise ValueError(f"Unsupported vectorizer kind: {kind}")


def _build_classifier(model_cfg: dict[str, Any]):
    kind = model_cfg.get("kind", "sgd")

    if kind == "sgd":
        return SGDClassifier(
            loss=model_cfg.get("loss", "log_loss"),
            alpha=model_cfg.get("alpha", 1e-5),
            max_iter=model_cfg.get("max_iter", 5000),
            tol=model_cfg.get("tol", 1e-4),
            class_weight=model_cfg.get("class_weight", "balanced"),
            random_state=model_cfg.get("random_state", 42),
            n_jobs=-1,
        )

    if kind == "linear_svc":
        return LinearSVC(
            C=model_cfg.get("C", 1.0),
            class_weight=model_cfg.get("class_weight", "balanced"),
            max_iter=model_cfg.get("max_iter", 5000),
            random_state=model_cfg.get("random_state", 42),
        )

    raise ValueError(f"Unsupported model kind: {kind}")


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    vec_cfg = config["vectorizer"]
    model_cfg = config["model"]

    vectorizer = _build_vectorizer(vec_cfg)
    classifier = _build_classifier(model_cfg)

    return Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
