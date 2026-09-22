from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecom_category_project.features.text import TextBuildConfig, build_text_column, extract_domain

RAW_COLUMNS = ["name", "description", "model", "type_prefix", "vendor", "url", "image_url"]


def _load_tree(tree_path: Path) -> pd.DataFrame:
    tree = pd.read_csv(tree_path)
    tree["root_category"] = tree["category"].str.split(" -> ").str[0]
    return tree


def _prepare_common(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in RAW_COLUMNS:
        if column not in work.columns:
            work[column] = ""
        work[column] = work[column].fillna("").astype(str)
    work["domain"] = work["url"].map(extract_domain)
    return work


def prepare_datasets(train_path: Path, test_path: Path, tree_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = _load_tree(tree_path)

    train = _prepare_common(pd.read_parquet(train_path))
    test = _prepare_common(pd.read_parquet(test_path))

    merge_cols = ["category_ind", "category", "root_category"]
    train = train.merge(tree[merge_cols], on="category_ind", how="left")

    text_config = TextBuildConfig(
        text_fields=["vendor", "name", "model", "type_prefix", "description"],
        include_domain=True,
    )
    train["full_text"] = build_text_column(train, text_config)
    test["full_text"] = build_text_column(test, text_config)

    train.to_csv(output_dir / "train_features.csv.gz", index=False)
    test.to_csv(output_dir / "test_features.csv.gz", index=False)
    tree.to_csv(output_dir / "tree_enriched.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare_datasets(args.train, args.test, args.tree, args.output_dir)
