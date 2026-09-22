"""EDA для соревнования Production ML Spring 2026.

Генерирует артефакты в ``docs/generated``:
* ``eda_summary.md``      — текстовый отчёт с ключевыми числами;
* ``label_counts.csv``    — распределение классов;
* набор PNG-графиков (дисбаланс, корневые категории, длины текстов,
  доля пропусков по полям, топ-домены, глубина дерева).

Главный вывод EDA, на который опирается всё решение: сильнейший дисбаланс
классов и длинный хвост (десятки классов с 1-5 примерами) — именно он
ограничивает macro-F1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"  # поддерживает кириллицу

RAW_TEXT_FIELDS = ["name", "description", "model", "type_prefix", "vendor"]


def _read_any(path: Path) -> pd.DataFrame:
    if str(path).endswith(".parquet.snappy") or str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save(fig, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)


def _plot_class_distribution(counts: pd.Series, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts.values)
    ax.set_yscale("log")
    ax.set_title("Распределение классов (log), отсортировано по частоте")
    ax.set_xlabel("класс (ранг)")
    ax.set_ylabel("число примеров (log)")
    _save(fig, out_dir, "class_distribution_log.png")


def _plot_root_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    if "root_category" not in df.columns:
        return
    vc = df["root_category"].value_counts().head(25)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(vc.index[::-1], vc.values[::-1])
    ax.set_title("Топ корневых категорий по числу товаров")
    ax.set_xlabel("число товаров")
    _save(fig, out_dir, "root_distribution.png")


def _plot_text_length(df: pd.DataFrame, out_dir: Path) -> None:
    source = "full_text" if "full_text" in df.columns else "name"
    if source not in df.columns:
        return
    lengths = df[source].fillna("").astype(str).str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths.clip(upper=lengths.quantile(0.99)), bins=60)
    ax.set_title(f"Длина текста ('{source}', обрезано по p99)")
    ax.set_xlabel("символов")
    ax.set_ylabel("товаров")
    _save(fig, out_dir, "text_length_hist.png")


def _plot_missing(df: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    fields = [c for c in RAW_TEXT_FIELDS + ["url", "image_url"] if c in df.columns]
    rates = {}
    for c in fields:
        col = df[c].fillna("").astype(str).str.strip()
        rates[c] = float((col == "").mean())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(rates.keys()), [v * 100 for v in rates.values()])
    ax.set_title("Доля пустых значений по полям")
    ax.set_ylabel("% пустых")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, out_dir, "missing_rate.png")
    return rates


def _plot_top_domains(df: pd.DataFrame, out_dir: Path) -> None:
    if "domain" not in df.columns:
        return
    vc = df["domain"].replace("", np.nan).dropna().value_counts().head(25)
    if vc.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(vc.index[::-1], vc.values[::-1])
    ax.set_title("Топ доменов-источников")
    ax.set_xlabel("число товаров")
    _save(fig, out_dir, "top_domains.png")


def _plot_tree_depth(tree: pd.DataFrame, out_dir: Path) -> None:
    levels = tree["category"].fillna("").str.split(" -> ").apply(len)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(levels, bins=range(1, int(levels.max()) + 2), align="left", rwidth=0.8)
    ax.set_title("Глубина дерева категорий")
    ax.set_xlabel("число уровней")
    ax.set_ylabel("число категорий")
    _save(fig, out_dir, "tree_depth.png")


def run_eda(train_path: Path, tree_path: Path, output_dir: Path, label_column: str = "category_ind") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _read_any(train_path)
    tree = pd.read_csv(tree_path)
    if "root_category" not in df.columns and "category" in tree.columns:
        tree_root = tree.copy()
        tree_root["root_category"] = tree_root["category"].str.split(" -> ").str[0]
        df = df.merge(tree_root[[label_column, "root_category"]], on=label_column, how="left")

    counts = df[label_column].value_counts().sort_values(ascending=False)
    counts.sort_index().to_csv(output_dir / "label_counts.csv", header=["count"])

    _plot_class_distribution(counts, output_dir)
    _plot_root_distribution(df, output_dir)
    _plot_text_length(df, output_dir)
    missing = _plot_missing(df, output_dir)
    _plot_top_domains(df, output_dir)
    _plot_tree_depth(tree, output_dir)

    n = len(df)
    n_classes = int(df[label_column].nunique())
    singletons = int((counts <= 1).sum())
    rare5 = int((counts <= 5).sum())
    imbalance = float(counts.max() / max(counts.min(), 1))
    img_avail = None
    if "image_url" in df.columns:
        img_avail = float(df["image_url"].fillna("").astype(str).str.startswith("http").mean())

    lines = [
        "# EDA — Production ML Spring 2026 (категоризация товаров)",
        "",
        f"- Объектов в train: **{n:,}**",
        f"- Уникальных классов (`{label_column}`): **{n_classes}**",
        f"- Классов с 1 примером (синглтоны): **{singletons}**",
        f"- Классов с <=5 примерами: **{rare5}**",
        f"- Дисбаланс (max/min частота класса): **{imbalance:,.0f}x**",
        f"- Самый частый класс: {counts.index[0]} ({counts.iloc[0]:,} примеров)",
    ]
    if img_avail is not None:
        lines.append(f"- Доля товаров с валидным `image_url`: **{img_avail*100:.1f}%**")
    lines += ["", "## Доля пустых значений по полям", ""]
    for k, v in missing.items():
        lines.append(f"- `{k}`: {v*100:.1f}% пустых")
    lines += [
        "",
        "## Ключевой вывод",
        "",
        "Длинный хвост редких классов — главная причина низкого macro-F1: каждый",
        "класс вносит равный вклад в метрику, а у десятков классов слишком мало",
        "данных. Поэтому решение делает упор на: (1) сильные семантические признаки",
        "(текстовые + картиночные эмбеддинги), (2) работу с дисбалансом",
        "(class_weight, oversampling хвоста), (3) иерархию дерева и (4) ансамбль.",
        "",
        "## Сгенерированные графики",
        "",
        "`class_distribution_log.png`, `root_distribution.png`, `text_length_hist.png`,",
        "`missing_rate.png`, `top_domains.png`, `tree_depth.png`.",
    ]
    (output_dir / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA artifacts written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label-column", default="category_ind")
    args = parser.parse_args()
    run_eda(args.train, args.tree, args.output_dir, args.label_column)
