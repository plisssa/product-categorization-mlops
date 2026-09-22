"""Ансамбль: смешать сохранённые матрицы вероятностей -> финальный submission."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecom_category_project.models.proba_io import blend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proba", nargs="+", required=True, help="paths to *.npz proba files")
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = blend(args.proba, args.weights)
    preds = result["classes"][result["proba"].argmax(axis=1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": result["ids"], "category_ind": preds}).to_csv(args.output, index=False)
    print(f"Blended {len(args.proba)} models -> {args.output}")


if __name__ == "__main__":
    main()
