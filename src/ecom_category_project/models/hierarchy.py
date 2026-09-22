"""Утилиты для работы с деревом категорий (tree.csv).

Дерево задаёт иерархию ``root -> ... -> leaf``. Целевая метка ``category_ind``
всегда соответствует листу, поэтому любая модель, предсказывающая лист, по
построению согласована с деревом. Дополнительно мы умеем:

* строить таргет верхнего уровня (root) — простой и точный классификатор;
* маскировать вероятности листьев несовместимых корней (иерархическое
  декодирование), что повышает macro-F1 на спорных классах.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CategoryTree:
    ind_to_path: dict[int, str]
    ind_to_root: dict[int, str]
    ind_to_depth: dict[int, int]
    root_to_id: dict[str, int]
    root_to_inds: dict[str, list[int]]

    @property
    def n_roots(self) -> int:
        return len(self.root_to_id)

    def root_of(self, ind: int) -> str:
        return self.ind_to_root.get(int(ind), "")

    def root_id_of(self, ind: int) -> int:
        return self.root_to_id.get(self.root_of(ind), -1)

    def roots_for_inds(self, inds: list[int]) -> list[str]:
        return [self.root_of(i) for i in inds]

    def mask_proba_by_root(
        self,
        proba: np.ndarray,
        classes: np.ndarray,
        predicted_roots: list[str],
    ) -> np.ndarray:
        """Обнулить вероятности листьев, не принадлежащих предсказанному корню.

        ``proba``: (n_samples, n_classes) выровнено с ``classes``.
        ``predicted_roots``: длина n_samples (например, из root-классификатора).
        """
        class_roots = np.array([self.ind_to_root.get(int(c), "") for c in classes])
        masked = proba.copy()
        for i, root in enumerate(predicted_roots):
            allowed = class_roots == root
            if allowed.any():
                masked[i, ~allowed] = 0.0
        return masked


def load_tree(tree_path: str | Path) -> pd.DataFrame:
    tree = pd.read_csv(tree_path)
    levels = tree["category"].fillna("").str.split(" -> ")
    tree["root_category"] = levels.str[0]
    tree["leaf_category"] = levels.str[-1]
    tree["depth"] = levels.apply(len)
    return tree


def build_category_tree(tree: pd.DataFrame) -> CategoryTree:
    ind_to_path = dict(zip(tree["category_ind"].astype(int), tree["category"].astype(str)))
    ind_to_root = dict(zip(tree["category_ind"].astype(int), tree["root_category"].astype(str)))
    ind_to_depth = dict(zip(tree["category_ind"].astype(int), tree["depth"].astype(int)))
    roots = sorted(set(ind_to_root.values()))
    root_to_id = {r: i for i, r in enumerate(roots)}
    root_to_inds: dict[str, list[int]] = {r: [] for r in roots}
    for ind, root in ind_to_root.items():
        root_to_inds[root].append(ind)
    return CategoryTree(ind_to_path, ind_to_root, ind_to_depth, root_to_id, root_to_inds)


def add_root_target(df: pd.DataFrame, tree: CategoryTree, label_column: str = "category_ind") -> pd.DataFrame:
    out = df.copy()
    out["root_category"] = out[label_column].map(tree.ind_to_root).fillna("")
    out["root_id"] = out["root_category"].map(tree.root_to_id).fillna(-1).astype(int)
    return out
