"""Сохранение/загрузка и взвешенное смешивание матриц вероятностей.

Каждый эксперимент может сохранить предсказанные вероятности по тесту в .npz
(``ids``, ``classes``, ``proba``). Ансамбль = выравнивание по объединению
классов и взвешенное усреднение — стандартный приём для буста на Kaggle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_proba(path: str | Path, ids: np.ndarray, classes: np.ndarray, proba: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids), classes=np.asarray(classes), proba=np.asarray(proba))


def load_proba(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {"ids": data["ids"], "classes": data["classes"], "proba": data["proba"]}


def blend(proba_files: list[str | Path], weights: list[float] | None = None) -> dict[str, np.ndarray]:
    """Выровнять по объединению классов и взвешенно усреднить вероятности."""
    parts = [load_proba(p) for p in proba_files]
    if weights is None:
        weights = [1.0] * len(parts)
    if len(weights) != len(parts):
        raise ValueError("weights length must match number of proba files")

    ids = parts[0]["ids"]
    for part in parts[1:]:
        if not np.array_equal(part["ids"], ids):
            raise ValueError("All proba files must share the same ids order")

    all_classes = sorted({int(c) for part in parts for c in part["classes"]})
    class_to_col = {c: i for i, c in enumerate(all_classes)}
    blended = np.zeros((len(ids), len(all_classes)), dtype=np.float64)
    wsum = float(sum(weights))

    for part, w in zip(parts, weights):
        cols = [class_to_col[int(c)] for c in part["classes"]]
        blended[:, cols] += (w / wsum) * part["proba"]

    return {"ids": ids, "classes": np.array(all_classes), "proba": blended}
