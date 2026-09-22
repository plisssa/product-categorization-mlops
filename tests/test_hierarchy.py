import numpy as np
import pandas as pd

from ecom_category_project.models.hierarchy import build_category_tree, load_tree


def _tree_frame():
    return pd.DataFrame(
        {
            "category_ind": [0, 1, 2],
            "category": ["Мебель -> Кровати", "Мебель -> Столы", "Красота -> Парфюм"],
        }
    )


def test_build_tree_roots_and_paths(tmp_path):
    path = tmp_path / "tree.csv"
    _tree_frame().to_csv(path, index=False)
    tree = build_category_tree(load_tree(path))
    assert tree.n_roots == 2
    assert tree.root_of(0) == "Мебель"
    assert tree.root_of(2) == "Красота"


def test_mask_proba_by_root_zeros_other_roots(tmp_path):
    path = tmp_path / "tree.csv"
    _tree_frame().to_csv(path, index=False)
    tree = build_category_tree(load_tree(path))
    classes = np.array([0, 1, 2])
    proba = np.array([[0.2, 0.3, 0.5]])  # argmax -> класс 2 (Красота)
    masked = tree.mask_proba_by_root(proba, classes, predicted_roots=["Мебель"])
    # класс 2 (Красота) обнулён -> argmax теперь среди Мебели (0/1)
    assert masked[0, 2] == 0.0
    assert masked.argmax(axis=1)[0] in (0, 1)
