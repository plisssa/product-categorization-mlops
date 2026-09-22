import numpy as np

from ecom_category_project.models.proba_io import blend, load_proba, save_proba


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "a.npz"
    save_proba(p, np.array([1, 2]), np.array([10, 20]), np.array([[0.9, 0.1], [0.2, 0.8]]))
    loaded = load_proba(p)
    assert loaded["ids"].tolist() == [1, 2]
    assert loaded["classes"].tolist() == [10, 20]


def test_blend_aligns_classes_and_weights(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    save_proba(a, np.array([1]), np.array([10, 20]), np.array([[0.8, 0.2]]))
    save_proba(b, np.array([1]), np.array([20, 30]), np.array([[0.7, 0.3]]))
    res = blend([a, b], weights=[1.0, 1.0])
    assert res["classes"].tolist() == [10, 20, 30]
    # класс 20 присутствует в обоих -> ненулевой и наибольший
    assert res["proba"].argmax(axis=1)[0] == 1
