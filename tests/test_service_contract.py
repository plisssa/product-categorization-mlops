import importlib

import pytest

pytest.importorskip("flask")
pytest.importorskip("prometheus_client")


def test_predict_contract(monkeypatch):
    # лёгкий предиктор-заглушка, чтобы не тянуть модель
    monkeypatch.setenv("MODEL_BACKEND", "local")
    app_module = importlib.import_module("service.app")

    class Stub:
        def predict(self, full_text: str) -> int:
            return 42

    monkeypatch.setattr(app_module, "_predictor", Stub())
    client = app_module.app.test_client()

    payload = {
        "url": "https://shop.ru/p/1",
        "texts": {"name": "Кровать", "vendor": "Орматек"},
        "image_url": "http://img/1.jpg",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.get_json() == {"category_ind": 42}

    assert app_module.app.test_client().get("/health").status_code == 200
