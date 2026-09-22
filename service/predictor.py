"""Бэкенды инференса: локальный (joblib) и Triton (ONNX MaaS)."""
from __future__ import annotations

from typing import Protocol


class Predictor(Protocol):
    def predict(self, full_text: str) -> int: ...


class LocalPredictor:
    """TF-IDF + линейная модель из joblib. Надёжно работает на CPU."""

    def __init__(self, model_path: str) -> None:
        import joblib

        self.model = joblib.load(model_path)

    def predict(self, full_text: str) -> int:
        return int(self.model.predict([full_text])[0])


class TritonPredictor:
    """Клиент Triton Inference Server (MaaS). Голова-классификатор в ONNX.

    Требует эмбеддинг текста (E5) на входе; здесь используется onnxruntime-энкодер,
    выгруженный в Triton как отдельная модель ensemble. Подробности — в triton/.
    """

    def __init__(self, url: str, model_name: str) -> None:
        import tritonclient.http as http

        self.http = http
        self.client = http.InferenceServerClient(url=url)
        self.model_name = model_name

    def predict(self, full_text: str) -> int:
        import numpy as np

        # ensemble 'ecom_ensemble' принимает сырой текст (BYTES) и возвращает class id
        text_in = self.http.InferInput("TEXT", [1, 1], "BYTES")
        text_in.set_data_from_numpy(np.array([[full_text.encode("utf-8")]], dtype=object))
        out = self.http.InferRequestedOutput("CATEGORY_IND")
        resp = self.client.infer(self.model_name, inputs=[text_in], outputs=[out])
        return int(resp.as_numpy("CATEGORY_IND").reshape(-1)[0])


def build_predictor(settings) -> Predictor:
    if settings.model_backend == "triton":
        return TritonPredictor(settings.triton_url, settings.triton_model)
    return LocalPredictor(settings.local_model_path)
