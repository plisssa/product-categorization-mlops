"""MaaS-гейтвей: REST /predict для категоризации товара.

Контракт (строго по заданию защиты):
  POST /predict
  {
    "url": "...",
    "texts": {"name": "...", "description": "...", "model": "...",
               "type_prefix": "...", "vendor": "..."},
    "image_url": "..."
  }
  -> 200 {"category_ind": 42}

Доп. ручки: GET /health (liveness), GET /metrics (Prometheus).
Порт берётся из переменной окружения SERVICE_PORT.
"""
from __future__ import annotations

import time

from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from service import metrics as M
from service.cache import PredictionCache
from service.config import SETTINGS
from service.featurize import request_to_full_text
from service.metrics import CACHE_HITS, CACHE_MISSES
from service.predictor import build_predictor

app = Flask(__name__)
_cache = PredictionCache(SETTINGS.redis_url, SETTINGS.cache_ttl)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = build_predictor(SETTINGS)
    return _predictor


@app.get("/health")
def health():
    return jsonify({"status": "ok", "backend": SETTINGS.model_backend}), 200


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.post("/predict")
def predict():
    start = time.perf_counter()
    try:
        payload = request.get_json(force=True) or {}
        if "texts" not in payload and "url" not in payload:
            M.REQUESTS.labels(status="400").inc()
            return jsonify({"error": "expected fields: url, texts, image_url"}), 400

        cache_key = _cache.key(payload)
        cached = _cache.get(cache_key)
        if cached is not None:
            CACHE_HITS.inc()
            M.REQUESTS.labels(status="200").inc()
            return jsonify({"category_ind": int(cached)}), 200
        CACHE_MISSES.inc()

        full_text = request_to_full_text(payload)
        category_ind = int(get_predictor().predict(full_text))

        _cache.set(cache_key, category_ind)
        M.PRED_CLASS.labels(category_ind=str(category_ind)).inc()
        M.LAST_SCORE.set(time.time())
        M.REQUESTS.labels(status="200").inc()
        return jsonify({"category_ind": category_ind}), 200
    except Exception as exc:
        M.REQUESTS.labels(status="500").inc()
        return jsonify({"error": str(exc)}), 500
    finally:
        M.LATENCY.observe(time.perf_counter() - start)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=SETTINGS.service_port)
