"""Prometheus-метрики сервиса."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("ecom_predict_requests_total", "Total /predict requests", ["status"])
LATENCY = Histogram("ecom_predict_latency_seconds", "Latency of /predict")
CACHE_HITS = Counter("ecom_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("ecom_cache_misses_total", "Cache misses")
PRED_CLASS = Counter("ecom_predicted_class_total", "Predicted class distribution", ["category_ind"])
LAST_SCORE = Gauge("ecom_last_request_unixtime", "Unix time of last request")
