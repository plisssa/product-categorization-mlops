"""Конфигурация сервиса из переменных окружения (12-factor)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    service_port: int = int(os.getenv("SERVICE_PORT", "8080"))
    model_backend: str = os.getenv("MODEL_BACKEND", "local")  # local | triton
    local_model_path: str = os.getenv("LOCAL_MODEL_PATH", "artifacts/serving/model.joblib")
    triton_url: str = os.getenv("TRITON_URL", "triton:8000")
    triton_model: str = os.getenv("TRITON_MODEL", "ecom_ensemble")
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    cache_ttl: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
    image_timeout: float = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "5"))


SETTINGS = Settings()
