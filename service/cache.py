"""Кэш предсказаний на Redis с мягкой деградацией (если Redis недоступен)."""
from __future__ import annotations

import hashlib
import json
from typing import Any


class PredictionCache:
    def __init__(self, redis_url: str, ttl: int) -> None:
        self.ttl = ttl
        self._client = None
        try:
            import redis

            self._client = redis.Redis.from_url(redis_url, socket_connect_timeout=1)
            self._client.ping()
        except Exception:
            self._client = None  # работаем без кэша

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return "pred:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> int | None:
        if not self._client:
            return None
        try:
            val = self._client.get(key)
            return int(val) if val is not None else None
        except Exception:
            return None

    def set(self, key: str, value: int) -> None:
        if not self._client:
            return
        try:
            self._client.setex(key, self.ttl, int(value))
        except Exception:
            pass
