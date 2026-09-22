"""Гарантирует наличие чекпойнта модели ВНУТРИ контейнера.

Качает model.joblib с S3 (MinIO) по кредам из окружения, если файла нет.
boto3 присутствует в образе (service/requirements.txt), поэтому от хоста
проверяющего ничего не требуется — креды и порт пробрасываются в контейнер
через docker-compose. Фолбэк: публичная ссылка MODEL_PUBLIC_URL.

Запускается из entrypoint перед стартом gunicorn (а не в момент запроса),
чтобы первый /predict не упирался в таймаут на скачивании ~450 МБ.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _download_s3(bucket: str, key: str, dst: str, endpoint: str) -> None:
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3.download_file(bucket, key, dst)


def _download_url(url: str, dst: str) -> None:
    import requests

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)


def ensure_model() -> int:
    backend = os.getenv("MODEL_BACKEND", "local")
    if backend != "local":
        print(f"[fetch_model] backend={backend}: локальная модель не нужна, пропуск.")
        return 0

    dst = os.getenv("LOCAL_MODEL_PATH", "artifacts/serving/model.joblib")
    path = Path(dst)
    if path.is_file() and path.stat().st_size > 0:
        print(f"[fetch_model] модель уже на месте: {dst}")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)

    bucket = os.getenv("S3_BUCKET", "e.puzyreva")
    prefix = os.getenv("S3_ARTIFACTS_PREFIX", "models").strip("/")
    key = f"{prefix}/model.joblib" if prefix else "model.joblib"
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://minio.v-efimov.tech")

    if os.getenv("AWS_ACCESS_KEY_ID"):
        try:
            print(f"[fetch_model] качаю s3://{bucket}/{key} -> {dst} (endpoint={endpoint})")
            _download_s3(bucket, key, dst, endpoint)
            print(f"[fetch_model] готово (S3): {path.stat().st_size} байт")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[fetch_model] S3 не удался: {exc}", file=sys.stderr)

    url = os.getenv("MODEL_PUBLIC_URL")
    if url:
        try:
            print(f"[fetch_model] качаю по публичной ссылке -> {dst}")
            _download_url(url, dst)
            print(f"[fetch_model] готово (URL): {path.stat().st_size} байт")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[fetch_model] публичная загрузка не удалась: {exc}", file=sys.stderr)

    print(
        f"[fetch_model] ОШИБКА: не удалось получить модель ({dst}). "
        "Задайте AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (или MODEL_PUBLIC_URL) "
        "в окружении/.env — они пробрасываются в контейнер через docker-compose.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(ensure_model())
