"""Загрузка/выгрузка артефактов модели на S3 (MinIO).

Требование защиты #5: данные для запуска сервиса (чекпойнт, словари) лежат на
внешнем хранилище. Ключи берутся из окружения (.env).

Примеры:
  python scripts/upload_artifacts_s3.py up   artifacts/serving/model.joblib  models/model.joblib
  python scripts/upload_artifacts_s3.py down models/model.joblib  artifacts/serving/model.joblib
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://minio.v-efimov.tech"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["up", "down"])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "e.puzyreva"))
    args = parser.parse_args()

    s3 = _client()
    if args.action == "up":
        s3.upload_file(args.src, args.bucket, args.dst)
        print(f"Uploaded {args.src} -> s3://{args.bucket}/{args.dst}")
    else:
        Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(args.bucket, args.src, args.dst)
        print(f"Downloaded s3://{args.bucket}/{args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
