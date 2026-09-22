#!/bin/sh
set -e
MODEL="${LOCAL_MODEL_PATH:-artifacts/serving/model.joblib}"
if [ ! -f "$MODEL" ]; then
  echo "[entrypoint] model not found at $MODEL — fetching..."
  mkdir -p "$(dirname "$MODEL")"
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
    python - <<'PY' || echo "[entrypoint] WARN: S3 download failed"
import os, boto3
m = os.environ.get("LOCAL_MODEL_PATH", "artifacts/serving/model.joblib")
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "https://minio.v-efimov.tech"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)
bucket = os.environ.get("S3_BUCKET", "e.puzyreva")
key = os.environ.get("S3_ARTIFACTS_PREFIX", "models").rstrip("/") + "/model.joblib"
print("[entrypoint] downloading s3://%s/%s" % (bucket, key))
s3.download_file(bucket, key, m)
print("[entrypoint] downloaded ->", m)
PY
  elif [ -n "${MODEL_PUBLIC_URL:-}" ]; then
    curl -fsSL "${MODEL_PUBLIC_URL}" -o "$MODEL" || echo "[entrypoint] WARN: public download failed"
  else
    echo "[entrypoint] WARN: нет AWS-кредов и MODEL_PUBLIC_URL — модель недоступна"
  fi
fi
exec gunicorn -w 2 -t 120 -b 0.0.0.0:"${SERVICE_PORT:-8080}" service.app:app
