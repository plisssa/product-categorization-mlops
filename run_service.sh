#!/usr/bin/env bash
# Поднимает весь сервис одной командой. Порт — из SERVICE_PORT (по умолчанию 8080).
# Модель скачивается ВНУТРИ контейнера сервиса (boto3) по кредам AWS_* из окружения.
#   ./run_service.sh           -> сервис + redis + мониторинг
#   ./run_service.sh triton    -> + Triton MaaS (ONNX)
set -euo pipefail
cd "$(dirname "$0")"

# Подгрузить .env, НЕ затирая уже заданные переменные окружения
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key=${line%%=*}; key=${key// /}
    val=${line#*=}
    case "$val" in \"*\") val=${val#\"}; val=${val%\"} ;; \'*\') val=${val#\'}; val=${val%\'} ;; esac
    if [ -z "${!key:-}" ]; then export "$key=$val"; fi
  done < .env
fi

export SERVICE_PORT="${SERVICE_PORT:-8080}"
export MODEL_BACKEND="${MODEL_BACKEND:-local}"

PROFILE_ARGS=""
[ "${1:-}" = "triton" ] && PROFILE_ARGS="--profile triton"

docker compose ${PROFILE_ARGS} up -d --build

# Дождаться готовности: на первом старте контейнер скачивает модель с S3 (~1-2 мин),
# поэтому возвращаем управление только когда сервис реально отвечает.
echo "Жду готовности сервиса на порту ${SERVICE_PORT}..."
for _ in $(seq 1 80); do
  if curl -fs "http://localhost:${SERVICE_PORT}/health" >/dev/null 2>&1; then
    echo "Service is up on port ${SERVICE_PORT}. Endpoints: /predict /health /metrics"
    exit 0
  fi
  sleep 3
done
echo "WARN: сервис не ответил за отведённое время. Логи: docker compose logs service"
exit 1
