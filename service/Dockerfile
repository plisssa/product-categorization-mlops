FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

COPY src /app/src
COPY service /app/service
RUN chmod +x /app/service/entrypoint.sh

ENV PYTHONPATH=/app:/app/src
ENV SERVICE_PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD curl -fs http://localhost:${SERVICE_PORT}/health || exit 1

# entrypoint скачивает модель внутри контейнера (если её нет), затем стартует gunicorn.
# Запускаем через `sh`, чтобы не зависеть от exec-бита файла (важно после распаковки zip/Windows).
ENTRYPOINT ["sh", "/app/service/entrypoint.sh"]
