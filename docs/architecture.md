# Архитектура решения

Диаграмма: [`architecture.svg`](architecture.svg).

Решение делится на три слоя.

## 1. Research / Kaggle (обучение)
`prepare_dataset` собирает из сырых полей `full_text` (vendor/name/model/
type_prefix/description + домен + токены URL) и обогащает данными дерева
категорий. Линейка моделей: TF-IDF + линейные → текстовые эмбеддинги E5 →
late fusion E5 + CLIP (картинки) → MLP-голова. Вероятности моделей смешиваются
(`blend_submission.py`) в финальный сабмишн. Все запуски логируются в **MLflow**
(параметры, метрики, модель), артефакты — на **S3/MinIO**. Лучший чекпойнт
выкладывается на S3 для сервиса.

## 2. Production / MaaS (сервис)
`./run_service.sh` поднимает один `docker compose`:
- **Flask-гейтвей** — REST `/predict` со строгим контрактом
  (`url`, `texts`, `image_url` → `category_ind`), сбор признаков, кэш, метрики.
  Порт берётся из `SERVICE_PORT`.
- **Redis** — кэш предсказаний (ключ = хэш запроса).
- **Triton Inference Server** — модель как ONNX-ансамбль
  (`preprocess → text_encoder(E5) → category_head → postprocess`). Бэкенд
  переключается `MODEL_BACKEND=triton|local`; по умолчанию `local` (TF-IDF
  joblib) — надёжно на CPU-сервере без GPU.

## 3. Observability & Automation
- **Prometheus** скрейпит `/metrics`, **Grafana** рисует дашборд,
  **Alertmanager** шлёт алерты по порогам (5xx, latency p95, недоступность).
- **Evidently** строит отчёт о дрифте данных (reference = train, current =
  свежие данные).
- **Airflow** DAG `ecom_retrain` (`@daily`): extract → features → drift →
  retrain → publish на S3.
- **GitLab CI/CD**: lint → test → build образа сервиса на каждый push.

## Технологические решения
- Зависимости — через **uv** (`pyproject.toml` + lock), extras `embeddings`/`dev`.
- Секреты только в `.env` (в git не попадают); данные/чекпойнты — на S3.
- Тяжёлые эмбеддинги считаются офлайн на GPU; в сервисе — ONNX/CPU или
  лёгкий TF-IDF-бэкенд, что гарантирует работу на сервере без GPU.
