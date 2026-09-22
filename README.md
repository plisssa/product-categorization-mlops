# E-commerce Product Categorization — Production ML Spring 2026

Сервис и исследовательский проект: предсказание категории товара (`category_ind`)
по его контенту — текстовым полям, картинке и URL. Метрика соревнования —
**macro F1-score**. Категории образуют дерево (`tree.csv`).

Решение состоит из двух частей:

1. **Research / Kaggle** — воспроизводимые эксперименты с трекингом в MLflow:
   от TF-IDF-бейзлайна до ансамбля «текстовые (E5) + картиночные (CLIP)
   эмбеддинги».
2. **Production / MaaS** — модель как отдельный сервис: REST `/predict`,
   развёртка на Triton (ONNX), кэш (Redis), мониторинг (Prometheus + Grafana +
   алерты), ML-мониторинг дрифта (Evidently), регулярные расчёты (Airflow),
   CI/CD (GitLab), артефакты на S3/MinIO.

**Итог:** macro-F1 на Kaggle вырос с **0.288** (первый бейзлайн) до **0.723**
(ансамбль TF-IDF + E5 + E5&CLIP) — подробности в разделе [9. Результаты](#9-результаты).

---

## 1. Быстрый старт

### 1.1. Окружение (uv)

```bash
uv sync                      # базовое окружение (research/CPU)
uv sync --extra embeddings   # + torch/transformers/CLIP (для эмбеддингов, GPU)
uv sync --extra dev          # + pytest/ruff/mypy
```

### 1.2. Данные

Положите файлы соревнования (с Kaggle) в `data/raw/`:

```
data/raw/train.parquet.snappy    # содержит метку category_ind
data/raw/test.parquet.snappy     # без меток, их и предсказываем
data/raw/tree.csv                # дерево категорий, уже в репозитории
```

Схема полей, заполненность и формат submission — в [`data/README.md`](data/README.md).

### 1.3. Воспроизводимый прогон (РК, CPU)

```bash
make prepare      # предобработка -> data/processed/*.csv.gz
make eda          # EDA-артефакты -> docs/generated/
make train        # обучение baseline + метрики
make submit       # submission.csv
# или всё сразу:
bash scripts/run_end2end.sh
```

### 1.4. Полный пайплайн на максимум качества (GPU)

```bash
uv sync --extra embeddings
bash scripts/train_full_pipeline.sh
# data -> эмбеддинги (E5 + CLIP) -> модели -> ансамбль -> submissions/ensemble_final.csv
```

### 1.5. Запуск сервиса (одной командой)

```bash
cp .env.example .env          # пропишите ключи S3/MinIO
SERVICE_PORT=8080 ./run_service.sh          # сервис(local) + redis + мониторинг
SERVICE_PORT=8080 ./run_service.sh triton   # + Triton MaaS (ONNX)
```

Проверка контракта:

```bash
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{
  "url": "https://ormatek.com/catalog/krovati/product/krovat-como-veda-4/",
  "texts": {"name": "Мягкая Кровать Орматек Como (Veda) 4", "vendor": "Орматек", "model": "Como (Veda) 4", "type_prefix": "Кровать"},
  "image_url": "http://d.mradx.net//ecomimg/914/21/9a84a4b6cbbc.jpeg"
}'
# -> {"category_ind": 42}
```

---

## 2. Контракт сервиса

| Метод | Путь | Назначение |
|------|------|-----------|
| POST | `/predict` | `{"url", "texts":{...}, "image_url"}` → `{"category_ind": <int>}` |
| GET  | `/health`  | liveness-проверка |
| GET  | `/metrics` | метрики Prometheus |

Порт задаётся переменной окружения `SERVICE_PORT`. Модель грузится с S3
(чекпойнт `models/model.joblib`); бэкенд выбирается `MODEL_BACKEND` (`local` |
`triton`).

---

## 3. Архитектура

См. диаграмму [`docs/architecture.svg`](docs/architecture.svg) и
[`docs/architecture.md`](docs/architecture.md).


---

## 4. Структура репозитория

```text
.
├── run_service.sh            # запуск всего сервиса одной командой (защита)
├── docker-compose.yml        # service + redis + prometheus + grafana + alertmanager + triton
├── pyproject.toml            # зависимости (uv), extras: embeddings / dev
├── Makefile                  # prepare/eda/train/submit/serve/test/lint
├── config.toml / .env.example
├── configs/experiments/*.toml   # конфиги воспроизводимых экспериментов
├── checkpoints/              # лёгкие головы на эмбеддингах (+ README про тяжёлую модель)
├── data/                     # raw / processed / images (в git не хранятся)
├── docs/                     # architecture.svg, REPORT.md, generated EDA
├── scripts/                  # run_end2end.sh, train_full_pipeline.sh, run_mlflow_sweep.sh, upload_artifacts_s3.py
├── src/ecom_category_project/
│   ├── data/        prepare_dataset.py, eda.py, download_images.py
│   ├── features/    text.py, embeddings.py
│   ├── models/      common.py, train_text_baseline.py, train_embedding_head.py,
│   │                make_submission.py, blend_submission.py, proba_io.py, hierarchy.py
│   └── tracking/    mlflow_utils.py
├── service/         app.py (/predict), predictor.py, cache.py, featurize.py, metrics.py, Dockerfile
├── triton/          model_repository/ (ensemble), prepare_models.py (ONNX)
├── monitoring/      prometheus/ alertmanager/ grafana/ drift/ (Evidently)
├── airflow/         dags/retrain_dag.py (регулярные расчёты)
├── mlflow_docker/   локальный MLflow + MinIO (compose)
└── tests/           pytest
```

---

## 5. Моделирование (как растёт macro-F1)

EDA (`docs/generated/eda_summary.md`): **111 405** товаров, **207** классов,
дисбаланс **31 563×**, 7 классов-синглтонов, у 30 классов ≤5 примеров,
картинка доступна у **100%** товаров (поля `vendor`/`url` почти всегда
заполнены, `name` пуст у 44%, `description` — у 65%). Главный ограничитель
macro-F1 — длинный хвост редких классов.

Линейка экспериментов (все логируются в MLflow):

| Этап | Признаки | Голова | Конфиг |
|------|----------|--------|--------|
| Baseline | TF-IDF (word) | SGD | `run03` |
| Classic+ | TF-IDF word+char | SGD/SVM | `run06` |
| Текст-эмбеддинги | E5-large | LogReg | `run10` |
| **Fusion** | **E5 + CLIP (late fusion)** | **MLP** | `run11` |
| **Ансамбль** | blend всех | — | `blend_submission.py` |

Работа с дисбалансом: `class_weight="balanced"`, аккуратный сплит редких
классов, иерархия дерева (`models/hierarchy.py`), смешивание вероятностей.

---

## 6. MLflow (трекинг)

```bash
docker compose -f mlflow_docker/compose.yml up -d     # MLflow + MinIO
export MLFLOW_TRACKING_URI=http://localhost:5000
make sweep   # серия запусков с логированием параметров/метрик/моделей
```

Каждый запуск логирует: все параметры конфига, метрики (accuracy, macro-F1),
саму модель и артефакты (метрики, распределение классов). Артефакты — на S3.

---

## 7. Production-компоненты 

| Компонент | Где | Запуск |
|-----------|-----|--------|
| MaaS / Triton (ONNX) | `triton/` | `python triton/prepare_models.py …` + `./run_service.sh triton` |
| Кэш (Redis) | `service/cache.py` | поднимается в compose |
| Мониторинг (Prometheus+Grafana) | `monitoring/` | Grafana: http://localhost:3000 |
| Алерты | `monitoring/prometheus/alerts.yml` | Alertmanager: http://localhost:9093 |
| ML-дрифт (Evidently) | `monitoring/drift/` | `python monitoring/drift/drift_report.py …` |
| Регулярные расчёты (Airflow) | `airflow/` | `docker compose -f airflow/docker-compose.yml up -d` |
| CI/CD | `.gitlab-ci.yml` | lint → test → build на каждый push |
| Артефакты на S3 | `scripts/upload_artifacts_s3.py` | `up` / `down` |

---

## 8. Тесты и качество кода

```bash
uv run pytest -q          # тесты
uv run ruff check .       # линтер
uv run mypy src           # типы
```

---

## 9. Результаты

Public leaderboard Kaggle, метрика macro-F1:

| # | Сабмишн | Что внутри | macro-F1 |
|---|---------|-----------|---------:|
| 1 | baseline | TF-IDF (word) по `name`+`vendor`, SGD | 0.288 |
| 2 | run03 | + `description`, домен и токены URL | 0.364 |
| 3 | run06 | TF-IDF word + char_wb, SGD, `class_weight="balanced"` | 0.387 |
| 4 | tfidf_v1 | подобранные гиперпараметры, обучение на полном train | 0.537 |
| 5 | ens_3way_v2 | ансамбль трёх текстовых моделей | 0.582 |
| 6 | e5_clip_solo | late fusion E5 + CLIP → MLP | 0.699 |
| 7 | **ensemble_clip** | **blend TF-IDF : E5-LogReg : E5+CLIP = 1 : 2 : 1.5** | **0.723** |

Что дало прирост:

- **char-н-граммы** (строки 2→3) — устойчивость к опечаткам и морфологии;
- **семантические эмбеддинги E5** вместо разреженных признаков (4→5);
- **картинки (CLIP)** — главный скачок, +0.12 (5→6): изображение есть у 100%
  товаров, тогда как `name` пуст у 44%, а `description` — у 65%;
- **смешивание вероятностей** разнородных моделей (6→7).

Узкое место метрики — длинный хвост: дисбаланс 31 563×, 7 классов-синглтонов.
Разбор ошибок и дальнейшие шаги — в [`docs/REPORT.md`](docs/REPORT.md).

---

## 10. Модель и артефакты

Чекпойнты разделены по весу — подробности в
[`checkpoints/README.md`](checkpoints/README.md).

| Что | Размер | Где |
|-----|-------:|-----|
| Головы на эмбеддингах (`e5_logreg`, `e5_clip_mlp`) | единицы МБ | `checkpoints/` — **в репозитории** |
| TF-IDF-модель сервиса `model.joblib` | ~460 МБ | ассет в [Releases](../../releases) + S3/MinIO |
| Лучший submission `ensemble_clip.csv` | 330 КБ | `submissions/` — в репозитории |
| Эмбеддинги, остальные submissions | ГБ | S3/MinIO, `python scripts/upload_artifacts_s3.py down` |
| EDA-графики, отчёт о дрифте | КБ | `docs/generated/` — в репозитории |

Головы лёгкие, потому что энкодеры в них не входят: E5 и CLIP скачиваются с
Hugging Face. А вот TF-IDF-модель self-contained — в ней весь словарь, поэтому
460 МБ, и в git она не кладётся: GitHub блокирует файлы больше 100 МиБ, а
бинарник в истории сделал бы любой `clone` неподъёмным.

Сервис достаёт `model.joblib` сам при старте (`service/entrypoint.sh`): сначала
пробует S3 по кредам `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, при их
отсутствии — прямую ссылку из `MODEL_PUBLIC_URL` (ассет релиза). Переменные
задаются в `.env` (шаблон — [`.env.example`](.env.example)); если они уже есть
в окружении, `.env` их **не** перетирает.

Обучить всё с нуля: `bash scripts/train_full_pipeline.sh` (нужен GPU), затем
`python scripts/upload_artifacts_s3.py up`.

---

## 11. Источники

**Данные**

- Датасет соревнования «Production ML Spring 2026» (Kaggle): `train.parquet.snappy`,
  `test.parquet.snappy`, `tree.csv`. Схема полей, заполненность и формат
  submission описаны в [`data/README.md`](data/README.md).
- Изображения товаров скачиваются по `image_url` из самого датасета
  (`src/ecom_category_project/data/download_images.py`), в репозитории не хранятся.

**Предобученные модели**

- [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
  — текстовые эмбеддинги. Wang et al., *Multilingual E5 Text Embeddings: A
  Technical Report*, [arXiv:2402.05672](https://arxiv.org/abs/2402.05672)
- [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
  — эмбеддинги изображений. Radford et al., *Learning Transferable Visual Models
  From Natural Language Supervision*, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

Условия использования — в карточках моделей на Hugging Face.

**Библиотеки**

[scikit-learn](https://scikit-learn.org/stable/) ·
[sentence-transformers](https://www.sbert.net/) ·
[transformers](https://huggingface.co/docs/transformers) ·
[LightGBM](https://lightgbm.readthedocs.io/) ·
[pandas](https://pandas.pydata.org/docs/) ·
[Flask](https://flask.palletsprojects.com/) ·
[gunicorn](https://docs.gunicorn.org/) ·
[uv](https://docs.astral.sh/uv/)

**Инфраструктура**

[MLflow](https://mlflow.org/docs/latest/index.html) ·
[Triton Inference Server](https://github.com/triton-inference-server/server) ·
[ONNX Runtime](https://onnxruntime.ai/docs/) ·
[Redis](https://redis.io/docs/latest/) ·
[Prometheus](https://prometheus.io/docs/) ·
[Grafana](https://grafana.com/docs/grafana/latest/) ·
[Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/) ·
[Evidently](https://docs.evidentlyai.com/) ·
[Apache Airflow](https://airflow.apache.org/docs/) ·
[MinIO](https://min.io/docs/minio/linux/index.html) ·
[Label Studio](https://labelstud.io/guide/) ·
[Docker Compose](https://docs.docker.com/compose/)

---

## 12. Лицензия

Код — [MIT](LICENSE). Данные соревнования и веса предобученных моделей
распространяются на условиях их правообладателей и в репозиторий не включены.
