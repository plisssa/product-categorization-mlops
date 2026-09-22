# Triton Inference Server (MaaS, ONNX)

Ансамбль `ecom_ensemble` принимает сырой текст и возвращает `category_ind`:

```
TEXT --preprocess(py: токенизация E5)--> INPUT_IDS, ATTENTION_MASK
     --text_encoder(ONNX E5, mean-pool+L2)--> EMBEDDING
     --category_head(ONNX sklearn-голова)--> PROBS
     --postprocess(py: argmax->classes.json)--> CATEGORY_IND
```

## Подготовка артефактов (ONNX)

```bash
uv sync --extra embeddings
python triton/prepare_models.py \
  --head artifacts/e5_logreg/models/embedding_head.joblib \
  --text-model intfloat/multilingual-e5-large --dim 1024
```

Это создаст:
- `model_repository/text_encoder/1/model.onnx`
- `model_repository/category_head/1/model.onnx`
- `model_repository/postprocess/1/classes.json`

## Запуск

```bash
./run_service.sh triton      # поднимет Triton + сервис с MODEL_BACKEND=triton
# проверка:
curl localhost:8000/v2/health/ready
```

> Python-бэкенды (`preprocess`) требуют `transformers` в окружении Triton.
> Это задаётся через `EXECUTION_ENV_PATH` (conda-pack) или кастомный образ —
> см. документацию TIS. По умолчанию сервис работает на надёжном CPU-бэкенде
> (`MODEL_BACKEND=local`, TF-IDF), а Triton включается профилем `triton`.
