# Чекпойнты

Здесь лежат обученные классификаторы. Разделены по весу: лёгкие головы — прямо
в git, тяжёлая TF-IDF-модель — в GitHub Releases.

## В репозитории

| Файл | Что это | Вход | macro-F1 (Kaggle) |
|------|---------|------|------------------:|
| `e5_logreg.joblib` | LogisticRegression на текстовых эмбеддингах | E5, 1024-d | — (в ансамбле) |
| `e5_clip_mlp.joblib` | MLP на late fusion текста и картинки | E5 + CLIP, 1536-d | 0.699 соло |

Эти файлы весят единицы мегабайт, потому что сами энкодеры в них не входят:
[E5](https://huggingface.co/intfloat/multilingual-e5-large) и
[CLIP](https://huggingface.co/openai/clip-vit-base-patch32) скачиваются с
Hugging Face при первом запуске.

```python
import joblib
from ecom_category_project.features.embeddings import compute_text_embeddings, compute_image_embeddings, fuse

head = joblib.load("checkpoints/e5_clip_mlp.joblib")
X = fuse(compute_text_embeddings(texts), compute_image_embeddings(image_paths))
pred = head.predict(X)
```

## Не в репозитории

| Файл | Что это | Размер | Где взять |
|------|---------|-------:|-----------|
| `model.joblib` | TF-IDF (word + char_wb) + SGD, self-contained | ~460 МБ | ассет [GitHub Releases](../../releases) или S3/MinIO |

Это модель, которую использует сервис: ей не нужны ни GPU, ни энкодеры, она
считает предсказание из голого текста. В git не кладётся — GitHub блокирует
файлы больше 100 МиБ.

Сервис скачивает её сам при старте (`service/entrypoint.sh`): сначала пробует
S3 по кредам `AWS_*`, при их отсутствии — прямую ссылку из `MODEL_PUBLIC_URL`.

```bash
# вручную, если нужно локально
mkdir -p artifacts/serving
curl -fsSL "$MODEL_PUBLIC_URL" -o artifacts/serving/model.joblib
```

## Лучший результат

macro-F1 **0.723** даёт не одна модель, а смесь вероятностей трёх:

```
TF-IDF : E5-LogReg : E5+CLIP-MLP  =  1 : 2 : 1.5
```

Веса подобраны по отложенной выборке. Сборка — `models/blend_submission.py`,
конфиг — `configs/ensemble.toml`. Чтобы воспроизвести, нужны все три чекпойнта
и посчитанные эмбеддинги (`scripts/train_full_pipeline.sh`, нужен GPU).
