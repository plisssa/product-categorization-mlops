# Данные

Файлы соревнования в git **не хранятся** (см. `.gitignore`) — они скачиваются
с Kaggle. В репозитории лежит только дерево категорий `raw/tree.csv`.

## Что нужно положить сюда

```
data/raw/train.parquet.snappy     # с Kaggle, содержит метку category_ind
data/raw/test.parquet.snappy      # с Kaggle, без меток — их и предсказываем
data/raw/tree.csv                 # дерево категорий, уже в репозитории
```

## Схема файлов

### `train.parquet.snappy` / `test.parquet.snappy`

Одна строка — один товар. Текстовые поля часто пустые: это главная особенность
датасета, из неё следует ставка на картинки (см. раздел «Заполненность»).

| Поле | Тип | Что это |
|------|-----|---------|
| `ID` | int | идентификатор товара (в `test`, нужен для submission) |
| `name` | str | название товара |
| `description` | str | текстовое описание |
| `vendor` | str | производитель / бренд |
| `model` | str | модель |
| `type_prefix` | str | тип товара («Кровать», «Смартфон», …) |
| `url` | str | ссылка на карточку товара в магазине |
| `image_url` | str | ссылка на изображение товара |
| `category_ind` | int | **целевая переменная**, только в `train` |

Заполненность полей в `train` (по `docs/generated/eda_summary.md`):
`url` и `image_url` — 100%, `vendor` — 94%, `name` — 56%, `description` — 35%,
`type_prefix` — 17%.

### `tree.csv`

Дерево категорий, 223 строки. Уровни разделены ` -> `:

```csv
category,category_ind
Бытовая техника -> Встраиваемая техника,0
Бытовая техника -> Климатическая техника,1
```

| Поле | Тип | Что это |
|------|-----|---------|
| `category` | str | полный путь по дереву, уровни через ` -> ` |
| `category_ind` | int | индекс категории, совпадает с меткой в `train` |

В `train` реально встречаются **207** из 223 категорий. Корень пути
(`root_category`) вытаскивается в `prepare_dataset.py` и используется для
иерархического маскирования вероятностей (`models/hierarchy.py`).

## Формат submission

```csv
ID,category_ind
0,42
1,173
```

37 136 строк — по одной на каждый товар из `test`. Готовый пример лежит в
`submissions/ensemble_clip.csv` — это лучший результат проекта (macro-F1 0.723).

Метрика соревнования — **macro F1-score**: все 207 классов вносят равный вклад
независимо от размера, поэтому редкие классы важны так же, как частые.

## Что генерируется автоматически

```
data/processed/train_features.csv.gz   # make prepare
data/processed/test_features.csv.gz    # + колонки domain, full_text, category, root_category
data/processed/tree_enriched.csv       # tree.csv + root_category
data/images/<sha1 от image_url>.jpg    # python -m ecom_category_project.data.download_images
```

Картинки (~110 тыс. файлов) качаются по `image_url` и в git тоже не попадают.
Эмбеддинги (`.npy`/`.npz`) и обученные модели лежат на S3/MinIO — см. раздел
«Модель и артефакты» в корневом `README.md`.
