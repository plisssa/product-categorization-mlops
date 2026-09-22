"""Семантические эмбеддинги: текст (E5) и картинки (CLIP).

Тяжёлый DL-стек (torch/transformers) импортируется ленивыми импортами внутри
функций, чтобы модуль можно было импортировать без GPU-зависимостей
(для тестов и линтинга). Эмбеддинги кэшируются на диск (.npy).

Запускать на GPU-машине:  uv sync --extra embeddings
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_IMAGE_MODEL = "openai/clip-vit-base-patch32"


def _device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def compute_text_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_TEXT_MODEL,
    batch_size: int = 128,
    device: str | None = None,
    cache_path: str | Path | None = None,
    e5_prefix: str = "query: ",
) -> np.ndarray:
    """E5-эмбеддинги текстов (L2-нормированные). Кэшируются в ``cache_path``."""
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=_device(device))
    # E5 требует префиксы query:/passage: — для классификации используем query.
    prepared = [f"{e5_prefix}{t}" for t in texts]
    emb = model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb = emb.astype(np.float32)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb


def compute_image_embeddings(
    image_paths: list[str | None],
    model_name: str = DEFAULT_IMAGE_MODEL,
    batch_size: int = 64,
    device: str | None = None,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """CLIP-эмбеддинги картинок. Для отсутствующих файлов -> нулевой вектор."""
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dev = _device(device)
    model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    dim = model.config.projection_dim
    out = np.zeros((len(image_paths), dim), dtype=np.float32)

    batch_imgs, batch_idx = [], []

    def flush() -> None:
        if not batch_imgs:
            return
        inputs = processor(images=batch_imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        out[batch_idx] = feats.cpu().numpy().astype(np.float32)
        batch_imgs.clear()
        batch_idx.clear()

    for i, p in enumerate(image_paths):
        if not p or not Path(p).exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
    return out


def fuse(text_emb: np.ndarray, image_emb: np.ndarray | None, image_weight: float = 1.0) -> np.ndarray:
    """Late fusion: конкатенация (опционально взвешенная) текст+картинка."""
    if image_emb is None:
        return text_emb
    return np.hstack([text_emb, image_weight * image_emb]).astype(np.float32)
