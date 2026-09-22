"""Превращение входного JSON /predict в признаки модели."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ecom_category_project.features.text import TextBuildConfig, build_text_column

_TEXT_CONFIG = TextBuildConfig(
    text_fields=["vendor", "name", "model", "type_prefix", "description"],
    include_domain=True,
    include_url_tokens=True,
)


def request_to_full_text(payload: dict[str, Any]) -> str:
    texts = payload.get("texts") or {}
    row = {
        "name": texts.get("name", ""),
        "description": texts.get("description", ""),
        "model": texts.get("model", ""),
        "type_prefix": texts.get("type_prefix", ""),
        "vendor": texts.get("vendor", ""),
        "url": payload.get("url", ""),
        "image_url": payload.get("image_url", ""),
    }
    df = pd.DataFrame([row])
    return build_text_column(df, _TEXT_CONFIG).iloc[0]
