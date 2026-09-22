from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import pandas as pd

TEXT_MARKERS = {
    "vendor": "vendor",
    "type_prefix": "type",
    "name": "name",
    "model": "model",
    "description": "desc",
    "url": "url",
    "image_url": "image",
    "domain": "domain",
    "url_tokens": "urltok",
}


@dataclass(slots=True)
class TextBuildConfig:
    text_fields: list[str]
    include_domain: bool = True
    include_url_tokens: bool = True
    description_max_len: int = 400
    url_token_limit: int = 12


def extract_domain(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower().replace("www.", "")


def extract_url_tokens(url: str, limit: int = 12) -> str:
    if not url:
        return ""

    parsed = urlparse(url)
    raw_path = f"{parsed.path} {parsed.query}".strip().lower()
    if not raw_path:
        return ""

    pieces = re.split(r"[/_\-?&=.:%+]+", raw_path)
    tokens: list[str] = []

    for piece in pieces:
        piece = re.sub(r"[^0-9a-zа-яё]+", "", piece, flags=re.IGNORECASE)
        if not piece:
            continue
        if piece.isdigit():
            continue
        if len(piece) < 2:
            continue
        tokens.append(piece)
        if len(tokens) >= limit:
            break

    return " ".join(tokens)


def _normalize_value(field: str, value: str, description_max_len: int) -> str:
    if value is None:
        return ""

    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return ""

    if field == "description":
        return text[:description_max_len]

    return text


def build_text_from_row(row: pd.Series, config: TextBuildConfig) -> str:
    parts: list[str] = []

    for field in config.text_fields:
        value = _normalize_value(field, row.get(field, ""), config.description_max_len)
        if value:
            marker = TEXT_MARKERS.get(field, field)
            parts.append(f"{marker}: {value}")

    if config.include_domain:
        domain = row.get("domain", "") or extract_domain(str(row.get("url", "")))
        if domain:
            parts.append(f"domain: {domain}")

    if config.include_url_tokens:
        url_tokens = row.get("url_tokens", "") or extract_url_tokens(
            str(row.get("url", "")),
            limit=config.url_token_limit,
        )
        if url_tokens:
            parts.append(f"urltok: {url_tokens}")

    return " ".join(parts).strip()


def build_text_column(df: pd.DataFrame, config: TextBuildConfig) -> pd.Series:
    work_df = df.copy()

    if config.include_domain and "domain" not in work_df.columns:
        work_df["domain"] = work_df["url"].fillna("").astype(str).map(extract_domain)

    if config.include_url_tokens and "url_tokens" not in work_df.columns:
        work_df["url_tokens"] = (
            work_df["url"]
            .fillna("")
            .astype(str)
            .map(lambda x: extract_url_tokens(x, limit=config.url_token_limit))
        )

    for field in config.text_fields:
        if field not in work_df.columns:
            work_df[field] = ""
        work_df[field] = work_df[field].fillna("").astype(str)

    return work_df.apply(lambda row: build_text_from_row(row, config), axis=1)
