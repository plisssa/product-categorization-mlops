from __future__ import annotations

import re
from typing import Any

import requests
from bs4 import BeautifulSoup

PRICE_SELECTORS = [
    ".price",
    "[itemprop='price']",
    ".product-price",
    ".price-current",
    ".final-price",
]
TITLE_SELECTORS = ["#productTitle", "h1", "[itemprop='name']"]
VENDOR_SELECTORS = [".brand", "[itemprop='brand']", ".vendor"]
DESCRIPTION_SELECTORS = ["#description", ".description", "[itemprop='description']"]


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node.get_text(" ", strip=True)
    return ""


def _parse_price(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"\d+(?:[\s,]\d+)*(?:\.\d+)?", text)
    if not match:
        return None
    normalized = match.group(0).replace(" ", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_product(url: str, timeout: int = 15) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "lxml")

    title = _first_text(soup, TITLE_SELECTORS)
    vendor = _first_text(soup, VENDOR_SELECTORS)
    description = _first_text(soup, DESCRIPTION_SELECTORS)
    price_text = _first_text(soup, PRICE_SELECTORS)

    return {
        "url": url,
        "title": title,
        "vendor": vendor,
        "description": description,
        "price": _parse_price(price_text),
    }
