"""Скачивание картинок товаров по ``image_url`` (источник веб — см. лекцию 5).

Особенности продакшен-сбора, которые тут учтены:
* битые/мертвые ссылки не валят процесс (считаем, пропускаем);
* дедупликация по sha1(url) — одинаковые URL качаем один раз;
* многопоточность с таймаутом;
* User-Agent, чтобы не отбивались простыми анти-бот фильтрами.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ecom-category-bot/1.0)"}


def url_to_filename(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg"


def download_one(url: str, out_dir: Path, timeout: float) -> tuple[str, bool]:
    if not url or not str(url).startswith("http"):
        return url, False
    target = out_dir / url_to_filename(url)
    if target.exists():
        return url, True
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return url, False
        content = resp.content
        if not content:
            return url, False
        target.write_bytes(content)
        return url, True
    except Exception:
        return url, False


def download_images(
    df: pd.DataFrame,
    url_column: str,
    out_dir: Path,
    max_workers: int = 16,
    timeout: float = 5.0,
    limit: int | None = None,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = df[url_column].dropna().astype(str)
    urls = urls[urls.str.startswith("http")].unique().tolist()
    if limit:
        urls = urls[:limit]

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(download_one, u, out_dir, timeout) for u in urls]
        for fut in as_completed(futures):
            _, success = fut.result()
            ok += int(success)
            fail += int(not success)
    stats = {"total": len(urls), "downloaded": ok, "failed": fail}
    print(f"Images: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="csv.gz with an image url column")
    parser.add_argument("--url-column", default="image_url")
    parser.add_argument("--out-dir", default=Path("data/images"), type=Path)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--timeout", default=5.0, type=float)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    download_images(frame, args.url_column, args.out_dir, args.workers, args.timeout, args.limit)
