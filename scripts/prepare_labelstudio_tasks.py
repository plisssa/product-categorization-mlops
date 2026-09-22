from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VISIBLE_COLUMNS = ["name", "description", "vendor", "url", "image_url"]


def main(input_path: Path, output_path: Path, limit: int) -> None:
    df = pd.read_csv(input_path)
    tasks = []
    for idx, row in df.head(limit).iterrows():
        data = {column: row.get(column, "") for column in VISIBLE_COLUMNS}
        data["sample_id"] = int(idx)
        tasks.append({"data": data})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(tasks, fp, ensure_ascii=False, indent=2)
    print(f"Saved {len(tasks)} tasks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    main(args.input, args.output, args.limit)
