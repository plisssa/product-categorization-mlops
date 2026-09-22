"""ML-мониторинг: детекция Data Drift с помощью Evidently (паттерн hw8).

Сравнивает распределение признаков reference (обучающие данные) и current
(свежие данные/логи запросов). Сохраняет HTML-отчёт и JSON-сводку; печатает
долю задрифтовавших признаков — её можно завести как метрику/алерт.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_FIELDS = ["name", "description", "model", "type_prefix", "vendor"]


def build_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    text = df.get("full_text", df.get("name", pd.Series([""] * len(df)))).fillna("").astype(str)
    out["text_len"] = text.str.len()
    out["n_empty_fields"] = sum(
        (df.get(f, pd.Series([""] * len(df))).fillna("").astype(str).str.strip() == "").astype(int)
        for f in _FIELDS
    )
    out["has_image"] = df.get("image_url", pd.Series([""] * len(df))).fillna("").astype(str).str.startswith("http").astype(int)
    if "domain" in df.columns:
        out["domain"] = df["domain"].fillna("unknown").astype(str)
    if "root_category" in df.columns:
        out["root_category"] = df["root_category"].fillna("unknown").astype(str)
    return out


def run_drift(reference_path: Path, current_path: Path, out_dir: Path) -> dict:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    ref = build_drift_features(pd.read_csv(reference_path))
    cur = build_drift_features(pd.read_csv(current_path))
    common = [c for c in ref.columns if c in cur.columns]
    ref, cur = ref[common], cur[common]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "drift_report.html"
    report.save_html(str(html_path))

    result = report.as_dict()
    drift = result["metrics"][0]["result"]
    summary = {
        "dataset_drift": bool(drift.get("dataset_drift", False)),
        "share_drifted": float(drift.get("share_of_drifted_columns", 0.0)),
        "n_drifted": int(drift.get("number_of_drifted_columns", 0)),
    }
    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Drift summary: {summary}")
    print(f"HTML report: {html_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("docs/generated/drift"), type=Path)
    args = parser.parse_args()
    run_drift(args.reference, args.current, args.out_dir)
