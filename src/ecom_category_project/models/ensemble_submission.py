from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_VERSION_RE = re.compile(r"_v(\d+)$")


def resolve_versioned_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = output_path.stem
    suffix = output_path.suffix

    if _VERSION_RE.search(stem):
        return output_path

    max_version = 0
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+){re.escape(suffix)}$")

    for existing in output_path.parent.iterdir():
        if not existing.is_file():
            continue
        match = pattern.match(existing.name)
        if match:
            max_version = max(max_version, int(match.group(1)))

    return output_path.parent / f"{stem}_v{max_version + 1}{suffix}"


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"ID", "category_ind"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return df[["ID", "category_ind"]].copy()


def build_ensemble(
    submission_paths: list[Path],
    fallback_path: Path | None = None,
) -> pd.DataFrame:
    if not submission_paths:
        raise ValueError("At least one submission path must be provided.")

    submissions = [load_submission(path) for path in submission_paths]

    base_ids = submissions[0]["ID"]
    for i, df in enumerate(submissions[1:], start=2):
        if not base_ids.equals(df["ID"]):
            raise ValueError(
                f"Submission #{i} has different ID ordering. "
                "All submissions must have identical ID column."
            )

    vote_df = pd.DataFrame({"ID": base_ids})
    for idx, df in enumerate(submissions, start=1):
        vote_df[f"pred_{idx}"] = df["category_ind"].values

    fallback_map: dict[int, int] = {}
    if fallback_path is not None:
        fallback_df = load_submission(fallback_path)
        if not base_ids.equals(fallback_df["ID"]):
            raise ValueError("Fallback submission has different ID ordering.")
        fallback_map = dict(zip(fallback_df["ID"], fallback_df["category_ind"]))

    pred_columns = [col for col in vote_df.columns if col.startswith("pred_")]
    final_preds: list[int] = []

    for _, row in vote_df.iterrows():
        votes = row[pred_columns].value_counts()
        top_count = votes.iloc[0]
        top_labels = votes[votes == top_count].index.tolist()

        if len(top_labels) == 1:
            chosen = int(top_labels[0])
        else:
            current_id = int(row["ID"])
            if current_id in fallback_map:
                chosen = int(fallback_map[current_id])
            else:
                chosen = int(top_labels[0])

        final_preds.append(chosen)

    return pd.DataFrame(
        {
            "ID": vote_df["ID"].astype(int),
            "category_ind": final_preds,
        }
    )


def main(
    submissions: list[Path],
    output: Path,
    fallback: Path | None = None,
) -> None:
    ensemble_df = build_ensemble(submissions, fallback_path=fallback)
    resolved_output = resolve_versioned_output_path(output)
    ensemble_df.to_csv(resolved_output, index=False)
    print(f"Ensemble submission saved to: {resolved_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submissions",
        nargs="+",
        required=True,
        type=Path,
        help="List of submission CSV files with columns ID,category_ind",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output ensemble submission CSV",
    )
    parser.add_argument(
        "--fallback",
        type=Path,
        default=None,
        help="Optional fallback submission used to break ties",
    )
    args = parser.parse_args()

    main(
        submissions=args.submissions,
        output=args.output,
        fallback=args.fallback,
    )
