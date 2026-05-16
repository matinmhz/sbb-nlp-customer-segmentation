"""Preprocessing: remove exact duplicate reviews (Author + Review).

Behaviour:
    - Loads the default file `data/trustpilot_reviews.xlsx` and operates in-place.
    - Normalizes `Author` and `Review` (strip and fill missing), drops reviews
        shorter than 20 characters, then removes exact duplicates on
        (`Author`, `Review`) keeping the first occurrence.
    - Creates a timestamped backup of the original file (written to the same
        directory, e.g. `trustpilot_reviews.backup.20260516_153012.xlsx`) before
        overwriting the original. If the atomic replace fails, a backup copy is
        written and the cleaned file is saved to the original path.

Notes:
    - The script intentionally runs with defaults (no CLI required). If you
        prefer a configurable CLI, I can restore `argparse` and `parse_args()`.
    - Short-review threshold (20 chars) and duplicate logic can be adjusted
        if you want different heuristics.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


def load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_excel(path)


def save_excel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def deduplicate_author_review(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Normalize text, drop short/empty reviews, and remove exact duplicates.

    Returns (cleaned_df, n_removed_duplicates, n_removed_short).
    Keeps the first occurrence of each (Author, Review) pair after removing short reviews.
    """
    work_df = df.copy()
    # Normalize author and review text
    work_df["Author"] = work_df.get("Author", "").fillna("").astype(str).str.strip()
    work_df["Review"] = work_df.get("Review", "").fillna("").astype(str).str.strip()

    # Drop empty or very short reviews (< 20 characters)
    review_lengths = work_df["Review"].str.len().fillna(0)
    short_mask = review_lengths < 20
    n_removed_short = int(short_mask.sum())
    if n_removed_short > 0:
        work_df = work_df.loc[~short_mask].copy().reset_index(drop=True)

    # Mark duplicates (keep first). If both fields empty this will also consider them duplicates;
    # that's intentional: exact duplicate rows will be removed.
    dup_mask = work_df.duplicated(subset=["Author", "Review"], keep="first")
    n_removed_dup = int(dup_mask.sum())
    if n_removed_dup > 0:
        cleaned = work_df.loc[~dup_mask].copy().reset_index(drop=True)
    else:
        cleaned = work_df
    return cleaned, n_removed_dup, n_removed_short


def main() -> None:
    # Default run: operate on data/trustpilot_reviews.xlsx without CLI args.
    inp = Path("data/trustpilot_reviews.xlsx")
    out = inp

    try:
        df = load_excel(inp)
    except FileNotFoundError:
        raise

    print(f"Loaded {len(df)} rows from {inp}")

    cleaned, n_removed_dup, n_removed_short = deduplicate_author_review(df)
    print(f"Removed short/empty reviews (<20 chars): {n_removed_short}")
    print(f"Exact duplicates removed (Author+Review): {n_removed_dup}")
    print(f"Rows after cleaning: {len(cleaned)}")

    # Create a timestamped backup and overwrite the original
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = inp.with_name(inp.stem + f".backup.{ts}" + inp.suffix)
    try:
        # attempt atomic replace: move original to backup, then write cleaned to original path
        inp.replace(backup_path)
        save_excel(cleaned, inp)
        print(f"Original file moved to backup: {backup_path}")
        print(f"Wrote cleaned file -> {inp}")
    except Exception:
        # Fallback: write backup copy and write cleaned output
        save_excel(df, backup_path)
        save_excel(cleaned, out)
        print(f"Saved backup copy: {backup_path}")
        print(f"Wrote cleaned file -> {out}")


if __name__ == "__main__":
    main()
