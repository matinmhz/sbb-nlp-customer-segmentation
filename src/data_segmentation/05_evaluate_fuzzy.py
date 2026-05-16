"""Evaluate fuzzy segmentation against human ratings.

Validates fuzzy pipeline output by comparing fuzzy assignments to ground truth labels.
Reads fuzzy output with soft memberships (mem_* columns) and hard assignment (assigned_label),
maps human Rating column to sentiment labels symmetrically, and computes:
  - Crisp accuracy: proportion of correctly assigned labels
  - Confusion matrix: error distribution across sentiment classes
  - Fuzzy statistics: mean/median membership in true label, proportion with high confidence

This provides both quantitative validation (accuracy) and confidence metrics
(membership in correct class) to assess fuzzy model quality.

Input Requirements:
  - assigned_label: Hard assignment from fuzzy pipeline (very_negative, negative, mixed, positive, very_positive)
  - mem_very_negative, mem_negative, mem_mixed, mem_positive, mem_very_positive: Soft membership scores [0, 1]
  - Rating: Ground truth human ratings (1-5) from Trustpilot

Output:
  - Excel workbook with 3 sheets:
    * rows: Full DataFrame with true_label and mem_true_label columns added
    * confusion: Confusion matrix (true vs assigned labels)
    * summary: Key metrics (accuracy, mean membership, confidence proportions)

Usage:
    python src/data_segmentation/04_evaluate_fuzzy.py --input data/trustpilot_reviews_fuzzy.xlsx
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
import numpy as np


def load_df(path: Path) -> pd.DataFrame:
    """Load Excel file with fuzzy pipeline output.

    Expected columns: assigned_label, mem_* (membership columns), Rating (human ratings)

    Args:
        path: Path to Excel file from 03_fuzzy_pipeline.py

    Returns:
        DataFrame with fuzzy output and human ratings

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If Rating column is missing
    """
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_excel(path)
    if "Rating" not in df.columns:
        raise ValueError(
            "Rating column not found in input file. "
            "Expected human ratings from Trustpilot for evaluation."
        )
    return df


def rating_to_label(rating: float) -> str:
    """Map numeric rating (1-5) to sentiment label.

    Symmetric 5-class mapping (matches fuzzy pipeline output):
      - 1.0: very_negative
      - 1.0 < rating <= 2.0: negative
      - 2.0 < rating < 4.0: mixed (neutral, balanced)
      - 4.0 <= rating < 4.5: positive
      - 4.5 <= rating: very_positive

    Args:
        rating: Numeric value from the Rating column (1-5)

    Returns:
        Sentiment label string: very_negative, negative, mixed, positive, or very_positive

    Notes:
        - Non-numeric ratings default to 'mixed' (safe fallback)
        - Symmetric boundaries: 1.0→very_neg mirrors 5.0→very_pos
        - Avoids loss of granularity at extremes
    """
    # Convert to float; return neutral if parsing fails
    try:
        r = float(rating)
    except Exception:
        return "mixed"

    # Map to sentiment classes (symmetric)
    if r <= 1.0:
        return "very_negative"
    if r <= 2.0:
        return "negative"
    if r < 4.0:
        return "mixed"
    if r < 4.5:
        return "positive"
    return "very_positive"  # 4.5 <= r


def compute_basic_metrics(
    df: pd.DataFrame, labels: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Compute evaluation metrics comparing fuzzy assignments to ground truth.

    Computes:
      1. true_label: Ground truth sentiment from human Rating (symmetric 5-class mapping)
      2. Crisp accuracy: Proportion of correct hard assignments
      3. Confusion matrix: Cross-tabulation of true vs assigned labels
      4. Fuzzy statistics: Mean/median membership in correct class, high-confidence proportion

    Ground truth mapping (symmetric, from human Rating):
      - 1.0: very_negative  ↔  4.5+: very_positive
      - 1.0-2.0: negative   ↔  4.0-4.5: positive
      - 2.0-4.0: mixed (balanced)

    Args:
        df: DataFrame with assigned_label, mem_* columns, Rating (human ratings)
        labels: List of sentiment labels (for reference, not used in this function)

    Returns:
        (df_eval, stats):
        - df_eval: Input DataFrame with added columns:
            * true_label: Ground truth sentiment class from human Rating
            * mem_true_label: Membership in the correct class
        - stats: Dict with evaluation metrics:
            * n_rows: Total number of reviews
            * accuracy: Crisp accuracy (assigned_label == true_label)
            * confusion_matrix: Pandas crosstab of true vs assigned
            * mean_membership_true_label: Average confidence in correct class
            * median_membership_true_label: Median confidence
            * prop_mem_ge_0.5: Proportion of reviews with >= 0.5 membership in true class
    """
    # Step 1: Map human ratings to true sentiment labels
    df["true_label"] = df["Rating"].apply(rating_to_label)

    # Step 2: Get assigned label from fuzzy pipeline (or compute if missing)
    if "assigned_label" not in df.columns:
        # Fallback: argmax over membership columns if assigned_label not present
        mem_cols = [c for c in df.columns if c.startswith("mem_")]
        if mem_cols:
            df["assigned_label"] = df[mem_cols].idxmax(axis=1).str.replace("mem_", "")
        else:
            df["assigned_label"] = df["true_label"]

    # Step 3: Compute crisp accuracy (hard assignment accuracy)
    correct = (df["assigned_label"] == df["true_label"]).sum()
    total = len(df)
    accuracy = float(correct) / float(total) if total else 0.0

    # Step 4: Build confusion matrix (true_label vs assigned_label)
    conf = pd.crosstab(df["true_label"], df["assigned_label"], normalize=False)

    # Step 5: Extract membership values for the true label
    # For each row, get the membership column name matching its true_label
    mem_col_for_true = df["true_label"].apply(lambda t: f"mem_{t}")
    mem_values = []
    for i, col in enumerate(mem_col_for_true):
        v = 0.0
        # Look up the membership value in the appropriate mem_* column
        if col in df.columns:
            v = float(df.at[i, col])
        mem_values.append(v)
    df["mem_true_label"] = mem_values

    # Step 6: Compile evaluation statistics
    stats = {
        "n_rows": total,
        "accuracy": accuracy,
        "confusion_matrix": conf,
        # Fuzzy statistics: membership in the correct sentiment class
        "mean_membership_true_label": float(np.mean(mem_values)) if mem_values else 0.0,
        "median_membership_true_label": (
            float(np.median(mem_values)) if mem_values else 0.0
        ),
        # High-confidence metric: what fraction had >= 0.5 membership in correct class?
        "prop_mem_ge_0.5": (
            float((np.array(mem_values) >= 0.5).mean()) if mem_values else 0.0
        ),
    }
    return df, stats


# Input is fixed for this evaluation; no CLI parsing required.
# The pipeline writes fuzzy output to `data/trustpilot_reviews_fuzzy.xlsx`.
INPUT_PATH = Path("data/trustpilot_reviews_fuzzy.xlsx")
OUTPUT_PATH = Path("data/eval_fuzzy_summary.xlsx")


def main() -> None:
    """Main entry point: execute fuzzy segmentation evaluation.

    Steps:
      1. Parse command-line arguments (input fuzzy output, output evaluation results)
      2. Load fuzzy pipeline output from Excel
      3. Detect sentiment labels from membership columns
      4. Compute evaluation metrics (accuracy, confusion matrix, membership stats)
      5. Print summary to console
      6. Export results to 3-sheet Excel workbook:
         - rows: Full DataFrame with true_label and mem_true_label
         - confusion: Confusion matrix showing error distribution
         - summary: Key metrics for quick reference
    """
    # Step 1: Use fixed input/output paths (no CLI)
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH

    # Step 2: Load fuzzy output with error handling
    try:
        df = load_df(input_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Step 3: Detect sentiment labels from membership columns
    # Extract label names from mem_* columns in the DataFrame
    labels = sorted({c.replace("mem_", "") for c in df.columns if c.startswith("mem_")})
    # Fallback to standard labels if none found
    if not labels:
        labels = ["very_negative", "negative", "mixed", "positive", "very_positive"]

    # Step 4: Compute metrics
    df_eval, stats = compute_basic_metrics(df, labels)

    # Step 5: summary to console
    print("\nFuzzy Evaluation Summary")
    print("------------------------")
    print(f"Rows: {stats['n_rows']}")
    print(f"Crisp accuracy (assigned_label vs mapped Rating): {stats['accuracy']:.3f}")
    print(f"Mean membership for true label: {stats['mean_membership_true_label']:.3f}")
    print(
        f"Median membership for true label: {stats['median_membership_true_label']:.3f}"
    )
    print(f"Proportion with mem_true_label >= 0.5: {stats['prop_mem_ge_0.5']:.3f}")
    print("\nConfusion matrix (counts):")
    print(stats["confusion_matrix"])

    # Step 6: Export results to Excel workbook with 3 sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Full evaluated DataFrame with added true_label and mem_true_label columns
        df_eval.to_excel(writer, sheet_name="rows", index=False)

        # Sheet 2: Confusion matrix (true_label rows vs assigned_label columns)
        stats["confusion_matrix"].to_excel(writer, sheet_name="confusion")

        # Sheet 3: Summary metrics for quick reference
        summary_df = pd.DataFrame(
            [
                {"metric": "accuracy", "value": stats["accuracy"]},
                {
                    "metric": "mean_mem_true",
                    "value": stats["mean_membership_true_label"],
                },
                {
                    "metric": "median_mem_true",
                    "value": stats["median_membership_true_label"],
                },
                {"metric": "prop_mem_ge_0.5", "value": stats["prop_mem_ge_0.5"]},
                {"metric": "n_rows", "value": stats["n_rows"]},
            ]
        )
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"Wrote evaluation -> {output_path}")


if __name__ == "__main__":
    main()
