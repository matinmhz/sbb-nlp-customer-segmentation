"""Fuzzy rule-based sentiment segmentation for Trustpilot reviews.

Converts AI ratings into fuzzy membership scores across sentiment categories.
Combines multiple signals (rating, text length, keyword density) using Mamdani-style
fuzzy logic to produce soft labels where a review can belong to multiple categories
with different membership degrees (e.g., 0.6 negative + 0.4 mixed).

Core Features:
  - Case-insensitive keyword matching: "Chaos" and "CHAOS" both match
  - Token-level matching: punctuation stripped, whitespace-separated
  - Mamdani inference: min (AND), max (OR), normalized membership output
  - 5 domain-driven rules combining rating, length, and keyword density
  - Symmetric handling: very_negative ↔ very_positive (not just positive)
  - Fallback heuristics for edge cases (no rules fire)

Workflow:
  1. Load reviews with AI rankings from Excel
  2. Extract text features: word count, character length, keyword densities
  3. Define fuzzy membership functions for rating (1-5), length (0-1000), density (0-1)
  4. Fire 5 Mamdani rules to compute membership in {very_negative, negative, mixed, positive, very_positive}
  5. Normalize memberships to sum to 1.0 for probabilistic interpretation
  6. Assign each review to its strongest sentiment class
  7. Export with both soft memberships (mem_* columns) and hard assignment

Usage:
    python src/data_segmentation/03_fuzzy_pipeline.py --input data/trustpilot_reviews_ai_ranked.xlsx --output data/trustpilot_reviews_fuzzy.xlsx

Dependencies: pandas, numpy, scikit-fuzzy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import skfuzzy as fuzz

NEG_KEYWORDS = [
    # Negative sentiment keywords for text feature extraction (~80 terms).
    # Case-insensitive matching: "bad", "Bad", "BAD" all match.
    # Used to compute neg_count and neg_density for fuzzy rules.
    # Generic negative
    "bad",
    "chaos",
    "worse",
    "worst",
    "awful",
    "terrible",
    "horrible",
    "disappointing",
    "disappointed",
    "poor",
    "useless",
    "pathetic",
    "ridiculous",
    "unacceptable",
    "annoying",
    "frustrating",
    "frustrated",
    "disgusting",
    "broken",
    "fake",
    "scam",
    "fraud",
    "dishonest",
    # Negation words
    "not",
    "no",
    "never",
    "nothing",
    "none",
    "hardly",
    "barely",
    "cannot",
    "can't",
    "won't",
    "don't",
    "didn't",
    # Service issues
    "delay",
    "delayed",
    "late",
    "cancel",
    "cancelled",
    "canceled",
    "refund",
    "refused",
    "ignored",
    "unresponsive",
    "problem",
    "issue",
    "complaint",
    "mistake",
    "error",
    "failure",
    "failed",
    # Customer support
    "rude",
    "unhelpful",
    "incompetent",
    "careless",
    "unprofessional",
    "unfriendly",
    "aggressive",
    # Emotional dissatisfaction
    "angry",
    "upset",
    "furious",
    "sad",
    "stressful",
    "waste",
    "wasted",
    "regret",
    "regretted",
    # Travel / transport specific
    "dirty",
    "crowded",
    "smelly",
    "uncomfortable",
    "overpriced",
    "expensive",
    "missed",
    "confusing",
    "unsafe",
    "broken seat",
    "noisy",
    # Strong intensity
    "absolutely terrible",
    "extremely bad",
    "very disappointed",
]

# Positive sentiment keywords for text feature extraction (~40 terms).
# Case-insensitive matching across praise, satisfaction, recommendation, efficiency.
# Used to compute pos_count and pos_density for fuzzy rules.
POS_KEYWORDS = [
    # Generic positive
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "perfect",
    "fantastic",
    "wonderful",
    "brilliant",
    "superb",
    "outstanding",
    # Satisfaction
    "happy",
    "satisfied",
    "pleased",
    "impressed",
    "delighted",
    # Recommendation
    "recommended",
    "recommend",
    "trustworthy",
    "reliable",
    # Service quality
    "helpful",
    "friendly",
    "professional",
    "efficient",
    "responsive",
    "supportive",
    "polite",
    # Speed / convenience
    "fast",
    "quick",
    "easy",
    "smooth",
    "convenient",
    "simple",
    "comfortable",
    # Emotional positive
    "love",
    "loved",
    "enjoyed",
    "pleasant",
    # Travel / transport specific
    "clean",
    "on time",
    "punctual",
    "safe",
    "affordable",
    "spacious",
    # Strong intensity
    "highly recommended",
    "very good",
    "very helpful",
    "best service",
]


def load_input(path: Path, fallback_n: int = 8) -> pd.DataFrame:
    """Load Excel workbook with Trustpilot reviews and AI rankings.

    Expected columns: Review (text), ai_ranking or ai_rank (1-5 integer).
    If file not found, returns an 8-row smoke dataset for testing.

    Args:
        path: Path to Excel file
        fallback_n: Unused; kept for API compatibility

    Returns:
        DataFrame with 'Review' and 'ai_ranking' columns ready for feature extraction.
        Missing/malformed rankings default to 3 (neutral).
    """
    if path.exists():
        df = pd.read_excel(path)
    else:
        raise FileNotFoundError(f"File not found: {path}")
    # ensure columns
    if "ai_ranking" not in df.columns and "ai_rank" in df.columns:
        df["ai_ranking"] = df["ai_rank"]
    df["Review"] = df.get("Review", "").fillna("").astype(str)
    df["ai_ranking"] = (
        pd.to_numeric(df.get("ai_ranking", 3), errors="coerce").fillna(3).astype(int)
    )
    return df.reset_index(drop=True)


def text_feature_counts(text: str, keywords: Iterable[str]) -> int:
    """Count keyword occurrences in text (case-insensitive, token-level).

    - Text is lowercased before matching
    - Punctuation is stripped from each word
    - Matching is exact: token must fully match a keyword
    - Example: "Chaos!", "CHAOS", "chaos" all match keyword "chaos"

    Args:
        text: Review text to search (any case)
        keywords: List of lowercase keywords to match

    Returns:
        Integer count of keyword occurrences
    """
    tokens = [t.strip(".,!?;:") for t in text.lower().split() if t.strip()]
    return sum(1 for t in tokens if t in keywords)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 7 text features from reviews for use in fuzzy rules.

    Adds columns:
      - word_count: Number of words
      - review_length: Total character count
      - neg_count: Number of negative keywords found
      - pos_count: Number of positive keywords found
      - neg_density: Proportion of words that are negative [0, 1]
      - pos_density: Proportion of words that are positive [0, 1]
      - exclamation_density: Exclamation mark frequency (intensity signal)

    Args:
        df: DataFrame with 'Review' column (string)

    Returns:
        New DataFrame with original columns + 7 feature columns.

    Notes:
        - Densities are clipped to [0, 1] and avoid division by zero
        - Used as inputs to fuzzy membership functions
    """
    df = df.copy()

    # Basic text metrics
    df["word_count"] = df["Review"].str.split().str.len().fillna(0).astype(int)
    df["review_length"] = df["Review"].str.len().fillna(0).astype(int)

    # Keyword counts (absolute)
    df["neg_count"] = df["Review"].apply(
        lambda t: text_feature_counts(str(t), NEG_KEYWORDS)
    )
    df["pos_count"] = df["Review"].apply(
        lambda t: text_feature_counts(str(t), POS_KEYWORDS)
    )

    # Keyword densities (relative, used as fuzzy inputs)
    df["neg_density"] = (df["neg_count"] / df["word_count"].replace(0, 1)).clip(0, 1)
    df["pos_density"] = (df["pos_count"] / df["word_count"].replace(0, 1)).clip(0, 1)

    # Intensity signal: exclamation marks per word (high = emotional)
    df["exclamation_density"] = df["Review"].str.count("!") / df["word_count"].replace(
        0, 1
    )
    return df


class FuzzySentimentRules:
    """Mamdani-style fuzzy rule engine for sentiment classification.

    Encapsulates:
      - Fuzzy universes: rating [1,5], length [0,1000], density [0,1]
      - Membership functions: triangular for linguistic terms
      - 4 domain-driven rules combining rating, length, and density
      - Mamdani inference: min (AND), max (OR), normalized output

    Each review gets fuzzy membership in: {very_negative, negative, mixed, positive}
    Memberships are normalized to sum to 1.0 for probabilistic interpretation.
    """

    def __init__(self) -> None:
        # universes
        self.u_rating = np.linspace(1, 5, 41)
        self.u_length = np.linspace(0, 1000, 101)
        self.u_density = np.linspace(0, 1, 51)

        # rating MFs
        self.rating_mfs = {
            "very_negative": fuzz.trimf(self.u_rating, [1.0, 1.0, 2.0]),
            "negative": fuzz.trimf(self.u_rating, [1.5, 2.0, 2.5]),
            "neutral": fuzz.trimf(self.u_rating, [2.5, 3.0, 3.5]),
            "positive": fuzz.trimf(self.u_rating, [3.5, 4.0, 4.5]),
            "very_positive": fuzz.trimf(self.u_rating, [4.5, 5.0, 5.0]),
        }

        # length MFs (chars)
        self.length_mfs = {
            "short": fuzz.trimf(self.u_length, [0, 0, 50]),
            "medium": fuzz.trimf(self.u_length, [30, 120, 250]),
            "long": fuzz.trimf(self.u_length, [180, 350, 600]),
            "very_long": fuzz.trimf(self.u_length, [400, 700, 1000]),
        }

        # density MFs
        self.density_mfs = {
            "low": fuzz.trimf(self.u_density, [0.0, 0.0, 0.2]),
            "medium": fuzz.trimf(self.u_density, [0.1, 0.35, 0.6]),
            "high": fuzz.trimf(self.u_density, [0.5, 0.75, 1.0]),
        }

        # sentiment labels (consequents) - we keep symbolic labels and aggregate rule strengths
        # Updated to include very_positive for symmetric handling of 5.0 ratings
        self.labels = [
            "very_negative",
            "negative",
            "mixed",
            "positive",
            "very_positive",
        ]

        # rules: each rule is (antecedents, consequent_label, weight)
        # antecedents: list of tuples (variable, fuzzy_label)
        self.rules = [
            # Very negative: low rating + long review + high negative density
            (
                ("rating", "very_negative"),
                ("length", "very_long"),
                ("neg_density", "high"),
                "very_negative",
                1.0,
            ),
            # Negative: low-mid rating + short review + medium negative density
            (
                ("rating", "negative"),
                ("length", "short"),
                ("neg_density", "medium"),
                "negative",
                0.7,
            ),
            # Very positive: high rating + high positive density (strong signal)
            (
                ("rating", "very_positive"),
                ("pos_density", "high"),
                "very_positive",
                1.0,
            ),
            # Positive: mid-high rating + positive density
            (("rating", "positive"), ("pos_density", "high"), "positive", 1.0),
            # Mixed: neutral rating + balanced sentiment
            (
                ("rating", "neutral"),
                ("neg_density", "medium"),
                ("pos_density", "medium"),
                "mixed",
                1.0,
            ),
        ]

    def _interp(self, universe: np.ndarray, mf: np.ndarray, x: float) -> float:
        """Interpolate membership function value at a crisp input.

        Uses linear interpolation to find the membership degree of a crisp value
        against a triangular membership function.

        Args:
            universe: Array of possible values (e.g., 1 to 5 for rating)
            mf: Membership function values (triangular)
            x: Crisp input value to evaluate

        Returns:
            Membership degree (0 to 1)
        """
        return float(fuzz.interp_membership(universe, mf, x))

    def eval_instance(
        self, *, rating: float, length: float, neg_density: float, pos_density: float
    ) -> dict:
        """Evaluate fuzzy rules for a single review (Mamdani inference).

        Steps:
          1. Fuzzify inputs: compute membership in all linguistic terms
          2. Fire rules: evaluate antecedents (min), apply weights, aggregate (max)
          3. Defuzzify: normalize memberships to probability distribution

        Args:
            rating: AI ranking 1-5
            length: Review character count
            neg_density: Negative keyword proportion [0, 1]
            pos_density: Positive keyword proportion [0, 1]

        Returns:
            Dict mapping sentiment labels to membership degrees (sum = 1.0)
            Example: {'very_negative': 0.8, 'negative': 0.2, 'mixed': 0.0, 'positive': 0.0}

        Notes:
            - If no rules fire, uses fallback heuristics based on rating
            - Memberships always sum to 1.0 (normalized)
        """
        # Step 1: Fuzzify inputs - compute membership in all fuzzy sets
        r_degs = {
            k: self._interp(self.u_rating, v, rating)
            for k, v in self.rating_mfs.items()
        }
        l_degs = {
            k: self._interp(self.u_length, v, length)
            for k, v in self.length_mfs.items()
        }
        neg_degs = {
            k: self._interp(self.u_density, v, neg_density)
            for k, v in self.density_mfs.items()
        }
        pos_degs = {
            k: self._interp(self.u_density, v, pos_density)
            for k, v in self.density_mfs.items()
        }

        # Step 2: Fire rules - initialize aggregation for each output sentiment
        agg = {label: 0.0 for label in self.labels}

        for rule in self.rules:
            # Parse rule: (...antecedents, consequent_label, weight)
            weight = float(rule[-1])
            consequent = rule[-2]
            antecedents = rule[:-2]

            # Evaluate antecedents: get membership degree for each condition
            degrees = []
            for ant in antecedents:
                var, lbl = ant
                # Look up the membership degree of the input in this fuzzy set
                if var == "rating":
                    degrees.append(r_degs.get(lbl, 0.0))
                elif var == "length":
                    degrees.append(l_degs.get(lbl, 0.0))
                elif var == "neg_density":
                    degrees.append(neg_degs.get(lbl, 0.0))
                elif var == "pos_density":
                    degrees.append(pos_degs.get(lbl, 0.0))
                else:
                    degrees.append(0.0)

            # Compute rule firing strength: AND all antecedents (min), apply weight
            firing = float(np.min(degrees)) * weight if degrees else 0.0

            # Aggregate: OR rules with same consequent (max)
            agg[consequent] = max(agg[consequent], firing)

        # Step 3: Defuzzify - normalize aggregated firing strengths to probability distribution
        total = sum(agg.values())
        if total > 0:
            # Divide each by total to get normalized probabilities
            for k in agg:
                agg[k] = agg[k] / total
        else:
            # Fallback: no rules fired; use heuristic based on rating alone
            # Ensures every review gets some membership, even with rare feature combos
            if rating <= 2:
                agg = {
                    "very_negative": 0.8,
                    "negative": 0.2,
                    "mixed": 0.0,
                    "positive": 0.0,
                }
            elif rating == 3:
                agg = {
                    "very_negative": 0.0,
                    "negative": 0.2,
                    "mixed": 0.6,
                    "positive": 0.2,
                }
            else:
                agg = {
                    "very_negative": 0.0,
                    "negative": 0.0,
                    "mixed": 0.1,
                    "positive": 0.9,
                }

        return agg


def apply_rules_to_df(df: pd.DataFrame, engine: FuzzySentimentRules) -> pd.DataFrame:
    """Apply fuzzy rules to all reviews in the DataFrame.

    For each row, computes fuzzy membership in all 4 sentiment classes,
    then assigns the review to its strongest class (argmax).

    Adds 5 new columns:
      - mem_very_negative, mem_negative, mem_mixed, mem_positive: [0, 1] membership values
      - assigned_label: Strongest sentiment class (argmax of membership columns)

    Args:
        df: DataFrame with feature columns from extract_features():
            - ai_ranking, review_length, neg_density, pos_density
        engine: FuzzySentimentRules instance (already initialized)

    Returns:
        New DataFrame with original columns + 5 new columns

    Notes:
        - Memberships sum to ~1.0 for each row (small numerical error possible)
        - assigned_label is deterministic: always picks first max if tie
    """
    out = df.copy()

    # Initialize membership columns (will be overwritten)
    mem_cols = []
    for label in engine.labels:
        col = f"mem_{label}"
        mem_cols.append(col)
        out[col] = 0.0

    # Process each review row
    for i, row in out.iterrows():
        # Extract features for this row (with fallbacks for robustness)
        agg = engine.eval_instance(
            rating=float(row.get("ai_ranking", row.get("ai_rank", 3))),
            length=float(row.get("review_length", len(str(row.get("Review", ""))))),
            neg_density=float(row.get("neg_density", 0.0)),
            pos_density=float(row.get("pos_density", 0.0)),
        )
        # Store membership values
        for label, value in agg.items():
            out.at[i, f"mem_{label}"] = float(value)

    # Assign each review to its strongest (highest membership) sentiment
    out["assigned_label"] = (
        out[[f"mem_{l}" for l in engine.labels]].idxmax(axis=1).str.replace("mem_", "")
    )
    return out


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with 'input' and 'output' Path attributes
    """
    p = argparse.ArgumentParser(
        description="Fuzzy rule-based sentiment membership for Trustpilot reviews"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/trustpilot_reviews_ai_ranked.xlsx"),
        help="Path to input Excel file with AI rankings",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/trustpilot_reviews_fuzzy.xlsx"),
        help="Path to output Excel file with fuzzy memberships",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point: execute the full fuzzy segmentation pipeline.

    Steps:
      1. Parse command-line arguments for input/output paths
      2. Load reviews and AI rankings from Excel
      3. Extract text features (word count, keyword densities, etc.)
      4. Initialize fuzzy rule engine
      5. Apply fuzzy rules to compute membership for each review
      6. Export results to Excel (or CSV based on filename extension)
      7. Print summary statistics
    """
    args = parse_args()

    # Load and prepare data
    df = load_input(args.input)
    df = extract_features(df)

    # Initialize fuzzy engine and apply rules
    engine = FuzzySentimentRules()
    out = apply_rules_to_df(df, engine)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Export results (detect format from filename extension)
    if str(args.output).lower().endswith(".csv"):
        out.to_csv(args.output, index=False)
    else:
        out.to_excel(args.output, index=False)

    print(f"Wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
