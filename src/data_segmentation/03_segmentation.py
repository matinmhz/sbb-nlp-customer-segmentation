"""LLM-assisted review ranking for Trustpilot samples.

This script loads the scraped review workbook, sends each review to an OpenAI
model, asks for a star rating, and writes a new Excel workbook with an
`ai_ranking` column alongside the original data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from textwrap import dedent

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "trustpilot_reviews.xlsx"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "trustpilot_reviews_ai_ranked.xlsx"
)
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


load_dotenv(DEFAULT_ENV_PATH, override=False)


def get_api_key() -> str:
    """Return the OpenAI API key from the environment.

    Returns:
        The first non-empty value found in OPENAI_API_KEY or OPENAI_TOKEN.

    Raises:
        RuntimeError: If neither environment variable is set.
    """

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TOKEN")
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY or OPENAI_TOKEN before running this script."
        )
    return api_key


def load_reviews(data_path: Path, limit: int = 5) -> pd.DataFrame:
    """Load the review workbook and return up to ``limit`` non-empty reviews.

    Args:
        data_path: Path to the Excel workbook that contains the reviews.
        limit: Maximum number of non-empty review rows to return. Defaults to 5.

    Returns:
        A DataFrame containing the first ``limit`` non-empty rows from the
        workbook.

    Raises:
        FileNotFoundError: If the workbook path does not exist.
        KeyError: If no case-insensitive ``Review`` column is found.
        ValueError: If the workbook contains no non-empty review text.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"Review workbook not found: {data_path}")

    frame = pd.read_excel(data_path)
    review_column = next(
        (column for column in frame.columns if str(column).strip().lower() == "review"),
        None,
    )
    if review_column is None:
        raise KeyError("Could not find a 'Review' column in the workbook.")

    filtered = frame.copy()
    filtered[review_column] = filtered[review_column].fillna("").astype(str).str.strip()
    filtered = filtered[filtered[review_column] != ""]

    if filtered.empty:
        raise ValueError("No non-empty review text found in the workbook.")

    return filtered.head(limit).reset_index(drop=True)


def build_prompt(review_text: str) -> str:
    """Build the model prompt for a single review text value.

    Args:
        review_text: The review text to send to the model.

    Returns:
        A prompt string that includes only the review text and rating rules.
    """

    return dedent(f"""
        You are a strict customer-review rater.

        Based only on this review text, assign one star rating from 1 to 5.
        Use this scale:
        - 1 = very negative / severe complaint
        - 2 = negative
        - 3 = mixed or neutral
        - 4 = positive
        - 5 = very positive / strong praise

        Return only valid JSON with these keys:
        - ai_ranking: integer from 1 to 5
        - sentiment: one of negative, mixed, positive
        - reasoning: a short explanation

        Review text:
        {review_text}
        """).strip()


def call_openai(prompt: str, model: str) -> str:
    """Send a prompt to OpenAI and return the JSON response text.

    Args:
        prompt: The prompt text built from the review text.
        model: The OpenAI model name to use.

    Returns:
        The raw JSON response content returned by the model.
    """

    client = OpenAI(api_key=get_api_key())
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You produce concise, structured analysis of customer reviews.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    message = response.choices[0].message.content
    return message.strip() if message else ""


def parse_ai_ranking(response_text: str) -> tuple[int, str, str]:
    """Parse the model JSON response into ranking, sentiment, and reasoning.

    Args:
        response_text: JSON string returned by the model.

    Returns:
        A tuple containing ``ai_ranking``, ``sentiment``, and ``reasoning``.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        KeyError: If ``ai_ranking`` is missing.
        ValueError: If ``ai_ranking`` is outside the 1 to 5 range.
    """

    data = json.loads(response_text)
    ranking = int(data["ai_ranking"])
    if ranking < 1 or ranking > 5:
        raise ValueError("ai_ranking must be between 1 and 5")

    sentiment = str(data.get("sentiment", "mixed")).strip().lower()
    reasoning = str(data.get("reasoning", "")).strip()
    return ranking, sentiment, reasoning


def rank_reviews(review_frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Rank each review with OpenAI and append the AI output columns.

    Args:
        review_frame: DataFrame with at least a ``Review`` column.
        model: The OpenAI model name to use.

    Returns:
        A copy of ``review_frame`` with ``ai_ranking``, ``ai_sentiment``, and
        ``ai_reasoning`` columns added.
    """

    ranked_frame = review_frame.copy()
    ai_rankings: list[int] = []
    sentiments: list[str] = []
    reasonings: list[str] = []

    for _, row in ranked_frame.iterrows():
        review_text = str(row.get("Review", "")).strip()
        prompt = build_prompt(review_text)
        response_text = call_openai(prompt, model)
        ranking, sentiment, reasoning = parse_ai_ranking(response_text)
        ai_rankings.append(ranking)
        sentiments.append(sentiment)
        reasonings.append(reasoning)

    ranked_frame["ai_ranking"] = ai_rankings
    ranked_frame["ai_sentiment"] = sentiments
    ranked_frame["ai_reasoning"] = reasonings
    return ranked_frame


def write_output(frame: pd.DataFrame, output_path: Path) -> None:
    """Write the ranked review DataFrame to an Excel file.

    Args:
        frame: DataFrame containing the original review rows and AI outputs.
        output_path: File path where the Excel workbook should be written.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the segmentation workflow.

    Returns:
        Parsed command-line arguments with defaults for the data path, model,
        limit, and output path.
    """

    parser = argparse.ArgumentParser(
        description="Send the first 167 Trustpilot reviews to OpenAI for fuzzy segmentation."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the scraped review workbook.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=167,
        help="Number of review rows to send to the model.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the new Excel file with AI rankings.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end review ranking workflow.

    Loads reviews, sends each review text to the model, parses the JSON
    response, and writes the ranked workbook to disk.
    """

    args = parse_args()
    review_frame = load_reviews(args.data_path, limit=args.limit)
    ranked_frame = rank_reviews(review_frame, args.model)
    write_output(ranked_frame, args.output_path)

    print(f"Wrote {len(ranked_frame)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
