from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


@dataclass
class Review:
    title: str
    rating: int | None
    date: str
    content: str


def fetch_reviews_html(url: str, timeout: int = 20) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    response.raise_for_status()
    return response.text


def parse_reviews(html: str, max_reviews: int | None = None) -> list[Review]:
    soup = BeautifulSoup(html, "html.parser")
    cards: Iterable = soup.select('article[data-service-review-card-paper="true"], article')
    reviews: list[Review] = []

    for card in cards:
        title_el = card.select_one('[data-service-review-title-typography="true"], h2, h3')
        title = title_el.get_text(" ", strip=True) if title_el else ""

        body_el = card.select_one('[data-service-review-text-typography="true"], p')
        content = body_el.get_text(" ", strip=True) if body_el else ""

        date_el = card.select_one("time")
        date = date_el.get("datetime", "") if date_el else ""

        rating = None
        rating_el = card.select_one('[data-service-review-rating]')
        if rating_el:
            raw_rating = rating_el.get("data-service-review-rating", "")
            if raw_rating.isdigit():
                rating = int(raw_rating)

        if not title and not content:
            continue

        reviews.append(Review(title=title, rating=rating, date=date, content=content))

        if max_reviews and len(reviews) >= max_reviews:
            break

    return reviews


def _selenium_fallback(url: str, wait_seconds: int = 6) -> str:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Selenium dependencies not installed. Run: pip install -r requirements-selenium.txt"
        ) from exc

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-agent={USER_AGENT}")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(url)
        driver.implicitly_wait(wait_seconds)
        return driver.page_source
    finally:
        driver.quit()


def run(url: str, output: str | None, max_reviews: int | None, use_selenium: bool) -> list[dict]:
    html = _selenium_fallback(url) if use_selenium else fetch_reviews_html(url)
    reviews = [asdict(item) for item in parse_reviews(html, max_reviews=max_reviews)]

    if output:
        with open(output, "w", encoding="utf-8") as file:
            json.dump(reviews, file, ensure_ascii=False, indent=2)

    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Trustpilot reviews from a URL.")
    parser.add_argument("url", help="Trustpilot URL, e.g. https://www.trustpilot.com/review/example.com")
    parser.add_argument("--output", help="Optional output JSON file path", default=None)
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument(
        "--use-selenium",
        action="store_true",
        help="Use Selenium fallback for pages where static requests parsing is insufficient",
    )
    args = parser.parse_args()

    if args.max_reviews is not None and args.max_reviews <= 0:
        raise ValueError("--max-reviews must be a positive integer")

    try:
        reviews = run(args.url, args.output, args.max_reviews, args.use_selenium)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reviews_scraped": len(reviews),
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
