import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from selenium.common.exceptions import NoSuchElementException


def _load_scraper_module():
    scraper_path = Path(__file__).resolve(
    ).parents[1] / "src" / "scraper" / "01_trustpilot_scraper.py"
    spec = importlib.util.spec_from_file_location(
        "trustpilot_scraper_module", scraper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRAPER = _load_scraper_module()


class TestResolveChromeDriverPath(unittest.TestCase):
    def test_prefers_explicit_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_driver = Path(tmp_dir) / "chromedriver"
            fake_driver.write_text("", encoding="utf-8")
            resolved = SCRAPER._resolve_chromedriver_path(str(fake_driver))
            self.assertEqual(resolved, str(fake_driver))

    def test_raises_when_no_candidates_exist(self):
        with patch.object(SCRAPER.shutil, "which", return_value=None), patch.dict(SCRAPER.os.environ, {}, clear=True), patch.object(SCRAPER.Path, "is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                SCRAPER._resolve_chromedriver_path(None)


class TestInputValidation(unittest.TestCase):
    def test_scrape_raises_for_invalid_limits(self):
        with self.assertRaises(ValueError):
            SCRAPER.scrape_trustpilot_reviews(
                "https://example.com", max_pages=0)

        with self.assertRaises(ValueError):
            SCRAPER.scrape_trustpilot_reviews(
                "https://example.com", max_reviews=0)


class TestTextExtractionHelpers(unittest.TestCase):
    def test_modal_text_prefers_longest_paragraph_and_cleans_markers(self):
        driver = MagicMock()
        modal = MagicMock()
        p1 = MagicMock()
        p2 = MagicMock()

        driver.find_element.return_value = modal
        modal.find_elements.return_value = [p1, p2]
        driver.execute_script.side_effect = [
            " short ",
            "This is the full review body See more",
        ]

        text = SCRAPER._extract_review_text_from_modal(driver)

        self.assertEqual(text, "This is the full review body")

    def test_article_extraction_returns_defaults_when_elements_missing(self):
        driver = MagicMock()
        article = MagicMock()

        article.find_element.side_effect = NoSuchElementException("missing")
        article.find_elements.side_effect = [[], [], []]

        result = SCRAPER._extract_review_from_article(driver, article)

        self.assertEqual(result["author"], "Unknown")
        self.assertEqual(result["title"], "")
        self.assertEqual(result["rating"], 0)
        self.assertEqual(result["date"], "")
        self.assertEqual(result["review"], "")


if __name__ == "__main__":
    unittest.main()
