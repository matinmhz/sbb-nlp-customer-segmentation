import unittest

from trustpilot_scraper import parse_reviews


SAMPLE_HTML = """
<html>
  <body>
    <article data-service-review-card-paper="true">
      <h2 data-service-review-title-typography="true">Great service</h2>
      <p data-service-review-text-typography="true">The trip was comfortable.</p>
      <time datetime="2026-01-10"></time>
      <div data-service-review-rating="5"></div>
    </article>
    <article data-service-review-card-paper="true">
      <h2 data-service-review-title-typography="true">Late train</h2>
      <p data-service-review-text-typography="true">The train was delayed.</p>
      <time datetime="2026-02-12"></time>
      <div data-service-review-rating="2"></div>
    </article>
  </body>
</html>
"""


class TestTrustpilotScraper(unittest.TestCase):
    def test_parse_reviews_extracts_expected_fields(self):
        reviews = parse_reviews(SAMPLE_HTML)

        self.assertEqual(len(reviews), 2)
        self.assertEqual(reviews[0].title, "Great service")
        self.assertEqual(reviews[0].content, "The trip was comfortable.")
        self.assertEqual(reviews[0].rating, 5)
        self.assertEqual(reviews[0].date, "2026-01-10")

    def test_parse_reviews_obeys_max_reviews(self):
        reviews = parse_reviews(SAMPLE_HTML, max_reviews=1)

        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].title, "Great service")


if __name__ == "__main__":
    unittest.main()
