# sbb-nlp-customer-segmentation

End-to-end NLP pipeline for fuzzy clustering and segmentation of SBB customer reviews.

## Trustpilot scraping setup

### 1) Open in Dev Container (VS Code)

This repository now includes a Python dev container at:

- `.devcontainer/devcontainer.json`

In VS Code, run **Dev Containers: Reopen in Container**.
It will install dependencies from `requirements-dev.txt` automatically.

### 2) Install dependencies locally (without dev container)

```bash
pip install -r requirements-dev.txt
```

### 3) Scrape Trustpilot reviews (requests-based)

```bash
python trustpilot_scraper.py "https://www.trustpilot.com/review/example.com" --output reviews.json --max-reviews 20
```

### 4) Optional Selenium fallback

If Trustpilot content is highly dynamic and requests parsing returns too little data:

```bash
pip install -r requirements-selenium.txt
python trustpilot_scraper.py "https://www.trustpilot.com/review/example.com" --use-selenium --output reviews.json
```

### 5) Run tests

```bash
python -m unittest discover -s tests
```
