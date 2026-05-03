# sbb-nlp-customer-segmentation

End-to-end NLP pipeline for fuzzy clustering and segmentation of SBB customer reviews.

## Trustpilot scraping setup

This project has two main steps:

1. Scrape Trustpilot reviews for SBB into an Excel workbook.
2. Run the segmentation script on the review text to create AI-based ratings.

### 1) Open in Dev Container (VS Code)

This repository now includes a Python dev container at:

- `.devcontainer/devcontainer.json`

In VS Code, run **Dev Containers: Reopen in Container**.
It will install all dependencies from `pyproject.toml` (including optional tooling) automatically.

### 2) Install dependencies locally (without dev container)

```bash
pip install -e .


```

### 3) Scrape Trustpilot reviews (Selenium)

```bash
python src/scraper/01_trustpilot_scraper.py
```

This writes an Excel file to:

- `data/trustpilot_reviews.xlsx`

That workbook is the input for the segmentation step.

### 4) Notes about generated output

Generated files are intentionally ignored from git:

- `data/*.xlsx`
- `src/scraper/.wdm/`

### 5) Run fuzzy segmentation with OpenAI

Create a local `.env` file in the project root with your OpenAI key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The `.env` file is already ignored by git, so it stays local.

Run the segmentation script from the project root:

```bash
source .env
python src/data_segmentation/03_segmentation.py
```

The script reads the first 167 non-empty reviews from `data/trustpilot_reviews.xlsx`, sends only the review text to the model, and writes the AI-ranked output to `data/trustpilot_reviews_ai_ranked.xlsx`.

In other words, the scraper collects the Trustpilot data first, then the segmentation script uses the saved Excel file as input and produces the ranked result file.

You can also install only formatting/linting tools with:

```bash
pip install -e .[dev]
```

### 6) Run tests

```bash
python -m unittest -v tests/test_01_trustpilot_scraper.py
```
