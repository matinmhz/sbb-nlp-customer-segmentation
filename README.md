# sbb-nlp-customer-segmentation

End-to-end NLP pipeline for fuzzy clustering and segmentation of SBB customer reviews.

## Trustpilot scraping setup

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

### 4) Notes about generated output

Generated files are intentionally ignored from git:

- `data/*.xlsx`
- `src/scraper/.wdm/`

You can also install only formatting/linting tools with:

```bash
pip install -e .[dev]
```

### 5) Run tests

```bash
python -m unittest -v tests/test_01_trustpilot_scraper.py
```
