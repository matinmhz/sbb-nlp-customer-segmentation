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

## Project logic and files

High-level pipeline (recommended order):

1. Scrape Trustpilot reviews into Excel
	- Script: `src/scraper/01_trustpilot_scraper.py`
	- Output: `data/trustpilot_reviews.xlsx`

2. (Optional) Produce AI ranking column
	- If you have a separate AI ranking step, produce `ai_ranking` column in the same file
	- Expected values: integer 1..5 (missing values treated as 3)

3. Run fuzzy segmentation (feature extraction + Mamdani FIS)
	- Script: `src/data_segmentation/03_fuzzy_pipeline.py`
	- Input: `data/trustpilot_reviews_ai_ranked.xlsx` (or `data/trustpilot_reviews.xlsx` with `ai_ranking`)
	- Output: `data/trustpilot_reviews_fuzzy.xlsx`
	- Produces soft membership columns: `mem_very_negative, mem_negative, mem_mixed, mem_positive, mem_very_positive`
	- Produces hard assignment column: `assigned_label`

4. Evaluate fuzzy output against ground truth ratings
	- Script: `src/data_segmentation/04_evaluate_fuzzy.py`
	- Input: `data/trustpilot_reviews_fuzzy.xlsx`
	- Output: `data/eval_fuzzy_summary.xlsx` (sheets: `rows`, `confusion`, `summary`)
	- Ground truth mapping: symmetric 5-class mapping (very_negative ↔ very_positive)

5. Visualize results (notebook)
	- Notebook: `src/data_segmentation/05_visualize_evaluation.ipynb`
	- It reads `data/eval_fuzzy_summary.xlsx` (preferred) or the fuzzy output file

Notes on logic:
- The fuzzy pipeline extracts 7 features (word_count, review_length, neg_count, pos_count, neg_density, pos_density, exclamation_density) and applies a Mamdani-style rule set to compute memberships.
- The system uses symmetric sentiment labels including `very_positive`; ensure evaluation and visualization use the same label set.
- Keyword matching is token-level by default; multi-word phrase entries may be ignored unless preprocessed (see `src/data_segmentation/03_fuzzy_pipeline.py`).

## Quick run commands

Activate virtualenv (if used):
```bash
source .venv/bin/activate
```

Run full pipeline (scrape → fuzzy → eval):
```bash
python src/scraper/01_trustpilot_scraper.py
python src/data_segmentation/03_fuzzy_pipeline.py --input data/trustpilot_reviews_ai_ranked.xlsx --output data/trustpilot_reviews_fuzzy.xlsx
python src/data_segmentation/04_evaluate_fuzzy.py --input data/trustpilot_reviews_fuzzy.xlsx --output data/eval_fuzzy_summary.xlsx
```

Open the notebook in VS Code or Jupyter to regenerate figures:
```bash
# in VS Code: open the project and open the notebook file
# or run in Jupyter Lab: jupyter lab src/data_segmentation/05_visualize_evaluation.ipynb
```

## Tests

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
