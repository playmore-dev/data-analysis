# Rossmann Promotion Hypothesis Analysis

A portfolio project demonstrating the **full hypothesis-testing workflow** on a real
retail dataset: data preparation → EDA → hypothesis formulation → statistical testing →
interpretation and reporting, implemented as a reproducible **marimo** notebook.

## Business problem

Rossmann (German drugstore chain, 1,115 stores) runs frequent promotions. Before deciding
how aggressively to use them, analysts asked:

> **Do Rossmann stores have significantly different average daily sales on promotional
> days compared with non-promotional days?**

and, if so: how large is the difference, how uncertain is the estimate, is it practically
meaningful, and what might otherwise explain it.

## Dataset

- **Rossmann Store Sales** (Kaggle competition): https://www.kaggle.com/competitions/rossmann-store-sales/data
- Files used (kept locally under `rossmann-store-sales/`):
  - `train.csv` — 1,017,209 daily store records (2013-01-01 → 2015-07-31, 1,115 stores)
  - `store.csv` — store-level metadata (type, assortment, competition, Promo2)
- `test.csv` / `sample_submission.csv` present but ignored (not needed for inference).

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the notebook interactively:

```bash
marimo edit promotion_analysis.py
```

or execute it headlessly (regenerates all figures and the results CSV):

```bash
marimo export html promotion_analysis.py --output output/notebook.html
```

Pre-registered hypotheses (two-sided, α = 0.05): **H₀: μP − μN = 0** vs **H₁: μP − μN ≠ 0**
for mean log daily sales on promo vs non-promo store-days.

## Methodology

1. **Preparation** — normalize `StateHoliday` coding, verify the store merge, drop
   closed days (172,817) and open zero-sales days (54) with a documented audit trail;
   derive Year/Month/Weekday features; add `LogSales` to handle strong right skew (1.60).
2. **EDA** — descriptive statistics, distribution and outlier review, promo vs non-promo
   comparisons, and confounder checks (day-of-week, month, holidays, store type).
3. **Testing**
   - **A. Welch's t-test** on log sales (unadjusted baseline; unequal variances allowed).
   - **B. Store-level paired t-test** — within-store mean log sales promo vs non-promo
     (1,115/1,115 stores have both conditions), plus Wilcoxon signed-rank robustness.
   Effects are back-transformed: uplift % = 100 × (exp(mean log diff) − 1).
4. **Interpretation** — significance vs magnitude, statistical vs practical significance,
   observational-data caveats (association ≠ causation).

## Main findings (from the executed analysis)

- Both tests **reject H₀** decisively (p-values far below 0.05).
- Promotional days average **€8,228.74** vs **€5,929.83** on non-promotional days —
  a raw difference of **+38.8%**.
- On the log scale the difference is **0.346**, i.e. ≈ **+41.4% uplift**
  (95% CI **40.3%–42.5%** from the store-paired estimate; unadjusted CI is
  (41.1%–41.6%), confirming the pairing doesn't materially change the answer).
- Effect sizes: Hedges' g = 0.890 (unadjusted); Cohen's dz = 2.66 (store-paired).
- The gap is stable across all 2.5 years, every day of the week where promo runs,
  and every store type.
- **Caveat:** promotions run Monday–Friday only and are planned by managers; the
  association is **not** causal evidence of incremental sales. Promo cost data would be
  needed to judge profitability.

See `report.md` for the full narrative report.

## Project structure

```
.
├── promotion_analysis.py        # marimo notebook (all analysis)
├── README.md                    # this file
├── report.md                    # standalone results report
├── requirements.txt             # pinned dependencies
├── instruction.txt              # project brief (supplied)
├── rossmann-store-sales/        # Kaggle dataset (not redistributed)
│   ├── train.csv
│   ├── store.csv
│   ├── test.csv                 # ignored
│   └── sample_submission.csv    # ignored
├── figs/                        # generated figures (01…05)
└── output/
    ├── notebook.html            # executed notebook snapshot
    └── hypothesis_test_results.csv  # machine-readable results
```
