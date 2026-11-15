# 🛒 Rossmann Store Sales — EDA & Sales Forecasting

Machine Learning coursework: exploratory data analysis and daily sales
forecasting for Rossmann drugstores, delivered as a [marimo](https://marimo.io)
notebook (`retail_sales_ml.py`) plus reusable Python modules.

## Deliverables

| File | Description |
|---|---|
| `retail_sales_ml.py` | **Marimo notebook** — EDA, feature engineering, model comparison, residual analysis, feature importance, submission export |
| `retail_sales_ml.html` | The notebook executed & exported to HTML (all outputs included) |
| `features.py` | Data loading, cleaning, leakage-safe feature engineering, temporal split |
| `models.py` | Model definitions (Ridge, RandomForest, HistGradientBoosting) + naive baseline |
| `evaluation.py` | Metrics (RMSPE/MAE/RMSE/MAPE), model comparison, residual plots, artifacts |
| `submission.csv` | Holdout predictions (`Store, Date, Sales, PredictedSales`) — 40,282 open store-days |
| `artifacts/model_metrics.json` | Final metrics for all models |
| `artifacts/plots/` | Residual diagnostics, importance plots |
| `rossmann-sales-data/` | Source data (`train.csv`, `store.csv`) |

## Problem & data

* `train.csv`: ~1.02 M daily store records (1,115 stores, 2013-01-01 → 2015-07-31)
  with `Sales`, `Customers`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday`.
* `store.csv`: static store attributes (`StoreType`, `Assortment`, competition
  distance/opening, `Promo2` intervals).
* **`test.csv` is not provided**, so the **final 6 weeks** of the timeline
  (2015-06-20 → 2015-07-31, 40,282 open store-days) are held out as an
  **out-of-time evaluation window**.

## Key design decisions

1. **Leakage-safe split.** Out-of-time holdout (train on the past, forecast the
   future) instead of a random row split.
2. **Modelling frame = open days only.** Closed days have deterministic
   `Sales == 0` (172,817 rows) and are predicted trivially. The 54 open-but-zero
   days are kept (legitimate trading days) but excluded from MAPE.
3. **`Customers` is excluded.** It correlates ~0.95 with sales but is only known
   after the fact — a textbook leakage feature.
4. **One-step-ahead daily forecast assumption.** All lag/rolling features
   (`sales_lag1/7/14/28`, `sales_rollmean7/28`, `sales_rollstd7`,
   `days_since_last_open`) are computed per store over its open-day history and
   shifted by one open day, so a row's own sales never enter its features.
5. **Log-space modelling.** Target is `log1p(Sales)` (raw sales are right-skewed);
   sales-scale lag features are also `log1p`-transformed so the linear model sees
   an approximately linear relationship. Predictions are inverted with `expm1`
   and clipped at zero.
6. **Training-split-only statistics.** Imputation (median) and one-hot categories
   are fitted on the training split; the holdout never influences preprocessing.

## Results (6-week out-of-time holdout, sales scale)

| Model | RMSPE ↓ | MAE (€) | RMSE (€) | MAPE % (zero-sales excl.) |
|---|---|---|---|---|
| Naive (last week, same weekday) | 0.4895 | 2,418 | 3,063 | 37.6 |
| Ridge (log target) | 0.1943 | 885 | 1,292 | 13.2 |
| **RandomForest (log target)** 🏆 | **0.1250** | **610** | **888** | **9.1** |
| HistGradientBoosting (log target) | 0.1282 | 625 | 901 | 9.3 |

* The RandomForest wins narrowly over the gradient boosting (~0.3 pp RMSPE);
  both cut the naive baseline's error by **~3.9×**.
* **What drives sales** (permutation importance on the holdout):
  `sales_rollmean28` (0.37 — each store's 28-day level), `Promo` (0.24),
  `sales_lag1` (0.14), `DayOfWeek` (0.11), `sales_rollmean7` (0.11), then
  rolling volatility, store type, `days_since_last_open`.
* Residual diagnostics (`artifacts/plots/`) show: predictions track actuals
  along the diagonal with mild under-prediction at extreme promo peaks
  (residuals centered near zero with a long left tail); the worst percentage
  errors occur on **Mondays after public-holiday closures** (no lag-7 analogue)
  and on **Saturdays**, while percentage errors are broadly uniform across store
  types (type `b` stores are the easiest).

## How to re-run

```bash
pip install marimo pandas numpy scikit-learn matplotlib
python -m marimo export html retail_sales_ml.py -o retail_sales_ml.html -f
```

or open interactively with `marimo edit retail_sales_ml.py`. Modules
(`features.py`, `models.py`, `evaluation.py`) can also be used standalone; see
their docstrings.

Runtime: ~12 minutes end-to-end (RandomForest fit dominates).

## Limitations

* No `test.csv` — the 6-week holdout proxies the real forecast task; production
  use would warrant rolling-origin cross-validation.
* Fixed hyperparameters (no tuning window); further gains likely from
  gradient-boosting libraries (LightGBM/XGBoost) with promo interaction features
  and longer rolling windows.
* External signals (weather, regional events, competitor promotions) absent.
* The daily one-step-ahead scenario means predictions assume yesterday's sales
  are known; multi-week-ahead forecasts would degrade autoregressive features.
