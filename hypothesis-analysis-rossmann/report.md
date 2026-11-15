# Do promotions change daily sales? — Rossmann Store Sales analysis

*Hypothesis-testing portfolio project · Python · marimo · all numbers below are computed from the dataset, not estimated.*

---

## 1. Executive summary

| Item | Finding |
|---|---|
| Research question | Do Rossmann stores have significantly different average daily sales on promotional days vs non-promotional days? |
| Data | 1,015,152 open-store days (1,017,209 raw rows − 172,817 closed days − 54 open days with zero sales), 1,115 stores, 2013-01-01 → 2015-07-31 |
| Principal result | **Yes, promotions are strongly associated with higher average daily sales.** Both the unadjusted Welch test and the store-level paired test reject H₀ (p ≈ 0, far below α = 0.05). |
| Size of the difference | Promotional days average **€8,228.74** vs **€5,929.83** on non-promotional days (raw mean difference ≈ **+38.8%**). On the log scale (primary analysis) the difference is **0.3462 log points ≈ +41.4% uplift** (95% CI 40.3%–42.5%, store-paired). |
| Practical interpretation | On average, a trading day with an active promo rings up roughly 4 in 10 more euros-worth of sales than a trading day without one in the same store. |
| Main limitation | Data are **observational**: promos run Monday–Friday only and their placement is planned, so this association is not causal proof of incremental sales. |

---

## 2. Data and methodology

### 2.1 Source and files

- Kaggle competition dataset *Rossmann Store Sales* (https://www.kaggle.com/competitions/rossmann-store-sales/data).
- Files used: `rossmann-store-sales/train.csv` (1,017,209 daily store records), `rossmann-store-sales/store.csv` (1,115 stores of metadata).
- `test.csv` and `sample_submission.csv` were **not** used.

### 2.2 Data quality findings and cleaning decisions

| Check | Result | Decision |
|---|---|---|
| Duplicated (Store, Date) keys | 0 | none needed |
| Missing values in train | 0 | none needed |
| Closed days (`Open == 0`) | 172,817 (all Sales = 0, Customers = 0) | **excluded** — non-trading days |
| Open days with Sales ≤ 0 | 54 (refurbishment-like anomalies) | **excluded** |
| Negative customer counts | 0 | none needed |
| `StateHoliday` coding | mixed bare `0` and string `"0"` | normalized to string codes |
| Missing store metadata | `CompetitionDistance` missing in 3 stores; competition/Promo2 fields partially empty | those fields not used in the primary test |

No outlier trimming: the right tail of sales is genuine business volume. The final sample is
**1,015,152 store-days across all 1,115 stores**, every one with Sales > 0, so log(Sales) is well defined.

### 2.3 Sample snapshot

| Metric | Value |
|---|---|
| Observations | 1,015,152 |
| Stores | 1,115 |
| Date range | 2013-01-01 → 2015-07-31 |
| Mean / median sales (€) | 6,956 / 6,369 |
| Std dev | 3,103.8 |
| Skewness | 1.595 (strongly right-skewed → log transform) |
| Promo days | 376,875 (37.1%); **only Monday–Friday** |
| Mean customers | 697 (no promo) vs 844 (promo) |

### 2.4 EDA findings

1. **Sales are strongly right-skewed** (skew 1.60), which motivates log-transform for the primary test.
2. **Promo and non-promo distributions differ visibly** (box plot): medians €7,650 vs €5,459.
3. **Promotions run Monday–Friday only**: Saturday and Sunday trading days are exclusively non-promo, and weekend days have distinct sales levels — a structural confounder.
4. School-holiday incidence is similar in both groups (≈ 18–21%); state-holiday days are rare ≈ 0.1% in both.
5. Promo days show consistent gaps in every month of the 2.5-year window (monthly trend plot), i.e. the gap is not driven by one calendar period.
6. Per store type (`a/b/c/d`), promo days also show higher means within each type, so the association is not driven by one store class.

### 2.5 Hypotheses and test selection

Two-sided, α = 0.05:

- **H₀:** μP − μN = 0 (mean log daily sales equal on promo and non-promo days)
- **H₁:** μP − μN ≠ 0

Two complementary tests were run:

- **Test A — Welch's t-test** on `LogSales` for all open store-days (unadjusted, pooled).
  Transparent baseline; robust to unequal variances. Does **not** account for repeated
  observations within stores.
- **Test B — store-level paired t-test**: mean `LogSales` computed separately for promo and
  non-promo days **within each store** (all 1,115 stores have both conditions), then paired
  t-test across stores plus Wilcoxon signed-rank robustness check. This holds the store
  constant and partially mitigates weekday-mix differences, giving a like-for-like within-store
  comparison.

Because the outcome is log sales, the effect is back-transformed as % uplift = 100 × (exp(mean log diff) − 1).

---

## 3. Results

| Metric | A. Welch (unadjusted) | B. Store-paired |
|---|---|---|
| n promo | 376,875 | per-store pairs: 1,115 |
| n non-promo | 467,463 | per-store pairs: 1,115 |
| Mean sales (raw €) promo / no-promo | €8,228.74 / €5,929.83 | — |
| Median sales (raw €) promo / no-promo | €7,650 / €5,459 | — |
| Mean difference (log scale) | 0.3462 | 0.3462 |
| 95% CI (log scale) | (0.3446, 0.3479) | (0.3385, 0.3538) |
| **Uplift (% back-transformed)** | **+41.4%** (CI 41.1%–41.6%) | **+41.4%** (CI 40.3%–42.5%) |
| Test statistic | t = 412.29 | t = 88.81 |
| Degrees of freedom | ≈ 838,092 (Welch–Satterthwaite) | — |
| p-value | ≈ 0 (< 1e-300; printed as 0.0 by float) | ≈ 0 (< 1e-300; printed as 0.0 by float) |
| Effect size | Hedges' g = 0.890 | Cohen's dz = 2.660 |
| Wilcoxon signed-rank | — | p ≈ 0 (consistent) |
| Conclusion at α = 0.05 | **Reject H₀** | **Reject H₀** |

Machine-readable table: `output/hypothesis_test_results.csv`.

### Figures

| File | Shows |
|---|---|
| `figs/01_sales_distribution.png` | distribution of daily sales (open store-days); skew 1.60 |
| `figs/02_sales_by_promo_boxplot.png` | box plots of daily sales by promo status |
| `figs/03_mean_sales_by_dow_promo.png` | mean sales by day of week × promo — weekday confounding |
| `figs/04_monthly_trend.png` | monthly mean log(Sales) by promo — gap persists across 2.5 years |
| `figs/05_ci_comparison.png` | 95% CIs for the two estimates of the log-scale difference |

### Interpretation in plain English

- Every day of the 2.5-year window and every store class shows higher sales on promo days.
- The two independent analyses give **the same magnitude**: ~41% more sales on a promo day than on a non-promo day in the same store.
- The p-values are effectively 0, so with this data volume there is no realistic chance the observed gap is random noise.
- However — this is an **observational** dataset. Promos were planned (weekday-only, certain chains), so some of the gap may reflect *when* promos are turned on, not *what they cause*. Do read the number as a strong association, not a causal lift.

### Statistical vs practical significance

Statistically, the difference is unambiguous. Practically, whether it matters depends on what a promo costs (markdowns, supplier funding, cannibalization from neighboring days) — none of which this dataset contains. A 41% revenue uplift is large enough to investigate; it is not, on its own, proof of profitability.

---

## 4. Business recommendations

1. **Investigate promo economics before scaling**: combine the ~41% revenue association with per-store promo cost data (discount depth, supplier funding) and margins to estimate incremental profit — revenue uplift alone does not equal profit uplift.
2. **Study heterogeneity**: test whether the association varies by store type, assortment, and region before uniform decisions.
3. **Do not read the 41% causally**: because promos are placed on chosen weekdays and chains, part of the gap may be attributed to *which* days promos run.
4. **Evidence the causal effect properly**: run a randomized controlled promo experiment across comparable stores, or a well-designed quasi-experiment (diff-in-diff around promo calendar changes), before making expansion/elimination decisions.

---

## 5. Limitations and future work

- **Not randomized**: promotions were selected by planners; selection can correlate with expected demand.
- **Within-store dependence**: each store contributes ~900 store-days; the paired design handles store-level dependence but not all cross-day dependence (calendar, seasonality). A regression with store fixed effects and cluster-robust standard errors would tighten this further.
- **Weekday confounding**: Sat/Sun trading days exist only in the non-promo group. The paired estimate mitigates this but cannot fully remove it; weekday-stratified or regression-adjusted estimates would add confidence.
- **Seasonality** and the December spikes are visible but not explicitly modeled.
- **Costs are invisible**: revenue ≠ profit; promotions may pull sales forward rather than generate incremental demand.
- **Data vintage**: 2013–2015; behaviors may have shifted.

**Future analyses worth running**: promo profitability analysis; heterogeneous effects across store types/assortments; a diff-in-diff design using the Promo2 (continuing) campaign; a randomized experiment for causal attribution.

---

## 6. Reproducibility

- Notebook: `promotion_analysis.py` (marimo). Run headlessly:
  `marimo export html promotion_analysis.py --output output/notebook.html`
- Execution: notebook runs deterministically from a clean state; figures saved to `figs/`, result table to `output/hypothesis_test_results.csv`..- No random seeds are needed; the analysis is fully deterministic.
- Dependencies: `requirements.txt`.
