# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Rossmann promotion hypothesis-testing analysis — marimo notebook.

Run headlessly:    marimo export html promotion_analysis.py --output output/notebook.html
Run interactively: marimo edit promotion_analysis.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Do Rossmann promotions change daily sales?

    **Research question:** Do Rossmann stores have significantly different average
    daily sales on promotional days compared with non-promotional days?

    - **Data:** `rossmann-store-sales/train.csv` + `rossmann-store-sales/store.csv`
      (`test.csv` / `sample_submission.csv` ignored).
    - **Design:** observational — association only, not causal proof.
    - **Primary test:** Welch's t-test on log(Sales), two-sided, α = 0.05,
      H₀: μP − μN = 0 vs H₁: μP − μN ≠ 0.
    - **Robustness:** store-level paired t-test + Wilcoxon signed-rank.
    """)
    return


@app.cell
def _():
    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    import seaborn as sns

    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="notebook")
    return mo, np, os, pd, plt, sns, st


@app.cell
def _(os):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "rossmann-store-sales")
    FIG_DIR = os.path.join(ROOT_DIR, "figs")
    OUT_DIR = os.path.join(ROOT_DIR, "output")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for _name in ("train.csv", "store.csv"):
        if not os.path.exists(os.path.join(DATA_DIR, _name)):
            raise FileNotFoundError(f"Dataset file not found: {DATA_DIR}/{_name}")
    return DATA_DIR, FIG_DIR, OUT_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stage 1 · Load data
    """)
    return


@app.cell
def _(DATA_DIR, os, pd):
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    STORE_PATH = os.path.join(DATA_DIR, "store.csv")

    REQ_COLS = {
        "Store",
        "DayOfWeek",
        "Date",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    }
    train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    missing_cols = REQ_COLS - set(train_raw.columns)
    if missing_cols:
        raise KeyError(f"train.csv is missing required columns: {sorted(missing_cols)}")
    train_raw["Date"] = pd.to_datetime(train_raw["Date"], errors="raise")
    store_meta = pd.read_csv(STORE_PATH)
    return STORE_PATH, TRAIN_PATH, store_meta, train_raw


@app.cell
def _(STORE_PATH, TRAIN_PATH, mo, os, store_meta, train_raw):
    def rel_path(path):
        return "/" + "/".join(path.replace(os.sep, "/").split("/")[-2:])

    mo.md(
        f"""
        | Dataset | Shape | Notes |
        |---|---|---|
        | train `{rel_path(TRAIN_PATH)}` | {train_raw.shape[0]:,} × {train_raw.shape[1]} | daily records, {train_raw.Store.nunique()} stores, {train_raw.Date.min():%Y-%m-%d} → {train_raw.Date.max():%Y-%m-%d} |
        | store `{rel_path(STORE_PATH)}` | {store_meta.shape[0]:,} × {store_meta.shape[1]} | store-level metadata |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stage 1 · Data quality assessment
    """)
    return


@app.cell
def _(pd, train_raw):
    quality_checks = pd.DataFrame(
        [
            (
                "duplicated (Store, Date) rows",
                int(train_raw.duplicated(["Store", "Date"]).sum()),
            ),
            ("missing values in train", int(train_raw.isna().sum().sum())),
            (
                "open days with Sales ≤ 0",
                int(((train_raw.Open == 1) & (train_raw.Sales <= 0)).sum()),
            ),
            ("closed days (all Sales = 0)", int((train_raw.Open == 0).sum())),
            ("negative customer counts", int((train_raw.Customers < 0).sum())),
            (
                "StateHoliday distinct codes",
                ", ".join(sorted(train_raw.StateHoliday.astype(str).unique())),
            ),
        ],
        columns=["check", "result"],
    )
    quality_checks
    return


@app.cell
def _(train_raw):
    promo_share_by_dow = train_raw.groupby("DayOfWeek").Promo.mean().round(2)
    promo_share_by_dow
    return (promo_share_by_dow,)


@app.cell(hide_code=True)
def _(mo, promo_share_by_dow, train_raw):
    mo.md(f"""
    ### Quality findings → cleaning decisions

    1. `StateHoliday` mixes bare `0` and the string `"0"` → normalized to string codes.
    2. **{int((train_raw.Open == 0).sum()):,} closed-store days** (all Sales = 0,
       Customers = 0) are not trading days → excluded from the comparison.
    3. Open days with Sales ≤ 0 (rare refurbishment-like anomalies) → excluded and counted.
    4. No duplicated (Store, Date) keys; no missing values in train. `store.csv`
       structural missings (competition / Promo2 fields) are not used in the primary test.
    5. **Promotions only run Monday–Friday** (Sat share = {promo_share_by_dow.loc[6]:.2f},
       Sun share = {promo_share_by_dow.loc[7]:.2f}) → promo status is confounded with
       day of week; the store-level paired design (Test B) partially mitigates
       store-level and weekday-mix differences.

    *No outlier trimming:* high-sales days are genuine business volume.
    """)
    return


@app.cell
def _(store_meta, train_raw):
    train_norm = train_raw.copy()
    train_norm["StateHoliday"] = train_norm.StateHoliday.astype(str).str.strip()
    assert set(train_norm.StateHoliday.unique()) <= {"0", "a", "b", "c"}

    df_merged = train_norm.merge(
        store_meta, on="Store", how="left", validate="many_to_one"
    )
    unmatched_rows = int(df_merged.StoreType.isna().sum())
    assert unmatched_rows == 0, f"{unmatched_rows} rows unmatched after store merge"
    return (df_merged,)


@app.cell
def _(df_merged, np):
    n_rows_start = len(df_merged)
    df_open = df_merged[df_merged.Open == 1].copy()
    n_rows_open = len(df_open)
    rows_removed_closed = n_rows_start - n_rows_open

    df_clean = df_open[df_open.Sales > 0].copy()
    rows_removed_open_zero = n_rows_open - len(df_clean)

    df_clean["Year"] = df_clean.Date.dt.year
    df_clean["Month"] = df_clean.Date.dt.month
    df_clean["Weekday"] = df_clean.Date.dt.dayofweek + 1  # 1 = Monday
    df_clean["PromoLabel"] = df_clean.Promo.map({0: "No promo", 1: "Promo"})
    df_clean["LogSales"] = np.log(df_clean.Sales.astype(float))

    assert df_clean.Sales.min() > 0, "log(Sales) undefined for non-positive sales"
    return (
        df_clean,
        n_rows_open,
        n_rows_start,
        rows_removed_closed,
        rows_removed_open_zero,
    )


@app.cell(hide_code=True)
def _(
    mo,
    n_rows_open,
    n_rows_start,
    rows_removed_closed,
    rows_removed_open_zero,
):
    mo.md(f"""
    **Filtering audit trail**

    | step | rows removed | rows remaining |
    |---|---|---|
    | start | — | {n_rows_start:,} |
    | drop closed days (Open = 0) | {rows_removed_closed:,} | {n_rows_open:,} |
    | drop open days with Sales ≤ 0 | {rows_removed_open_zero:,} | {n_rows_open - rows_removed_open_zero:,} |

    Final analytical sample: **{n_rows_open - rows_removed_open_zero:,} open-store days**
    across 1,115 stores, all with Sales > 0, so log(Sales) is defined.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stage 2 · Exploratory Data Analysis
    """)
    return


@app.cell
def _(df_clean, pd):
    descriptive = pd.DataFrame(
        {
            "observations": len(df_clean),
            "stores": df_clean.Store.nunique(),
            "mean_sales_EUR": round(df_clean.Sales.mean(), 2),
            "median_sales_EUR": round(df_clean.Sales.median(), 2),
            "std_sales_EUR": round(df_clean.Sales.std(), 2),
            "p25": round(df_clean.Sales.quantile(0.25), 2),
            "p75": round(df_clean.Sales.quantile(0.75), 2),
            "iqr": round(df_clean.Sales.quantile(0.75) - df_clean.Sales.quantile(0.25), 2),
            "skew": round(df_clean.Sales.skew(), 3),
            "mean_log_sales": round(df_clean.LogSales.mean(), 3),
            "std_log_sales": round(df_clean.LogSales.std(), 3),
            "mean_customers": round(df_clean.Customers.mean(), 1),
            "promo_days": int((df_clean.Promo == 1).sum()),
            "promo_share": round(float((df_clean.Promo == 1).mean()), 3),
        },
        index=["value"],
    )
    descriptive.T
    return


@app.cell
def _(df_clean):
    promo_compare = (
        df_clean.groupby("PromoLabel")
        .agg(
            n=("Sales", "size"),
            mean_sales=("Sales", "mean"),
            median_sales=("Sales", "median"),
            std_sales=("Sales", "std"),
            mean_log_sales=("LogSales", "mean"),
            mean_customers=("Customers", "mean"),
        )
        .round(2)
    )
    mean_diff_sales = float(
        promo_compare.loc["Promo", "mean_sales"]
        - promo_compare.loc["No promo", "mean_sales"]
    )
    mean_diff_pct = float(
        mean_diff_sales / promo_compare.loc["No promo", "mean_sales"] * 100
    )
    promo_compare, mean_diff_sales, mean_diff_pct
    return mean_diff_pct, promo_compare


@app.cell
def _(FIG_DIR, df_clean, plt, sns):
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    sns.histplot(df_clean.Sales, bins=60, ax=ax_hist)
    ax_hist.set(
        title="Distribution of daily sales (open store-days)",
        xlabel="Sales (€)",
        ylabel="Days",
    )
    fig_hist.tight_layout()
    fig_hist.savefig(
        f"{FIG_DIR}/01_sales_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_hist)
    return


@app.cell
def _(FIG_DIR, df_clean, plt, sns):
    fig_box, ax_box = plt.subplots(figsize=(6.5, 4.2))
    sns.boxplot(
        data=df_clean,
        x="PromoLabel",
        y="Sales",
        hue="PromoLabel",
        order=["No promo", "Promo"],
        palette="Set2",
        legend=False,
        showfliers=False,
        ax=ax_box,
    )
    ax_box.set(
        title="Daily sales by promotion status (outliers hidden)",
        xlabel="",
        ylabel="Sales (€)",
    )
    fig_box.tight_layout()
    fig_box.savefig(
        f"{FIG_DIR}/02_sales_by_promo_boxplot.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_box)
    return


@app.cell
def _(FIG_DIR, df_clean, plt):
    ax_dow = df_clean.groupby(["DayOfWeek", "PromoLabel"]).Sales.mean().unstack().plot(
        kind="bar",
        figsize=(8, 4),
        ylabel="Mean sales (€)",
        legend=True,
    )
    ax_dow.set(
        title="Mean sales by day of week × promotion (confounder check)",
        xlabel="Day of week (1 = Mon … 7 = Sun)",
    )
    ax_dow.legend(title="Promo")
    fig_dow = ax_dow.figure
    fig_dow.tight_layout()
    fig_dow.savefig(
        f"{FIG_DIR}/03_mean_sales_by_dow_promo.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_dow)
    return


@app.cell
def _(FIG_DIR, df_clean, plt):
    ts = (
        df_clean.set_index("Date")
        .groupby("Promo")
        .LogSales.resample("ME")
        .mean()
        .unstack(0)
        .rename(columns={0: "no_promo", 1: "promo"})
    )
    fig_ts, ax_ts = plt.subplots(figsize=(9, 4))
    ax_ts.plot(ts.index, ts["no_promo"], label="No promo", color="tab:green")
    ax_ts.plot(ts.index, ts["promo"], label="Promo", color="tab:orange")
    ax_ts.set(
        title="Monthly mean log(Sales) trend",
        xlabel="Date",
        ylabel="Mean log(Sales)",
    )
    ax_ts.legend()
    fig_ts.tight_layout()
    fig_ts.savefig(f"{FIG_DIR}/04_monthly_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ts)
    return


@app.cell
def _(df_clean):
    confounders = (
        df_clean.assign(StateHolidayFlag=(df_clean.StateHoliday != "0").astype(int))
        .groupby("PromoLabel")
        .agg(
            mean_sales=("Sales", "mean"),
            mean_customers=("Customers", "mean"),
            weekday_share=("Weekday", lambda w: (w <= 5).mean()),
            schoolholiday_share=("SchoolHoliday", "mean"),
            stateholiday_share=("StateHolidayFlag", "mean"),
        )
        .round(3)
    )
    dow_composition = (
        df_clean.groupby("PromoLabel")
        .Weekday.value_counts(normalize=True)
        .unstack()
        .round(3)
    )
    dow_composition = dow_composition.set_axis(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], axis=1
    )
    sales_by_storetype = (
        df_clean.groupby(["StoreType", "PromoLabel"]).Sales.mean().unstack().round(0)
    )
    confounders, dow_composition, sales_by_storetype
    return (confounders,)


@app.cell(hide_code=True)
def _(confounders, mo):
    mo.md(f"""
    **EDA takeaway on confounders.** Promo days concentrate on Monday–Thursday while
    closed Sundays pull down the non-promo mean; school-holiday incidence is similar
    in both groups (≈{confounders.loc['Promo', 'schoolholiday_share']:.0%} vs
    {confounders.loc['No promo', 'schoolholiday_share']:.0%}), and state-holiday days
    are rare in either (~0.1%). Because promo status correlates with weekday and each
    store contributes many days, the pooled daily comparison (Test A) can overweight
    the promotion association. The store-level paired design (Test B) holds each store
    constant, giving a like-for-like estimate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stage 3 · Hypothesis testing

    Two-sided, α = 0.05.

    - **H₀:** μP − μN = 0
    - **H₁:** μP − μN ≠ 0

    where μP, μN are mean log daily sales on promotional / non-promotional days for
    open stores. We work on the **log scale** because raw Sales is right-skewed;
    effects are back-transformed to % uplift: 100 × (exp(mean log diff) − 1).
    """)
    return


@app.cell
def _(df_clean, np, st):
    grp_promo = df_clean.LogSales[df_clean.Promo == 1]
    grp_no = df_clean.LogSales[df_clean.Promo == 0]

    welch_result = st.ttest_ind(grp_promo, grp_no, equal_var=False)
    n1, n0 = grp_promo.count(), grp_no.count()
    v1, v0 = grp_promo.var(ddof=1), grp_no.var(ddof=1)
    welch_se = float(np.sqrt(v1 / n1 + v0 / n0))
    welch_df = float(
        (v1 / n1 + v0 / n0) ** 2
        / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
    )
    welch_diff_log = float(grp_promo.mean() - grp_no.mean())
    welch_err = float(st.t.ppf(0.975, welch_df)) * welch_se
    welch_ci_log = (welch_diff_log - welch_err, welch_diff_log + welch_err)

    pooled_sd = np.sqrt(
        ((grp_promo.count() - 1) * grp_promo.var(ddof=1)
         + (grp_no.count() - 1) * grp_no.var(ddof=1))
        / (grp_promo.count() + grp_no.count() - 2)
    )
    hedges_g = (
        welch_diff_log
        / pooled_sd
        * (1 - 3 / (4 * (grp_promo.count() + grp_no.count()) - 9))
    )

    welch_uplift_pct = (np.exp(welch_diff_log) - 1) * 100
    welch_ci_pct = (
        (np.exp(welch_ci_log[0]) - 1) * 100,
        (np.exp(welch_ci_log[1]) - 1) * 100,
    )
    n_promo_grp = int(grp_promo.count())
    n_no_promo_grp = int(grp_no.count())
    return (
        hedges_g,
        n_no_promo_grp,
        n_promo_grp,
        welch_ci_log,
        welch_ci_pct,
        welch_df,
        welch_diff_log,
        welch_result,
        welch_uplift_pct,
    )


@app.cell
def _(df_clean, pd):
    per_store = (
        df_clean.groupby(["Store", "Promo"]).LogSales.agg(["mean", "size"]).unstack()
    )
    stores_both = per_store[
        per_store[("mean", 0)].notna() & per_store[("mean", 1)].notna()
    ]
    df_store_paired = pd.DataFrame(
        {
            "Store": stores_both.index,
            "n_no_promo": stores_both[("size", 0)].astype(int),
            "n_promo": stores_both[("size", 1)].astype(int),
            "mean_no_promo": stores_both[("mean", 0)],
            "mean_promo": stores_both[("mean", 1)],
        }
    ).reset_index(drop=True)
    df_store_paired["diff_log_sales"] = (
        df_store_paired.mean_promo - df_store_paired.mean_no_promo
    )
    assert (df_store_paired[["n_promo", "n_no_promo"]] > 0).all().all()
    return (df_store_paired,)


@app.cell
def _(df_store_paired, np, st):
    paired_diff = df_store_paired.diff_log_sales
    n_stores_paired = int(len(paired_diff))

    paired_t_result = st.ttest_rel(
        df_store_paired.mean_promo, df_store_paired.mean_no_promo
    )
    se_diff = paired_diff.std(ddof=1) / np.sqrt(n_stores_paired)
    t_crit = st.t.ppf(0.975, n_stores_paired - 1)
    paired_diff_log = float(paired_diff.mean())
    paired_ci_log = (
        paired_diff_log - t_crit * se_diff,
        paired_diff_log + t_crit * se_diff,
    )
    paired_uplift_pct = (np.exp(paired_diff_log) - 1) * 100
    paired_ci_pct = (
        (np.exp(paired_ci_log[0]) - 1) * 100,
        (np.exp(paired_ci_log[1]) - 1) * 100,
    )
    wilcoxon_result = st.wilcoxon(paired_diff)
    paired_dz = paired_diff_log / paired_diff.std(ddof=1)
    return (
        n_stores_paired,
        paired_ci_log,
        paired_ci_pct,
        paired_diff_log,
        paired_dz,
        paired_t_result,
        paired_uplift_pct,
        wilcoxon_result,
    )


@app.cell
def _(
    FIG_DIR,
    hedges_g,
    n_no_promo_grp,
    n_promo_grp,
    paired_ci_log,
    paired_ci_pct,
    paired_dz,
    paired_t_result,
    paired_uplift_pct,
    pd,
    plt,
    welch_ci_log,
    welch_ci_pct,
    welch_df,
    welch_diff_log,
    welch_result,
    welch_uplift_pct,
):
    results_table = pd.DataFrame(
        [
            {
                "analysis": "A. Welch t-test on daily log sales (unadjusted)",
                "n_promo": n_promo_grp,
                "n_non_promo": n_no_promo_grp,
                "test_statistic": round(float(welch_result.statistic), 2),
                "df": round(welch_df, 0),
                "p_value": float(welch_result.pvalue),
                "mean_diff_log": round(welch_diff_log, 4),
                "95CI_diff_log": f"({welch_ci_log[0]:.4f}, {welch_ci_log[1]:.4f})",
                "uplift_pct": round(welch_uplift_pct, 1),
                "95CI_uplift_pct": f"({welch_ci_pct[0]:.1f}, {welch_ci_pct[1]:.1f})",
                "effect_size": f"Hedges' g = {hedges_g:.3f}",
                "reject_H0_at_0.05": bool(welch_result.pvalue < 0.05),
            },
            {
                "analysis": "B. Paired t-test on store-level mean log sales",
                "n_promo": "store-level",
                "n_non_promo": "store-level",
                "test_statistic": round(float(paired_t_result.statistic), 2),
                "df": None,
                "p_value": float(paired_t_result.pvalue),
                "mean_diff_log": round((paired_ci_log[0] + paired_ci_log[1]) / 2, 4),
                "95CI_diff_log": f"({paired_ci_log[0]:.4f}, {paired_ci_log[1]:.4f})",
                "uplift_pct": round(paired_uplift_pct, 1),
                "95CI_uplift_pct": f"({paired_ci_pct[0]:.1f}, {paired_ci_pct[1]:.1f})",
                "effect_size": f"Cohen's dz = {paired_dz:.3f}",
                "reject_H0_at_0.05": bool(paired_t_result.pvalue < 0.05),
            },
        ]
    )

    welch_center = (welch_ci_log[0] + welch_ci_log[1]) / 2
    paired_center = (paired_ci_log[0] + paired_ci_log[1]) / 2
    ci_labels = [
        f"Welch (unadjusted)\nuplift = {welch_uplift_pct:.1f}%",
        f"Store-paired\nuplift = {paired_uplift_pct:.1f}%",
    ]
    ci_centers = [welch_center, paired_center]
    ci_errors = [
        [welch_center - welch_ci_log[0], paired_center - paired_ci_log[0]],
        [welch_ci_log[1] - welch_center, paired_ci_log[1] - paired_center],
    ]
    assert min(ci_centers[0] - welch_ci_log[0], welch_ci_log[1] - ci_centers[0]) >= 0
    assert min(
        ci_centers[1] - paired_ci_log[0], paired_ci_log[1] - ci_centers[1]
    ) >= 0

    fig_ci, ax_ci = plt.subplots(figsize=(7, 3.5))
    ax_ci.errorbar(
        ci_labels, ci_centers, yerr=ci_errors, fmt="o", capsize=5, color="tab:blue"
    )
    ax_ci.axhline(0, color="red", linestyle="--", alpha=0.6)
    ax_ci.set(
        title="95% CI for mean log-sales difference (promo − no promo)",
        ylabel="Difference in mean log(Sales)",
    )
    fig_ci.tight_layout()
    fig_ci.savefig(f"{FIG_DIR}/05_ci_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ci)
    results_table
    return (results_table,)


@app.cell
def _(OUT_DIR, results_table):
    results_table.to_csv(f"{OUT_DIR}/hypothesis_test_results.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stage 4 · Results and interpretation
    """)
    return


@app.cell
def _(
    df_store_paired,
    hedges_g,
    mean_diff_pct,
    mo,
    n_no_promo_grp,
    n_promo_grp,
    n_stores_paired,
    paired_ci_pct,
    paired_diff_log,
    paired_dz,
    paired_t_result,
    paired_uplift_pct,
    promo_compare,
    welch_ci_pct,
    welch_diff_log,
    welch_result,
    welch_uplift_pct,
    wilcoxon_result,
):
    n_promo_span = (
        f"{int(df_store_paired.n_promo.min())}–{int(df_store_paired.n_promo.max())}"
    )
    results_summary = mo.md(
        f"""
        ### Interpretation

        **Test A — Welch's t-test (unadjusted, pooled daily records).**
        Mean sales are higher on promo days: €{promo_compare.loc['Promo', 'mean_sales']:,.0f}
        vs €{promo_compare.loc['No promo', 'mean_sales']:,.0f}
        (raw mean difference +{mean_diff_pct:.1f}%). On the log scale the difference is
        {welch_diff_log:.4f}, i.e. ≈ **{welch_uplift_pct:+.1f}%** uplift
        (95% CI ({welch_ci_pct[0]:.1f}%, {welch_ci_pct[1]:.1f}%)).
        n = {n_promo_grp:,} promo vs {n_no_promo_grp:,} non-promo open store-days.
        Welch t = {welch_result.statistic:.1f}, p = {welch_result.pvalue:.3g} — far
        below α = 0.05, so we **reject H₀**. Hedges' g = {hedges_g:.2f} on the log
        scale is not trivial, but note the imbalance: with {n_promo_grp + n_no_promo_grp:,}
        records even modest effects become detectable, so statistical significance
        alone is not evidence of commercial importance.

        **Test B — Store-level paired comparison** ({n_stores_paired} stores have both
        promo and non-promo observations; per-store promo-day counts range from
        {n_promo_span}). Within-store mean log difference = {paired_diff_log:.4f},
        ≈ **{paired_uplift_pct:+.1f}%**
        (95% CI ({paired_ci_pct[0]:.1f}%, {paired_ci_pct[1]:.1f}%)).
        Paired t = {paired_t_result.statistic:.1f}, p = {paired_t_result.pvalue:.3g};
        Wilcoxon signed-rank p = {wilcoxon_result.pvalue:.3g}; Cohen's dz = {paired_dz:.2f}.
        This is the more defensible estimate: it holds each store constant and
        compares like with like days within the same store.

        **Statistical vs practical significance.** The uplift is statistically
        detectable but its *practical* value depends on promo costs and margins,
        which we cannot see in this data. And because promotions are planned
        (weekday-only), stores differ in promo intensity, and other calendar
        effects remain uncontrolled, the association is **not causal evidence**.
        """
    )
    results_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Limitations and next steps

    - **Observational data**: promotions were chosen by store managers on weekdays;
      no randomization means we cannot read this difference as a causal effect.
    - **Within-store dependence**: ~943 store-days each; per-store clustering is only
      partially handled by the paired design — a mixed-effects or cluster-robust
      regression would tighten uncertainty bands further.
    - **Weekday confounding**: promos run Mon–Fri only, so Saturday and Sunday
      (usually stronger trading days) sit exclusively in the non-promo group.
      The paired estimate mitigates but cannot fully remove this.
    - **Costs are invisible**: higher revenue ≠ higher profit; promo cost data
      (discounts, lost margin) would be needed to judge profitability.
    - **Heterogeneity**: store type (a/b/c/d) and assortment may moderate the promo
      effect; a per-store-type breakdown is a natural extension.

    **Recommended next steps.** (1) Quantify promo economics by combining this sales
    uplift with per-store promo cost data. (2) Examine heterogeneous effects across
    store types and weekends-opportunity stores. (3) Most decisively: run a
    randomized controlled promo experiment across similar stores to estimate the
    true incremental (causal) lift.
    """)
    return


if __name__ == "__main__":
    app.run()
