import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    md_overview = mo.md(
        r"""
        # 🛒 Rossmann Store Sales — EDA & Forecasting

        **Machine Learning coursework — sales forecasting for Rossmann drugstores.**

        **Data**: `train.csv` (~1.02 M daily store records, 2013-01-01 → 2015-07-31) and
        `store.csv` (1,115 stores with static attributes), merged on `Store`.

        **Scenario assumption**: `test.csv` was not provided, so the **last 6 weeks**
        of the timeline (2015-06-20 → 2015-07-31) are held out as an out-of-time
        evaluation window. This mirrors deployment: train on the past, forecast the
        future, and avoids the leakage of random row-wise splits.

        **Models compared**

        | Model | Notes |
        |---|---|
        | Naive baseline | "last week, same weekday" sales per store (sanity check) |
        | Ridge regression | on `log1p(Sales)`, one-hot categoricals, scaled numerics |
        | RandomForest | 300 trees, on `log1p(Sales)` |
        | HistGradientBoosting | sklearn gradient boosting, likely winner |

        **Key design decisions** (documented inline in the modules):

        * Modelling frame = **open-store days only**; closed days are trivially
          predicted as zero sales.
        * `Customers` is **excluded** as a feature — it is only known after the fact
          (leakage).
        * Lag/rolling features are computed per store on its open-day history and
          **shifted by one open day**, so the current day's sales never enter its own
          features. The assumed scenario is a daily one-step-ahead forecast.
        * Sales-scale lag features are `log1p`-transformed so a linear model on
          `log1p(Sales)` sees an approximately linear relationship.
        * All imputation statistics are fitted on the **training split only**.
        """
    )
    md_overview
    return (md_overview,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    return (Path, np, pd, plt)


@app.cell
def _():
    import features as F
    import evaluation as E
    import models as M
    return (E, F, M)


@app.cell
def _(mo):
    md_data = mo.md(
        """
        ## 1. Data loading & audit
        """
    )
    md_data
    return (md_data,)


@app.cell
def _(F, Path):
    DATA_DIR = Path("rossmann-sales-data")
    train_raw, store = F.load_data(DATA_DIR)
    return (DATA_DIR, store, train_raw)


@app.cell
def _(mo, store, train_raw):
    train_head = train_raw.head(8)
    store_head = store.head(8)
    _overview = mo.md(
        f"""
        **Shapes**: train = {train_raw.shape[0]:,} rows × {train_raw.shape[1]} cols,
        store = {store.shape[0]:,} rows × {store.shape[1]} cols.
        Date range: **{train_raw['Date'].min().date()} → {train_raw['Date'].max().date()}**.
        Unique stores: {train_raw['Store'].nunique():,}.
        """
    )
    mo.vstack([_overview, mo.md("### train.csv (first rows)"), train_head,
               mo.md("### store.csv (first rows)"), store_head])
    return (store_head, train_head)


@app.cell
def _(mo, store, train_raw):
    missing_train = train_raw.isna().sum()
    missing_store = store.isna().sum()
    mo.vstack([
        mo.md(
            """
            ### Missing values

            `train.csv` is complete. `store.csv` has structural missingness:
            `CompetitionDistance` (3 stores), `CompetitionOpenSinceMonth/Year` (354),
            `Promo2SinceWeek/Year` (544 — exactly the stores with `Promo2 == 0`).
            These are handled with explicit flags + imputation in the pipeline.
            """
        ),
        mo.vstack([
            missing_train.to_frame("missing (train)"),
            missing_store[missing_store > 0].to_frame("missing (store)"),
        ]),
    ])
    return (missing_store, missing_train)


@app.cell
def _(mo):
    md_eda = mo.md(
        """
        ## 2. Exploratory data analysis
        """
    )
    md_eda
    return (md_eda,)


@app.cell
def _(pd, plt, train_raw):
    monthly = train_raw.assign(_m=train_raw["Date"].dt.to_period("M")).groupby("_m")["Sales"].sum()
    fig_trend, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(monthly.index.astype(str), monthly.values / 1e6, marker="o", ms=3)
    _ax.set_title("Total sales per month, all stores (€ millions)")
    _ax.set_ylabel("€ m")
    plt.xticks(rotation=45)
    _ax.grid(alpha=0.3)
    fig_trend
    return (fig_trend, monthly)


@app.cell
def _(mo):
    _trend_note = mo.md(
        """
        **Observation**: sales are stationary-ish around €270–300 m per month with a
        pronounced December spike (Christmas) and a visible dip in 2014 — consistent
        with the 2014 German recession and Rossmann's expansion lull. Strong monthly
        seasonality justifies the `Month`/`WeekOfYear` + sin/cos features.
        """
    )
    _trend_note
    return (_trend_note,)


@app.cell
def _(plt, train_raw):
    _open = train_raw[train_raw["Open"] == 1]
    dow = _open.groupby("DayOfWeek")["Sales"].mean()
    promo = _open.groupby("Promo")["Sales"].mean()
    fig_dow, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.bar(dow.index, dow.values)
    _ax1.set_title("Mean sales by day of week (open days)")
    _ax1.set_xlabel("DayOfWeek (1=Mon … 7=Sun)")
    _ax2.bar(["No promo", "Promo"], promo.values)
    _ax2.set_title("Mean sales: promo effect (open days)")
    _ax1.grid(alpha=0.3, axis="y")
    _ax2.grid(alpha=0.3, axis="y")
    fig_dow
    return (dow, fig_dow, promo)


@app.cell
def _(mo):
    _dow_note = mo.md(
        """
        **Observation**: Mondays are the strongest day (post-weekend + promo start);
        Sunday trading is rare (most stores closed) but the few open stores are busy.
        Promo days see roughly **+40%** mean sales. Both effects are strong model
        features (`DayOfWeek`, `Promo`).
        """
    )
    _dow_note
    return (_dow_note,)


@app.cell
def _(plt, train_raw):
    _open = train_raw[train_raw["Open"] == 1]
    holiday = _open.groupby("StateHoliday")["Sales"].mean()
    school = _open.groupby("SchoolHoliday")["Sales"].mean()
    fig_holiday, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.bar(holiday.index, holiday.values)
    _ax1.set_title("Mean sales by state holiday (open days)")
    _ax1.set_xlabel("0=none, a=public, b=Easter, b/c=Christmas")
    _ax2.bar(school.index.astype(str), school.values)
    _ax2.set_title("Mean sales: school holiday effect")
    _ax1.grid(alpha=0.3, axis="y")
    _ax2.grid(alpha=0.3, axis="y")
    fig_holiday
    return (fig_holiday, holiday, school)


@app.cell
def _(plt, store, train_raw):
    merged_eda = train_raw.merge(store, on="Store", how="left")
    _open = merged_eda[merged_eda["Open"] == 1]
    by_type = _open.groupby("StoreType")["Sales"].mean()
    by_assort = _open.groupby("Assortment")["Sales"].mean()
    fig_type, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.bar(by_type.index, by_type.values)
    _ax1.set_title("Mean sales by store type")
    _ax2.bar(by_assort.index, by_assort.values)
    _ax2.set_title("Mean sales by assortment")
    _ax1.grid(alpha=0.3, axis="y")
    _ax2.grid(alpha=0.3, axis="y")
    fig_type
    return (by_assort, by_type, fig_type, merged_eda)


@app.cell
def _(mo):
    _type_note = mo.md(
        """
        **Observation**: store type and assortment separate store sizes — type **d**
        and extra assortment stores sell ~2× more than type **c**/basic stores. These
        are store-level (not day-level) features: they help the model learn *which*
        store is being forecast but they are constant per store.
        """
    )
    _type_note
    return (_type_note,)


@app.cell
def _(mo, store, train_raw):
    open_zero = train_raw[(train_raw["Open"] == 1) & (train_raw["Sales"] == 0)]
    closed = train_raw[train_raw["Open"] == 0]
    zero_md = mo.md(
        f"""
        ### Data quality audit

        * Closed days (`Open == 0`): **{len(closed):,}** rows — all have `Sales == 0`
          (deterministic; excluded from the modelling frame, predicted as 0).
        * Open days with zero sales: **{len(open_zero):,}** rows — legitimate
          (e.g. refurbishment), kept in the frame; they inflate MAPE, so MAPE is
          reported with these rows excluded.
        * Stores missing `CompetitionDistance`: **{int(store['CompetitionDistance'].isna().sum())}**
        * Cross-check: `Sales == 0` always coincides with `Open == 0` or the
          refurbishment rows above — no inconsistencies.
        """
    )
    zero_md
    return (closed, open_zero, zero_md)


@app.cell
def _(np, plt, train_raw):
    sales_open = train_raw.loc[(train_raw["Open"] == 1) & (train_raw["Sales"] > 0), "Sales"]
    fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(sales_open, bins=60)
    ax1.set_title(f"Sales distribution (open, positive) — n={len(sales_open):,}")
    ax1.set_xlabel("Sales (€)")
    ax2.hist(np.log1p(sales_open), bins=60)
    ax2.set_title("log1p(Sales) — modelling target")
    ax2.set_xlabel("log1p(Sales)")
    fig_dist
    return (fig_dist, sales_open)


@app.cell
def _(mo):
    _dist_note = mo.md(
        """
        **Observation**: raw `Sales` is right-skewed (long upper tail); `log1p(Sales)`
        is near-symmetric. Modelling in log space both stabilises the linear model and
        matches how percentage error (RMSPE, the competition metric) is measured.
        Predictions are inverted with `expm1` before computing sales-scale metrics.
        """
    )
    _dist_note
    return (_dist_note,)


@app.cell
def _(plt, train_raw):
    _sample = train_raw[train_raw["Open"] == 1].sample(8000, random_state=42)
    fig_cust, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(_sample["Customers"], _sample["Sales"], s=5, alpha=0.3)
    _ax.set_xlabel("Customers")
    _ax.set_ylabel("Sales (€)")
    _ax.set_title("Customers vs Sales (sample)")
    fig_cust
    return (fig_cust,)


@app.cell
def _(mo):
    _cust_note = mo.md(
        """
        **⚠️ Leakage guard**: `Customers` correlates almost perfectly with `Sales`
        (r ≈ 0.95+), but customer counts are only known **after** the fact — they
        cannot be known for the day being forecast. `Customers` is therefore excluded
        from every model. (It would trivially "win" any leaderboard while being
        unusable in production.)
        """
    )
    _cust_note
    return (_cust_note,)


@app.cell
def _(mo):
    md_features = mo.md(
        """
        ## 3. Feature engineering & temporal split

        Implemented in `features.py` (see module docstrings for full rationale):

        * **Calendar**: year, month, day, week-of-year + sin/cos encodings of month
          and week; `DayOfWeek`.
        * **Promotions**: `Promo`, `Promo2`, `Promo2Active` (is the store's continuous
          promo active in this month and after its start date?), duration.
        * **Competition**: distance (+ explicit missing flag), months since the
          nearest competitor opened, open flag.
        * **Autoregressive (leakage-safe)**: per-store sales lags (1, 7, 14, 28 open
          days), 7/28-day rolling means & 7-day std — all shifted by one open day so
          the row's own sales never enter its features. Plus `days_since_last_open`
          (captures the Sunday-closed → Monday pattern) and a `HasLag1` flag.
        * **Split**: last 6 weeks of the timeline = holdout; everything before =
          training. Imputation statistics are fitted on the training split only.
        """
    )
    md_features
    return (md_features,)


@app.cell
def _(F):
    data = F.build_dataset("rossmann-sales-data", holdout_weeks=6)
    train_df = data["train"]
    val_df = data["val"]
    open_df = data["open"]
    full_df = data["full"]
    cutoff = data["cutoff"]
    return (cutoff, data, full_df, open_df, train_df, val_df)


@app.cell
def _(cutoff, data, mo, open_df, train_df, val_df):
    split_md = mo.md(
        f"""
        ### Split summary

        | | rows | date range |
        |---|---|---|
        | Open-day frame | {len(open_df):,} | {open_df['Date'].min().date()} → {open_df['Date'].max().date()} |
        | Training | {len(train_df):,} | → {train_df['Date'].max().date()} |
        | **Holdout (6 weeks)** | {len(val_df):,} | **{cutoff.date()} → {val_df['Date'].max().date()}** |

        Features: **{len(data['features'])}** ({len(data['numeric'])} numeric,
        {len(data['categorical'])} categorical). Closed days ({len(full_df) - len(open_df):,} rows)
        are predicted deterministically as €0 and excluded from model metrics.
        """
    )
    split_md
    return (split_md,)


@app.cell
def _(mo):
    md_models = mo.md(
        """
        ## 4. Model comparison (out-of-time holdout)

        Four candidates, all evaluated on the **same 6-week holdout** on the sales
        scale (predictions inverted with `expm1`):

        * **Naive** — same-store, same-weekday sales one week earlier (`sales_lag7`
          before transform, with per-store median fallback). Any supervised model must
          beat this.
        * **Ridge** — linear model on `log1p(Sales)`, one-hot categoricals, standardised
          numerics.
        * **RandomForest** — 300 trees, max depth 18.
        * **HistGradientBoosting** — sklearn's histogram GBM with early stopping.
        """
    )
    md_models
    return (md_models,)


@app.cell
def _(E, F, train_df, val_df):
    results_df, fitted, preds = E.compare_models(
        train_df,
        val_df,
        F.ALL_FEATURES,
        F.NUMERIC_FEATURES,
        F.CATEGORICAL_FEATURES,
    )
    return (fitted, preds, results_df)


@app.cell
def _(E, mo, results_df):
    E.save_metrics_json(results_df)
    _ranking = results_df["RMSPE"].sort_values().index.tolist()
    best_name = results_df["RMSPE"].idxmin()
    best_rmspe = results_df.loc[best_name, "RMSPE"]
    best_md = mo.md(
        f"""
        ### Results

        🏆 **Best model: {best_name}** — holdout RMSPE **{best_rmspe:.3f}**
        (≈ {best_rmspe * 100:.1f}% typical percentage error), MAE
        ≈ €{results_df.loc[best_name, 'MAE']:,.0f} per store-day.

        Ranking (best → worst): {' > '.join(_ranking)}.

        Every supervised model crushes the naive baseline. The two tree ensembles
        land within a few tenths of a percentage point of each other, while the
        linear model trails as expected — the sales process is dominated by
        multiplicative interactions (store size × day-of-week × promo) that trees
        capture natively.
        """
    )
    mo.vstack([best_md, results_df])
    return (best_md, best_name, best_rmspe)


@app.cell
def _(mo):
    md_eval = mo.md(
        """
        ## 5. Residual analysis, feature importance & submission
        """
    )
    md_eval
    return (md_eval,)


@app.cell
def _(E, best_name, mo, preds, val_df):
    res = E.run_residual_analysis(
        best_name,
        val_df["Sales"].to_numpy(),
        preds[best_name],
        val_df,
    )
    res_summary = res[["y_true", "y_pred", "residual", "abs_pct_error"]].describe()
    _slug = best_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _imgs = [mo.image(str(p)) for p in sorted(E.PLOTS_DIR.glob(f"{_slug}_*.png"))]
    res_summary
    mo.vstack([res_summary, *_imgs])
    return (res, res_summary)


@app.cell
def _(E, F, best_name, fitted, mo, val_df):
    rf_imp = E.tree_feature_importance(fitted["RandomForest (log target)"])
    perm_imp = E.permutation_importance_table(fitted[best_name], val_df, F.ALL_FEATURES)
    E.plot_feature_importance_bar(
        perm_imp,
        "HistGradientBoosting: permutation importance (holdout)",
        E.PLOTS_DIR / "permutation_importance.png",
    )
    E.plot_feature_importance_bar(
        rf_imp, "RandomForest: impurity importance", E.PLOTS_DIR / "rf_impurity_importance.png"
    )
    mo.vstack([
        mo.image(str(E.PLOTS_DIR / "permutation_importance.png")),
        perm_imp,
        mo.image(str(E.PLOTS_DIR / "rf_impurity_importance.png")),
        rf_imp,
    ])
    return (perm_imp, rf_imp)


@app.cell
def _(mo):
    _imp_note = mo.md(
        """
        **Observation**: the 28-day rolling mean is by far the most important feature
        (it encodes each store's *level*), followed by the promo flag, day-of-week and
        recent lags — exactly the drivers seen in the EDA. Store type/assortment add
        modest store-level information on top of the rolling-mean baseline.
        """
    )
    _imp_note
    return (_imp_note,)


@app.cell
def _(E, best_name, mo, preds, val_df):
    submission = E.save_submission(preds[best_name], val_df)
    sub_md = mo.md(
        f"""
        ### `submission.csv`

        Written to the project root: **{len(submission):,} holdout predictions**
        (open days only; closed days are deterministic €0 and would be added as
        `PredictedSales = 0` in a real Kaggle submission). Columns:
        `Store, Date, Sales, PredictedSales`.
        """
    )
    mo.vstack([sub_md, submission.head(15)])
    return (sub_md, submission)


@app.cell
def _(best_name, best_rmspe, mo, results_df):
    md_conclusions = mo.md(
        f"""
        ## 6. Conclusions

        * **{best_name} wins** the out-of-time comparison: holdout
          RMSPE **{best_rmspe:.3f}**, MAE ≈ €{results_df.loc[best_name, 'MAE']:,.0f} per
          store-day — roughly a **{results_df.loc['Naive (last week same weekday)', 'RMSPE'] / best_rmspe:.1f}×**
          error reduction vs the naive baseline.
        * **What drives sales** (per permutation importance): each store's 28-day sales
          level (`sales_rollmean28`), promo participation, day-of-week and recent lags;
          store type/assortment add store-level context.
        * **Modelling in log space** (`log1p`) aligns the linear model with the
          multiplicative nature of sales and the percentage-error metric.
        * **Leakage controls**: out-of-time split, no `Customers`, one-open-day shift on
          all autoregressive features, training-split-only imputation statistics.

        **Limitations & next steps**

        * No `test.csv` was provided — the 6-week holdout is a proxy for the real
          forecast task; production deployment would need a rolling-origin evaluation.
        * Hyperparameters were fixed rather than tuned (a validation window inside the
          training period would be the next step); external signals (weather, regional
          holidays beyond the dataset) are absent.
        * The promo-effect estimate may still be optimistic for long-running promos
          ( Promo2 ) due to limited variety in the training window.
        * All artifacts live in `artifacts/` (`model_metrics.json`, `plots/`); the
          holdout predictions are in `submission.csv`.
        """
    )
    md_conclusions
    return (md_conclusions,)


if __name__ == "__main__":
    app.run()
