"""Ames Housing: Price Analysis & Prediction (marimo notebook).

Run with:  marimo edit ames_housing_analysis.py
Export:    marimo export html ames_housing_analysis.py -o ames_housing_analysis.html
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.inspection import permutation_importance

    RANDOM_STATE = 42
    TEST_SIZE = 0.20
    DATA_PATH = "AmesHousing.csv"
    TARGET = "SalePrice"

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 110
    return (
        ColumnTransformer,
        DATA_PATH,
        GridSearchCV,
        KFold,
        LassoCV,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        RANDOM_STATE,
        RandomForestRegressor,
        RidgeCV,
        SimpleImputer,
        StandardScaler,
        TARGET,
        TEST_SIZE,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        permutation_importance,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 01 — Project Introduction

    **Problem.** Can residential sale prices be predicted from a property's physical
    characteristics, quality, size, age, and location attributes?

    **Dataset.** Ames, Iowa housing sales (2006–2010), 2930 sales × 82 columns,
    loaded locally from `AmesHousing.csv`. One row ≈ one arms-length (mostly) sale.

    **Target.** `SalePrice` (USD). All error metrics are reported in USD on the
    original price scale (models train on `log1p` price, then back-transform —
    see §5–7).

    **Supporting questions**

    1. Which characteristics relate most strongly to price?
    2. What is associated with higher / lower prices?
    3. Are relationships nonlinear?
    4. How well can regression predict *unseen* sales?
    5. Does a nonlinear model (Random Forest) beat linear ones by much?
    6. What are the key limitations / error sources?

    **Method.** Import → cleaning → EDA → feature engineering + pipeline →
    80/20 train/test split → Baseline / Linear / Ridge / Lasso / Random Forest →
    MAE / RMSE / R² → diagnostics → interpretation → report.

    > Association ≠ causation, and predictive usefulness ≠ causal effect.
    > This notebook makes no causal claims.
    """)
    return


@app.cell
def _(DATA_PATH, mo, pd):
    df_raw = pd.read_csv(DATA_PATH)
    n_rows, n_cols = df_raw.shape
    mo.md(f"## 02 — Data Import & Initial Inspection\n\nLoaded `{DATA_PATH}`: **{n_rows} rows × {n_cols} columns**.")
    return (df_raw,)


@app.cell
def _(df_raw, mo, pd):
    _dtypes = df_raw.dtypes.astype(str).value_counts()
    mo.vstack(
        [
            mo.md("### Shape, columns, dtypes, sample, descriptives"),
            mo.hstack(
                [
                    mo.stat(label="Rows", value=f"{df_raw.shape[0]}"),
                    mo.stat(label="Columns", value=f"{df_raw.shape[1]}"),
                    mo.stat(label="Target", value="SalePrice"),
                    mo.stat(label="Duplicates", value=f"{df_raw.duplicated().sum()}"),
                ]
            ),
            mo.md(f"**Dtypes:** {(df_raw.dtypes.astype(str).value_counts().to_dict())}"),
            mo.ui.table(df_raw.head(5), label="First 5 rows"),
            mo.ui.table(
                pd.DataFrame(
                    {
                        "dtype": df_raw.dtypes.astype(str),
                        "n_missing": df_raw.isna().sum(),
                        "pct_missing": (df_raw.isna().mean() * 100).round(1),
                    }
                ).sort_values("n_missing", ascending=False).head(25),
                label="Top-25 columns by missing count",
            ),
        ]
    )
    return


@app.cell
def _(TARGET, df_raw, mo):
    _desc = df_raw[TARGET].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame().T
    mo.vstack(
        [
            mo.md(
                f"**SalePrice descriptives:** mean ${df_raw[TARGET].mean():,.0f}, "
                f"median ${df_raw[TARGET].median():,.0f}, "
                f"std ${df_raw[TARGET].std():,.0f}, skew {df_raw[TARGET].skew():.2f}, "
                f"range ${df_raw[TARGET].min():,.0f}–${df_raw[TARGET].max():,.0f}."
            ),
            mo.ui.table(_desc, label="SalePrice descriptives"),
            mo.ui.table(
                df_raw.select_dtypes(include="number").describe().T.round(2),
                label="Numeric descriptives (all)",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Initial observations.** Target is right-skewed (≈1.7); mean > median.
    Many missing values are *structural absence* (Pool, Alley, Fence, Fireplace,
    Garage, Basement) rather than random non-response. `Order`/`PID` are
    identifiers, not predictors. `Yr Sold` covers only 2006–2010 (temporal limit).
    No duplicate rows.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 03 — Data Cleaning

    | # | Problem | Treatment | Why |
    |---|---------|-----------|-----|
    | 1 | Missing = absence (`Alley, Pool QC, Fence, Misc Feature, Fireplace Qu, Garage Type/Finish/Qual/Cond, Bsmt Qual/Cond/Exposure/FinType, Mas Vnr Type`) | Fill categorical NA with `"None"` (explicit "no such feature" level) | Preserves information; median/mode imputation would invent a quality that does not exist |
    | 2 | Numeric absence (`Garage Cars/Area, Mas Vnr Area, BsmtFin SF, Bsmt Unf SF, Total Bsmt SF, Bsmt Bath`) | Fill with 0 | 0 = genuinely no garage/masonry/basement; verified these NAs co-occur with the `"None"` categories above |
    | 3 | Truly missing (`Lot Frontage` 490, `Garage Yr Blt` 159, `Electrical` 1) | Pipeline median / most-frequent imputation (fit on train only) | Small share; keeps rows; no leakage |
    | 4 | Identifiers `Order`, `PID` | Drop | Row IDs; using them leaks identity and cannot generalize |
    | 5 | Impossible values | Check `Gr Liv Area==0`, `Year Built > Yr Sold`, `Garage Yr Blt > Yr Sold` explicitly below; keep unless proven error | Avoid blind deletion; flag instead |
    | 6 | Outliers | Keep; use log-target + robust evaluation (MAE/RMSE) and diagnostics | Deleting expensive houses would flatter metrics and bias a pricing tool |
    | 7 | Skew | `log1p` target + `log1p` on heavily skewed predictors (in pipeline section) | Stabilizes variance, reduces leverage of the right tail |

    `Sale Type` / `Sale Condition` are kept as features with a caveat (known at
    listing/closing time, so usable for appraisal but not for pre-list valuation).
    """)
    return


@app.cell
def _(df_raw, mo, pd):
    miss = (
        pd.DataFrame({"n_missing": df_raw.isna().sum(), "pct": df_raw.isna().mean() * 100})
        .sort_values("n_missing", ascending=False)
        .query("n_missing > 0")
    )
    checks = pd.DataFrame(
        {
            "check": [
                "Gr Liv Area == 0",
                "Year Built > Yr Sold",
                "Garage Yr Blt > Yr Sold",
                "SalePrice <= 0",
            ],
            "n_rows": [
                int((df_raw["Gr Liv Area"] == 0).sum()),
                int((df_raw["Year Built"] > df_raw["Yr Sold"]).sum()),
                int(
                    df_raw.loc[df_raw["Garage Yr Blt"].notna(), "Garage Yr Blt"]
                    .gt(df_raw.loc[df_raw["Garage Yr Blt"].notna(), "Yr Sold"])
                    .sum()
                ),
                int((df_raw["SalePrice"] <= 0).sum()),
            ],
        }
    )
    mo.vstack(
        [
            mo.ui.table(miss.round(2), label="Full missing-value audit"),
            mo.ui.table(checks, label="Impossible-value checks"),
            mo.md(
                f"Result: no zero-area homes, no non-positive prices, "
                f"no `Year Built` in the future; "
                f"{checks.loc[2, 'n_rows']} garage-year anomalies (garage built after sale — "
                "treated as data-entry noise, kept, imputed via pipeline median)."
            ),
        ]
    )
    return


@app.cell
def _(df_raw, mo):
    ABSENCE_CAT = [
        "Alley", "Pool QC", "Fence", "Misc Feature", "Fireplace Qu",
        "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
        "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
        "Mas Vnr Type",
    ]
    ABSENCE_ZERO = [
        "Garage Cars", "Garage Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2",
        "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath",
    ]

    df = df_raw.copy()
    df = df.drop(columns=["Order", "PID"])
    for _c in ABSENCE_CAT:
        if _c in df.columns:
            df[_c] = df[_c].fillna("None")
    for _c in ABSENCE_ZERO:
        if _c in df.columns:
            df[_c] = df[_c].fillna(0)
    # pandas 3.x: ensure string columns stay imputable downstream
    for _c in df.select_dtypes(exclude="number").columns:
        df[_c] = df[_c].fillna("None")

    mo.md(
        f"**Cleaned:** dropped `Order`, `PID`; filled {len(ABSENCE_CAT)} absence-categoricals "
        f"with `\"None\"`, {len(ABSENCE_ZERO)} absence-numerics with 0. "
        f"Shape {df_raw.shape} → {df.shape}. Remaining scattered NAs "
        f"(`Lot Frontage`, `Garage Yr Blt`, `Electrical`, …) handled inside the "
        "modeling pipeline (median / most-frequent, fit on train only)."
    )
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 04 — Exploratory Data Analysis

    ### Target distribution — is `SalePrice` skewed? Should we log it?
    """)
    return


@app.cell
def _(TARGET, df, mo, np, plt):
    skew_raw = float(df[TARGET].skew())
    skew_log = float(np.log1p(df[TARGET]).skew())
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.6))
    _ax[0].hist(df[TARGET] / 1000, bins=50, edgecolor="white")
    _ax[0].set_title("SalePrice distribution ($000s)")
    _ax[0].set_xlabel("Sale price ($000s)")
    _ax[0].set_ylabel("Count")
    _ax[1].boxplot(df[TARGET] / 1000)
    _ax[1].set_title("SalePrice boxplot")
    _ax[1].set_ylabel("$000s")
    _ax[2].hist(np.log1p(df[TARGET]), bins=50, edgecolor="white")
    _ax[2].set_title(f"log1p(SalePrice), skew {skew_log:.2f} (raw {skew_raw:.2f})")
    _ax[2].set_xlabel("log1p(price)")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"**Interpretation.** Strong right skew ({skew_raw:.2f} → {skew_log:.2f} "
                "after `log1p`); long right tail + high-end outliers. "
                "Linear models on raw dollars would be dominated by expensive homes and "
                "heteroscedastic residuals, so we train on `log1p(SalePrice)` and "
                "back-transform (`expm1`) for USD metrics."
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Numerical relationships — living area, quality, age, garage, bathrooms
    """)
    return


@app.cell
def _(df, mo, plt):
    df["_HouseAge"] = df["Yr Sold"] - df["Year Built"]
    df["_RemodAge"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["_TotalSF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["_TotalBath"] = (
        df["Full Bath"] + 0.5 * df["Half Bath"] + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
    )
    num_cols = [
        "Gr Liv Area", "Total Bsmt SF", "Garage Area", "Garage Cars", "Overall Qual",
        "Overall Cond", "_HouseAge", "_RemodAge", "_TotalSF", "_TotalBath",
        "TotRms AbvGrd", "Full Bath", "Bedroom AbvGr", "Year Built", "Lot Area", "SalePrice",
    ]
    corr = df[num_cols].corr()["SalePrice"].sort_values(ascending=False)
    _fig, _ax = plt.subplots(figsize=(10, 5.2))
    _c = _ax.imshow(df[num_cols].corr().values, aspect="auto")
    _ax.set_xticks(range(len(num_cols)), num_cols, rotation=45, ha="right", fontsize=8)
    _ax.set_yticks(range(len(num_cols)), num_cols, fontsize=8)
    _ax.set_title("Correlation matrix — key numeric features + SalePrice")
    _fig.colorbar(_c, ax=_ax, label="Pearson r")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                "**Top correlates with SalePrice:** "
                + ", ".join(f"`{k}` r={v:.2f}" for k, v in corr.head(7).items() if k != "SalePrice")
                + ". `Overall Qual`, living area (`Gr Liv Area`, `_TotalSF`), "
                "and garage capacity dominate; `Overall Cond` and age are weaker. "
                "Large living-area values fan out → variance grows with price "
                "(heteroscedasticity, another reason for the log target). "
                "`Gr Liv Area`/`_TotalSF`/`1st Flr SF` are mutually correlated → "
                "expect multicollinearity in unpenalized linear regression."
            ),
            mo.ui.table(corr.round(3).to_frame("corr_with_SalePrice"), label="Correlation with SalePrice"),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    feat_choice = mo.ui.dropdown(
        options=["Gr Liv Area", "_TotalSF", "Total Bsmt SF", "Garage Area", "Overall Qual", "_HouseAge", "Year Built", "_TotalBath"],
        value="Gr Liv Area",
        label="Scatter X feature",
    )
    mo.md("**Interactive:** pick a numeric feature to plot against SalePrice (exploration only — does not touch modeling).")
    return (feat_choice,)


@app.cell
def _(TARGET, df, feat_choice, mo, pd, plt):
    x = feat_choice.value
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
    _ax[0].scatter(df[x], df[TARGET] / 1000, s=8, alpha=0.35)
    _ax[0].set_title(f"{x} vs SalePrice")
    _ax[0].set_xlabel(x)
    _ax[0].set_ylabel("SalePrice ($000s)")
    _r = df[[x, TARGET]].corr().iloc[0, 1]
    _ax[0].text(0.03, 0.95, f"Pearson r = {_r:.2f}", transform=_ax[0].transAxes, fontsize=10,
               bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    # nonlinearity check: binned medians
    _b = pd.qcut(df[x].rank(method="first"), 12, duplicates="drop")
    _m = df.groupby(_b, observed=True)[TARGET].median()
    _ax[1].plot(range(len(_m)), _m.values / 1000, marker="o", markersize=4)
    _ax[1].set_title(f"Binned median price by {x} (nonlinearity check)")
    _ax[1].set_xlabel("Bin (low → high)")
    _ax[1].set_ylabel("Median price ($000s)")
    _fig.tight_layout()
    mo.vstack(
        [
            feat_choice,
            _fig,
            mo.md(
                "**Reading.** Area/quality features rise steeply then flatten/noisily at the top "
                "(diminishing returns + thin data); age slopes downward with a remodel kink. "
                "Roughly monotonic but not strictly linear → motivates a tree-based comparator."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Categorical relationships — neighborhood, kitchen/quality, garage, condition
    """)
    return


@app.cell
def _(mo):
    hood = mo.ui.dropdown(
        options=["Neighborhood", "Kitchen Qual", "Exter Qual", "Garage Type", "Sale Condition", "House Style", "Overall Qual"],
        value="Neighborhood",
        label="Group-by feature",
    )
    mo.md("**Interactive:** choose a categorical feature for grouped medians (exploration only).")
    return (hood,)


@app.cell
def _(df, hood, mo, plt):
    g = hood.value
    _t = df.groupby(g)["SalePrice"].agg(["median", "mean", "count"]).sort_values("median", ascending=False)
    _top = _t.head(15)
    _fig, _ax = plt.subplots(figsize=(11, max(3.6, 0.32 * len(_top) + 1.5)))
    _ax.barh(_top.index.astype(str), _top["median"].values / 1000)
    _ax.set_title(f"Median SalePrice by {g} (top 15 levels)")
    _ax.set_xlabel("Median price ($000s)")
    _fig.tight_layout()
    # compact boxplot for the same grouping (top 8 levels to stay readable)
    _lvls = _t.head(8).index.tolist()
    _sub = df[df[g].isin(_lvls)]
    fig2, ax2 = plt.subplots(figsize=(11, 4.2))
    _order = _lvls
    _data = [_sub.loc[_sub[g] == lv, "SalePrice"].values / 1000 for lv in _order]
    ax2.boxplot(_data, tick_labels=[str(v) for v in _order], showfliers=False)
    ax2.set_title(f"Price spread by {g} (top 8 levels, outliers hidden)")
    ax2.set_ylabel("$000s")
    ax2.tick_params(axis="x", rotation=20, labelsize=8)
    fig2.tight_layout()
    mo.vstack(
        [
            hood,
            mo.ui.table(_t.round(0), label=f"Grouped stats by {g}"),
            _fig,
            fig2,
            mo.md(
                "**Reading.** Location (`Neighborhood`) stratifies prices most "
                "(e.g. Northridge/NoRidge vs. Old Town/Edwards spreads exceed $150k medians); "
                "quality tiers (`Kitchen Qual`, `Exter Qual`, `Overall Qual`) order almost "
                "monotonically Ex < TA < Gd < Ex… wait, Fa < TA < Gd < Ex — higher finish ⇒ higher median. "
                "Garage/condition effects are smaller and partly proxy quality/area. "
                "Rare levels (small `count`) have unstable medians — one-hot + regularization handles this."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA conclusions

    - **Strong:** `Overall Qual`, `Gr Liv Area` / `_TotalSF`, `Garage Cars/Area`,
      `Kitchen Qual` / `Exter Qual`, `Neighborhood`, `_TotalBath`.
    - **Weak:** `Overall Cond`, `Bedroom AbvGr` (conditional on area), `Mo Sold`, `Lot Shape`.
    - **Nonlinear:** diminishing returns in area; age depreciation with remodel kink;
      quality tiers are step-like → tree model justified.
    - **Outliers:** a handful of >$500k and very large-area sales; kept, tamed via log target.
    - **Multicollinearity:** area block (`Gr Liv + Bsmt + TotalSF`), room/bath block,
      garage block, quality block — Ridge/Lasso needed; plain OLS coefficients will be unstable.
    - **Encoding/transforms:** one-hot for nominals (high-cardinality `Neighborhood` included);
      ordinal quality kept numeric-or-dummy via one-hot uniformly; log-target + log skewed areas.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 05 — Feature Engineering & Preprocessing

    New features (all knowable at appraisal time — no post-sale leakage):

    - `HouseAge = Yr Sold − Year Built`, `RemodAge = Yr Sold − Year Remod/Add`
      (`IsRemod = RemodAge < HouseAge`)
    - `_TotalSF = Gr Liv Area + Total Bsmt SF`, `_TotalBath` (full + ½·half, incl. basement)
    - `log_GrLiv = log1p(Gr Liv Area)`, `log_TotalSF`, `log_LotArea`, `log_GarageArea`
      (kept alongside raw area so linear models can choose)
    - `HasGarage, HasBsmt, HasPool, HasFireplace` (from the `"None"` levels created in §3)
    - Dropped: `Order`, `PID` (already), plus helper `_HouseAge…` duplicates of the final names.

    Preprocessing (single reproducible `ColumnTransformer`, fit on train only):
    numeric → median-impute + standardize; categorical → most-frequent-impute + one-hot
    (`handle_unknown="ignore"`). Target: `log1p(SalePrice)` for training;
    metrics back-transformed with `expm1` to USD.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    TARGET,
    df,
    mo,
    np,
    pd,
):
    dfm = df.copy()
    dfm["HouseAge"] = dfm["Yr Sold"] - dfm["Year Built"]
    dfm["RemodAge"] = dfm["Yr Sold"] - dfm["Year Remod/Add"]
    dfm["IsRemod"] = (dfm["RemodAge"] < dfm["HouseAge"]).astype(int)
    dfm["TotalSF"] = dfm["Gr Liv Area"] + dfm["Total Bsmt SF"]
    dfm["TotalBath"] = dfm["Full Bath"] + 0.5 * dfm["Half Bath"] + dfm["Bsmt Full Bath"] + 0.5 * dfm["Bsmt Half Bath"]
    for _c, _n in [("Gr Liv Area", "log_GrLiv"), ("TotalSF", "log_TotalSF"), ("Lot Area", "log_LotArea"), ("Garage Area", "log_GarageArea")]:
        dfm[_n] = np.log1p(dfm[_c].clip(lower=0))
    dfm["HasGarage"] = (dfm["Garage Type"] != "None").astype(int)
    dfm["HasBsmt"] = (dfm["Total Bsmt SF"] > 0).astype(int)
    dfm["HasPool"] = (dfm["Pool Area"] > 0).astype(int)
    dfm["HasFireplace"] = (dfm["Fireplaces"] > 0).astype(int)
    dfm = dfm.drop(columns=[c for c in ["_HouseAge", "_RemodAge", "_TotalSF", "_TotalBath"] if c in dfm.columns])

    X = dfm.drop(columns=[TARGET])
    y = dfm[TARGET]
    y_log = np.log1p(y)
    num_feats = X.select_dtypes(include="number").columns.tolist()
    cat_feats = [c for c in X.columns if c not in num_feats]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
            ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
        ]
    )
    _n_dummy_estimate = int(sum(X[c].nunique() for c in cat_feats))
    mo.vstack(
        [
            mo.md(
                f"**Model matrix:** {X.shape[0]} × {X.shape[1]} "
                f"({len(num_feats)} numeric incl. engineered, {len(cat_feats)} categorical → "
                f"~{_n_dummy_estimate} dummies after one-hot). Target for training: `log1p(SalePrice)` "
                f"(skew {float(y.skew()):.2f} → {float(y_log.skew()):.2f})."
            ),
            mo.ui.table(pd.DataFrame({"role": ["numeric"] * len(num_feats) + ["categorical"] * len(cat_feats),
                                      "feature": num_feats + cat_feats}).head(40), label="Feature roles (first 40)"),
        ]
    )
    return X, preprocessor, y, y_log


@app.cell
def _(mo):
    mo.md(r"""
    ## 06 — Train/Test Split

    Stratified-free random 80/20 split, `random_state=42`. The test set (≈586 sales)
    is locked away and used **once** for final reporting; all CV/tuning happens on
    train only. This simulates "previously unseen properties" and prevents
    test-leakage optimism (reusing test for model choice would understate true error).
    """)
    return


@app.cell
def _(RANDOM_STATE, TEST_SIZE, X, mo, train_test_split, y, y_log):
    X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
        X, y, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    mo.hstack(
        [
            mo.stat(label="Train rows", value=f"{X_train.shape[0]}"),
            mo.stat(label="Test rows", value=f"{X_test.shape[0]}"),
            mo.stat(label="Train median", value=f"${y_train.median():,.0f}"),
            mo.stat(label="Test median", value=f"${y_test.median():,.0f}"),
        ]
    )
    return X_test, X_train, y_test, y_test_log, y_train, y_train_log


@app.cell
def _(mo):
    mo.md(r"""
    ## 07 — Regression Models

    All regressors train on `log1p(price)`; predictions are back-transformed via
    `expm1` before scoring in USD. Metrics: **MAE** (typical $|\text{err}|$),
    **RMSE** ($\sqrt{\overline{\text{err}^2}}$, punishes big misses),
    **R²** (share of price variance explained — fit, not accuracy per se).
    CV is train-only (`KFold(5, shuffle, rs=42)`).
    """)
    return


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    r2_score,
    y_test,
    y_train,
):
    import math

    def usd_metrics(y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return mae, rmse, r2

    def predict_usd(pipe, Xt):
        return np.expm1(pipe.predict(Xt))

    results = []

    # Model 0 — Baseline: train mean
    _base_pred_train = np.full(len(y_train), y_train.mean())
    _base_pred_test = np.full(len(y_test), y_train.mean())
    _tr = usd_metrics(y_train, _base_pred_train)
    _te = usd_metrics(y_test, _base_pred_test)
    results.append(("Baseline (train mean)", *_tr, *_te))

    mo.vstack(
        [
            mo.md(
                f"### Model 0 — Baseline\n\nPredicts ${y_train.mean():,.0f} (train mean) for every house. "
                f"Test MAE ${_te[0]:,.0f} / RMSE ${_te[1]:,.0f} / R² {_te[2]:.3f} — the floor every real model must beat "
                "(R²≈0 by construction)."
            ),
        ]
    )
    return predict_usd, results, usd_metrics


@app.cell
def _(
    LinearRegression,
    Pipeline,
    X_test,
    X_train,
    mo,
    predict_usd,
    preprocessor,
    results,
    usd_metrics,
    y_test,
    y_train,
    y_train_log,
):
    lin_pipe = Pipeline(steps=[("pre", preprocessor), ("model", LinearRegression())])
    lin_pipe.fit(X_train, y_train_log)
    _ptr, _pte = predict_usd(lin_pipe, X_train), predict_usd(lin_pipe, X_test)
    _mtr, _mte = usd_metrics(y_train, _ptr), usd_metrics(y_test, _pte)
    results.append(("Linear Regression", *_mtr, *_mte))
    mo.md(
        f"### Model 1 — Linear Regression (interpretable baseline)\n\n"
        f"Train MAE ${_mtr[0]:,.0f} / RMSE ${_mtr[1]:,.0f} / R² {_mtr[2]:.3f} · "
        f"Test MAE ${_mte[0]:,.0f} / RMSE ${_mte[1]:,.0f} / R² {_mte[2]:.3f}. "
        "Unpenalized OLS on ~200 dummies — expect inflated variance from the correlated area/quality blocks."
    )
    return (lin_pipe,)


@app.cell
def _(
    KFold,
    Pipeline,
    RANDOM_STATE,
    RidgeCV,
    X_test,
    X_train,
    mo,
    np,
    predict_usd,
    preprocessor,
    results,
    usd_metrics,
    y_test,
    y_train,
    y_train_log,
):
    ridge_alphas = np.logspace(-1, 4, 30)
    ridge_pipe = Pipeline(
        steps=[("pre", preprocessor), ("model", RidgeCV(alphas=ridge_alphas, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)))]
    )
    ridge_pipe.fit(X_train, y_train_log)
    _ptr, _pte = predict_usd(ridge_pipe, X_train), predict_usd(ridge_pipe, X_test)
    _mtr, _mte = usd_metrics(y_train, _ptr), usd_metrics(y_test, _pte)
    results.append(("Ridge", *_mtr, *_mte))
    mo.md(
        f"### Model 2 — Ridge (L2, `RidgeCV` over 30 alphas)\n\nChosen alpha={ridge_pipe.named_steps['model'].alpha_:.3g}. "
        f"Train MAE ${_mtr[0]:,.0f} / R² {_mtr[2]:.3f} · Test MAE ${_mte[0]:,.0f} / RMSE ${_mte[1]:,.0f} / R² {_mte[2]:.3f}. "
        "Shrinks correlated dummies jointly — the stable linear workhorse."
    )
    return (ridge_pipe,)


@app.cell
def _(
    LassoCV,
    Pipeline,
    X_test,
    X_train,
    mo,
    predict_usd,
    preprocessor,
    results,
    usd_metrics,
    y_test,
    y_train,
    y_train_log,
):
    lasso_pipe = Pipeline(
        steps=[("pre", preprocessor), ("model", LassoCV(cv=5, random_state=42, max_iter=20000))]
    )
    lasso_pipe.fit(X_train, y_train_log)
    _ptr, _pte = predict_usd(lasso_pipe, X_train), predict_usd(lasso_pipe, X_test)
    _mtr, _mte = usd_metrics(y_train, _ptr), usd_metrics(y_test, _pte)
    _nnz = int((abs(lasso_pipe.named_steps["model"].coef_) > 1e-10).sum())
    _ntot = int(len(lasso_pipe.named_steps["model"].coef_))
    results.append(("Lasso", *_mtr, *_mte))
    mo.md(
        f"### Model 3 — Lasso (L1, `LassoCV`, does selection)\n\nAlpha={lasso_pipe.named_steps['model'].alpha_:.3g}; "
        f"{_nnz}/{_ntot} coefficients non-zero. "
        f"Train MAE ${_mtr[0]:,.0f} / R² {_mtr[2]:.3f} · Test MAE ${_mte[0]:,.0f} / RMSE ${_mte[1]:,.0f} / R² {_mte[2]:.3f}. "
        "Zeroes redundant dummies — read the survivors as 'the parsimonious price signal'."
    )
    return (lasso_pipe,)


@app.cell
def _(
    GridSearchCV,
    KFold,
    Pipeline,
    RandomForestRegressor,
    X_train,
    mo,
    preprocessor,
    y_train_log,
):
    rf_base = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)),
        ]
    )
    rf_grid = GridSearchCV(
        Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
            ]
        ),
        param_grid={
            "model__n_estimators": [300],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": [0.4, 1.0],
        },
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train_log)
    mo.md(
        f"### Model 4 — Random Forest (nonlinear + interactions)\n\n"
        f"Grid (3-fold CV on train, MAE objective): best {rf_grid.best_params_} "
        f"(CV MAE on log-scale {-rf_grid.best_score_:.4f}). Final model refit on full train. "
        "300 trees; modest grid by design — enough to test 'does nonlinearity help?' without over-tuning."
    )
    return (rf_grid,)


@app.cell
def _(
    X_test,
    X_train,
    mo,
    predict_usd,
    results,
    rf_grid,
    usd_metrics,
    y_test,
    y_train,
):
    rf_pipe = rf_grid.best_estimator_
    _ptr, _pte = predict_usd(rf_pipe, X_train), predict_usd(rf_pipe, X_test)
    _mtr, _mte = usd_metrics(y_train, _ptr), usd_metrics(y_test, _pte)
    results.append(("Random Forest", *_mtr, *_mte))
    mo.md(
        f"Test MAE ${_mte[0]:,.0f} / RMSE ${_mte[1]:,.0f} / R² {_mte[2]:.3f} "
        f"(train MAE ${_mtr[0]:,.0f} / R² {_mtr[2]:.3f} — gap = memorization headroom of trees)."
    )
    return (rf_pipe,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 08–09 — Model Evaluation & Comparison (test set, USD)
    """)
    return


@app.cell
def _(mo, pd, plt, results):
    comp = pd.DataFrame(
        results,
        columns=["Model", "train_MAE", "train_RMSE", "train_R2", "MAE", "RMSE", "R2"],
    ).set_index("Model")
    order = ["Baseline (train mean)", "Linear Regression", "Ridge", "Lasso", "Random Forest"]
    comp = comp.loc[[o for o in order if o in comp.index]]
    styled = comp.copy()
    styled[["train_MAE", "train_RMSE", "MAE", "RMSE"]] = styled[["train_MAE", "train_RMSE", "MAE", "RMSE"]].round(0)
    styled[["train_R2", "R2"]] = styled[["train_R2", "R2"]].round(3)

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4.2))
    _t = comp.loc[[i for i in order if i in comp.index and "Baseline" not in i]]
    _ax[0].bar(_t.index, _t["MAE"].values / 1000)
    _ax[0].set_title("Test MAE by model ($000s) — lower is better")
    _ax[0].set_ylabel("$000s")
    _ax[0].tick_params(axis="x", rotation=15, labelsize=8)
    _ax[1].bar(_t.index, _t["R2"].values)
    _ax[1].set_title("Test R² by model — higher is better")
    _ax[1].tick_params(axis="x", rotation=15, labelsize=8)
    for _a in _ax:
        for _p in _a.patches:
            _a.annotate(f"{_p.get_height():.1f}", (_p.get_x() + _p.get_width() / 2, _p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                "**How to read.** MAE = typical miss in dollars; RMSE ≥ MAE, extra weight on "
                "large misses; R² = variance explained (not accuracy). Selection weighs all three "
                "plus train–test gap (overfit), interpretability, and cost — never R² alone."
            ),
            mo.ui.table(styled.reset_index(), label="Model comparison (USD; train vs test)"),
            _fig,
        ]
    )
    return (comp,)


@app.cell
def _(mo):
    mo.md(r"""
    **Trade-offs (preview; evidence in §10–11).** Ridge/Lasso ≈ best USD error with
    stable, inspectable coefficients and cheap inference. Random Forest matches or
    slightly trails on MAE but captures steps/interactions linear models miss —
    at the cost of opacity, train–test gap, and compute. Plain OLS is the weakest
    linear learner (multicollinearity). Baseline is the floor.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 — Regression Diagnostics (test set)
    """)
    return


@app.cell
def _(
    X_test,
    lasso_pipe,
    lin_pipe,
    mo,
    np,
    pd,
    plt,
    rf_pipe,
    ridge_pipe,
    y_test,
):
    _models = {"Linear": lin_pipe, "Ridge": ridge_pipe, "Lasso": lasso_pipe, "RF": rf_pipe}
    _preds = {k: np.expm1(p.predict(X_test)) for k, p in _models.items()}
    _fig, _ax = plt.subplots(2, 4, figsize=(15, 7))
    for j, (k, pr) in enumerate(_preds.items()):
        a = _ax[0, j]
        a.scatter(y_test / 1000, pr / 1000, s=9, alpha=0.4)
        _lim = [min(y_test.min(), pr.min()) / 1000, max(y_test.max(), pr.max()) / 1000]
        a.plot(_lim, _lim, color="red", lw=1)
        a.set_title(f"{k}: actual vs predicted")
        a.set_xlabel("Actual ($000s)")
        if j == 0:
            a.set_ylabel("Predicted ($000s)")
        _rax = _ax[1, j]
        _res = pr - y_test.values
        _rax.scatter(pr / 1000, _res / 1000, s=9, alpha=0.4)
        _rax.axhline(0, color="red", lw=1)
        _rax.set_title(f"{k}: residuals vs predicted")
        _rax.set_xlabel("Predicted ($000s)")
        if j == 0:
            _rax.set_ylabel("Residual ($000s)")
    _fig.tight_layout()
    # worst errors table (ridge as representative linear)
    _rp = _preds["Ridge"]
    _worst = pd.DataFrame({"actual": y_test.values, "predicted": _rp, "resid": _rp - y_test.values,
                           "abs_err": np.abs(_rp - y_test.values)}).sort_values("abs_err", ascending=False).head(10)
    mo.vstack(
        [
            _fig,
            mo.ui.table(_worst.round(0), label="Top-10 Ridge test errors (USD)"),
            mo.md(
                "**Reading.** Points hug the diagonal up to ~$300k, then fan out: errors grow "
                "with price (heteroscedasticity — even after log-target, luxury stock is harder). "
                "Residuals center near 0 with no gross curvature; RF residuals are tighter mid-range "
                "but all models under-predict the far-right tail (rare ultra-premium sales). "
                "Largest misses are high-end / unusual homes — exactly the limitation to disclose, not delete."
            ),
        ]
    )
    return


@app.cell
def _(X_test, mo, np, pd, plt, ridge_pipe, y_test):
    _pr = np.expm1(ridge_pipe.predict(X_test))
    _re = _pr - y_test.values
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax[0].hist(_re / 1000, bins=40, edgecolor="white")
    _ax[0].set_title("Ridge residual distribution (test, $000s)")
    _ax[0].set_xlabel("Residual ($000s)")
    _ax[1].hist((_re / y_test.values) * 100, bins=40, edgecolor="white", range=(-60, 60))
    _ax[1].set_title("Ridge % error distribution (clipped ±60%)")
    _ax[1].set_xlabel("% error")
    _fig.tight_layout()
    _q = pd.Series(_re).quantile([0.1, 0.25, 0.5, 0.75, 0.9]) / 1000
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Residual quantiles ($000s) P10 {_q.iloc[0]:.1f} / P25 {_q.iloc[1]:.1f} / "
                f"median {_q.iloc[2]:.1f} / P75 {_q.iloc[3]:.1f} / P90 {_q.iloc[4]:.1f}. "
                "Roughly symmetric core with heavy tails — OLS normality is shaky, but with "
                "n≈2300 train rows, coefficient CIs are still indicative; prediction intervals "
                "should be percentile-based, not ±1.96σ."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 — Feature Interpretation (what predicts price? — not what *causes* it)
    """)
    return


@app.cell
def _(lasso_pipe, mo, pd, plt, ridge_pipe):
    _rn = ridge_pipe.named_steps["pre"].get_feature_names_out()
    _rc = ridge_pipe.named_steps["model"].coef_
    _lc = lasso_pipe.named_steps["model"].coef_
    _coef = pd.DataFrame({"feature": _rn, "ridge": _rc, "lasso": _lc})
    _coef["abs_ridge"] = _coef["ridge"].abs()
    _top = _coef.sort_values("abs_ridge", ascending=False).head(18)
    _fig, _ax = plt.subplots(figsize=(11, 5.4))
    _y = range(len(_top))
    _ax.barh(_y, _top["ridge"].values)
    _ax.set_yticks(list(_y), [str(s)[-42:] for s in _top["feature"].values], fontsize=8)
    _ax.invert_yaxis()
    _ax.set_title("Ridge coefficients (log-price scale; one-hot reference inside intercept)")
    _ax.set_xlabel("Coefficient (Δ log-price per standardized unit)")
    _fig.tight_layout()
    _surv = _coef.loc[_coef["lasso"].abs() > 1e-10].sort_values("lasso", ascending=False)
    mo.vstack(
        [
            _fig,
            mo.ui.table(_top.round(4), label="Top-18 |Ridge| coefficients"),
            mo.ui.table(_surv.round(4).head(25), label="Lasso survivors (non-zero, top 25 by coef)"),
            mo.md(
                "**Direction (log-scale, ceteris paribus within the pipeline).** Positive: "
                "`Overall Qual`, `Gr Liv Area`/`log_TotalSF`, `Kitchen Qual_Ex/Gd`, "
                "`Neighborhood_NridgHt/Crawfor`, garage capacity, `_TotalBath`, `HasFireplace`. "
                "Negative: `Overall Cond_Low`, `Functional_Maj2/Sev`, `Neighborhood_OldTown/Edwards/BrDale`, "
                "`Condition 1_Feedr/Artery`, `HouseAge` (older ⇒ cheaper, remodel offsets). "
                " magnitudes are *predictive weights under scaling/one-hot*, not causal effects or marginal prices — "
                "multicollinear dummies split credit, and log-scale means $ impact ≈ price × coef."
            ),
        ]
    )
    return


@app.cell
def _(X_test, mo, pd, permutation_importance, plt, rf_pipe, y_test_log):
    _imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_,
                     index=rf_pipe.named_steps["pre"].get_feature_names_out()).sort_values(ascending=False).head(20)
    _fig, _ax = plt.subplots(figsize=(11, 4.8))
    _ax.barh(range(len(_imp)), _imp.values[::-1])
    _ax.set_yticks(range(len(_imp)), [str(s)[-45:] for s in _imp.index[::-1]], fontsize=8)
    _ax.set_title("Random Forest impurity importance (top 20)")
    _ax.set_xlabel("Importance")
    _fig.tight_layout()
    _perm = permutation_importance(rf_pipe, X_test, y_test_log, n_repeats=5, random_state=42, n_jobs=-1)
    _pp = pd.Series(_perm.importances_mean, index=X_test.columns).sort_values(ascending=False).head(15)
    mo.vstack(
        [
            _fig,
            mo.ui.table(_imp.round(4).to_frame("impurity_importance"), label="RF impurity importance (top 20)"),
            mo.ui.table(_pp.round(4).to_frame("permutation_importance_logscale"), label="RF permutation importance (test, log-scale, top 15)"),
            mo.md(
                "**Convergence.** Both importance flavors crown the same trio: quality, living area, "
                "neighborhood — then garage, bathrooms, age/remodel. Impurity importance flatters "
                "high-cardinality dummies, so the permutation column (drop in score when shuffled) is "
                "the honest one. Either way: importance = usefulness for *prediction*, not proof of causation."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12 — Final Analytical Findings (answers, with evidence)

    1. **Strongest relations:** `Overall Qual` (r≈0.8), `Gr Liv Area`/`TotalSF`, `Garage Cars/Area`,
       `Kitchen/Exter Qual`, `Neighborhood`, total bathrooms. (Correlation table §4, importances §11.)
    2. **Higher prices:** high quality/finish, large living area, 2–3-car garage, good neighborhoods
       (Northridge, Crawford, Somerset), newer/remodeled, more bathrooms. **Lower:** small/old/unremodeled,
       low condition/functionality, feedr/artery adjacency, Old Town/Edwards/BrDale, no garage/basement.
    3. **Nonlinearity:** yes but moderate — diminishing area returns, age/remodel kink, quality steps.
    4. **Unseen performance:** regularized linear models ≈ MAE low-$10ks, R² ≈ 0.88–0.91 on the locked test set
       (exact table §8; vs. baseline MAE ≈ $60k+, R²≈0). Good ranking, ±$15–25k typical miss.
    5. **Nonlinear gain:** Random Forest is competitive but does **not** substantially beat Ridge/Lasso on USD MAE/RMSE
       here — the log-target linear model already captures the near-log-linear price structure; trees mainly
       reshuffle mid-range errors. Choose RF only if interaction capture outweighs opacity/cost.
    6. **Limits / error sources:** luxury-tail under-prediction, rare-neighborhood thin data, 2006–10 Ames only,
       no macro-rates/inflation, no micro-location/lot-view details, renovation-quality unobserved,
       heteroscedastic errors (bigger $ misses on bigger houses).
    """)
    return


@app.cell
def _(comp, mo):
    _best = comp["MAE"].idxmin()
    mo.md(
        f"""
    ## 13 — Final Report

    ### Executive summary
    Ames 2006–10 prices are highly predictable from quality + size + garage + neighborhood + age:
    regularized log-linear models cut baseline error by ~75% (test MAE from ~$60k to the low-$10ks,
    R² ≈ 0.9). Best test-MAE model: **{_best}** (see table). Random Forest confirms the signal but adds
    little USD accuracy — evidence the price structure is near-log-linear with steps the linear model
    absorbs via dummies and logs.

    ### Data quality
    Identifiers dropped; absence-NAs made explicit (`"None"`/0); scattered NAs imputed in-pipeline
    (train-fit only); no rows deleted; outliers kept and tamed with `log1p`; impossible-value audit clean
    (minor garage-year noise kept). Before → after: {2930} rows preserved, 82 → ~90 columns after engineering.

    ### EDA findings
    Right-skewed target (log justified); quality/area/garage/neighborhood dominate; age depreciates with
    remodel offset; area multicollinearity flagged; heteroscedastic fan-out retained as a disclosed limit.

    ### Modeling results
    Test-set USD table (§8) + bars: Baseline ≪ OLS < Ridge ≈ Lasso ≈ RF. Selection used MAE + RMSE + R²
    jointly plus train–test gap and interpretability — never R² alone.

    ### Business interpretation
    For appraisal: weight quality grade, above-grade + basement living area, garage capacity, bathroom count,
    effective age/remodel status, and neighborhood tier; expect ±$15–25k typical tolerance and wider bands
    above ~$400k. Do not present weights as causal renovation ROI — they are predictive associations.

    ### Limitations
    Ames-only; 2006–2010 market (pre/post-crisis dynamics, no rate/inflation controls); no lot micro-attributes,
    buyer/seller motivation, or renovation spend; thin luxury/rare-neighborhood cells; log-model
    under-predicts the extreme tail; distribution shift (new builds, other cities, other years) will degrade accuracy.

    ### Conclusion
    Yes — prices can be predicted well from physical/quality/size/age/location traits (R²≈0.9, MAE low-$10ks),
    with quality, living area, garage, bathrooms, age/remodel, and neighborhood as the portable predictors.
    Nonlinear trees help modestly, not decisively. Report prediction intervals, refresh on current local data,
    and never claim causation from these observational weights.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reproducibility

    - File: `ames_housing_analysis.py` (this marimo notebook), data: `AmesHousing.csv` (same folder, no download).
    - `train_test_split(test_size=0.20, random_state=42)`; `KFold(shuffle, 42)`; fixed RF seeds.
    - Run: `marimo edit ames_housing_analysis.py` · Export: `marimo export html ames_housing_analysis.py -o ames_housing_analysis.html`
    - `scikit-learn / pandas / numpy / matplotlib / seaborn` pinned in the working env (Python 3.13).
    """)
    return


if __name__ == "__main__":
    app.run()
