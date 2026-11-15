"""
features.py
===========
Feature engineering utilities for the Rossmann store sales forecasting project.

Every function here is a pure transformation of a pandas DataFrame and uses ONLY
information available at (or before) the date of each row being predicted, so the
same code path can be applied to the training set and to the out-of-time holdout
without leaking future information.

Design decisions (documented per the project instructions):
  * Modelling frame = open-store days only (Open == 1). Closed days have a
    deterministic sales value of 0 and are predicted trivially.
  * `StateHoliday` arrives as a mixed int/str column ("0" vs 0); it is normalised
    to a consistent string categorical.
  * `Customers` is deliberately NOT used as a feature: it is only available after
    the fact (you cannot know how many customers will show up before they do),
    so using it would leak the target.
  * Lag / rolling features are computed per store over that store's open-day
    history, shifted by one open day so that the current day's sales never enter
    its own features. The assumed forecasting scenario is a daily one-step-ahead
    forecast (yesterday's sales are known), which matches the assignment's
    suggestion of "previous-day sales" as a feature.
  * Sales-scale lag features are log1p-transformed so that a linear model on
    log1p(Sales) sees a multiplicative, approximately linear relationship;
    tree models are unaffected by a monotonic transform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Feature definitions
# --------------------------------------------------------------------------

CALENDAR_FEATURES: list[str] = [
    "Year", "Month", "Day", "WeekOfYear",
    "MonthSin", "MonthCos", "WeekSin", "WeekCos",
    "DayOfWeek",
]

PROMO_FEATURES: list[str] = [
    "Promo", "Promo2", "Promo2Active", "Promo2DurationWeeks",
]

COMPETITION_FEATURES: list[str] = [
    "CompetitionDistance", "CompetitionDistanceMissing",
    "CompetitionOpen", "CompetitionOpenKnown", "CompetitionOpenMonths",
]

LAG_WINDOWS: tuple[int, ...] = (7, 28)
LAGS: tuple[int, ...] = (1, 7, 14, 28)

LAG_FEATURES: list[str] = (
    [f"sales_lag{k}" for k in LAGS]
    + [f"sales_rollmean{w}" for w in LAG_WINDOWS]
    + ["sales_rollstd7", "days_since_last_open", "HasLag1"]
)

NUMERIC_FEATURES: list[str] = (
    CALENDAR_FEATURES + PROMO_FEATURES + COMPETITION_FEATURES + LAG_FEATURES
)

CATEGORICAL_FEATURES: list[str] = ["StoreType", "Assortment", "StateHoliday"]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "Sales"


# --------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw train and store tables."""
    data_dir = Path(data_dir)
    # StateHoliday is a mixed int/str column in the raw file ("0" vs 0) ->
    # force str dtype at load time to avoid the DtypeWarning and surprises.
    train = pd.read_csv(
        data_dir / "train.csv", parse_dates=["Date"], dtype={"StateHoliday": str}
    )
    store = pd.read_csv(data_dir / "store.csv")
    return train, store


def clean_and_merge(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw tables and merge store attributes onto daily records.

    Cleaning steps
    --------------
    1. Normalise ``StateHoliday`` (mixed int 0 / str "0" in the raw file) to a
       consistent string categorical: "0", "a", "b", "c".
    2. Cast store attribute columns to categorical dtype.
    3. Left-merge store attributes onto the daily table (every store key is
       present in both tables, verified upstream).
    """
    train = train.copy()
    store = store.copy()

    train["StateHoliday"] = train["StateHoliday"].astype(str).replace("0", "0")
    # values present: "0", "a", "b", "c" -- keep as plain strings; the one-hot
    # encoder in models.py will fit categories on the training split only.

    store["StoreType"] = store["StoreType"].astype(str)
    store["Assortment"] = store["Assortment"].astype(str)
    store["PromoInterval"] = store["PromoInterval"].fillna("")

    df = train.merge(store, on="Store", how="left", validate="many_to_one")

    if df["StoreType"].isna().any():
        raise ValueError("Store merge produced unmatched Store keys")
    return df


def open_day_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to open-store days (the modelling frame).

    Closed days (Open == 0) always have Sales == 0 and are predicted
    deterministically; they are handled outside the model. Rows that are open
    with zero sales (54 in the raw data) are kept: they are legitimate open
    trading days.
    """
    return df.loc[df["Open"] == 1].copy()


# --------------------------------------------------------------------------
# Calendar / store-static features
# --------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived only from the row's own date."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype("int64")
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["WeekSin"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52.0)
    df["WeekCos"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52.0)
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Competition features from static store attributes.

    * ``CompetitionDistance`` may be missing for a handful of stores -> an
      explicit missing flag is added; the model pipeline median-imputes the
      distance (statistics fitted on the training split only).
    * ``CompetitionOpen`` / ``CompetitionOpenMonths`` measure how long the
      nearest competitor has been open at the row's date. Stores without a
      recorded competition open date get a ``CompetitionOpenKnown`` flag of 0.
    """
    df = df.copy()

    df["CompetitionDistanceMissing"] = df["CompetitionDistance"].isna().astype("int64")

    raw_months = (
        (df["Year"] - df["CompetitionOpenSinceYear"]) * 12
        + (df["Month"] - df["CompetitionOpenSinceMonth"])
    )
    df["CompetitionOpenKnown"] = raw_months.notna().astype("int64")
    df["CompetitionOpenMonths"] = raw_months.fillna(0).clip(lower=0, upper=240)
    df["CompetitionOpen"] = raw_months.ge(0).astype("int64")  # NaN -> 0
    return df


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Promo2 (continuing promotion) participation features.

    ``Promo2Active`` is 1 when the store runs Promo2 and the row's month is one
    of the store's PromoInterval months and the promo start date has passed.
    """
    df = df.copy()

    since_year = df["Promo2SinceYear"].fillna(0)
    since_week = df["Promo2SinceWeek"].fillna(0)
    start = pd.to_datetime(
        {"year": since_year.astype("int64"), "month": 1, "day": 1},
        errors="coerce",
    ) + pd.to_timedelta((since_week.astype("int64") - 1) * 7, unit="D")

    started = df["Date"].ge(start)  # NaT -> False
    month_abbr = df["Date"].dt.strftime("%b")
    in_interval = pd.Series(
        [m in iv.split(",") for m, iv in zip(month_abbr, df["PromoInterval"])],
        index=df.index,
    )

    df["Promo2Active"] = (
        df["Promo2"].eq(1) & started & in_interval
    ).astype("int64")
    df["Promo2DurationWeeks"] = np.where(
        df["Promo2Active"].eq(1),
        ((df["Date"] - start).dt.days // 7),
        0,
    ).astype("int64")
    return df


# --------------------------------------------------------------------------
# Lag / rolling features (store-level time series, open days only)
# --------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, sales_col: str = TARGET) -> pd.DataFrame:
    """Per-store lag and rolling statistics over the store's open-day history.

    The input must already be restricted to open days. Rows are processed per
    store in date order:

    * ``sales_lag{k}``        - sales k open-days earlier
    * ``sales_rollmean{w}``   - mean of the trailing ``w`` open days, shifted by
                                one open day (current day never enters its own
                                features)
    * ``sales_rollstd7``      - std of the trailing 7 open days, shifted by one
    * ``days_since_last_open``- calendar-day gap since the previous open day
    * ``HasLag1``             - flag whether the store has prior open-day
                                history in the frame (0 for the first open day
                                of a store; missing lags are median-imputed
                                downstream on training-split statistics only)

    Early rows of a store have no history -> NaN features, which the model
    pipeline imputes using training-split statistics only (no leakage).
    """
    df = df.sort_values(["Store", "Date"], kind="mergesort").copy()
    g = df.groupby("Store", sort=False)[sales_col]

    for k in LAGS:
        df[f"sales_lag{k}"] = g.shift(k)

    # Keep a raw (untransformed) copy of the one-week lag for the naive
    # "last week same-weekday sales" baseline used in evaluation.py.
    df["naive_lag7_raw"] = df["sales_lag7"]

    shifted = g.shift(1)
    for w in LAG_WINDOWS:
        roll = shifted.groupby(df["Store"]).rolling(w, min_periods=1).mean()
        df[f"sales_rollmean{w}"] = roll.droplevel(0).reindex(df.index)
    rollstd = shifted.groupby(df["Store"]).rolling(7, min_periods=2).std()
    df["sales_rollstd7"] = rollstd.droplevel(0).reindex(df.index)

    df["days_since_last_open"] = (
        df.groupby("Store")["Date"].diff().dt.days
    )
    df["HasLag1"] = df["sales_lag1"].notna().astype("int64")

    # Sales-scale features are log1p-transformed: this makes the relationship
    # with log1p(Sales) approximately linear for the linear model, and is a
    # monotonic (hence split-invariant) change for tree models.
    for col in [f"sales_lag{k}" for k in LAGS] + [f"sales_rollmean{w}" for w in LAG_WINDOWS]:
        df[col] = np.log1p(df[col])
    df["sales_rollstd7"] = np.log1p(df["sales_rollstd7"])

    return df


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame, holdout_weeks: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Out-of-time split: the last ``holdout_weeks`` of the timeline are held
    out; everything before is training data. This mirrors how the model would
    be deployed (train on the past, forecast the future) and avoids the
    leakage inherent to random row-wise splits.
    """
    cutoff = df["Date"].max() - pd.Timedelta(weeks=holdout_weeks) + pd.Timedelta(days=1)
    train = df.loc[df["Date"] < cutoff].copy()
    val = df.loc[df["Date"] >= cutoff].copy()
    return train, val, cutoff


# --------------------------------------------------------------------------
# One-shot assembly
# --------------------------------------------------------------------------

def build_dataset(
    data_dir: str | Path,
    holdout_weeks: int = 6,
) -> dict:
    """Load, clean, engineer features and split the Rossmann data.

    Returns a dict with the merged/cleaned full frame, the open-day modelling
    frame, the train/holdout frames, engineered features and the cutoff date.
    """
    train_raw, store = load_data(data_dir)
    merged = clean_and_merge(train_raw, store)

    merged = add_calendar_features(merged)
    merged = add_competition_features(merged)
    merged = add_promo2_features(merged)

    open_df = open_day_frame(merged)
    open_df = add_lag_features(open_df)

    train, val, cutoff = temporal_split(open_df, holdout_weeks=holdout_weeks)

    return {
        "full": merged,
        "open": open_df,
        "train": train,
        "val": val,
        "cutoff": cutoff,
        "features": ALL_FEATURES,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    }
