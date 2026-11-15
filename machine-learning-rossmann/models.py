"""
models.py
===========
Model definitions and training utilities for the Rossmann sales forecasting
project.

The target is ``log1p(Sales)``; all predictions are inverted back to the sales
scale with ``np.expm1`` before computing metrics. Three model families are
compared, plus a naive seasonal baseline:

1. ``make_naive_baseline``  - "last week same weekday" sales per store (no ML).
2. ``make_linear_log``      - Ridge regression on log1p(Sales) with a
                              ColumnTransformer (impute + scale numerics,
                              one-hot categoricals).
3. ``make_random_forest``   - RandomForestRegressor with the same encoding.
4. ``make_hist_gb``         - HistGradientBoostingRegressor, the likely winner.

All pipelines are wrapped in ``transformed_target_regressor``-style logic
implemented manually: we fit on ``log1p(Sales)`` and invert with ``expm1``, so
there is exactly one transformation path and it is easy to audit.

Hyperparameters are modest on purpose: the dataset is large (over 1.7M rows
including closed days) and the goal is a defensible, fast comparison rather
than a leaderboard submission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build the shared column transformer.

    * Numeric features: median-impute missing values, then (optionally) scale.
      Scaling matters for the linear model; trees are scale-invariant but we
      keep the same encoder for comparability.
    * Categorical features: one-hot encode, with unseen categories in the
      holdout mapped to all-zero rows (``handle_unknown="ignore"``) so the
      pipeline never crashes on the holdout.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
        n_jobs=-1,
    )


def make_linear_log(
    numeric_features: list[str],
    categorical_features: list[str],
    alpha: float = 1.0,
) -> Pipeline:
    """Ridge regression on log1p(Sales)."""
    return Pipeline(
        [
            (
                "prep",
                _make_preprocessor(numeric_features, categorical_features, scale_numeric=True),
            ),
            ("model", Ridge(alpha=alpha, random_state=42)),
        ]
    )


def make_random_forest(
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int = 300,
    max_depth: int = 18,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
) -> Pipeline:
    """Random forest on log1p(Sales)."""
    return Pipeline(
        [
            (
                "prep",
                _make_preprocessor(numeric_features, categorical_features, scale_numeric=False),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=0.5,
                    n_jobs=n_jobs,
                    random_state=42,
                ),
            ),
        ]
    )


def make_hist_gb(
    numeric_features: list[str],
    categorical_features: list[str],
    max_iter: int = 400,
    learning_rate: float = 0.08,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.5,
) -> Pipeline:
    """HistGradientBoostingRegressor on log1p(Sales).

    Trees natively handle NaNs, so the numeric branch still median-imputes for
    pipeline consistency (the imputer is a no-op on complete columns), and
    categoricals are one-hot encoded.
    """
    return Pipeline(
        [
            (
                "prep",
                _make_preprocessor(numeric_features, categorical_features, scale_numeric=False),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
            ),
        ]
    )


# --------------------------------------------------------------------------
# Fitting / prediction helpers
# --------------------------------------------------------------------------

def fit_pipeline(pipe: Pipeline, train_df: pd.DataFrame, features: list[str]) -> Pipeline:
    """Fit the pipeline on log1p(Sales)."""
    pipe.fit(train_df[features], np.log1p(train_df["Sales"]))
    return pipe


def predict_sales(pipe: Pipeline, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Predict on the sales scale (expm1 of the log-space prediction), clipped
    at zero so negative log-space predictions never yield negative sales."""
    log_pred = pipe.predict(df[features])
    return np.maximum(np.expm1(log_pred), 0.0)


def naive_baseline_predictions(val_df: pd.DataFrame) -> np.ndarray:
    """Naive seasonal baseline: for each open holdout day, predict the sales
    the same store recorded on the same weekday one week earlier
    (``sales_lag7`` before the log transform). Missing history -> store median
    sales over the training window.

    This baseline is an important sanity check: any supervised model must beat
    it, otherwise the extra complexity buys nothing.
    """
    pred = val_df["naive_lag7_raw"].copy()
    fallback = val_df["naive_store_median"]
    pred = pred.fillna(fallback).fillna(0.0)
    return np.maximum(pred.to_numpy(), 0.0)
