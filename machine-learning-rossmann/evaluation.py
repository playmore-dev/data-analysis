"""
evaluation.py
==============
Metrics, model comparison, residual diagnostics and artifact helpers for the
Rossmann sales forecasting project.

Metrics
-------
* RMSPE (the Kaggle Rossmann competition metric, evaluated on the sales scale)
* MAE  - average absolute error in daily sales per store
* RMSE - punishes large errors; useful alongside MAE
* MAPE - relative error, excluded here on purpose? No: reported but caveated,
  because open days with zero sales (54 rows) make MAPE explode; we report
  MAPE excluding exact zeros and say so explicitly.

All artifacts are written under ``artifacts/`` and the final holdout
predictions are saved as ``submission.csv`` in the project root.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering for scripts and notebook exports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

import models as M

ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared percentage error (Kaggle Rossmann metric)."""
    return float(np.sqrt(np.mean(((y_true - y_pred) / np.maximum(y_true, 1e-9)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE excluding exact zero true sales (54 such rows exist among open days;
    they would otherwise make the metric explode)."""
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSPE": rmspe(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%) (zero-sales excluded)": mape(y_true, y_pred),
    }


# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------

def prepare_baseline_column(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the naive-baseline fallback (per-store median sales on open days
    in the training window) to the holdout frame."""
    med = train_df.groupby("Store")["Sales"].median()
    out = val_df.copy()
    out["naive_store_median"] = out["Store"].map(med)
    return out


def compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Fit every candidate model and evaluate on the out-of-time holdout.

    Returns (results table, fitted pipelines, holdout predictions per model).
    """
    val_df = prepare_baseline_column(train_df, val_df)
    y_true = val_df["Sales"].to_numpy()

    model_factories: dict[str, Callable[[], Pipeline | None]] = {
        "Naive (last week same weekday)": lambda: None,
        "Ridge (log target)": lambda: M.make_linear_log(numeric_features, categorical_features),
        "RandomForest (log target)": lambda: M.make_random_forest(numeric_features, categorical_features),
        "HistGradientBoosting (log target)": lambda: M.make_hist_gb(numeric_features, categorical_features),
    }

    results: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    preds: dict[str, np.ndarray] = {}

    for name, factory in model_factories.items():
        if name.startswith("Naive"):
            y_pred = M.naive_baseline_predictions(val_df)
        else:
            pipe = fit_with_timing(factory(), train_df, features)
            fitted[name] = pipe
            y_pred = M.predict_sales(pipe, val_df, features)
        preds[name] = y_pred
        row = {"model": name, **evaluate(y_true, y_pred)}
        results.append(row)
        print(f"[compare_models] {name}: RMSPE={row['RMSPE']:.4f}  MAE={row['MAE']:.2f}")

    results_df = pd.DataFrame(results).set_index("model")
    return results_df, fitted, preds


def fit_with_timing(pipe: Pipeline, train_df: pd.DataFrame, features: list[str]) -> Pipeline:
    """Fit a pipeline on log1p(Sales) and print how long it took."""
    import time

    t0 = time.perf_counter()
    M.fit_pipeline(pipe, train_df, features)
    dt = time.perf_counter() - t0
    print(f"[fit] {pipe.steps[-1][0]!r} fitted in {dt:.1f}s on {len(train_df):,} rows")
    return pipe


# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------

def get_feature_names(pipe: Pipeline) -> list[str]:
    prep = pipe.named_steps["prep"]
    return list(prep.get_feature_names_out())


def tree_feature_importance(pipe: Pipeline, top: int = 20) -> pd.DataFrame:
    """Impurity-based importances for a fitted RandomForest pipeline."""
    model = pipe.named_steps["model"]
    names = get_feature_names(pipe)
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


def permutation_importance_table(
    pipe: Pipeline,
    val_df: pd.DataFrame,
    features: list[str],
    sample_size: int = 15_000,
    n_repeats: int = 3,
    seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Permutation importance on the holdout for a fitted pipeline, computed on
    a random sample of holdout rows and measured by the drop in R^2 on the
    log1p(Sales) target."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(val_df), size=min(sample_size, len(val_df)), replace=False)
    sample = val_df.iloc[idx]
    y = np.log1p(sample["Sales"].to_numpy())

    r = permutation_importance(
        pipe, sample[features], y, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs
    )
    imp = pd.DataFrame(
        {"feature": features, "importance_mean": r.importances_mean,
         "importance_std": r.importances_std}
    )
    return imp.sort_values("importance_mean", ascending=False).head(20).reset_index(drop=True)


# --------------------------------------------------------------------------
# Residual analysis & plots
# --------------------------------------------------------------------------

def residual_frame(y_true: np.ndarray, y_pred: np.ndarray, val_df: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true,
            "abs_pct_error": np.abs((y_pred - y_true) / np.maximum(y_true, 1e-9)),
        },
        index=val_df.index,
    )
    res["DayOfWeek"] = val_df["DayOfWeek"]
    res["Month"] = val_df["Month"]
    res["Promo"] = val_df["Promo"]
    res["StoreType"] = val_df["StoreType"]
    res["sales"] = y_true
    return res


def plot_pred_vs_actual(res: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = res.sample(n=min(10_000, len(res)), random_state=42)
    ax.scatter(sample["y_true"], sample["y_pred"], s=4, alpha=0.25)
    lim = max(sample["y_true"].max(), sample["y_pred"].max()) * 1.02
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residual_hist(res: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res["residual"], bins=80)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Residual (predicted - actual sales)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residual_by_group(res: pd.DataFrame, group: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    grp = res.groupby(group)["abs_pct_error"].mean()
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_ylabel("Mean absolute percentage error")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance_bar(imp: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    col = "importance" if "importance" in imp.columns else "importance_mean"
    show = imp.iloc[::-1]
    ax.barh(show["feature"], show[col])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_residual_analysis(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Produce all diagnostic plots for a model and return the residual frame."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    res = residual_frame(y_true, y_pred, val_df)

    plot_pred_vs_actual(res, f"{model_name}: predicted vs actual", PLOTS_DIR / f"{slug}_pred_vs_actual.png")
    plot_residual_hist(res, f"{model_name}: residual distribution", PLOTS_DIR / f"{slug}_residual_hist.png")
    plot_residual_by_group(res, "DayOfWeek", f"{model_name}: MAPE by day of week", PLOTS_DIR / f"{slug}_mape_by_dow.png")
    plot_residual_by_group(res, "Month", f"{model_name}: MAPE by month", PLOTS_DIR / f"{slug}_mape_by_month.png")
    plot_residual_by_group(res, "StoreType", f"{model_name}: MAPE by store type", PLOTS_DIR / f"{slug}_mape_by_storetype.png")
    return res


# --------------------------------------------------------------------------
# Submission & artifacts
# --------------------------------------------------------------------------

def save_submission(
    preds: np.ndarray, val_df: pd.DataFrame, path: Path = Path("submission.csv")
) -> pd.DataFrame:
    """Write holdout predictions as a submission-style CSV
    (Store, Date, Sales, PredictedSales)."""
    sub = pd.DataFrame(
        {
            "Store": val_df["Store"].to_numpy(),
            "Date": val_df["Date"].dt.strftime("%Y-%m-%d"),
            "Sales": val_df["Sales"].to_numpy(),
            "PredictedSales": np.round(preds, 1),
        }
    ).sort_values(["Date", "Store"])
    sub.to_csv(path, index=False)
    return sub


def save_metrics_json(
    results_df: pd.DataFrame, extra: dict | None = None, path: Path = ARTIFACTS_DIR / "model_metrics.json"
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"models": results_df.reset_index().to_dict(orient="records")}
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
