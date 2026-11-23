"""
Phase 3 figures. Reads ONLY the saved result files (no retraining, nothing recomputed
except simple plotting transforms), so the same files can feed the dashboard later.
Run:  python src/make_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402

MODELS = [("baseline", "Persistence baseline", "#7f7f7f"),
          ("random_forest", "Random Forest (Bagging)", "#2ca02c"),
          ("xgboost", "XGBoost (Boosting)", "#d62728"),
          ("stacking", "Stacking (RF + XGB -> Ridge)", "#1f77b4")]
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def load():
    pred = pd.read_csv(du.RESULTS_DIR / "predictions_validation_test.csv", parse_dates=["date", "next_date"])
    return (pred, pd.read_csv(du.RESULTS_DIR / "metrics_all.csv"),
            pd.read_csv(du.RESULTS_DIR / "feature_importance.csv"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(du.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_actual_vs_pred(pred, split, name, approach="return", zoom=None):
    d = pred[pred.split == split]
    ncols = 2 if zoom else 1
    fig, axes = plt.subplots(1, ncols, figsize=(15 if zoom else 11, 4.6), gridspec_kw={"width_ratios": [2, 1]} if zoom else None)
    axes = np.atleast_1d(axes)
    for ax, sub, title in [(axes[0], d, f"{split.capitalize()} period: actual vs predicted next closing price")] + (
            [(axes[1], d.iloc[-zoom:], f"Zoom: last {zoom} observations")] if zoom else []):
        ax.plot(sub.next_date, sub.actual_next_price, color="black", lw=1.6, label="Actual")
        for key, label, color in MODELS:
            ax.plot(sub.next_date, sub[f"{approach}__{key}__price"], color=color, lw=1.0, alpha=0.85, label=label)
        ax.set_title(title); ax.set_ylabel("Closing price (LKR)"); ax.tick_params(axis="x", rotation=25)
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, name)


def fig_errors(pred, split, name):
    d = pred[pred.split == split]
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (key, label, color) in zip(axes, MODELS):
        err = d.actual_next_price - d[f"return__{key}__price"]
        ax.plot(d.next_date, err, color=color, lw=0.9)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel("Error (LKR)"); ax.set_title(f"{label}   (MAE {err.abs().mean():.3f})", fontsize=9, loc="left")
    axes[-1].set_xlabel("Date predicted"); fig.suptitle(f"Prediction error over time, {split} set (actual - predicted)", y=1.0)
    save(fig, name)


def fig_residuals(pred, split, name):
    d = pred[pred.split == split]
    fig, axes = plt.subplots(2, len(MODELS), figsize=(15, 6.5))
    for j, (key, label, color) in enumerate(MODELS):
        p = d[f"return__{key}__price"]
        r = d.actual_next_price - p
        axes[0, j].hist(r, bins=40, color=color, alpha=0.85)
        axes[0, j].axvline(0, color="black", lw=0.8)
        axes[0, j].set_title(label, fontsize=9)
        axes[0, j].set_xlabel("Residual (LKR)"); axes[0, j].set_ylabel("Count" if j == 0 else "")
        axes[1, j].scatter(p, r, s=6, color=color, alpha=0.6)
        axes[1, j].axhline(0, color="black", lw=0.8)
        axes[1, j].set_xlabel("Predicted price (LKR)"); axes[1, j].set_ylabel("Residual (LKR)" if j == 0 else "")
    fig.suptitle(f"Residual analysis, {split} set: distribution (top) and residual vs predicted price (bottom)", y=1.0)
    save(fig, name)


def fig_metric_bars(metrics, name):
    m = metrics[metrics.approach == "return"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    labels = [l for _, l, _ in MODELS]
    x = np.arange(len(MODELS)); w = 0.38
    for ax, col, title in [(axes[0], "MAE", "MAE (LKR) - lower is better"), (axes[1], "RMSE", "RMSE (LKR) - lower is better"),
                           (axes[2], "DirAcc_excl_unchanged", "Directional accuracy, excl. unchanged days")]:
        for i, (split, hatch) in enumerate([("validation", ""), ("test", "//")]):
            vals = [m[(m.model == l) & (m.split == split)][col].iloc[0] for l in labels]
            vals = [np.nan if v is None else v for v in vals]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, color=[c for *_, c in MODELS], hatch=hatch, edgecolor="white",
                          alpha=0.55 if split == "validation" else 0.95)
            for b_, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}" if col != "DirAcc_excl_unchanged" else f"{v:.1%}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(["Baseline", "RF", "XGBoost", "Stacking"]); ax.set_title(title)
    axes[0].bar([], [], color="grey", alpha=0.55, label="validation (light)"); axes[0].bar([], [], color="grey", hatch="//", label="test (hatched)")
    axes[0].legend(fontsize=8)
    axes[2].axhline(0.5, color="black", lw=0.7, ls="--")
    axes[2].text(0, 0.02, "n/a\n(predicts\nno change)", ha="center", fontsize=8)
    save(fig, name)


def fig_importance(imp, name):
    r = imp[imp.approach == "return"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for i, (kind_key, kind_title) in enumerate([("built-in", "built-in importance"), ("permutation", "permutation importance (validation)")]):
        for j, model in enumerate(["Random Forest (Bagging)", "XGBoost (Boosting)"]):
            s = r[(r.model == model) & r.importance_type.str.startswith(kind_key)].sort_values("importance").tail(12)
            color = "#2ca02c" if j == 0 else "#d62728"
            axes[i, j].barh(s.feature, s.importance, color=color)
            axes[i, j].set_title(f"{model}: {kind_title}", fontsize=9)
    fig.suptitle("Feature importance, return-based models (top 12 features)", y=1.0)
    save(fig, name)


def fig_return_vs_raw(pred, metrics, name):
    t = pred[pred.split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2, 1.2]})
    axes[0].plot(t.next_date, t.actual_next_price, color="black", lw=1.6, label="Actual")
    for key, label, color in MODELS[1:]:
        axes[0].plot(t.next_date, t[f"rawprice__{key}__price"], color=color, lw=1.0, label=f"{label} - raw-price target")
    axes[0].set_title("SECONDARY experiment, test set: models predicting the raw closing price directly")
    axes[0].set_ylabel("Closing price (LKR)"); axes[0].legend(fontsize=8, loc="upper left"); axes[0].tick_params(axis="x", rotation=25)
    mt = metrics[metrics.split == "test"]
    names = [l for _, l, _ in MODELS[1:]]
    x = np.arange(len(names)); w = 0.38
    ret = [mt[(mt.approach == "return") & (mt.model == n)].RMSE.iloc[0] for n in names]
    raw = [mt[(mt.approach == "rawprice") & (mt.model == n)].RMSE.iloc[0] for n in names]
    base = mt[(mt.approach == "return") & (mt.model == "Persistence baseline")].RMSE.iloc[0]
    for off, vals, lab, col in [(-0.5, ret, "log-return target (primary)", "#1f77b4"), (0.5, raw, "raw-price target (secondary)", "#ff7f0e")]:
        bars = axes[1].bar(x + off * w, vals, w, label=lab, color=col)
        for b_, v in zip(bars, vals):
            axes[1].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].axhline(base, color="grey", ls="--", lw=1, label=f"persistence baseline ({base:.2f})")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["RF", "XGBoost", "Stacking"]); axes[1].set_title("Test RMSE (LKR): the two target definitions")
    axes[1].legend(fontsize=8)
    save(fig, name)


def main():
    du.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pred, metrics, imp = load()
    fig_actual_vs_pred(pred, "test", "01_actual_vs_predicted_test.png", zoom=60)
    fig_actual_vs_pred(pred, "validation", "02_actual_vs_predicted_validation.png", zoom=60)
    fig_errors(pred, "test", "03_prediction_errors_test.png")
    fig_residuals(pred, "test", "04_residual_analysis_test.png")
    fig_metric_bars(metrics, "05_metric_comparison.png")
    fig_importance(imp, "06_feature_importance.png")
    fig_return_vs_raw(pred, metrics, "07_return_vs_rawprice_experiment.png")


if __name__ == "__main__":
    main()
