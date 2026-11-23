"""
Evaluation helpers. EVERY metric is computed on prices in LKR.

Return-based models predict a log return, which is converted to a price with
    predicted_price = close_t * exp(predicted_log_return)
before any metric is calculated.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def reconstruct_price(close_t, predicted_log_return):
    """The agreed formula: predicted_price = Price[t] * exp(predicted_log_return)."""
    return np.asarray(close_t) * np.exp(np.asarray(predicted_log_return))


def directional_accuracy(close_t, actual_next, predicted_next) -> tuple[float, float]:
    """
    Direction = sign(next price - today's price): +1 up, -1 down, 0 unchanged.

    Including unchanged : every row counts; a prediction of "no change" is correct on a flat day.
    Excluding unchanged : only rows where the actual price moved; a model that never
                          predicts a direction (the persistence baseline) is returned as NaN.
    """
    actual_dir = np.sign(np.asarray(actual_next) - np.asarray(close_t))
    pred_dir = np.sign(np.asarray(predicted_next) - np.asarray(close_t))
    hit = actual_dir == pred_dir
    incl = float(hit.mean())
    moved = actual_dir != 0
    excl = float(hit[moved].mean()) if not (pred_dir == 0).all() else float("nan")
    return incl, excl


def evaluate_prices(close_t, actual_next, predicted_next) -> dict:
    """Full metric set for one model on one set of observations."""
    close_t, actual_next, predicted_next = map(np.asarray, (close_t, actual_next, predicted_next))
    da_incl, da_excl = directional_accuracy(close_t, actual_next, predicted_next)
    # R2 of the implied one-step return: tells us whether the model beats "no change"
    # on the thing that is actually hard to forecast (the price MOVE, not the price level).
    actual_ret = np.log(actual_next / close_t)
    implied_ret = np.log(predicted_next / close_t)
    return {
        "n": int(len(actual_next)),
        "MAE": float(mean_absolute_error(actual_next, predicted_next)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_next, predicted_next))),
        "R2": float(r2_score(actual_next, predicted_next)),
        "DirAcc_incl_unchanged": da_incl,
        "DirAcc_excl_unchanged": da_excl,
        "Return_R2": float(r2_score(actual_ret, implied_ret)),
    }


def block_bootstrap_mae_gain(actual, baseline_pred, model_pred, block=20, n_boot=2000, seed=42):
    """
    How much smaller is the model's MAE than the baseline's, with a 95% interval?
    A moving-block bootstrap keeps neighbouring days together because errors in a
    time series are not independent. Returns (gain_%, ci_low_%, ci_high_%).
    A positive interval that excludes 0 means the model is reliably better.
    """
    actual, baseline_pred, model_pred = map(np.asarray, (actual, baseline_pred, model_pred))
    diff = np.abs(actual - baseline_pred) - np.abs(actual - model_pred)   # >0 means model better
    base_mae = np.abs(actual - baseline_pred).mean()
    n = len(diff)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(diff.mean() / base_mae * 100), float(lo / base_mae * 100), float(hi / base_mae * 100))


def residual_summary(actual, predicted) -> dict:
    """Residual = actual - predicted (LKR). Positive = the model under-predicted."""
    r = np.asarray(actual) - np.asarray(predicted)
    lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    return {
        "mean_residual": float(r.mean()),
        "std_residual": float(r.std(ddof=1)),
        "median_abs_residual": float(np.median(np.abs(r))),
        "p95_abs_residual": float(np.percentile(np.abs(r), 95)),
        "max_abs_residual": float(np.abs(r).max()),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "lag1_autocorrelation": lag1,
    }
