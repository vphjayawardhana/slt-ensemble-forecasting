"""
Shared helpers for Phase 3 (and later the Streamlit dashboard).

Everything here reads the Phase 2 output (data/processed/daily_model_ready.csv),
so no preprocessing is repeated or changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SPLITS = ["train", "validation", "test"]


def load_model_ready() -> tuple[pd.DataFrame, list[str]]:
    """Return the Phase 2 table (chronological) and the 26 return-approach feature names."""
    df = pd.read_csv(PROCESSED_DIR / "daily_model_ready.csv", parse_dates=["date", "next_date"])
    meta = json.loads((PROCESSED_DIR / "phase2_metadata.json").read_text())
    assert df["date"].is_monotonic_increasing, "data must stay in chronological order"
    return df, meta["feature_columns"]


def raw_price_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set for the SECONDARY raw-price experiment.

    A model that must output a price LEVEL needs to see price levels, so here the
    scale-free ratios from Phase 2 are converted back into price levels
    (e.g. price_lag5 = close_lag5_ratio * close_t). These are exactly the same
    historical numbers (nothing new, nothing from the future); only the scale differs.
    """
    c = df["close_t"]
    out = pd.DataFrame(index=df.index)
    out["close_t"] = c
    for k in (1, 2, 3, 5, 10, 20):
        out[f"price_lag{k}"] = df[f"close_lag{k}_ratio"] * c
    for w in (5, 10, 20):
        out[f"ma{w}"] = c / df[f"close_over_ma{w}"]
    out["open"] = df["open_over_close"] * c
    out["high"] = df["high_over_close"] * c
    out["low"] = df["low_over_close"] * c
    out["hl_range"] = df["hl_range_over_close"] * c
    for col in ["log_ret_lag0", "log_ret_lag1", "log_ret_lag2", "log_ret_lag3", "log_ret_lag4",
                "volatility_5", "volatility_20", "log_volume", "volume_missing",
                "volume_over_ma20", "change_pct", "day_of_week", "month"]:
        out[col] = df[col]
    return out


def split_index(df: pd.DataFrame, name: str):
    """Boolean mask of the rows belonging to a split."""
    return df["split"] == name
