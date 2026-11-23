"""
Phase 2 - Data preprocessing and feature engineering
Project: Forecasting Sri Lanka Telecom stock prices with Ensemble Learning

What this script does (and does NOT do)
---------------------------------------
* Reads ONLY TeleCom_2009_2023_Daily.csv (read-only; the file is never rewritten).
* Cleans the data, builds historical features and the targets.
* Splits chronologically 70% / 15% / 15% (no shuffling).
* Saves a model-ready table to data/processed/.
* Does NOT train any model.

Time-series rule used everywhere
--------------------------------
A row t represents "the market at the close of trading observation t".
Every feature uses only rows <= t (shift with positive numbers, backward rolling
windows). The target looks one row ahead (t+1) and is never used as a feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CSV_NAME = "TeleCom_2009_2023_Daily.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]      # the "ml" folder
OUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FRAC = 0.70        # first 70% of labelled rows
VAL_END_FRAC = 0.85      # next 15% (0.70 -> 0.85) is validation; last 15% is test

LAGS = [1, 2, 3, 5, 10, 20]       # price lags
MA_WINDOWS = [5, 10, 20]          # moving-average windows (in trading observations)
RET_LAGS = [0, 1, 2, 3, 4]        # log return today (0) and 1..4 observations ago
VOL_WINDOWS = [5, 20]             # rolling volatility windows


# --------------------------------------------------------------------------
# 1. Locating and reading the data
# --------------------------------------------------------------------------
def find_csv(cli_path: str | None = None) -> Path:
    """Look for the CSV in ml/data/ first, then in the ml/ folder itself."""
    candidates = [Path(cli_path)] if cli_path else [
        PROJECT_ROOT / "data" / CSV_NAME,
        PROJECT_ROOT / CSV_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {CSV_NAME}. Looked in: {[str(c) for c in candidates]}")


def sha256(path: Path) -> str:
    """Fingerprint of the file, used to prove the original CSV is unchanged."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


_VOLUME_MULTIPLIER = {"K": 1e3, "M": 1e6, "B": 1e9}


def parse_volume(value) -> float:
    """'9.90K' -> 9900.0, '1.25M' -> 1250000.0, blank -> NaN (kept missing for now)."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    if text in ("", "-"):
        return np.nan
    suffix = text[-1].upper()
    if suffix in _VOLUME_MULTIPLIER:
        return float(text[:-1]) * _VOLUME_MULTIPLIER[suffix]
    return float(text)


def load_and_clean(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (raw, df).
    raw = the file exactly as stored (all strings), kept only for quality checks.
    df  = a cleaned COPY: parsed dates, numeric columns, chronological order.
    """
    raw = pd.read_csv(csv_path, dtype=str)       # read as text: nothing is guessed
    df = raw.copy()

    # Dates are month/day/year (verified in Phase 1). Sort oldest -> newest.
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)

    # 'Price' in this file is the CLOSING price; we rename it to Close for clarity.
    df = df.rename(columns={"Price": "Close"})
    for col in ["Close", "Open", "High", "Low"]:
        df[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False))
    df["Change_pct"] = pd.to_numeric(df["Change %"].str.replace("%", "", regex=False))
    df["Volume_raw"] = df["Vol."].map(parse_volume)   # still contains the 18 NaNs

    df = df[["Date", "Open", "High", "Low", "Close", "Volume_raw", "Change_pct"]]

    # Hard stops: these would silently break a time-series pipeline.
    assert df["Date"].is_unique, "duplicate dates found"
    assert df["Date"].is_monotonic_increasing, "dates are not sorted"
    assert (df[["Open", "High", "Low", "Close"]] > 0).all().all(), "non-positive price found"
    return raw, df


# --------------------------------------------------------------------------
# 2. Data-quality report (inspection only: nothing is changed or removed)
# --------------------------------------------------------------------------
def quality_report(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    q1, q3 = log_ret.quantile([0.25, 0.75])
    iqr = q3 - q1
    gaps = df["Date"].diff().dt.days
    return {
        "rows": int(len(df)),
        "first_date": str(df["Date"].min().date()),
        "last_date": str(df["Date"].max().date()),
        "raw_missing_per_column": raw.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_dates": int(df["Date"].duplicated().sum()),
        "zero_volume_rows": int((df["Volume_raw"] == 0).sum()),
        "high_below_low": int((df["High"] < df["Low"]).sum()),
        "high_below_open_or_close": int((df["High"] < df[["Open", "Close"]].max(axis=1)).sum()),
        "low_above_open_or_close": int((df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()),
        "abs_log_return_gt_10pct": int((log_ret.abs() > 0.10).sum()),
        "log_return_iqr_outliers_1.5x": int(((log_ret < q1 - 1.5 * iqr) | (log_ret > q3 + 1.5 * iqr)).sum()),
        "largest_calendar_gap_days": int(gaps.max()),
        "outlier_policy": "inspected only; genuine market outliers and zero-volume rows are KEPT",
    }


# --------------------------------------------------------------------------
# 3. Chronological split
# --------------------------------------------------------------------------
def assign_split(n_rows: int) -> np.ndarray:
    """
    The LAST row has no 'next observation', so it cannot be labelled.
    We split the remaining (n_rows - 1) labelled rows by position, in time order:
        first 70% -> train, next 15% -> validation, final 15% -> test.
    The split is decided on the FULL series, before any rows are dropped for
    missing features, so the boundaries do not move.
    """
    n_labelled = n_rows - 1
    train_end = int(n_labelled * TRAIN_FRAC)
    val_end = int(n_labelled * VAL_END_FRAC)
    position = np.arange(n_rows)
    return np.where(position < train_end, "train",
           np.where(position < val_end, "validation",
           np.where(position < n_labelled, "test", "unlabelled_last_row")))


# --------------------------------------------------------------------------
# 4. Feature engineering (all features are known at the close of observation t)
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, volume_fill: float) -> pd.DataFrame:
    """
    Build every feature from the FULL chronological series.
    Because the whole history is passed in, a validation/test row at time t
    automatically uses earlier observations (even from the training period) for its
    lags and rolling windows, exactly as in real forecasting.

    volume_fill : the TRAINING-ONLY median used to fill missing volumes.
    """
    close = df["Close"]
    f = pd.DataFrame(index=df.index)

    # Log return from observation t-1 to t. It is known at the close of t.
    log_ret = np.log(close / close.shift(1))

    # (a) Price lags as ratios to today's close. Ratios keep the feature scale-free,
    #     so the tree models are not tied to the 19-63 LKR price range seen in training.
    for k in LAGS:
        f[f"close_lag{k}_ratio"] = close.shift(k) / close

    # (b) Moving averages. The window ends at today's row and looks only backward.
    #     min_periods=w means the value stays NaN until a full window exists (no partial windows).
    for w in MA_WINDOWS:
        f[f"close_over_ma{w}"] = close / close.rolling(w, min_periods=w).mean()

    # (c) Historical returns: today's and the previous four observations'.
    #     (log_ret_lag0 is mathematically -log(close_lag1_ratio); the overlap is harmless for trees.)
    for k in RET_LAGS:
        f[f"log_ret_lag{k}"] = log_ret.shift(k)

    # (d) Rolling volatility = standard deviation of past log returns.
    for w in VOL_WINDOWS:
        f[f"volatility_{w}"] = log_ret.rolling(w, min_periods=w).std()

    # (e) Intraday information of observation t, expressed relative to its close.
    f["open_over_close"] = df["Open"] / close
    f["high_over_close"] = df["High"] / close
    f["low_over_close"] = df["Low"] / close
    f["hl_range_over_close"] = (df["High"] - df["Low"]) / close

    # (f) Volume. Missing values are filled with the training-only median (volume_fill);
    #     a 0/1 flag records which rows were originally missing. log1p tames the heavy skew
    #     and keeps zero-volume days valid (log1p(0) = 0).
    volume = df["Volume_raw"].fillna(volume_fill)
    f["log_volume"] = np.log1p(volume)
    f["volume_missing"] = df["Volume_raw"].isna().astype(int)
    f["volume_over_ma20"] = (volume / volume.rolling(20, min_periods=20).mean()).replace([np.inf, -np.inf], np.nan)

    # (g) Reported percentage change of the close (same-day information).
    f["change_pct"] = df["Change_pct"]

    # (h) Calendar features of the current observation (0 = Monday ... 4 = Friday).
    f["day_of_week"] = df["Date"].dt.dayofweek
    f["month"] = df["Date"].dt.month
    return f


# --------------------------------------------------------------------------
# 5. Targets + assembling the model-ready table
# --------------------------------------------------------------------------
def assemble(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Attach identifiers and the two targets to the features."""
    close = df["Close"]
    out = pd.DataFrame({"date": df["Date"], "split": df["split"], "close_t": close})
    out = pd.concat([out, feats], axis=1)
    out["next_date"] = df["Date"].shift(-1)                          # date of the observation being predicted
    out["target_log_return"] = np.log(close.shift(-1) / close)       # PRIMARY target
    out["target_price_next"] = close.shift(-1)                       # SECONDARY (raw-price) experiment
    return out


def run_pipeline(csv_path: str | None = None, save: bool = True) -> dict:
    csv = find_csv(csv_path)
    hash_before = sha256(csv)

    raw, df = load_and_clean(csv)
    quality = quality_report(raw, df)

    df["split"] = assign_split(len(df))

    # Imputation value from TRAINING rows only (never from validation/test).
    volume_fill = float(df.loc[df["split"] == "train", "Volume_raw"].median())

    feats = build_features(df, volume_fill)
    full = assemble(df, feats)
    feature_cols = list(feats.columns)

    # Drop rows only where necessary:
    #   - warm-up rows whose lag/rolling features cannot be computed yet (start of history)
    #   - the very last row, which has no next observation to predict
    has_features = full[feature_cols].notna().all(axis=1)
    has_target = full["target_log_return"].notna()
    model_ready = full[has_features & has_target].copy()

    dropped_warmup = full.index[~has_features].tolist()
    dropped_no_target = full.index[~has_target].tolist()

    result = dict(csv=csv, raw=raw, df=df, feats=feats, full=full, model_ready=model_ready,
                  feature_cols=feature_cols, volume_fill=volume_fill, quality=quality,
                  dropped_warmup=dropped_warmup, dropped_no_target=dropped_no_target,
                  hash_before=hash_before)

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model_ready.to_csv(OUT_DIR / "daily_model_ready.csv", index=False, date_format="%Y-%m-%d")
        meta = build_metadata(result)
        (OUT_DIR / "phase2_metadata.json").write_text(json.dumps(meta, indent=2))
    result["hash_after"] = sha256(csv)         # must equal hash_before
    return result


def build_metadata(r: dict) -> dict:
    mr, full = r["model_ready"], r["full"]
    split_info = {}
    for name in ["train", "validation", "test"]:
        part_full, part = full[full["split"] == name], mr[mr["split"] == name]
        split_info[name] = {
            "rows_before_dropping": int(len(part_full)),
            "rows_after_dropping": int(len(part)),
            "first_date": str(part["date"].min().date()),
            "last_date": str(part["date"].max().date()),
        }
    return {
        "source_csv": r["csv"].name,
        "source_csv_sha256": r["hash_before"],
        "rows_in_csv": int(len(full)),
        "rows_model_ready": int(len(mr)),
        "dropped_warmup_rows": len(r["dropped_warmup"]),
        "dropped_last_row_without_target": len(r["dropped_no_target"]),
        "volume_imputation_value_train_median": r["volume_fill"],
        "primary_target": "target_log_return = log(Close[t+1] / Close[t])",
        "price_reconstruction": "predicted_price = close_t * exp(predicted_log_return)",
        "secondary_target": "target_price_next = Close[t+1]",
        "feature_columns": r["feature_cols"],
        "splits": split_info,
        "data_quality": r["quality"],
    }


# --------------------------------------------------------------------------
# 6. Console summary
# --------------------------------------------------------------------------
def print_summary(r: dict) -> None:
    meta = build_metadata(r)
    mr = r["model_ready"]
    print("=" * 70, "\nPHASE 2 SUMMARY\n", "=" * 70, sep="")
    print(f"Source file           : {r['csv']}")
    print(f"Rows in CSV           : {meta['rows_in_csv']}")
    print(f"Dropped (warm-up)     : {meta['dropped_warmup_rows']} rows at the start of history")
    print(f"Dropped (no target)   : {meta['dropped_last_row_without_target']} row (the last observation)")
    print(f"Rows model-ready      : {meta['rows_model_ready']}")
    print(f"Volume fill (train median): {r['volume_fill']}")
    print("\nSplits:")
    for name, s in meta["splits"].items():
        print(f"  {name:<10} before drop={s['rows_before_dropping']:>5}  after drop={s['rows_after_dropping']:>5}  "
              f"{s['first_date']} -> {s['last_date']}")
    print(f"\nFeatures ({len(r['feature_cols'])}):")
    for c in r["feature_cols"]:
        print("  -", c)
    print("\nMissing values in model-ready table (per column, non-zero only):")
    miss = mr.isna().sum()
    print(miss[miss > 0].to_string() if miss.any() else "  none (0 missing values in every column)")
    print("\nMissing volumes BEFORE imputation:", int(r["df"]["Volume_raw"].isna().sum()),
          "| flagged rows kept:", int(mr["volume_missing"].sum()))
    print("\nData-quality report:")
    for k, v in r["quality"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 preprocessing + feature engineering")
    parser.add_argument("--csv", default=None, help="optional explicit path to the daily CSV")
    args = parser.parse_args()
    result = run_pipeline(args.csv)
    print_summary(result)
    print("\nOriginal CSV unchanged:", result["hash_before"] == result["hash_after"])
    print("Saved to:", OUT_DIR)
