"""
Phase 2 verification: proves (by running code on the real data) that no future
information or target leakage is present. Prints PASS/FAIL for every check and
saves the log to data/processed/leakage_verification.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing as pp  # noqa: E402

LOG: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    LOG.append((bool(ok), name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if detail else ""))


def main() -> int:
    r = pp.run_pipeline(save=False)
    df, feats, full, mr = r["df"], r["feats"], r["full"], r["model_ready"]
    fill, cols = r["volume_fill"], r["feature_cols"]
    n = len(df)
    n_lab = n - 1
    a, b = int(n_lab * pp.TRAIN_FRAC), int(n_lab * pp.VAL_END_FRAC)

    # 1. Original CSV untouched
    check(r["hash_before"] == r["hash_after"], "Original CSV unchanged (SHA-256 before == after)", r["hash_before"][:16] + "...")

    # 2. Truncation test: features at t must be identical when ONLY data up to t exists.
    rng = np.random.default_rng(42)
    valid = mr.index.to_numpy()
    must = {valid[0], a - 1, a, b - 1, b, n_lab - 1, n - 1}
    sample = sorted(set(rng.choice(valid, 250, replace=False).tolist()) | {int(i) for i in must})
    bad = []
    for t in sample:
        past_only = pp.build_features(df.iloc[: t + 1].reset_index(drop=True), fill).iloc[-1].to_numpy(float)
        if not np.allclose(past_only, feats.iloc[t].to_numpy(float), equal_nan=True):
            bad.append(t)
    check(not bad, f"Truncation test: features at t identical using data up to t only ({len(sample)} origins incl. all split boundaries)",
          f"mismatches: {bad[:5]}" if bad else "0 mismatches")

    # 3. Perturbation test: wreck every FUTURE row (x1000); features at t must not move.
    bad = []
    for t in rng.choice(valid, 60, replace=False).tolist():
        d2 = df.copy()
        d2.loc[t + 1:, ["Open", "High", "Low", "Close", "Volume_raw", "Change_pct"]] *= 1000
        f2 = pp.build_features(d2, fill).iloc[t].to_numpy(float)
        if not np.allclose(f2, feats.iloc[t].to_numpy(float), equal_nan=True):
            bad.append(t)
    check(not bad, "Perturbation test: multiplying all future OHLCV by 1000 leaves features at t unchanged (60 origins)",
          f"mismatches: {bad[:5]}" if bad else "0 mismatches")

    # 4. Target definition recomputed independently
    c = df["Close"].to_numpy()
    idx = mr.index.to_numpy()
    expected = np.log(c[idx + 1] / c[idx])
    check(np.allclose(mr["target_log_return"].to_numpy(), expected, atol=1e-12),
          "target_log_return == log(Close[t+1]/Close[t]) for every model-ready row",
          f"max abs diff = {np.abs(mr['target_log_return'].to_numpy() - expected).max():.2e}")
    recon = mr["close_t"].to_numpy() * np.exp(mr["target_log_return"].to_numpy())
    check(np.allclose(recon, c[idx + 1], atol=1e-9),
          "Reconstruction close_t * exp(target) == actual next close",
          f"max abs diff = {np.abs(recon - c[idx + 1]).max():.2e} LKR")
    check((mr["target_price_next"].to_numpy() == c[idx + 1]).all() and (mr["next_date"].to_numpy() == df["Date"].to_numpy()[idx + 1]).all(),
          "target_price_next and next_date point to the very next row in the file")

    # 5. Target never used as a feature; no feature is a disguised copy of it
    check(not any(k in c_.lower() for c_ in cols for k in ("target", "next")), "No feature column is named/derived from target or next-row data")
    tr = mr[mr["split"] == "train"]
    corr = tr[cols].corrwith(tr["target_log_return"]).abs().sort_values(ascending=False)
    check(corr.max() < 0.5, "No feature is suspiciously correlated with the target (leak would show |r| ~ 1)",
          "top |corr| on train: " + ", ".join(f"{k}={v:.3f}" for k, v in corr.head(3).items()))

    # 6. Split integrity
    dates = {s: mr.loc[mr["split"] == s, "date"] for s in ["train", "validation", "test"]}
    check(mr["date"].is_monotonic_increasing and mr["date"].is_unique, "Model-ready rows are strictly chronological, no duplicate dates")
    check(dates["train"].max() < dates["validation"].min() and dates["validation"].max() < dates["test"].min(),
          "Train < validation < test in time, no overlap",
          f"train ends {dates['train'].max().date()} | val {dates['validation'].min().date()}..{dates['validation'].max().date()} | test starts {dates['test'].min().date()}")
    check((full["split"] == "train").sum() == a and (full["split"] == "validation").sum() == b - a and (full["split"] == "test").sum() == n_lab - b,
          "Split sizes equal 70/15/15 of the 3,238 labelled rows",
          f"{a} / {b - a} / {n_lab - b}")

    # 7. Imputation uses training data only
    train_median = float(df.loc[:a - 1, "Volume_raw"].median())
    check(fill == train_median, "Volume fill value equals the median of TRAINING rows only",
          f"train median={train_median} | median of all rows={df['Volume_raw'].median()} | val+test median={df.loc[a:, 'Volume_raw'].median()}")
    miss_pos = df.index[df["Volume_raw"].isna()]
    check(len(miss_pos) == 18 and (mr["volume_missing"].sum() == 18), "All 18 missing volumes were flagged and kept (indicator column)",
          f"missing rows are all inside training: {bool((miss_pos < a).all())}")

    # 8. Validation/test rows really use earlier history across the split boundaries
    for name, t in [("first validation row", a), ("first test row", b)]:
        lag_close = feats.loc[t, "close_lag20_ratio"] * df.loc[t, "Close"]
        src = t - 20
        check(np.isclose(lag_close, df.loc[src, "Close"]) and df.loc[src, "Date"] < df.loc[t, "Date"],
              f"{name} ({df.loc[t, 'Date'].date()}) uses the close from 20 observations earlier ({df.loc[src, 'Date'].date()}, split={df.loc[src, 'split']})")

    # 9. Nothing but warm-up and last row removed; outliers and zero volume kept
    expected_drop = set(range(len(r["dropped_warmup"]))) | {n - 1}
    check(set(r["dropped_warmup"]) == set(range(20)) and r["dropped_no_target"] == [n - 1] and len(expected_drop) == 21,
          "Only the first 20 warm-up rows and the final unlabelled row were dropped",
          f"warm-up indices {min(r['dropped_warmup'])}..{max(r['dropped_warmup'])}, no-target index {r['dropped_no_target']}")
    check(mr[cols + ["target_log_return"]].notna().all().all() and np.isfinite(mr[cols + ["target_log_return"]].to_numpy(float)).all(),
          "No NaN or infinite values in features or target after preprocessing")
    zero_vol_kept = int(((df["Volume_raw"] == 0) & df.index.isin(mr.index)).sum())
    check(zero_vol_kept == int((df["Volume_raw"] == 0).sum()), "All zero-volume rows retained",
          f"{zero_vol_kept} of {(df['Volume_raw'] == 0).sum()}")
    big = int((mr["target_log_return"].abs() > 0.10).sum())
    check(mr["target_log_return"].abs().max() == full["target_log_return"].abs().max(), "Largest market moves retained (no clipping)",
          f"|target|>10% rows in model-ready table: {big}; max |target| = {mr['target_log_return'].abs().max():.4f}")

    passed = sum(ok for ok, *_ in LOG)
    print(f"\n{passed}/{len(LOG)} checks passed")
    out = pp.OUT_DIR / "leakage_verification.txt"
    pp.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}" for ok, name, detail in LOG)
                   + f"\n\n{passed}/{len(LOG)} checks passed\n")
    return 0 if passed == len(LOG) else 1


if __name__ == "__main__":
    sys.exit(main())
