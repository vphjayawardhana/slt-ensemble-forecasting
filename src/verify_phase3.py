"""
Phase 3 verification (run AFTER train_models.py):  python src/verify_phase3.py
Loads the SAVED models and result files and checks, with real computations,
that results are leak-free, consistent and reproducible without retraining.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du                       # noqa: E402
import ensemble_models as em                  # noqa: E402  (must be imported before joblib.load)
from evaluation import evaluate_prices, reconstruct_price  # noqa: E402

LOG = []


def check(ok, name, detail=""):
    LOG.append((bool(ok), name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if detail else ""))


def main() -> int:
    df, feat_ret = du.load_model_ready()
    X_raw = du.raw_price_feature_table(df)
    reg = json.loads((du.MODELS_DIR / "model_registry.json").read_text())
    metrics = pd.read_csv(du.RESULTS_DIR / "metrics_all.csv")
    pred = pd.read_csv(du.RESULTS_DIR / "predictions_validation_test.csv", parse_dates=["date", "next_date"])
    masks = {s: (df.split == s).to_numpy() for s in du.SPLITS}
    n_tr = int(masks["train"].sum())

    # 1. every saved model loads
    models = {k: joblib.load(du.MODELS_DIR / v) for k, v in reg["registry"].items()}
    check(len(models) == 7 and all((du.MODELS_DIR / v).exists() for v in reg["registry"].values()),
          "All 7 model files exist and load", ", ".join(reg["registry"].values()))

    # 2. stacking folds: strictly expanding and strictly chronological
    for approach in ["return", "rawprice"]:
        st = models[f"{approach}_stacking"]
        folds = st.fold_info_
        ok = True
        for i, f in enumerate(folds):
            tr_end = f["n_train"]
            ps = int(f["predict_rows"].split("..")[0])
            ok &= (tr_end <= ps) and (i == 0 or tr_end == folds[i - 1]["n_train"] + folds[i - 1]["n_predict"])
            ok &= pd.Timestamp(f["train_end_date"]) < pd.Timestamp(f["predict_start_date"])
        covered = ~np.isnan(st.oof_predictions_).any(axis=1)
        check(ok and covered.sum() == n_tr - folds[0]["n_train"],
              f"[{approach}] stacking folds expand chronologically; each block predicted by models trained only on EARLIER rows",
              f"{len(folds)} folds; first train window {folds[0]['n_train']} rows; OOF rows = {int(covered.sum())} of {n_tr}; "
              f"fold1 trains to {folds[0]['train_end_date']}, predicts {folds[0]['predict_start_date']}..{folds[0]['predict_end_date']}; "
              f"last fold predicts to {folds[-1]['predict_end_date']}")

    # 3. perturbation test of the OOF logic: corrupt every training target from the START of fold 3's
    #    prediction block onward. Folds 1-3 only train on rows BEFORE that point, so their OOF predictions
    #    must not change; fold 4 trains on the corrupted rows, so it MUST change (proves the test has teeth).
    st = models["return_stacking"]
    Xtr, ytr = df.loc[masks["train"], feat_ret], df.loc[masks["train"], "target_log_return"].to_numpy()
    f3_start = int(st.fold_info_[2]["predict_rows"].split("..")[0])
    f3_end = int(st.fold_info_[2]["predict_rows"].split("..")[1]) + 1
    y_bad = ytr.copy(); y_bad[f3_start:] = np.random.default_rng(0).normal(0, 5, len(y_bad) - f3_start)
    probe = em.WalkForwardStackingRegressor(st.rf_params, st.xgb_params)
    oof_bad = probe.compute_oof(Xtr, y_bad)
    rows_early = np.arange(len(oof_bad))[:f3_end]
    rows_early = rows_early[~np.isnan(st.oof_predictions_[rows_early]).any(axis=1)]
    same_early = np.allclose(oof_bad[rows_early], st.oof_predictions_[rows_early])
    differs_fold4 = not np.allclose(oof_bad[f3_end:f3_end + 50], st.oof_predictions_[f3_end:f3_end + 50])
    check(same_early and differs_fold4, "Perturbation test: corrupting FUTURE training targets leaves earlier OOF predictions unchanged",
          f"targets corrupted from row {f3_start}; folds 1-3 identical: {same_early}; fold 4 changed (test has teeth): {differs_fold4}")

    # 4. stacking learned from training targets only
    check(np.allclose(st.oof_target_, ytr), "Stacking meta-model targets are the TRAIN split's targets only",
          f"{len(st.oof_target_)} rows = train rows")

    # 5. tuning logs contain validation scores only
    logs = sorted(du.RESULTS_DIR.glob("tuning_*.csv"))
    bad = [f.name for f in logs if any("test" in c.lower() for c in pd.read_csv(f, nrows=1).columns)]
    check(len(logs) == 6 and not bad, "Tuning logs hold validation scores only (no test columns)", f"{len(logs)} tuning files")

    # 6. same observations for every model
    p = pred
    cols_price = [c for c in p.columns if c.endswith("__price")]
    check(len(cols_price) == 8 and not p[cols_price].isna().any().any() and (p.split == "validation").sum() == 486 and (p.split == "test").sum() == 486,
          "All 4 models x 2 approaches predict the same 486 validation + 486 test observations (no gaps)",
          f"{len(cols_price)} prediction columns")
    check(np.allclose(p["return__baseline__price"], p["close_t"]), "Persistence baseline predicts exactly today's close")
    ok = all(np.allclose(p[f"return__{k}__price"], p["close_t"] * np.exp(p[f"return__{k}__log_return"]))
             for k in ["baseline", "random_forest", "xgboost", "stacking"])
    check(ok, "Reconstruction formula holds: price = close_t * exp(predicted log return) for every return-based model")

    # 7. reload saved models and recompute metrics WITHOUT retraining
    worst = 0.0
    for approach, X in [("return", df[feat_ret]), ("rawprice", X_raw)]:
        for key, label in [("random_forest", "Random Forest (Bagging)"), ("xgboost", "XGBoost (Boosting)"),
                           ("stacking", "Stacking (RF + XGB -> Ridge)")]:
            m = models[f"{approach}_{key}"]
            for split in ["validation", "test"]:
                mk = masks[split]
                close, actual = df.loc[mk, "close_t"].to_numpy(), df.loc[mk, "target_price_next"].to_numpy()
                out = m.predict(X[mk])
                price = reconstruct_price(close, out) if approach == "return" else out
                got = evaluate_prices(close, actual, price)
                ref = metrics[(metrics.approach == approach) & (metrics.model == label) & (metrics.split == split)].iloc[0]
                worst = max(worst, abs(got["MAE"] - ref.MAE), abs(got["RMSE"] - ref.RMSE), abs(got["R2"] - ref.R2))
    check(worst < 1e-9, "Saved models reproduce every reported MAE/RMSE/R2 exactly after reloading (no retraining)", f"max abs difference = {worst:.2e}")

    # 8. metric sanity vs independent computation
    for split in ["validation", "test"]:
        mk = masks[split]
        close, actual = df.loc[mk, "close_t"].to_numpy(), df.loc[mk, "target_price_next"].to_numpy()
        ref = metrics[(metrics.approach == "return") & (metrics.model == "Persistence baseline") & (metrics.split == split)].iloc[0]
        check(np.isclose(ref.MAE, np.abs(actual - close).mean()) and np.isclose(ref.DirAcc_incl_unchanged, (actual == close).mean()),
              f"Persistence {split}: MAE = mean|price change|; direction accuracy (incl.) = share of unchanged days",
              f"MAE {ref.MAE:.4f}, unchanged share {(actual == close).mean():.3f}")

    # 9. feature order saved with models matches the data
    check(list(models["return_random_forest"].feature_names_in_) == feat_ret and
          list(models["rawprice_xgboost"].feature_names_in_) == list(X_raw.columns),
          "Saved models expect the same feature names/order as the Phase 2 pipeline")

    passed = sum(ok for ok, *_ in LOG)
    print(f"\n{passed}/{len(LOG)} checks passed")
    (du.RESULTS_DIR / "phase3_verification.txt").write_text(
        "\n".join(f"[{'PASS' if ok else 'FAIL'}] {n} {d}" for ok, n, d in LOG) + f"\n\n{passed}/{len(LOG)} checks passed\n")
    return 0 if passed == len(LOG) else 1


if __name__ == "__main__":
    sys.exit(main())
