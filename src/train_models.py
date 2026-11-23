"""
Phase 3 - Model training and evaluation  (run:  python src/train_models.py)

Rules followed
--------------
* Fit on TRAIN only. Choose hyper-parameters on VALIDATION only.
* The TEST rows are used exactly once, at the very end, to report final scores.
* Every model is scored on the same validation and test observations, on prices in LKR.
* Two experiments:
    "return"   (PRIMARY)   model predicts log return -> price = close_t * exp(prediction)
    "rawprice" (SECONDARY) model predicts the next closing price directly
"""
from __future__ import annotations

import itertools
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du                                    # noqa: E402
from ensemble_models import (PersistenceBaseline, WalkForwardStackingRegressor,  # noqa: E402
                             make_random_forest, make_xgboost, SEED)
from evaluation import (block_bootstrap_mae_gain, evaluate_prices, reconstruct_price,  # noqa: E402
                        residual_summary)

# ---------------- small, CPU-friendly search grids (validation-based) -------------
RF_GRID = {"max_depth": [3, 6, 10, 15], "min_samples_leaf": [5, 20, 60, 150, 300], "max_features": [0.2, 0.4, 0.7, 1.0]}
RF_N_ESTIMATORS = 300
XGB_GRID = {"max_depth": [1, 2, 3, 4], "learning_rate": [0.01, 0.03, 0.1], "min_child_weight": [1, 10, 30, 60]}
XGB_CHECKPOINTS = [50, 100, 200, 400, 800, 1600]       # n_estimators values compared in one fit
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]

NAMES = {"baseline": "Persistence baseline", "random_forest": "Random Forest (Bagging)",
         "xgboost": "XGBoost (Boosting)", "stacking": "Stacking (RF + XGB -> Ridge)"}


def to_price(approach: str, close_t, prediction):
    """Return approach: prediction is a log return. Raw-price approach: prediction is already a price."""
    return reconstruct_price(close_t, prediction) if approach == "return" else np.asarray(prediction)


def val_scores(approach, close_va, actual_va, prediction):
    m = evaluate_prices(close_va, actual_va, to_price(approach, close_va, prediction))
    return m["RMSE"], m["MAE"]


def grid(d: dict):
    keys = list(d)
    for values in itertools.product(*d.values()):
        yield dict(zip(keys, values))


# ------------------------------------------------------------------ tuning
def tune_random_forest(approach, X_tr, y_tr, X_va, close_va, actual_va):
    rows = []
    for p in grid(RF_GRID):
        model = make_random_forest({**p, "n_estimators": RF_N_ESTIMATORS}).fit(X_tr, y_tr)
        rmse, mae = val_scores(approach, close_va, actual_va, model.predict(X_va))
        rows.append({**p, "n_estimators": RF_N_ESTIMATORS, "val_RMSE": rmse, "val_MAE": mae})
    log = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    best = {k: (None if pd.isna(v) else v) for k, v in log.iloc[0].drop(["val_RMSE", "val_MAE"]).items()}
    best["max_depth"] = int(best["max_depth"]); best["min_samples_leaf"] = int(best["min_samples_leaf"])
    best["n_estimators"] = int(best["n_estimators"]); best["max_features"] = float(best["max_features"])
    return best, log


def tune_xgboost(approach, X_tr, y_tr, X_va, close_va, actual_va):
    rows = []
    for p in grid(XGB_GRID):
        model = make_xgboost({**p, "n_estimators": max(XGB_CHECKPOINTS)}).fit(X_tr, y_tr)
        for k in XGB_CHECKPOINTS:      # one fit, several candidate numbers of boosting rounds
            pred = model.predict(X_va, iteration_range=(0, k))
            rmse, mae = val_scores(approach, close_va, actual_va, pred)
            rows.append({**p, "n_estimators": k, "val_RMSE": rmse, "val_MAE": mae})
    log = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    best = log.iloc[0].drop(["val_RMSE", "val_MAE"]).to_dict()
    best = {"max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
            "min_child_weight": int(best["min_child_weight"]), "n_estimators": int(best["n_estimators"])}
    return best, log


# ------------------------------------------------------------------ one experiment
def run_approach(approach, X, y, masks, df, results_dir):
    t0 = time.time()
    X_tr, X_va, X_te = X[masks["train"]], X[masks["validation"]], X[masks["test"]]
    y_tr = y[masks["train"]]
    close_va = df.loc[masks["validation"], "close_t"].to_numpy()
    actual_va = df.loc[masks["validation"], "target_price_next"].to_numpy()
    dates_tr = df.loc[masks["train"], "date"].to_numpy()
    print(f"\n=== Approach: {approach} | train={len(X_tr)} val={len(X_va)} test={len(X_te)} | features={X.shape[1]}")

    print("  tuning Random Forest on validation ...")
    rf_params, rf_log = tune_random_forest(approach, X_tr, y_tr, X_va, close_va, actual_va)
    print("   best RF :", rf_params, f"| val RMSE {rf_log.val_RMSE.iloc[0]:.4f}")
    print("  tuning XGBoost on validation ...")
    xgb_params, xgb_log = tune_xgboost(approach, X_tr, y_tr, X_va, close_va, actual_va)
    print("   best XGB:", xgb_params, f"| val RMSE {xgb_log.val_RMSE.iloc[0]:.4f}")

    # final standalone models: fitted on TRAIN only
    rf = make_random_forest(rf_params).fit(X_tr, y_tr)
    xgb = make_xgboost(xgb_params).fit(X_tr, y_tr)

    print("  building walk-forward out-of-fold predictions for stacking ...")
    stack = WalkForwardStackingRegressor(rf_params, xgb_params)
    stack.compute_oof(X_tr, y_tr, dates_tr)
    stack.fit_base_models(X_tr, y_tr)
    alpha_rows = []
    for a in RIDGE_ALPHAS:                    # Ridge strength chosen on VALIDATION
        stack.fit_meta(a)
        rmse, mae = val_scores(approach, close_va, actual_va, stack.predict(X_va))
        alpha_rows.append({"ridge_alpha": a, "val_RMSE": rmse, "val_MAE": mae})
    alpha_log = pd.DataFrame(alpha_rows).sort_values("val_RMSE").reset_index(drop=True)
    best_alpha = float(alpha_log.ridge_alpha.iloc[0])
    stack.fit_meta(best_alpha)
    print("   best Ridge alpha:", best_alpha, f"| val RMSE {alpha_log.val_RMSE.iloc[0]:.4f}")

    rf_log.to_csv(results_dir / f"tuning_{approach}_random_forest.csv", index=False)
    xgb_log.to_csv(results_dir / f"tuning_{approach}_xgboost.csv", index=False)
    alpha_log.to_csv(results_dir / f"tuning_{approach}_stacking_ridge_alpha.csv", index=False)
    stack.meta_weights().to_csv(results_dir / f"stacking_weights_{approach}.csv", index=False)
    pd.DataFrame(stack.fold_info_).to_csv(results_dir / f"stacking_folds_{approach}.csv", index=False)

    models = {"baseline": PersistenceBaseline(), "random_forest": rf, "xgboost": xgb, "stacking": stack}

    # permutation importance on VALIDATION (test stays untouched)
    imp_rows = []
    for key, model in [("random_forest", rf), ("xgboost", xgb)]:
        imp = model.feature_importances_
        for f_, v in zip(X.columns, imp):
            imp_rows.append({"approach": approach, "model": NAMES[key], "feature": f_,
                             "importance_type": "built-in (RF: impurity, XGB: gain)", "importance": float(v)})
        yv = y[masks["validation"]]
        pi = permutation_importance(model, X_va, yv, scoring="neg_mean_squared_error",
                                    n_repeats=10, random_state=SEED, n_jobs=1)
        for f_, v in zip(X.columns, pi.importances_mean):
            imp_rows.append({"approach": approach, "model": NAMES[key], "feature": f_,
                             "importance_type": "permutation (validation, rise in MSE)", "importance": float(v)})
    print(f"  approach '{approach}' finished in {time.time() - t0:.0f}s")
    return {"models": models, "params": {"random_forest": rf_params, "xgboost": xgb_params, "ridge_alpha": best_alpha},
            "importance": imp_rows, "stack_folds": stack.fold_info_,
            "X": {"validation": X_va, "test": X_te}}


def main():
    t_start = time.time()
    np.random.seed(SEED)
    du.RESULTS_DIR.mkdir(exist_ok=True); du.FIGURES_DIR.mkdir(parents=True, exist_ok=True); du.MODELS_DIR.mkdir(exist_ok=True)

    df, feat_ret = du.load_model_ready()
    masks = {s: du.split_index(df, s).to_numpy() for s in du.SPLITS}
    X_ret = df[feat_ret]
    X_raw = du.raw_price_feature_table(df)
    y_ret = df["target_log_return"].to_numpy()
    y_raw = df["target_price_next"].to_numpy()
    print("rows:", {s: int(m.sum()) for s, m in masks.items()})

    runs = {"return": run_approach("return", X_ret, y_ret, masks, df, du.RESULTS_DIR),
            "rawprice": run_approach("rawprice", X_raw, y_raw, masks, df, du.RESULTS_DIR)}

    # ------------------------------------------------ final evaluation (validation + test)
    metric_rows, resid_rows = [], []
    pred_table = df.loc[masks["validation"] | masks["test"], ["date", "next_date", "split", "close_t",
                                                              "target_price_next", "target_log_return"]].rename(
        columns={"target_price_next": "actual_next_price", "target_log_return": "actual_log_return"}).copy()
    for approach, run in runs.items():
        base_price = {}
        for split in ["validation", "test"]:
            m = masks[split]
            close, actual = df.loc[m, "close_t"].to_numpy(), df.loc[m, "target_price_next"].to_numpy()
            base_price[split] = close.copy()                       # persistence prediction
        for key, model in run["models"].items():
            parts = []
            for split in ["validation", "test"]:
                m = masks[split]
                close, actual = df.loc[m, "close_t"].to_numpy(), df.loc[m, "target_price_next"].to_numpy()
                pred_raw = model.predict(run["X"][split])
                price = to_price(approach, close, pred_raw) if key != "baseline" else close.copy()
                met = evaluate_prices(close, actual, price)
                if key != "baseline":
                    gain, lo, hi = block_bootstrap_mae_gain(actual, base_price[split], price)
                    met.update({"MAE_gain_vs_baseline_pct": gain, "MAE_gain_CI95_low": lo, "MAE_gain_CI95_high": hi})
                metric_rows.append({"approach": approach, "model": NAMES[key], "split": split, **met})
                resid_rows.append({"approach": approach, "model": NAMES[key], "split": split,
                                   **residual_summary(actual, price)})
                parts.append(price)
            # val rows come first, then test rows: the same order as pred_table
            pred_table[f"{approach}__{key}__price"] = np.concatenate(parts)
            if approach == "return":
                pred_table[f"return__{key}__log_return"] = np.concatenate(
                    [np.zeros(len(X_)) if key == "baseline" else model.predict(X_)
                     for X_ in (run["X"]["validation"], run["X"]["test"])])

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(du.RESULTS_DIR / "metrics_all.csv", index=False)
    pd.DataFrame(resid_rows).to_csv(du.RESULTS_DIR / "residual_summary.csv", index=False)
    pred_table.to_csv(du.RESULTS_DIR / "predictions_validation_test.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame([r for run in runs.values() for r in run["importance"]]).to_csv(
        du.RESULTS_DIR / "feature_importance.csv", index=False)

    # ------------------------------------------------ summary + best model (chosen on VALIDATION)
    summary = {}
    for approach in runs:
        sub = metrics[metrics.approach == approach]
        val, tst = sub[sub.split == "validation"].set_index("model"), sub[sub.split == "test"].set_index("model")
        summary[approach] = {
            "ranking_by_validation_RMSE": val["RMSE"].sort_values().index.tolist(),
            "ranking_by_test_RMSE": tst["RMSE"].sort_values().index.tolist(),
            "best_on_validation": val["RMSE"].idxmin(),
            "best_on_test_for_information": tst["RMSE"].idxmin(),
            "selected_hyperparameters": runs[approach]["params"],
        }
    # directional context: how often does the price actually rise / fall / stay flat?
    ctx = {}
    for split in ["validation", "test"]:
        d = np.sign(df.loc[masks[split], "target_price_next"] - df.loc[masks[split], "close_t"])
        ctx[split] = {"up_share": float((d > 0).mean()), "down_share": float((d < 0).mean()),
                      "unchanged_share": float((d == 0).mean())}
    summary["actual_direction_shares"] = ctx
    (du.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------ save models + registry
    registry = {"persistence_baseline": "persistence_baseline.joblib"}
    joblib.dump(PersistenceBaseline(), du.MODELS_DIR / "persistence_baseline.joblib")
    for approach, run in runs.items():
        for key in ["random_forest", "xgboost", "stacking"]:
            fname = f"{approach}_{key}.joblib"
            joblib.dump(run["models"][key], du.MODELS_DIR / fname)
            registry[f"{approach}_{key}"] = fname
    meta = {
        "registry": registry,
        "return_features": feat_ret,
        "rawprice_features": list(X_raw.columns),
        "hyperparameters": {a: r["params"] for a, r in runs.items()},
        "stacking_walk_forward_folds": {a: r["stack_folds"] for a, r in runs.items()},
        "fit_data": "train split only (2009-08-07 to 2019-01-18)",
        "seed": SEED,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "how_to_predict": {"return": "price = close_t * exp(model.predict(X[return_features]))",
                           "rawprice": "price = model.predict(X[rawprice_features])  (use data_utils.raw_price_feature_table)"},
        "total_runtime_seconds": round(time.time() - t_start, 1),
    }
    (du.MODELS_DIR / "model_registry.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {time.time() - t_start:.0f}s. Models -> {du.MODELS_DIR}, results -> {du.RESULTS_DIR}")


if __name__ == "__main__":
    main()
