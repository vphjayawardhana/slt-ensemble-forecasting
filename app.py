"""
Streamlit dashboard: Forecasting Sri Lanka Telecom stock prices with ensemble learning.

Run from the project folder:   streamlit run app.py

The dashboard never trains anything. It only READS the files produced in Phases 2-3
(models/, results/, data/processed/) and the original CSV. Every number on screen is
computed from those files. This is a historical back-test, NOT financial advice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import ensemble_models  # noqa: E402,F401  (must be imported before the saved models are loaded)
import data_utils as du  # noqa: E402
import preprocessing as pp  # noqa: E402

MODELS_DIR, RESULTS_DIR, PROCESSED_DIR = du.MODELS_DIR, du.RESULTS_DIR, du.PROCESSED_DIR

BASE, RF, XGB, STK = ("Persistence baseline", "Random Forest (Bagging)", "XGBoost (Boosting)",
                      "Stacking (RF + XGB -> Ridge)")
MODEL_ORDER = [BASE, RF, XGB, STK]
KEYS = {BASE: "baseline", RF: "random_forest", XGB: "xgboost", STK: "stacking"}
COLORS = {BASE: "#7f7f7f", RF: "#2a9d8f", XGB: "#e76f51", STK: "#264653"}
SPLIT_COLORS = {"train": "#dbe9f6", "validation": "#fdebc8", "test": "#f9d6d5"}
SPLIT_NOTE = {
    "train": "TRAINING period: the models were fitted on these days (in-sample, so errors look optimistic).",
    "validation": "VALIDATION period: used to choose hyper-parameters (so scores here are optimistic).",
    "test": "TEST period: not used for fitting or tuning. This is the honest out-of-sample back-test.",
}
DISCLAIMER = ("This dashboard shows a **historical back-test** for a university assignment. "
              "It is **not** a live forecast, **not** financial advice and not a trading system.")

st.set_page_config(page_title="SLT Ensemble Forecasting", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load_results() -> dict:
    out = {
        "metrics": pd.read_csv(RESULTS_DIR / "metrics_all.csv"),
        "preds": pd.read_csv(RESULTS_DIR / "predictions_validation_test.csv", parse_dates=["date", "next_date"]),
        "importance": pd.read_csv(RESULTS_DIR / "feature_importance.csv"),
        "resid": pd.read_csv(RESULTS_DIR / "residual_summary.csv"),
        "summary": json.loads((RESULTS_DIR / "summary.json").read_text()),
        "registry": json.loads((MODELS_DIR / "model_registry.json").read_text()),
        "meta": json.loads((PROCESSED_DIR / "phase2_metadata.json").read_text()),
        "weights": pd.read_csv(RESULTS_DIR / "stacking_weights_return.csv"),
    }
    return out


@st.cache_data(show_spinner=False)
def load_model_ready() -> tuple[pd.DataFrame, list[str]]:
    return du.load_model_ready()


@st.cache_data(show_spinner=False)
def load_raw_csv():
    """The original CSV, read-only. Returns None if the file cannot be found."""
    try:
        path = pp.find_csv()
    except FileNotFoundError:
        return None
    raw, df = pp.load_and_clean(path)
    df["Volume"] = df["Volume_raw"]
    return raw, df, path.name


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    reg = json.loads((MODELS_DIR / "model_registry.json").read_text())["registry"]
    return {k: joblib.load(MODELS_DIR / v) for k, v in reg.items()}


def fig_to_st(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def shade_splits(ax, md: pd.DataFrame):
    for s, g in md.groupby("split"):
        ax.axvspan(g["date"].min(), g["date"].max(), color=SPLIT_COLORS[s], alpha=0.6, lw=0, label=f"{s} period")


def return_metrics(res, split):
    m = res["metrics"].query("approach == 'return' and split == @split").set_index("model")
    return m.loc[MODEL_ORDER]


# --------------------------------------------------------------------------- pages
def page_overview(res, md):
    st.title("Forecasting Sri Lanka Telecom Stock Prices Using Ensemble Learning")
    st.subheader("A comparative study of Random Forest, XGBoost and Stacking models")
    st.warning(DISCLAIMER)
    meta = res["meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records in CSV", f"{meta['rows_in_csv']:,}")
    c2.metric("Model-ready rows", f"{meta['rows_model_ready']:,}")
    c3.metric("First date", str(md["date"].min().date()))
    c4.metric("Last forecast target", str(md["next_date"].max().date()))

    st.markdown("### Forecasting objective")
    st.markdown(
        "Predict the **next trading observation's closing price** from information available at the close of "
        "day *t*. The stock market is closed on weekends and holidays, so 'next' means the next available "
        "trading session, not necessarily the next calendar day. "
        "This is **historical one-step-ahead forecasting**, not live prediction.")

    st.markdown("### Dataset")
    st.markdown(
        f"* File: `{meta['source_csv']}` (daily prices of Sri Lanka Telecom, columns Date, Price, Open, High, Low, Vol., Change %).\n"
        f"* {meta['rows_in_csv']:,} records; {meta['dropped_warmup_rows']} warm-up rows are dropped (lags and moving "
        f"averages need history) and the last row has no next day, leaving {meta['rows_model_ready']:,} model-ready rows.\n"
        f"* Missing trading volumes are filled with the **training-set median** ({meta['volume_imputation_value_train_median']:,.0f}), "
        "and a flag column marks the filled days.\n"
        "* Only this daily file is used. No other data and no synthetic data.")
    split_tbl = (md.groupby("split").agg(rows=("date", "size"), first_day=("date", "min"), last_day=("date", "max"))
                 .loc[du.SPLITS])
    split_tbl["first_day"] = split_tbl["first_day"].dt.date
    split_tbl["last_day"] = split_tbl["last_day"].dt.date
    st.markdown("**Chronological split (70 / 15 / 15, never shuffled)**")
    st.dataframe(split_tbl, width="stretch")

    st.markdown("### Target used by the models")
    st.markdown(
        "The main models predict the **log return** `log(Price[t+1] / Price[t])` and the price is rebuilt as "
        "`Price[t] × exp(predicted return)`. Tree models cannot predict prices outside the range seen in training "
        f"(training maximum 63.1 LKR, test prices go up to {md.loc[md['split'] == 'test', 'target_price_next'].max():.2f} LKR), "
        "so predicting the return is the safer design. A raw-price experiment is also reported.")

    st.markdown("### What is ensemble learning?")
    a, b, c = st.columns(3)
    a.markdown("**Bagging: Random Forest**  \nMany decision trees are trained on random resamples of the rows "
               "(bootstrap) and random subsets of features. Their predictions are **averaged**, which reduces noise.")
    b.markdown("**Boosting: XGBoost**  \nSmall trees are built **one after another**; each new tree tries to correct "
               "the errors left by the previous ones. The learning rate controls the step size.")
    c.markdown("**Stacking**  \nRandom Forest and XGBoost make predictions, and a **Ridge regression** learns how to "
               "combine them. The combiner learns only from *walk-forward* out-of-fold predictions, never from "
               "shuffled K-fold, so nothing from the future leaks in.")
    st.markdown("A **persistence baseline** (tomorrow's close = today's close) is the benchmark every model must beat.")
    st.info("Use the sidebar to open Data Exploration, Ensemble Model Comparison and Prediction.")


def page_explore(res, md):
    st.title("Data Exploration")
    loaded = load_raw_csv()
    if loaded is None:
        st.error("The original CSV `TeleCom_2009_2023_Daily.csv` was not found in the project folder or in `data/`. "
                 "Place it there to see this page.")
        return
    raw, df, name = loaded
    st.caption(f"Source: {name} (read only; never modified).")

    st.subheader("Dataset preview")
    n = st.slider("Rows to show", 5, 50, 10)
    side = st.radio("Show", ["First rows", "Last rows"], horizontal=True)
    st.dataframe(raw.head(n) if side == "First rows" else raw.tail(n), width="stretch")
    st.caption("Raw file, exactly as stored (text). The cleaned version below is used for plots.")

    d0, d1 = df["Date"].min().to_pydatetime(), df["Date"].max().to_pydatetime()
    rng = st.slider("Date range", d0, d1, (d0, d1), format="YYYY-MM-DD")
    v = df[(df["Date"] >= rng[0]) & (df["Date"] <= rng[1])]
    if v.empty:
        st.warning("No data in this date range.")
        return

    st.subheader("Closing-price history")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    mdv = md[(md["date"] >= rng[0]) & (md["date"] <= rng[1])]
    if not mdv.empty:
        shade_splits(ax, mdv)
    ax.plot(v["Date"], v["Close"], color="#264653", lw=1.2)
    ax.set_ylabel("Close (LKR)")
    ax.legend(loc="upper left", fontsize=8)
    fig_to_st(fig)

    st.subheader("Opening, high and low prices")
    cols = st.multiselect("Series", ["Open", "High", "Low", "Close"], default=["Open", "High", "Low"])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for c, col in zip(cols, ["#2a9d8f", "#e76f51", "#8d99ae", "#264653"]):
        ax.plot(v["Date"], v[c], lw=0.9, label=c)
    ax.set_ylabel("LKR")
    ax.legend(fontsize=8)
    fig_to_st(fig)

    st.subheader("Trading volume")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.bar(v["Date"], v["Volume"].fillna(0), width=2, color="#457b9d")
    miss = v[v["Volume"].isna()]
    ax.scatter(miss["Date"], np.zeros(len(miss)), color="red", s=18, zorder=3, label=f"missing volume ({len(miss)})")
    ax.set_yscale("symlog")
    ax.set_ylabel("Volume (log-style axis)")
    ax.legend(fontsize=8)
    fig_to_st(fig)

    st.subheader("Daily percentage changes")
    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(v["Date"], v["Change_pct"], lw=0.6, color="#6d597a")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Change %")
    a.pyplot(fig, clear_figure=True)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(v["Change_pct"].dropna(), bins=60, color="#6d597a")
    ax.set_xlabel("Change %")
    ax.set_ylabel("days")
    b.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Moving averages")
    wins = st.multiselect("Windows (trading observations)", [5, 10, 20, 50, 100], default=[5, 10, 20])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(df["Date"], df["Close"], color="#bbbbbb", lw=0.8, label="Close")
    for w in wins:
        ax.plot(df["Date"], df["Close"].rolling(w).mean(), lw=1, label=f"MA{w}")
    ax.set_xlim(rng)
    ax.set_ylim(v["Close"].min() * 0.95, v["Close"].max() * 1.05)
    ax.legend(fontsize=8, ncol=5)
    fig_to_st(fig)
    st.caption("Moving averages use only the past (a window ending on each day).")

    st.subheader("Correlation analysis")
    a, b = st.columns(2)
    corr = v[["Open", "High", "Low", "Close", "Volume", "Change_pct"]].corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    a.pyplot(fig, clear_figure=True)
    plt.close(fig)
    feats = res["registry"]["return_features"]
    cc = md[feats + ["target_log_return"]].corr()["target_log_return"].drop("target_log_return").sort_values()
    top = pd.concat([cc.head(6), cc.tail(6)])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh(top.index, top.values, color=["#e76f51" if x < 0 else "#2a9d8f" for x in top.values])
    ax.set_title("Features most correlated with the next-day log return", fontsize=9)
    b.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("Prices are highly correlated with each other (they move together), but next-day returns "
               "have only weak linear correlation with any single feature. That is why the task is hard.")

    st.subheader("Missing-value summary")
    mv = pd.DataFrame({"missing": raw.isna().sum(), "percent": (raw.isna().mean() * 100).round(2)})
    cleaned = df.drop(columns=["Volume"]).isna().sum()
    mv["missing after cleaning"] = [int(cleaned.get({"Price": "Close", "Vol.": "Volume_raw", "Change %": "Change_pct"}
                                                    .get(i, i), 0)) for i in mv.index]
    st.dataframe(mv, width="stretch")
    st.write(f"Duplicate dates: {int(df['Date'].duplicated().sum())}. Missing volumes: {int(df['Volume'].isna().sum())} "
             "(filled with the training median in modelling, with a missing-flag feature).")


def interpretation(res, split):
    m = return_metrics(res, split)
    best = m["RMSE"].idxmin()
    lines = []
    if best == BASE:
        lines.append(f"On the **{split}** set the **persistence baseline** has the lowest RMSE ({m.loc[BASE, 'RMSE']:.4f}). "
                     "No ensemble beat 'tomorrow = today'.")
    else:
        lines.append(f"On the **{split}** set **{best}** has the lowest RMSE ({m.loc[best, 'RMSE']:.4f} vs "
                     f"{m.loc[BASE, 'RMSE']:.4f} for the baseline, {(1 - m.loc[best, 'RMSE'] / m.loc[BASE, 'RMSE']) * 100:.1f}% lower).")
    for k in MODEL_ORDER[1:]:
        g = (1 - m.loc[k, "MAE"] / m.loc[BASE, "MAE"]) * 100
        row = m.loc[k]
        ci = ""
        if pd.notna(row.get("MAE_gain_CI95_low")):
            ci = f" (95% interval {row['MAE_gain_CI95_low']:+.1f}% to {row['MAE_gain_CI95_high']:+.1f}%)"
        lines.append(f"* {k}: MAE {'lower' if g > 0 else 'higher'} than the baseline by {abs(g):.1f}%{ci}.")
    if split == "test":
        lines.append("The test period (2021-2023) is much more volatile than training, and the weak "
                     "day-to-day pattern the models learned faded, so large moves are missed. "
                     "Differences between models are small; intervals that include zero mean the gap to the baseline "
                     "cannot be told apart from noise.")
    else:
        lines.append("Validation scores are optimistic because hyper-parameters were tuned on this period.")
    return "\n\n".join(lines)


def page_compare(res, md):
    st.title("Ensemble Model Comparison")
    st.caption(DISCLAIMER)
    split = st.radio("Evaluation period", ["test", "validation"], horizontal=True,
                     help="Test is the honest out-of-sample result; validation was used for tuning.")
    m = return_metrics(res, split)
    best_rmse = m["RMSE"].idxmin()
    best_mae = m["MAE"].idxmin()
    c1, c2 = st.columns(2)
    c1.success(f"Best by RMSE on {split}: **{best_rmse}**")
    c2.success(f"Best by MAE on {split}: **{best_mae}**")

    st.subheader("Metrics (prices in LKR, same observations for every model)")
    show = m[["n", "MAE", "RMSE", "R2", "DirAcc_excl_unchanged", "MAE_gain_vs_baseline_pct",
              "MAE_gain_CI95_low", "MAE_gain_CI95_high"]].rename(columns={
        "DirAcc_excl_unchanged": "Direction acc. (price moved)", "MAE_gain_vs_baseline_pct": "MAE gain vs baseline %",
        "MAE_gain_CI95_low": "gain 95% low", "MAE_gain_CI95_high": "gain 95% high"})
    show = show.reset_index().rename(columns={"model": "Model"})
    st.dataframe(show.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}",
                                    "Direction acc. (price moved)": "{:.1%}", "MAE gain vs baseline %": "{:+.2f}",
                                    "gain 95% low": "{:+.1f}", "gain 95% high": "{:+.1f}"}, na_rep="n/a"),
                 width="stretch")
    st.caption("A positive gain means lower MAE than the baseline. The 95% interval comes from a block bootstrap "
               "of the daily errors; an interval that contains 0 means the difference is not reliable.")

    st.subheader("Performance comparison")
    cols = st.columns(3)
    for col, metric in zip(cols, ["MAE", "RMSE", "R2"]):
        fig, ax = plt.subplots(figsize=(4, 3.2))
        vals = m[metric]
        bars = ax.bar([k.split(" (")[0] for k in MODEL_ORDER], vals.values, color=[COLORS[k] for k in MODEL_ORDER])
        ax.bar_label(bars, fmt="%.3f" if metric != "R2" else "%.4f", fontsize=7, padding=2)
        if metric == "R2":
            ax.set_ylim(max(0, vals.min() - 0.01), min(1.0, vals.max() + 0.003))
        ax.set_title(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"), fontsize=9)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        col.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.subheader("Actual vs predicted closing price")
    p = res["preds"].query("split == @split")
    models = st.multiselect("Models to draw", MODEL_ORDER[1:], default=[STK])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(p["next_date"], p["actual_next_price"], color="black", lw=1.4, label="Actual")
    ax.plot(p["next_date"], p["return__baseline__price"], color=COLORS[BASE], lw=0.9, ls="--", label=BASE)
    for k in models:
        ax.plot(p["next_date"], p[f"return__{KEYS[k]}__price"], color=COLORS[k], lw=1, label=k)
    ax.set_ylabel("Close (LKR)")
    ax.legend(fontsize=8)
    fig_to_st(fig)
    st.caption("Lines join consecutive trading observations; non-trading periods (such as the 2020 closure) appear as straight segments.")

    st.subheader("Prediction errors and residual analysis")
    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k in MODEL_ORDER:
        err = (p["actual_next_price"] - p[f"return__{KEYS[k]}__price"]).abs().rolling(20).mean()
        ax.plot(p["next_date"], err, color=COLORS[k], lw=1.1, label=k.split(" (")[0])
    ax.set_title("Absolute error, 20-day rolling mean (LKR)", fontsize=9)
    ax.legend(fontsize=7)
    a.pyplot(fig, clear_figure=True)
    plt.close(fig)
    sel = b.selectbox("Residual histogram for", MODEL_ORDER, index=3)
    resid = p["actual_next_price"] - p[f"return__{KEYS[sel]}__price"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(resid, bins=50, color=COLORS[sel])
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("residual = actual - predicted (LKR)")
    b.pyplot(fig, clear_figure=True)
    plt.close(fig)
    rs = res["resid"].query("approach == 'return' and split == @split").set_index("model").loc[MODEL_ORDER]
    st.dataframe(rs.drop(columns=["approach", "split"]).style.format("{:.3f}"), width="stretch")
    st.caption("Residuals are strongly peaked with long tails (large positive kurtosis): most days are predicted "
               "closely and a few large moves are badly missed.")

    st.subheader("Feature importance")
    fi = res["importance"].query("approach == 'return'")
    a, b = st.columns(2)
    mname = a.selectbox("Model", [RF, XGB])
    itype = b.selectbox("Importance type", sorted(fi["importance_type"].unique()))
    t = fi[(fi["model"] == mname) & (fi["importance_type"] == itype)].nlargest(12, "importance").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(t["feature"], t["importance"], color=COLORS[mname])
    ax.set_title(f"{mname}: top 12 features ({itype})", fontsize=9)
    fig_to_st(fig)
    st.caption("Importance describes what the model used, not what causes price moves. Permutation importances are small and noisy.")
    w = res["weights"]
    st.markdown("**Stacking weights** (Ridge meta-learner on the base predictions)")
    st.dataframe(w, width="stretch")

    st.subheader("Interpretation (generated from the results shown above)")
    st.markdown(interpretation(res, split))

    with st.expander("Secondary experiment: raw next-price target"):
        r = res["metrics"].query("approach == 'rawprice' and split == @split").set_index("model").loc[MODEL_ORDER]
        st.dataframe(r[["MAE", "RMSE", "R2"]].style.format("{:.4f}"), width="stretch")
        st.write("Tree models cannot predict above the highest price seen in training, so when prices rise past "
                 "that level the raw-price models fall behind. This is why the main design predicts the return.")


def page_predict(res, md):
    st.title("Prediction (historical back-test)")
    st.warning(DISCLAIMER + " Every forecast below is for a **past** date whose real outcome is already known. "
               "Nothing here predicts the future.")
    models = load_models()
    feats = res["registry"]["return_features"]
    raw_feats = res["registry"]["rawprice_features"]

    split = st.radio("Period", ["test", "validation", "train"], horizontal=True)
    pool = md[md["split"] == split].reset_index(drop=True)
    st.info(SPLIT_NOTE[split])
    dates = pool["date"].dt.date.tolist()
    chosen = st.select_slider("Forecast origin (the day whose close is known)", options=dates, value=dates[len(dates) // 2])
    row = pool[pool["date"].dt.date == chosen]
    idx = row.index[0]
    r = row.iloc[0]
    st.write(f"Information up to the close of **{chosen}** (close {r['close_t']:.2f} LKR) is used to forecast "
             f"the next trading observation, **{r['next_date'].date()}**.")

    pick = st.multiselect("Models", MODEL_ORDER, default=MODEL_ORDER)
    use_raw = st.checkbox("Also show the secondary raw-price models", value=False)

    X = pool[feats].iloc[[idx]]
    rows = []
    for k in pick:
        key = KEYS[k]
        mdl = models["persistence_baseline"] if k == BASE else models[f"return_{key}"]
        lr = float(mdl.predict(X)[0])
        rows.append((k + " (return-based)", r["close_t"] * np.exp(lr), lr))
    if use_raw:
        Xr = du.raw_price_feature_table(pool).iloc[[idx]]
        for k in pick:
            if k == BASE:
                continue
            p_ = float(models[f"rawprice_{KEYS[k]}"].predict(Xr[raw_feats])[0])
            rows.append((k + " (raw-price)", p_, np.log(p_ / r["close_t"])))
    out = pd.DataFrame(rows, columns=["model", "predicted_next_close", "predicted_log_return"])
    actual = float(r["target_price_next"])
    out["actual_next_close"] = actual
    out["error (actual - predicted)"] = actual - out["predicted_next_close"]
    out["abs_error"] = out["error (actual - predicted)"].abs()
    out["direction"] = np.where(actual == r["close_t"], "unchanged",
                                np.where(np.sign(out["predicted_log_return"]) == np.sign(actual - r["close_t"]),
                                         "correct", "wrong"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Close on origin day", f"{r['close_t']:.2f}")
    c2.metric("Actual next close", f"{actual:.2f}", f"{actual - r['close_t']:+.2f}")
    if len(out):
        best = out.loc[out["abs_error"].idxmin()]
        c3.metric("Smallest absolute error", f"{best['abs_error']:.3f}")
        c3.caption(f"Closest: {best['model']}")
    st.dataframe(out.style.format({"predicted_next_close": "{:.3f}", "predicted_log_return": "{:+.4f}",
                                   "actual_next_close": "{:.2f}", "error (actual - predicted)": "{:+.3f}",
                                   "abs_error": "{:.3f}"}), width="stretch")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(out["model"].str.replace(" (return-based)", "", regex=False).str.replace(" (raw-price)", " raw", regex=False),
           out["predicted_next_close"], color="#8ecae6")
    ax.axhline(actual, color="black", lw=1.5, label=f"actual {actual:.2f}")
    ax.axhline(r["close_t"], color="#7f7f7f", lw=1, ls="--", label=f"today's close {r['close_t']:.2f}")
    lo = min(out["predicted_next_close"].min(), actual, r["close_t"])
    hi = max(out["predicted_next_close"].max(), actual, r["close_t"])
    pad = max((hi - lo) * 0.3, 0.05)
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.set_ylabel("LKR")
    ax.legend(fontsize=8)
    fig_to_st(fig)

    st.subheader("Context: 60 trading observations around the origin")
    lo_i, hi_i = max(0, idx - 40), min(len(pool), idx + 21)
    ctx = pool.iloc[lo_i:hi_i]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(ctx["date"], ctx["close_t"], color="#264653", lw=1.2, label="Close")
    ax.scatter([r["next_date"]], [actual], color="black", zorder=4, label="actual next close")
    for k in pick:
        o = out[out["model"].str.startswith(k)].iloc[0]
        ax.scatter([r["next_date"]], [o["predicted_next_close"]], color=COLORS[k], marker="x", s=70, zorder=5, label=k.split(" (")[0])
    ax.axvline(r["date"], color="#999", ls=":", lw=1)
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylabel("LKR")
    fig_to_st(fig)

    # consistency check against the saved back-test table (validation/test only)
    saved = res["preds"]
    s = saved[saved["date"].dt.date == chosen]
    if len(s) and split != "train":
        diffs = []
        for k in pick:
            if k == BASE:
                continue
            v = out[out["model"] == k + " (return-based)"]["predicted_next_close"].iloc[0]
            diffs.append(abs(v - float(s[f"return__{KEYS[k]}__price"].iloc[0])))
        if diffs:
            st.caption(f"Check: these live predictions differ from the saved back-test table by at most {max(diffs):.1e} LKR "
                       "(float32 rounding in the saved CSV).")
    with st.expander("How this works"):
        st.markdown(
            "The page loads the **real trained models** from `models/` and the **real historical features** from "
            "`data/processed/daily_model_ready.csv`. The features for a date use only information up to that day's close. "
            "The models predict a log return; the price is `close × exp(return)`. "
            "Dates in the training period are in-sample (the models saw them), validation dates were used for tuning, "
            "and only test dates are a fair out-of-sample check.")


# --------------------------------------------------------------------------- main
def main():
    try:
        res = load_results()
        md, _ = load_model_ready()
    except FileNotFoundError as e:
        st.error(f"Missing project file: {e}. Run the Phase 2 and Phase 3 scripts first (see README.md).")
        st.stop()
    st.sidebar.title("SLT Ensemble Forecasting")
    page = st.sidebar.radio("Page", ["1. Overview", "2. Data Exploration", "3. Ensemble Model Comparison", "4. Prediction"])
    st.sidebar.caption("Historical back-test for a university assignment. Not financial advice.")
    v = res["registry"].get("versions", {})
    st.sidebar.caption(f"Models trained with Python {v.get('python', '?')}, scikit-learn {v.get('scikit_learn', '?')}, "
                       f"xgboost {v.get('xgboost', '?')}.")
    {"1. Overview": page_overview, "2. Data Exploration": page_explore,
     "3. Ensemble Model Comparison": page_compare, "4. Prediction": page_predict}[page](res, md)


main()
