# Forecasting Sri Lanka Telecom Stock Prices Using Ensemble Learning

**A comparative study of Random Forest (Bagging), XGBoost (Boosting) and a walk-forward Stacking ensemble**

This project contains **Phase 2 (preprocessing and feature engineering)**, **Phase 3 (model training and evaluation)**, **Phase 4 (Jupyter notebook)**, **Phase 5 (Streamlit dashboard)** and **Phase 6-8 (testing notes and written reports)**.

> This is a historical back-test for a university assignment. It is **not financial advice** and not a live trading system.

**Authoritative results.** All numbers in this README come from the run the student executed on their own **MacBook Air M1** (Python 3.11.18). The earlier Linux run made by the AI assistant is kept only as a clearly labelled *reference* (section 8).

---

## 1. What is in the folder

```
ml_phase3/
├── README.md                     <- this file
├── requirements.txt              <- minimum versions
├── requirements-lock.txt         <- EXACT versions used for the authoritative Mac run (Python 3.11)
├── TeleCom_2009_2023_Daily.csv   <- YOUR original data (never modified, never in a ZIP)
├── app.py                        <- Phase 5: Streamlit dashboard (4 pages)
├── notebooks/ensemble_learning_assignment.ipynb   <- Phase 4 notebook
├── report/                       project_summary.md, presentation_script.md, ai_usage_statement.md
├── src/                          preprocessing, verify_leakage, data_utils, evaluation,
│                                 ensemble_models, train_models, make_plots, verify_phase3, compare_runs
├── data/processed/               daily_model_ready.csv, phase2_metadata.json, leakage_verification.txt
├── models/                       7 saved models (.joblib) + model_registry.json   (your Mac run)
├── results/                      metrics, predictions, residuals, importances, tuning logs, summary.json
│   └── figures/                  01 ... 07 PNG figures
├── logs/                         training / verification logs
└── reference_runs/               Linux runs by the AI assistant, for comparison ONLY
```

## 2. Setup on a MacBook Air M1 with Python 3.11

No GPU or CUDA is needed. The authoritative run used this environment:

| Item | Value |
|---|---|
| Machine | MacBook Air M1 (Apple Silicon), macOS 27 |
| Python | 3.11.18, installed with **uv** (Homebrew does not support macOS 27: `brew install python@3.11 libomp` failed with "unknown or unsupported macOS version") |
| Packages | pandas 3.1.1, numpy 2.4.6, scikit-learn 1.9.1, xgboost 3.2.0 (full list in `requirements-lock.txt`) |
| Folder | `~/Desktop/ml_phase3` (no spaces in the path) |

```bash
cd ~/Desktop/ml_phase3
uv python install 3.11                 # if Python 3.11 is not installed yet
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version                       # must print Python 3.11.x
uv pip install -r requirements-lock.txt
uv pip install jupyterlab ipykernel streamlit==1.67.0   # notebook + dashboard (not needed for training)
python -m ipykernel install --user --name ml_phase3 --display-name "Python 3.11 (ml_phase3)"
```

### XGBoost and the OpenMP library (libomp)

On macOS, XGBoost needs `libomp.dylib`. Without it, `import xgboost` fails with *"Library not loaded: @rpath/libomp.dylib"*.
Normally `brew install libomp` fixes this, but Homebrew does not support macOS 27. The workaround that was proposed is to point XGBoost at the copy of libomp that scikit-learn already ships inside the virtual environment, for example by symlinking `sklearn/.libs/libomp.dylib` into `xgboost/lib/`. A conda-forge environment (`conda install -c conda-forge xgboost libomp`) is an alternative.
**The student reported that the fix worked, but the exact method used was not recorded here; please add it in this line:** `fix used: ________`.
Check with: `python -c "import xgboost; print(xgboost.__version__)"` (expect 3.2.0).

## 3. Run order

Run from `~/Desktop/ml_phase3` with the virtual environment active.

```bash
python src/preprocessing.py     # Phase 2 (seconds)
python src/verify_leakage.py    # expect "19/19 checks passed"
python src/train_models.py      # Phase 3: ~105 s on the M1 (training time recorded in the model registry)
python src/make_plots.py        # regenerates the 7 figures
python src/verify_phase3.py     # expect "13/13 checks passed"
python -m jupyterlab notebooks/ensemble_learning_assignment.ipynb   # Phase 4: choose the kernel 'Python 3.11 (ml_phase3)', then Restart Kernel and Run All
streamlit run app.py            # Phase 5: dashboard at http://localhost:8501
```

* The original CSV is only **read**; scripts verify its SHA-256 hash is unchanged.
* **You do not need to retrain.** The notebook loads the saved models and results. It does not overwrite anything in `models/` or `results/`.
* Saved `.joblib` files only load with the package versions in `requirements-lock.txt`.
* Later phases (dashboard) must add `src/` to `sys.path` and `import ensemble_models` before loading models.

## 4. Method in brief

| Item | Choice |
|---|---|
| Data | `TeleCom_2009_2023_Daily.csv` only: 3,239 rows, 2009-07-08 to 2023-04-12 |
| Task | One-step-ahead forecast of the next trading observation's closing price |
| Primary target | `log(Price[t+1] / Price[t])`; predicted price = `Price[t] * exp(predicted log return)` |
| Secondary target | Raw next closing price (extra experiment; shows the extrapolation limit of tree models) |
| Features | 26 historical features known at the close of day *t* (lags, moving averages, returns, volatility, intraday ratios, volume, calendar) |
| Split | Chronological 70 / 15 / 15: train 2,246 rows (2009-08-07 to 2019-01-18), validation 486 (2019-01-21 to 2021-03-23), test 486 (2021-03-24 to 2023-04-11) |
| Fitting / tuning | Fit on train only; hyper-parameters chosen on validation only; test used once for final scores |
| Models | Persistence baseline; Random Forest (bagging); XGBoost (boosting); Stacking (RF + XGB -> Ridge) |
| Stacking | 5 expanding-window walk-forward folds inside the training set (first window 898 rows); Ridge learns only from out-of-fold predictions |
| Metrics | MAE, RMSE, R2 on prices in LKR; directional accuracy including and excluding unchanged days |

**Beginner-friendly model notes**

* **Persistence baseline** - "tomorrow's close = today's close". Nothing to learn; any model must beat it to be useful.
* **Random Forest (bagging)** - many decision trees, each trained on a random resample of the rows and a random subset of features at every split; the forest *averages* the trees, which reduces noise. Key settings: `n_estimators` (number of trees), `max_depth` (tree size), `min_samples_leaf` (minimum rows per leaf; larger = smoother), `max_features`.
* **XGBoost (boosting)** - small trees built *one after another*, each correcting the mistakes of the previous ones. Key settings: `n_estimators` (rounds), `learning_rate` (step size), `max_depth`, `min_child_weight` (caution on small leaves), `subsample`/`colsample_bytree` (random sampling), `reg_lambda` (shrinkage).
* **Stacking** - Random Forest and XGBoost make predictions; a Ridge regression learns how to combine them. Out-of-fold predictions come from an expanding window (train on the past, predict the next block), never from shuffled K-fold, so no model predicts a row it trained on and no model sees the future.

## 5. Authoritative results (student's MacBook Air M1, Python 3.11.18; return-based primary experiment; prices in LKR)

| Split | Model | MAE | RMSE | R2 | Dir. acc. (days the price moved) |
|---|---|---|---|---|---|
| Validation | Persistence baseline | 0.5247 | 0.8022 | 0.9675 | n/a |
| | Random Forest | 0.5087 | 0.7817 | 0.9692 | 64.1% |
| | XGBoost | 0.5057 | 0.7768 | 0.9695 | 64.1% |
| | Stacking | **0.5051** | **0.7750** | **0.9697** | 64.9% |
| Test | Persistence baseline | **1.3903** | **2.4675** | **0.9806** | n/a |
| | Random Forest | 1.4019 | 2.4834 | 0.9804 | 55.0% |
| | XGBoost | 1.4120 | 2.5032 | 0.9801 | 58.8% |
| | Stacking | 1.4068 | 2.4886 | 0.9803 | 57.4% |

Selected hyper-parameters (validation only): Random Forest `max_depth=15, min_samples_leaf=150, max_features=1.0, n_estimators=300`; XGBoost `max_depth=2, learning_rate=0.03, min_child_weight=1, n_estimators=400`; Ridge `alpha=10`. Training time 105.2 s.
Raw-price experiment and bootstrap intervals: see `results/metrics_all.csv` and the notebook (they are computed from your files, not copied here).

**Reading the results honestly**

* **Stacking is best on validation** (chosen before looking at test): MAE lower than the baseline by about 3.7% (RF 3.1%, XGBoost 3.6%).
* **On test the persistence baseline is best.** Ensemble test MAE is higher than the baseline by 0.8% (RF), 1.6% (XGBoost) and 1.2% (stacking).
* The notebook computes block-bootstrap intervals for these differences from `metrics_all.csv`; judge reliability from there.
* The models call the *direction* correctly more often than chance on days the price moved, but wrongly-sized large moves dominate MAE/RMSE.
* Raw-price trees cannot predict beyond the training price range (maximum 63.1 LKR; test prices reach 113.75).
* **Hyper-parameter ties.** Three Random Forest candidates (`max_depth` 6, 10, 15) have identical validation scores: `min_samples_leaf=150` already limits depth to 6, so they are the same model. The best XGBoost setting beat the runner-up by 0.12% in validation RMSE (a near-tie). Different settings on different machines are therefore expected and not a bug.
* Mac predictions equal the saved CSV to 1.9e-06, which is float32 rounding in the CSV.

## 6. Verification (authoritative Mac run)

* Reported by the student from their Mac: the saved models were reloaded and reproduced `predictions_validation_test.csv` (max difference 1.89e-06, float32 rounding); `model_registry.json` matches `summary.json`.
* `verify_leakage.py` (19 checks) and `verify_phase3.py` (13 checks): run them on your Mac and read `data/processed/leakage_verification.txt` and `results/phase3_verification.txt`; the notebook appendix prints both files.

## 7. What was and was not verified (notebook and dashboard)

* The notebook was executed end to end (26 code cells, no errors, 19 figures) in a **Python 3.11.15 Linux sandbox** against the **Linux reference files**, using the locked package versions.
* It has **not been run on your Mac files**. Its section 0 compares your files with the Mac values above and prints either "OK" or the differences.
* Every number in the notebook is computed from files on disk (the only typed numbers are the Mac values used for that check).
* The student ran the notebook on the Mac (all cells executed in JupyterLab). The assistant did not see its printed outputs, so the Section 0 check result is for the student to confirm.
* **Dashboard (`app.py`)**: tested in the Linux sandbox with Streamlit's automated test runner (`AppTest`): all four pages and the main controls (period switch, model selection, raw-price option, every period on the Prediction page) ran with no exceptions, and the pages were loaded in a headless browser. This used the **Linux reference files**, not the Mac files. It has **not** been run on the Mac yet. The sidebar shows the Python/library versions recorded in the models, so you can see which run it is reading.
* The Prediction page prints a check comparing its live predictions with `predictions_validation_test.csv` (expected difference about 1e-6, float32 rounding).

## 8. Reference only: earlier Linux run by the AI assistant (NOT authoritative)

These runs were made in the AI assistant's Linux sandbox. They are kept to show cross-platform differences and are **not** the reported results.

### Linux run history

| Run | Environment | Status |
|---|---|---|
| Initial-grid run | Python 3.12 | Superseded. Several chosen settings sat on the edge of the search grids, so the grids were widened once (decided before test metrics were viewed). Same conclusions. Logs kept in `logs/python312_runs/`. |
| Runs A, B, C | Python 3.12.3, numpy 2.5.4, xgboost 3.4.1 | Reference only. A, B and C gave **byte-identical** metrics and predictions; B and C were identical in all 15 result files and all 7 models. Results of B kept in `reference_runs/python312_run_B/`. |
| Linux Python 3.11 run (reference) | **Python 3.11.15, numpy 2.4.6, xgboost 3.2.0** | Reference only; superseded by the Mac run. |

**Why the Python 3.11 run is final.** You are working on Python 3.11, and the exact package set of runs A to C cannot be installed there (numpy 2.5.4 requires Python 3.12 or newer). Models saved under those versions might not load in your environment, and the dashboard in a later phase depends on loading them. The 3.11 run uses a package set you can install exactly (`requirements-lock.txt`).
The choice was made for **environment compatibility, not for scores**. The two environments give the same conclusions: Random Forest results are identical; XGBoost-based numbers differ slightly (for example test RMSE of XGBoost 2.4908 here vs 2.5215 in the reference run), and the XGBoost/Ridge settings chosen on validation differ. The likely cause is the different XGBoost version; I did not isolate it from the numpy/pandas differences.

**Reproducibility evidence**
* **Final run vs an independent Python 3.11 replicate** (separate folder, same code, locked versions): **15 of 15** shared result files and **7 of 7** saved models are byte-identical; only the runtime field in `model_registry.json` differs (`logs/run_comparison_final_vs_python311_replicate.txt`).
* **Python 3.12 runs A, B, C:** A and B have byte-identical metrics and predictions (only A's metrics, predictions and log survived, because B overwrote the rest); B and C are identical in all 15 result files and all 7 models (`logs/python312_runs/run_comparison_report.txt`).
* **Phase 2:** re-running `preprocessing.py` gave a byte-identical `daily_model_ready.csv`, and the processed data has the same SHA-256 under Python 3.11 and 3.12.
* **Across environments the results are not identical** (3 of 16 result files identical between the Linux Python 3.11 run and the Python 3.12 reference run), which is why exact versions are pinned (`logs/final_vs_python312_reference.txt`).
* `requirements-lock.txt` was installed into a fresh Python 3.11.15 environment and imported successfully.

### Linux verification summary

* `verify_leakage.py` (Phase 2): 19 of 19 checks passed (truncation and perturbation tests, target definition, training-only imputation, chronological split).
* `verify_phase3.py` (Phase 3): 13 of 13 checks passed (walk-forward fold chronology, perturbation test of the out-of-fold predictions, same observations for all models, reconstruction formula, saved models reloaded and reproducing every reported MAE/RMSE/R2 with maximum difference 0.0).
* Console output of every check: `logs/verify_phase2_console.txt`, `logs/verify_phase3_console.txt`.

## 9. Limitations and unverified items

* **Fitting uses the training split only.** Models are not refitted on train + validation, so test-period models are 2+ years out of date.
* **Test is a single regime** (2021-2023, much more volatile than training); do not generalise beyond it.
* Hyper-parameter search is small; ties and near-ties exist (section 5), and XGBoost results can differ slightly across machines and library versions.
* Permutation importances are tiny and noisy; built-in importances are descriptive, not causal.
* The validation actual-vs-predicted figure draws straight lines across the March-May 2020 market closure (no observations exist there).
* The exact libomp fix used on the Mac was not recorded (section 2).
* The notebook has not been run on the Mac files (section 7). The Mac raw-price metrics, bootstrap intervals and importances were not shown to the assistant, so they are not quoted in this README.
* The optional Random Forest + ARIMA hybrid is **not implemented** (future work; explained in the notebook, section 11).
* The written reports quote only the verified Mac numbers above; figures that were never shown to the assistant (Mac raw-price metrics, bootstrap intervals, importances) are left to the notebook and dashboard to display.
* The report's LightGBM/CatBoost comparison is conceptual; those models were not trained.
