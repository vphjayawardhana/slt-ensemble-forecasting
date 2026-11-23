# Project Summary

**Forecasting Sri Lanka Telecom Stock Prices Using Ensemble Learning: A Comparative Study of Random Forest, XGBoost and Stacking Models**

*All results below come from the run on the student's MacBook Air M1 (Python 3.11.18, scikit-learn 1.9.1, XGBoost 3.2.0). This is a historical back-test for a university assignment, not financial advice.*

## 1. Aim
To forecast the **next trading observation's closing price** of Sri Lanka Telecom from information known at the close of the current day, and to compare three ensemble methods (bagging, boosting, stacking) with a simple **persistence baseline** ("tomorrow's close = today's close").

## 2. Data
Only the supplied file `TeleCom_2009_2023_Daily.csv` was used: 3,239 daily records from 8 July 2009 to 12 April 2023 (Date, Price, Open, High, Low, Vol., Change %). 18 volumes are missing; there are no duplicate dates. No other or synthetic data was used. The original file was never modified (its SHA-256 hash is checked by the scripts).

## 3. Method
* **Target.** The main models predict the log return `log(Price[t+1] / Price[t])`; the price is rebuilt as `Price[t] x exp(predicted return)`. All metrics are computed on prices in LKR. A second experiment predicts the raw next price directly.
* **Features (26).** Price-lag ratios (1, 2, 3, 5, 10, 20), close-to-moving-average ratios (5, 10, 20), past log returns, rolling volatility, intraday open/high/low ratios and range, volume (log, missing flag, ratio to its 20-day average), Change %, day of week and month. All are known at the close of day *t*. Features are scale-free so that the models are not tied to the price level.
* **Missing volume.** Filled with the training-set median (6,030) plus a missing flag.
* **Chronological split (70/15/15, never shuffled).** Train 2,246 rows (2009-08-07 to 2019-01-18), validation 486 (2019-01-21 to 2021-03-23), test 486 (2021-03-24 to 2023-04-11). Models are fitted on training only, hyper-parameters are chosen on validation only, and the test set is used once.
* **Leakage checks.** Truncation and perturbation tests (changing future data must not change a feature), a lag-boundary test and a check that imputation uses training data only.
* **Models.** *Random Forest* (bagging: many trees on bootstrap samples, predictions averaged). *XGBoost* (boosting: small trees built one after another, each correcting the previous errors, controlled by the learning rate and number of rounds). *Stacking*: Random Forest and XGBoost predictions combined by a Ridge regression that learns only from **expanding-window walk-forward out-of-fold predictions** (5 folds, first window 898 rows), never from shuffled K-fold.
* **Metrics.** MAE, RMSE, R², directional accuracy (on days the price moved) and a block-bootstrap interval for the MAE improvement over the baseline.

## 4. Results (return-based models, prices in LKR)

| Split | Model | MAE | RMSE | R² | Direction acc. |
|---|---|---|---|---|---|
| Validation | Persistence baseline | 0.5247 | 0.8022 | 0.9675 | n/a |
| | Random Forest | 0.5087 | 0.7817 | 0.9692 | 64.1% |
| | XGBoost | 0.5057 | 0.7768 | 0.9695 | 64.1% |
| | Stacking | **0.5051** | **0.7750** | **0.9697** | 64.9% |
| Test | Persistence baseline | **1.3903** | **2.4675** | **0.9806** | n/a |
| | Random Forest | 1.4019 | 2.4834 | 0.9804 | 55.0% |
| | XGBoost | 1.4120 | 2.5032 | 0.9801 | 58.8% |
| | Stacking | 1.4068 | 2.4886 | 0.9803 | 57.4% |

Selected settings (chosen on validation): Random Forest `max_depth=15, min_samples_leaf=150, max_features=1.0, n_estimators=300`; XGBoost `max_depth=2, learning_rate=0.03, min_child_weight=1, n_estimators=400`; Ridge `alpha=10`.

## 5. Interpretation
1. **Stacking was best on validation**, lowering MAE by about 3.7% and RMSE by about 3.4% against the baseline.
2. **On the test set the baseline was best.** The ensembles' MAE was higher by 0.8% (Random Forest), 1.6% (XGBoost) and 1.2% (stacking). The differences are small, and the notebook and dashboard show the bootstrap intervals for them.
3. **Why validation results did not carry over (observations, not proof).** The test period (2021-2023) was much more volatile than training, and the weak pattern the models learned (a rise today tends to be followed by a fall) faded: the correlation between today's and tomorrow's return was about -0.22 in training, -0.13 in validation and +0.04 in test, computed in the notebook. Validation scores are also optimistic because settings were tuned there.
4. **Direction.** The models called the direction right on 55-59% of test days when the price moved, better than chance, but they miss the size of large moves, which dominates MAE and RMSE.
5. **Raw price vs return.** Tree models cannot predict above the highest price seen in training (63.1 LKR), while test prices reached 113.75 LKR, so raw-price models fail in the test period. Predicting the return avoids this. (Raw-price figures are in the notebook and dashboard.)
6. **Ties and near-ties.** Three Random Forest candidates (`max_depth` 6, 10, 15) are the same model because `min_samples_leaf=150` already limits the depth. The best XGBoost setting beat the runner-up by only 0.12% in validation RMSE. Different machines may therefore select slightly different settings; this was observed between the Linux reference run and the Mac run.

## 6. Brief comparison: Random Forest, XGBoost, LightGBM, CatBoost
*(Conceptual comparison only; LightGBM and CatBoost were **not** trained in this project.)*

| | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| Idea | Bagging: independent trees averaged | Boosting: sequential trees, level-wise growth | Boosting: sequential trees, leaf-wise growth, histogram binning | Boosting with ordered boosting and symmetric trees |
| Strength | Robust, few settings, hard to overfit | Accurate, regularised, widely used | Very fast on large data | Strong defaults, best with categorical features |
| Weakness | Cannot extrapolate; less accurate than boosting on some tasks | More settings to tune; needs libomp on macOS | Leaf-wise growth can overfit small data | Slower to train on some tasks |
| Fit here | Used (bagging) | Used (boosting) | Similar to XGBoost; future work | Little benefit (no categorical features); future work |

## 7. Limitations
* One stock and one test regime; do not generalise.
* Models were not refitted on train + validation, so test-period models are 2+ years old.
* Small hyper-parameter search with ties; XGBoost can differ slightly between computers and library versions.
* Feature importance is descriptive, not causal.
* Predicting a price is not the same as making a profitable trade; costs and risk were not modelled.

## 8. Future work
Refit on train + validation; leakage-safe **Random Forest + ARIMA hybrid** (not implemented because it needs ARIMA fitted on an expanding window of past residuals at every forecast origin, which is hard to verify, and was left out rather than done wrongly); LightGBM and CatBoost; probabilistic (interval) forecasts; more stocks and periods.

## 9. Reproducibility
Exact package versions are in `requirements-lock.txt`; the run order is in `README.md`. The notebook and dashboard read the saved models and results and do not retrain.
