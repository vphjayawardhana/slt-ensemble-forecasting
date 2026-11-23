"""
The models used in the project.

NOTE for later: saved .joblib files refer to the classes in THIS module, so the
Streamlit dashboard must add the `src` folder to sys.path and `import ensemble_models`
before loading the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42


# --------------------------------------------------------------------------
# Model 4: persistence baseline
# --------------------------------------------------------------------------
class PersistenceBaseline:
    """
    "Tomorrow's close = today's close."
    In log-return form this means the predicted return is always 0, so
    close_t * exp(0) = close_t. It has nothing to learn.
    """

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


# --------------------------------------------------------------------------
# Model 1: Random Forest (Bagging)
# --------------------------------------------------------------------------
def make_random_forest(params: dict) -> RandomForestRegressor:
    """
    BAGGING = "Bootstrap AGGregatING".
      * Each tree is trained on a random bootstrap sample (rows drawn WITH replacement).
      * Each split inside a tree only considers a random subset of features.
      * The forest's prediction is the AVERAGE of all trees, which cancels out much of
        the noise of any single tree (it reduces variance).
    Important settings
      n_estimators     number of trees (more = more stable, slower)
      max_depth        how deep each tree may grow (shallower = simpler, less over-fitting)
      min_samples_leaf minimum rows in a leaf (bigger = smoother predictions)
      max_features     share of features tried at each split (smaller = more diverse trees)
    """
    return RandomForestRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", 1.0),
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,            # CPU only; works on Apple Silicon without any GPU
    )


# --------------------------------------------------------------------------
# Model 2: XGBoost (Boosting)
# --------------------------------------------------------------------------
def make_xgboost(params: dict) -> XGBRegressor:
    """
    BOOSTING = build small trees ONE AFTER ANOTHER; each new tree tries to fix the
    errors (residuals) left by all previous trees.
    Important settings
      n_estimators      number of boosting rounds (trees added in sequence)
      learning_rate     how big a step each new tree takes (small = slow, careful learning)
      max_depth         size of each tree (small trees = "weak learners")
      min_child_weight  minimum amount of data a leaf needs (bigger = more cautious)
      subsample / colsample_bytree  random share of rows / features per tree (reduces over-fitting)
      reg_lambda        L2 penalty that shrinks leaf values
    """
    return XGBRegressor(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 3),
        min_child_weight=params.get("min_child_weight", 1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective="reg:squarederror",
        tree_method="hist",   # fast CPU algorithm
        n_jobs=4,
        random_state=SEED,
        verbosity=0,
    )


# --------------------------------------------------------------------------
# Model 3: Stacking with chronological (walk-forward) out-of-fold predictions
# --------------------------------------------------------------------------
class WalkForwardStackingRegressor:
    """
    Base learners : Random Forest + XGBoost
    Meta-learner  : Ridge regression (on standardised base predictions)

    Why not scikit-learn's StackingRegressor? It builds out-of-fold predictions with
    K-fold cross-validation, which lets a model train on FUTURE rows to predict PAST
    rows. For time series that leaks information. This class instead uses an
    EXPANDING WINDOW:

        fold 1: train on rows [0 .. a)      -> predict block 1 = rows [a .. b)
        fold 2: train on rows [0 .. b)      -> predict block 2 = rows [b .. c)
        fold 3: train on rows [0 .. c)      -> predict block 3 = rows [c .. d)   ...

    Every out-of-fold (OOF) prediction therefore comes from models that only saw
    EARLIER rows. The Ridge meta-model learns from these OOF predictions only, so it
    never sees a prediction made by a model that trained on that row's own target.
    The first `initial_train_frac` of rows has no OOF prediction and is used only as
    the starting training window for fold 1.

    After the meta-model is fitted, the base learners are re-fitted on ALL training rows
    and are used for validation/test predictions.
    """

    def __init__(self, rf_params: dict, xgb_params: dict, ridge_alpha: float = 1.0,
                 initial_train_frac: float = 0.40, n_folds: int = 5):
        self.rf_params = rf_params
        self.xgb_params = xgb_params
        self.ridge_alpha = ridge_alpha
        self.initial_train_frac = initial_train_frac
        self.n_folds = n_folds
        self.base_names = ["random_forest", "xgboost"]

    def _new_base_models(self) -> dict:
        return {"random_forest": make_random_forest(self.rf_params),
                "xgboost": make_xgboost(self.xgb_params)}

    # ---- step 1: chronological out-of-fold predictions -------------------
    def compute_oof(self, X: pd.DataFrame, y: np.ndarray, dates=None):
        n = len(X)
        first = int(n * self.initial_train_frac)
        edges = np.linspace(first, n, self.n_folds + 1).astype(int)
        oof = np.full((n, len(self.base_names)), np.nan)
        self.fold_info_ = []
        for k in range(self.n_folds):
            train_end, pred_start, pred_end = edges[k], edges[k], edges[k + 1]
            assert train_end <= pred_start          # training window ends before the block begins
            models = self._new_base_models()
            for j, name in enumerate(self.base_names):
                models[name].fit(X.iloc[:train_end], y[:train_end])
                oof[pred_start:pred_end, j] = models[name].predict(X.iloc[pred_start:pred_end])
            info = {"fold": k + 1, "train_rows": f"0..{train_end - 1}", "n_train": int(train_end),
                    "predict_rows": f"{pred_start}..{pred_end - 1}", "n_predict": int(pred_end - pred_start)}
            if dates is not None:
                info["train_end_date"] = str(pd.Timestamp(dates[train_end - 1]).date())
                info["predict_start_date"] = str(pd.Timestamp(dates[pred_start]).date())
                info["predict_end_date"] = str(pd.Timestamp(dates[pred_end - 1]).date())
            self.fold_info_.append(info)
        self.oof_predictions_ = oof
        self.oof_target_ = np.asarray(y)
        return oof

    # ---- step 2: base learners on the full training set ------------------
    def fit_base_models(self, X: pd.DataFrame, y: np.ndarray):
        self.base_models_ = self._new_base_models()
        for m in self.base_models_.values():
            m.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    # ---- step 3: Ridge meta-model trained ONLY on OOF predictions --------
    def fit_meta(self, alpha: float | None = None):
        if alpha is not None:
            self.ridge_alpha = alpha
        mask = ~np.isnan(self.oof_predictions_).any(axis=1)
        self.meta_ = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=self.ridge_alpha))])
        self.meta_.fit(self.oof_predictions_[mask], self.oof_target_[mask])
        return self

    def fit(self, X: pd.DataFrame, y, dates=None):
        y = np.asarray(y)
        self.compute_oof(X, y, dates)
        self.fit_base_models(X, y)
        return self.fit_meta()

    def base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.base_models_[n].predict(X) for n in self.base_names])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_.predict(self.base_predictions(X))

    def meta_weights(self) -> pd.DataFrame:
        """Ridge weights in the original units of the base predictions."""
        scaler, ridge = self.meta_.named_steps["scale"], self.meta_.named_steps["ridge"]
        raw = ridge.coef_ / scaler.scale_
        return pd.DataFrame({"base_learner": self.base_names,
                             "weight_on_prediction": raw,
                             "standardised_coefficient": ridge.coef_}).assign(intercept=float(
            ridge.intercept_ - (raw * scaler.mean_).sum()))
