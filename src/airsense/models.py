"""The forecasting model for one horizon: a point forecast plus an 80% interval.

Three LightGBM models per horizon
    point   predicts log(1 + PM2.5), squared error. Working in log space stops the
            few extreme winter days (300+ µg/m³) from dominating training, and
            matches how pollution varies: in percentages more than in absolutes.
    lo, hi  quantile regression at the 10th and 90th percentiles, giving a range
            the true value should fall inside 80% of the time.

Conformal calibration of the interval
    Quantile models are often over-confident: their 80% interval contains the
    truth less than 80% of the time. Conformalized quantile regression (CQR)
    fixes this using the validation set: measure how far outside the interval
    the actual values fell, take the 80th percentile of that distance, and widen
    every interval by it. With some assumptions, this guarantees ~80% coverage
    on new data. evaluate.py reports coverage before and after.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from airsense import config

COMMON = dict(learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1)
MAX_TREES = 4000
EARLY_STOP = 200


def _fit(objective_kwargs, X, y, X_val=None, y_val=None, n_estimators=MAX_TREES):
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **COMMON, **objective_kwargs)
    if X_val is None:
        model.fit(X, y)
    else:
        model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return model


@dataclass
class HorizonModel:
    horizon: int
    features: list[str]
    point: lgb.LGBMRegressor
    lo: lgb.LGBMRegressor
    hi: lgb.LGBMRegressor
    conformal_margin: float          # how much to widen the raw interval (µg/m³)
    n_trees: dict

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        point = np.expm1(self.point.predict(X)).clip(min=0)
        lo_raw, hi_raw = self.lo.predict(X), self.hi.predict(X)
        lo_raw, hi_raw = np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw)   # quantile models can cross
        return pd.DataFrame({
            "forecast": point,
            "lo": (lo_raw - self.conformal_margin).clip(min=0),
            "hi": hi_raw + self.conformal_margin,
            "lo_raw": lo_raw.clip(min=0),
            "hi_raw": hi_raw,
        }, index=df.index)


def conformal_margin(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """CQR: the (finite-sample corrected) quantile of how far y falls outside [lo, hi]."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level))


def train_horizon(train: pd.DataFrame, valid: pd.DataFrame, horizon: int, features: list[str]) -> HorizonModel:
    """Fit with early stopping on validation, then calibrate the interval on validation."""
    a_lo, a_hi = config.QUANTILES
    Xt, Xv = train[features], valid[features]
    point = _fit({"objective": "l2"}, Xt, np.log1p(train.target), Xv, np.log1p(valid.target))
    lo = _fit({"objective": "quantile", "alpha": a_lo}, Xt, train.target, Xv, valid.target)
    hi = _fit({"objective": "quantile", "alpha": a_hi}, Xt, train.target, Xv, valid.target)
    lo_v, hi_v = lo.predict(Xv), hi.predict(Xv)
    margin = conformal_margin(np.minimum(lo_v, hi_v), np.maximum(lo_v, hi_v), valid.target.to_numpy(), a_hi - a_lo)
    trees = {"point": point.best_iteration_, "lo": lo.best_iteration_, "hi": hi.best_iteration_}
    return HorizonModel(horizon, features, point, lo, hi, margin, trees)


def refit(model: HorizonModel, data: pd.DataFrame) -> HorizonModel:
    """Retrain on new data with the same settings (tree counts and interval margin).
    Used for scheduled retraining, where there's no fresh validation set to tune on."""
    a_lo, a_hi = config.QUANTILES
    X = data[model.features]
    return HorizonModel(
        model.horizon, model.features,
        _fit({"objective": "l2"}, X, np.log1p(data.target), n_estimators=max(1, model.n_trees["point"])),
        _fit({"objective": "quantile", "alpha": a_lo}, X, data.target, n_estimators=max(1, model.n_trees["lo"])),
        _fit({"objective": "quantile", "alpha": a_hi}, X, data.target, n_estimators=max(1, model.n_trees["hi"])),
        model.conformal_margin, model.n_trees,
    )
