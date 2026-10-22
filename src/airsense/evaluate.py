"""Forecast metrics.

All comparisons use the same rows: days where the persistence baseline exists
(PM2.5 was measured on the forecast day). Otherwise a model could look better
just by being scored on easier days.

    mae / rmse            average error in µg/m³ (RMSE punishes big misses more)
    skill_vs_persistence  share of persistence's error removed (see baselines.skill)
    skill_vs_best_baseline  the same, against whichever baseline was best at that horizon
    category_accuracy     forecast lands in the right CPCB PM2.5 category
    poor_alert_*          "Poor or worse" (> 90 µg/m³) as a yes/no alert:
                          precision = alerts that were right, recall = bad days warned about
    interval_coverage     share of actual values inside the 80% prediction interval (target: 0.80)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airsense import config
from airsense.baselines import skill
from airsense.data import pm25_category


def _alert_scores(pred: np.ndarray, actual: np.ndarray) -> dict:
    p, a = pred > config.POOR_THRESHOLD, actual > config.POOR_THRESHOLD
    tp = int((p & a).sum())
    return {"precision": round(tp / max(1, p.sum()), 3), "recall": round(tp / max(1, a.sum()), 3)}


def point_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = pred - actual
    cats_p = [pm25_category(x) for x in pred]
    cats_a = [pm25_category(x) for x in actual]
    alert = _alert_scores(pred, actual)
    return {
        "mae": round(float(np.abs(err).mean()), 2),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
        "bias": round(float(err.mean()), 2),
        "category_accuracy": round(float(np.mean([x == y for x, y in zip(cats_p, cats_a, strict=True)])), 3),
        "poor_alert_precision": alert["precision"],
        "poor_alert_recall": alert["recall"],
    }


def interval_metrics(lo: np.ndarray, hi: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "interval_coverage": round(float(((actual >= lo) & (actual <= hi)).mean()), 3),
        "interval_width": round(float((hi - lo).mean()), 1),
    }


def evaluate_horizon(fc: pd.DataFrame) -> dict:
    """fc: one row per forecast with columns actual, forecast, lo, hi, persistence, mean7, last_year, target_date."""
    fc = fc[fc.persistence.notna()]
    y = fc.actual.to_numpy()
    model = {**point_metrics(fc.forecast.to_numpy(), y), **interval_metrics(fc.lo.to_numpy(), fc.hi.to_numpy(), y)}
    if "lo_raw" in fc:
        raw = interval_metrics(fc.lo_raw.to_numpy(), fc.hi_raw.to_numpy(), y)
        model["interval_coverage_uncalibrated"] = raw["interval_coverage"]
    out = {"rows": len(fc), "model": model}
    for b in ["persistence", "mean7", "last_year"]:
        ok = fc[b].notna()
        out[b] = point_metrics(fc.loc[ok, b].to_numpy(), fc.loc[ok, "actual"].to_numpy())
    out["skill_vs_persistence"] = round(skill(model["mae"], out["persistence"]["mae"]), 3)
    # The toughest comparison: whichever baseline did best at this horizon.
    best = min(["persistence", "mean7", "last_year"], key=lambda b: out[b]["mae"])
    out["best_baseline"] = best
    out["skill_vs_best_baseline"] = round(skill(model["mae"], out[best]["mae"]), 3)

    # Before vs during the COVID-19 lockdown
    for name, mask in [("before_lockdown", fc.target_date < config.LOCKDOWN_START),
                       ("lockdown", fc.target_date >= config.LOCKDOWN_START)]:
        part = fc[mask]
        m = float(np.abs(part.forecast - part.actual).mean())
        p = float(np.abs(part.persistence - part.actual).mean())
        out[name] = {"rows": len(part), "model_mae": round(m, 2), "persistence_mae": round(p, 2),
                     "skill": round(skill(m, p), 3), "mean_pm25": round(float(part.actual.mean()), 1)}
    return out


def per_city(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc[fc.persistence.notna()]
    g = fc.groupby("city", observed=True)
    out = pd.DataFrame({
        "days": g.size(),
        "mean_pm25": g.actual.mean(),
        "model_mae": g.apply(lambda x: np.abs(x.forecast - x.actual).mean()),
        "persistence_mae": g.apply(lambda x: np.abs(x.persistence - x.actual).mean()),
    })
    out["skill"] = 1 - out.model_mae / out.persistence_mae
    return out.round(3).sort_values("skill", ascending=False)
