"""Monitoring and retraining, replayed over the test year.

The question
    The model goes live on 1 July 2019. Over the next year, Indian cities go
    through a monsoon, a winter smog season and, from 25 March 2020, a national
    COVID-19 lockdown that cut traffic and industry. When should the model be
    retrained, and how would we know?

The replay
    We step through the year one week at a time, using only what would have
    been known on that date, and compare three retraining policies:
        static     train once on everything before 1 July 2019, never retrain
        monthly    retrain on the 1st of every month with all data so far
        triggered  retrain only when a monitoring signal fires (see below)

Three monitoring signals, computed every week
    1. naive_psi       Population Stability Index of today's PM2.5 vs the
                       training data. The textbook drift check (alarm at > 0.2).
    2. seasonal_shift  how far this week's PM2.5 is from the SAME city in the
                       SAME month of the training years, in standard deviations
                       (a z-score). Removes the normal seasonal swing.
    3. skill_28d       the model's skill vs persistence over the last 28 days
                       of forecasts whose actual value is now known. This is
                       the one that measures what we care about: is the model
                       still adding value?

    The triggered policy retrains when skill_28d falls below 0 (the model is
    losing to "tomorrow = today") or |seasonal_shift| exceeds 1.0, with at least
    28 days between retrains.

Run:  python -m airsense.monitor     (after train.py)
"""

from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

from airsense import config
from airsense.baselines import skill
from airsense.features import build_features, load_weather
from airsense.models import refit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HORIZON = 1
PSI_ALARM = 0.2
SHIFT_ALARM = 1.0
SKILL_ALARM = 0.0
COOLDOWN_DAYS = 28
SKILL_WINDOW_DAYS = 28


# ---------------------------------------------------------------------- drift measures
def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index over deciles of the reference distribution.
    Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift."""
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(current) == 0:
        return float("nan")
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / len(reference) + 1e-4
    q = np.histogram(current, edges)[0] / len(current) + 1e-4
    return float(np.sum((q - p) * np.log(q / p)))


class SeasonalBaseline:
    """Each city's typical log(PM2.5) and spread for each calendar month, from training data."""

    def __init__(self, history: pd.DataFrame):
        h = history.dropna(subset=["pm25"]).assign(month=lambda d: d.date.dt.month, logpm=lambda d: np.log1p(d.pm25))
        self.stats = h.groupby(["city", "month"]).logpm.agg(["mean", "std"])

    def zscores(self, days: pd.DataFrame) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([days.city.astype(str), days.date.dt.month])
        s = self.stats.reindex(keys)
        return pd.Series((np.log1p(days.pm25.to_numpy()) - s["mean"].to_numpy()) / s["std"].to_numpy(), index=days.index)


# --------------------------------------------------------------------------- replay
def replay(clean: pd.DataFrame, table: pd.DataFrame, base_model, policy: str):
    """Forecast every test day under one retraining policy. Returns forecasts and retrain dates."""
    history = clean[clean.date < config.TEST_START]
    seasonal = SeasonalBaseline(history)
    reference = history.pm25.to_numpy(dtype=float)

    def known_before(day):          # rows whose actual value is known on `day`
        return table[(table.target.notna()) & (table.target_date < day)]

    model = refit(base_model, known_before(config.TEST_START))
    retrains, weeks, preds = [], [], []
    last_retrain = config.TEST_START
    end = table.target_date.max()
    week_start = config.TEST_START
    while week_start <= end:
        week_end = week_start + pd.Timedelta(days=7)
        # ----- monitoring, using only information available at week_start
        recent = clean[(clean.date >= week_start - pd.Timedelta(days=7)) & (clean.date < week_start)]
        done = pd.concat(preds) if preds else pd.DataFrame(columns=["target_date", "forecast", "actual", "persistence"])
        window = done[(done.target_date < week_start) & (done.target_date >= week_start - pd.Timedelta(days=SKILL_WINDOW_DAYS))]
        window = window.dropna(subset=["actual", "persistence"])
        sig = {
            "week": week_start,
            "naive_psi": psi(reference, recent.pm25.to_numpy(dtype=float)),
            "seasonal_shift": float(np.nanmean(seasonal.zscores(recent))) if len(recent) else np.nan,
            "skill_28d": skill(float(np.abs(window.forecast - window.actual).mean()),
                               float(np.abs(window.persistence - window.actual).mean())) if len(window) > 50 else np.nan,
        }
        # ----- retraining decision
        due = False
        if policy == "monthly":
            due = week_start.month != (week_start - pd.Timedelta(days=7)).month and week_start > config.TEST_START
        elif policy == "triggered":
            cooled = (week_start - last_retrain).days >= COOLDOWN_DAYS
            fired = (sig["skill_28d"] < SKILL_ALARM) or (abs(sig["seasonal_shift"]) > SHIFT_ALARM)
            due = cooled and bool(fired)
        if due:
            model = refit(model, known_before(week_start))
            retrains.append(str(week_start.date()))
            last_retrain = week_start
        sig["retrained"] = due
        weeks.append(sig)
        # ----- forecast this week's target days
        rows = table[(table.target_date >= week_start) & (table.target_date < week_end) & table.target.notna()]
        if len(rows):
            p = model.predict(rows)
            preds.append(pd.DataFrame({"target_date": rows.target_date, "city": rows.city.astype(str),
                                       "forecast": p.forecast, "actual": rows.target, "persistence": rows.pm25_lag0}))
        week_start = week_end
    return pd.concat(preds, ignore_index=True), retrains, pd.DataFrame(weeks)


def summarise(fc: pd.DataFrame) -> dict:
    fc = fc.dropna(subset=["persistence"])
    out = {}
    for name, mask in [("year", slice(None)), ("before_lockdown", fc.target_date < config.LOCKDOWN_START),
                       ("lockdown", fc.target_date >= config.LOCKDOWN_START)]:
        part = fc[mask]
        mae = float(np.abs(part.forecast - part.actual).mean())
        out[name] = {"mae": round(mae, 2),
                     "skill": round(skill(mae, float(np.abs(part.persistence - part.actual).mean())), 3)}
    return out


def plot(weeks: pd.DataFrame, retrains: list[str], path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    panels = [("naive_psi", "Naive drift (PSI)", PSI_ALARM, "alarm > 0.2"),
              ("seasonal_shift", "Season-adjusted shift (z)", -SHIFT_ALARM, "alarm < -1"),
              ("skill_28d", "Model skill vs persistence,\nlast 28 days", SKILL_ALARM, "alarm < 0")]
    for ax, (col, label, line, note) in zip(axes, panels, strict=True):
        ax.plot(weeks.week, weeks[col], marker="o", ms=3)
        ax.axhline(line, ls=":", color="red", lw=1)
        ax.text(weeks.week.iloc[0], line, f" {note}", color="red", fontsize=7, va="bottom")
        ax.axvspan(config.LOCKDOWN_START, weeks.week.max(), color="grey", alpha=0.15)
        ax.set_ylabel(label, fontsize=8)
    for r in retrains:
        for ax in axes:
            ax.axvline(pd.Timestamp(r), color="green", lw=0.8, alpha=0.6)
    axes[0].set_title("Weekly monitoring over the test year (grey = lockdown, green = triggered retrain)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    clean = pd.read_parquet(config.CLEAN_PATH)
    bundle = joblib.load(config.BUNDLE_PATH)
    table = build_features(clean, HORIZON, load_weather())
    base = bundle["models"][HORIZON]

    results, all_weeks = {}, None
    for policy in ["static", "monthly", "triggered"]:
        fc, retrains, weeks = replay(clean, table, base, policy)
        results[policy] = {**summarise(fc), "retrains": len(retrains), "retrain_dates": retrains}
        print(f"{policy:<9} MAE {results[policy]['year']['mae']:.2f} | lockdown MAE {results[policy]['lockdown']['mae']:.2f}"
              f" | retrains {len(retrains)}")
        if policy == "triggered":
            all_weeks, trig = weeks, retrains

    w = all_weeks
    lock = w.week >= config.LOCKDOWN_START
    alarms = {
        "weeks": len(w),
        "naive_psi_alarm_weeks": int((w.naive_psi > PSI_ALARM).sum()),
        "naive_psi_alarm_weeks_before_lockdown": int(((w.naive_psi > PSI_ALARM) & ~lock).sum()),
        "seasonal_alarm_weeks": int((w.seasonal_shift.abs() > SHIFT_ALARM).sum()),
        "seasonal_alarm_weeks_before_lockdown": int(((w.seasonal_shift.abs() > SHIFT_ALARM) & ~lock).sum()),
        "seasonal_shift_mean_before_lockdown": round(float(w.seasonal_shift[~lock].mean()), 2),
        "seasonal_shift_mean_lockdown": round(float(w.seasonal_shift[lock].mean()), 2),
        "first_seasonal_alarm_after_lockdown": next((str(d.date()) for d, s in zip(w.week, w.seasonal_shift, strict=True)
                                                     if d >= config.LOCKDOWN_START and abs(s) > SHIFT_ALARM), None),
        "min_skill_28d": round(float(w.skill_28d.min()), 3),
        "skill_alarm_weeks": int((w.skill_28d < SKILL_ALARM).sum()),
    }
    out = {"horizon_days": HORIZON, "policies": results, "monitoring": alarms}
    (config.REPORTS_DIR / "monitoring.json").write_text(json.dumps(out, indent=2))
    w.to_csv(config.REPORTS_DIR / "monitoring_weekly.csv", index=False)
    plot(w, trig, config.REPORTS_DIR / "monitoring.png")
    print(json.dumps(alarms, indent=2))


if __name__ == "__main__":
    main()
