"""Produce a 3-day outlook for one city, as of any date in the history.

This is the serving code path used by the API. It reuses build_features() on
the city's history up to the as-of date, so the model sees features computed
exactly as in training.

Why "as of a date" instead of "today"?
    The dataset ends on 1 July 2020, so there is no live sensor feed. Choosing an
    as-of date replays what the service would have said on that morning. Hooking
    up a live feed (e.g. the CPCB or OpenAQ API) would only change where the
    history comes from; see docs/EXERCISES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from airsense import config
from airsense.data import pm25_category
from airsense.features import build_features, load_weather


@dataclass
class DayForecast:
    horizon: int
    date: str
    pm25: float
    lo: float
    hi: float
    category: str


class Forecaster:
    def __init__(self, bundle: dict, clean: pd.DataFrame, weather: pd.DataFrame | None = None):
        self.bundle = bundle
        self.models = bundle["models"]
        self.cities = bundle["cities"]
        self.history = clean
        self.weather = weather if bundle.get("with_weather") else None
        self.first_day, self.last_day = clean.date.min(), clean.date.max()

    @classmethod
    def from_disk(cls) -> "Forecaster":
        return cls(joblib.load(config.BUNDLE_PATH), pd.read_parquet(config.CLEAN_PATH), load_weather())

    def forecast(self, city: str, as_of: pd.Timestamp | None = None) -> dict:
        if city not in self.cities:
            raise KeyError(city)
        as_of = pd.Timestamp(as_of or self.last_day).normalize()
        if not (self.first_day <= as_of <= self.last_day):
            raise ValueError(f"as_of must be between {self.first_day:%Y-%m-%d} and {self.last_day:%Y-%m-%d}")

        # Only the past: everything after the as-of date is invisible to the forecast.
        hist = self.history[(self.history.city == city) & (self.history.date <= as_of)]
        days = []
        for h, model in sorted(self.models.items()):
            table = build_features(hist, h, self.weather, cities=self.cities)
            row = table[table.date == as_of]
            p = model.predict(row).iloc[0]
            days.append(DayForecast(h, str((as_of + pd.Timedelta(days=h)).date()), round(float(p.forecast), 1),
                                    round(float(p.lo), 1), round(float(p.hi), 1), pm25_category(p.forecast)))
        today = hist[hist.date == as_of].pm25
        observed = None if today.empty or pd.isna(today.iloc[0]) else round(float(today.iloc[0]), 1)
        return {"city": city, "as_of": str(as_of.date()), "observed_pm25": observed,
                "forecasts": [d.__dict__ for d in days]}
