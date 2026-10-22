"""Turn the daily history into a supervised forecasting table.

One row = one city on one day t (the day the forecast is MADE), with:
    features   computed from days <= t only
    target     PM2.5 on day t + horizon

Point-in-time correctness
    A forecast made on the evening of day t can only use what was measured
    up to day t. Every feature below uses .shift(k) with k >= 0 or a rolling
    window that ends at t, on a complete daily calendar (see data.py), so
    "lag 1" always means "yesterday". tests/test_features.py checks this by
    rebuilding features with the future deleted and confirming nothing changes.

Same function for training and serving
    The API (forecast.py) calls build_features() on the history up to the
    as-of date and takes the last row. One implementation, so the features the
    model was trained on are exactly the features it gets in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airsense import config

FEATURE_DESCRIPTIONS = {
    **{f"pm25_lag{k}": f"PM2.5 {k} day(s) before the forecast day" if k else "PM2.5 on the forecast day"
       for k in range(7)},
    "pm25_mean3": "PM2.5 average, last 3 days",
    "pm25_mean7": "PM2.5 average, last 7 days",
    "pm25_mean14": "PM2.5 average, last 14 days",
    "pm25_mean30": "PM2.5 average, last 30 days",
    "pm25_std7": "PM2.5 variability, last 7 days",
    "pm25_change1": "change in PM2.5 since the day before",
    "pm25_last_year": "PM2.5 around the target date one year earlier",
    "pm10": "PM10 on the forecast day",
    "pm10_mean7": "PM10 average, last 7 days",
    "no2": "NO2 on the forecast day",
    "nox": "NOx on the forecast day",
    "co": "CO on the forecast day",
    "so2": "SO2 on the forecast day",
    "o3": "O3 on the forecast day",
    "nh3": "NH3 on the forecast day",
    "target_doy_sin": "season of the target date (sine of day of year)",
    "target_doy_cos": "season of the target date (cosine of day of year)",
    "target_dow": "day of week of the target date",
    "city": "city",
}
FEATURES = list(FEATURE_DESCRIPTIONS)

# Optional weather features (only used if data/raw/weather.csv exists; see scripts/fetch_weather.py)
WEATHER_VARS = ["temp_mean", "wind_mean", "rh_mean", "precip_sum"]
WEATHER_FEATURES = [f"{v}_today" for v in WEATHER_VARS] + [f"{v}_target" for v in WEATHER_VARS]


def _city_features(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = g.sort_values("date")
    pm = g.pm25
    out = pd.DataFrame({"city": g.city.values, "date": g.date.values}, index=g.index)
    for k in range(7):
        out[f"pm25_lag{k}"] = pm.shift(k)
    for w in (3, 7, 14, 30):
        out[f"pm25_mean{w}"] = pm.rolling(w, min_periods=max(2, w // 2)).mean()
    out["pm25_std7"] = pm.rolling(7, min_periods=4).std()
    out["pm25_change1"] = pm - pm.shift(1)
    # Same season last year: the 15 days centred on (target date - 365 days), i.e. days
    # t-372+h .. t-358+h. Written as a TRAILING window on a series shifted by 358-h days, so
    # computing it needs no rows after day t. (An earlier version used rolling(center=True),
    # which reads rows after t; it gave slightly different values in serving, where those rows
    # don't exist yet. tests/test_features.py::test_features_do_not_see_the_future caught it.)
    out["pm25_last_year"] = pm.shift(358 - horizon).rolling(15, min_periods=5).mean()
    for col in ["pm10", "no2", "nox", "co", "so2", "o3", "nh3"]:
        out[col] = g[col]
    out["pm10_mean7"] = g.pm10.rolling(7, min_periods=3).mean()

    out["target_date"] = out.date + pd.Timedelta(days=horizon)
    out["target"] = pm.shift(-horizon)          # the ONLY column that looks forward
    return out


def build_features(clean: pd.DataFrame, horizon: int, weather: pd.DataFrame | None = None,
                   cities: list[str] | None = None) -> pd.DataFrame:
    """Feature table for one horizon. Rows with an unknown target are kept
    (the API needs them to forecast the future); training drops them.

    `cities` fixes the list of city categories. The API passes the training list,
    so a table built for ONE city still encodes that city exactly as in training."""
    cities = sorted(cities or clean.city.unique())
    parts = [_city_features(g, horizon) for _, g in clean.groupby("city")]
    df = pd.concat(parts, ignore_index=True)

    doy = df.target_date.dt.dayofyear
    df["target_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["target_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["target_dow"] = df.target_date.dt.dayofweek
    df["city"] = pd.Categorical(df.city, categories=cities)

    if weather is not None:
        w = weather[["city", "date", *WEATHER_VARS]]
        df["city"] = df.city.astype(str)
        df = df.merge(w.rename(columns={v: f"{v}_today" for v in WEATHER_VARS}), on=["city", "date"], how="left")
        # Weather ON the target day. In real use this comes from a weather FORECAST; here we use
        # what was later observed, i.e. a perfect forecast. Results with it are an upper bound.
        df = df.merge(w.rename(columns={"date": "target_date", **{v: f"{v}_target" for v in WEATHER_VARS}}),
                      on=["city", "target_date"], how="left")
        df["city"] = pd.Categorical(df.city.astype(str), categories=cities)
    return df


def feature_list(with_weather: bool = False) -> list[str]:
    return FEATURES + (WEATHER_FEATURES if with_weather else [])


def split(df: pd.DataFrame):
    """Train / validation / test by TARGET date. Rows without a known target are dropped."""
    df = df[df.target.notna()]
    train = df[df.target_date < config.VALID_START]
    valid = df[(df.target_date >= config.VALID_START) & (df.target_date < config.TEST_START)]
    test = df[df.target_date >= config.TEST_START]
    return train, valid, test


def load_weather():
    if config.WEATHER_PATH.exists():
        return pd.read_csv(config.WEATHER_PATH, parse_dates=["date"])
    return None
