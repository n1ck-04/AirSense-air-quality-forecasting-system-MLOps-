"""Cleaning, categories and, most importantly, point-in-time features."""

import numpy as np
import pandas as pd
import pytest

from airsense import config
from airsense.data import clean, pm25_category
from airsense.features import FEATURES, WEATHER_FEATURES, build_features, feature_list, split


def _raw(values, start="2015-01-01", city="X", skip=()):
    days = pd.date_range(start, periods=len(values), freq="D")
    df = pd.DataFrame({"city": city, "date": days, "pm25": values})
    for col in ["pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3"]:
        df[col] = 1.0
    return df[~df.date.isin(pd.to_datetime(list(skip)))]


def test_clean_fills_calendar_gaps_and_drops_negatives():
    raw = _raw([10.0, -5.0, 30.0, 40.0], skip=["2015-01-03"])          # one day missing, one negative
    out = clean(raw, min_history_start=pd.Timestamp("2016-01-01"))
    assert len(out) == 4                                                   # the missing day is back, as NaN
    assert out.pm25.isna().sum() == 2


def test_clean_drops_cities_without_enough_history():
    raw = pd.concat([_raw([1.0] * 10, city="Old"), _raw([1.0] * 10, start="2019-06-01", city="New")])
    assert set(clean(raw).city) == {"Old"}


@pytest.mark.parametrize("value,expected", [(0, "Good"), (30, "Good"), (30.1, "Satisfactory"), (90, "Moderate"),
                                            (90.5, "Poor"), (250, "Very Poor"), (600, "Severe"), (np.nan, None)])
def test_pm25_categories_follow_cpcb_breakpoints(value, expected):
    assert pm25_category(value) == expected


def test_lags_are_calendar_days_not_rows():
    """With a gap in the data, 'lag 1' must be NaN, not the value from before the gap."""
    raw = _raw([10.0, 20.0, 30.0, 40.0], skip=["2015-01-03"])
    f = build_features(clean(raw, min_history_start=pd.Timestamp("2016-01-01")), horizon=1)
    jan4 = f[f.date == "2015-01-04"].iloc[0]
    assert jan4.pm25_lag0 == 40 and np.isnan(jan4.pm25_lag1) and jan4.pm25_lag2 == 20


def test_target_is_pm25_horizon_days_later(clean_df):
    for h in config.HORIZONS:
        f = build_features(clean_df, h)
        city = clean_df.city.iloc[0]
        c = clean_df[clean_df.city == city].set_index("date").pm25
        row = f[(f.city == city)].dropna(subset=["target"]).iloc[100]
        assert row.target == c[row.date + pd.Timedelta(days=h)]
        assert row.target_date == row.date + pd.Timedelta(days=h)


def test_features_do_not_see_the_future(clean_df):
    """THE key test. Features for days up to a cut-off must be identical whether or
    not later data exists. Any feature that peeks ahead would change."""
    cut = pd.Timestamp("2018-06-30")
    for h in config.HORIZONS:
        full = build_features(clean_df, h)
        past = build_features(clean_df[clean_df.date <= cut], h)
        cols = [c for c in FEATURES if c != "city"]          # city becomes part of the index
        a = full[full.date <= cut].set_index(["city", "date"])[cols]
        b = past.set_index(["city", "date"])[cols]
        assert len(a) == len(b) > 1000
        pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_one_city_table_matches_full_table(clean_df):
    """Serving builds features for one city; they must equal training's, including the city code."""
    cities = sorted(clean_df.city.unique())
    full = build_features(clean_df, 1)
    one = build_features(clean_df[clean_df.city == cities[-1]], 1, cities=cities)
    a = full[full.city == cities[-1]].reset_index(drop=True)[FEATURES]
    pd.testing.assert_frame_equal(a, one.reset_index(drop=True)[FEATURES])
    assert list(one.city.cat.categories) == cities


def test_split_is_by_target_date_and_drops_unknown_targets(clean_df):
    train, valid, test = split(build_features(clean_df, 3))
    assert train.target_date.max() < config.VALID_START <= valid.target_date.min()
    assert valid.target_date.max() < config.TEST_START <= test.target_date.min()
    assert train.target.notna().all() and test.target.notna().all()


def test_weather_features_are_joined_for_today_and_target_day(clean_df):
    w = clean_df[["city", "date"]].copy()
    w["temp_mean"] = w.date.dt.dayofyear.astype(float)      # easy to check: temperature = day of year
    w["wind_mean"] = w["rh_mean"] = w["precip_sum"] = 1.0
    f = build_features(clean_df, 2, weather=w)
    row = f.dropna(subset=["temp_mean_today", "temp_mean_target"]).iloc[50]
    assert row.temp_mean_today == row.date.dayofyear
    assert row.temp_mean_target == row.target_date.dayofyear
    assert feature_list(True) == FEATURES + WEATHER_FEATURES
