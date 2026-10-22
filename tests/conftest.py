"""Shared fixtures: a small FAKE dataset and a model trained on it, built once per test run.

Tests never touch data/, models/ or reports/, so they run on a fresh clone with
no download, and can't be broken by (or break) the real pipeline outputs.
"""

import pytest

from airsense import config
from airsense.data import clean, standardise
from airsense.features import FEATURES, build_features, split
from airsense.models import train_horizon
from airsense.synthetic import make_city_day


@pytest.fixture(scope="session")
def clean_df():
    return clean(standardise(make_city_day(n_cities=3, seed=1)))


@pytest.fixture(scope="session")
def bundle(clean_df):
    models = {}
    for h in config.HORIZONS:
        train, valid, _ = split(build_features(clean_df, h))
        models[h] = train_horizon(train, valid, h, FEATURES)
    return {"models": models, "features": FEATURES, "cities": sorted(clean_df.city.unique()),
            "trained_until": str(config.VALID_START.date()), "with_weather": False}
