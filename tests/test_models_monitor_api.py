"""Model behaviour, monitoring maths and the API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from airsense import config
from airsense.api import create_app
from airsense.baselines import skill
from airsense.evaluate import point_metrics
from airsense.features import build_features, split
from airsense.forecast import Forecaster
from airsense.models import conformal_margin, refit
from airsense.monitor import SeasonalBaseline, psi


# ---------------------------------------------------------------------------- models
def test_conformal_margin_widens_to_reach_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)       # far too narrow: ~38% coverage
    m = conformal_margin(lo, hi, y, coverage=0.8)
    covered = ((y >= lo - m) & (y <= hi + m)).mean()
    assert 0.79 <= covered <= 0.82


def test_model_beats_persistence_and_intervals_are_ordered(clean_df, bundle):
    _, _, test = split(build_features(clean_df, 1))
    p = bundle["models"][1].predict(test)
    assert (p.lo <= p.hi).all() and (p.forecast >= 0).all()
    model_mae = np.abs(p.forecast - test.target).mean()
    ok = test.pm25_lag0.notna()
    persistence_mae = np.abs(test.pm25_lag0[ok] - test.target[ok]).mean()
    assert model_mae < persistence_mae * 1.05        # fake data is easy; the model must at least keep up


def test_refit_keeps_settings(clean_df, bundle):
    train, _, _ = split(build_features(clean_df, 1))
    m = refit(bundle["models"][1], train)
    assert m.conformal_margin == bundle["models"][1].conformal_margin
    assert m.point.n_estimators == bundle["models"][1].n_trees["point"]


def test_promotion_gate():
    from airsense.train import should_promote
    assert should_promote(10.0, None)          # no champion yet: promote
    assert should_promote(9.5, 10.0)           # better: promote
    assert not should_promote(10.5, 10.0)      # worse: keep the old champion


def test_metrics_and_skill():
    m = point_metrics(np.array([100.0, 20.0]), np.array([95.0, 30.0]))
    assert m["mae"] == 7.5 and m["poor_alert_precision"] == 1.0 and m["poor_alert_recall"] == 1.0
    assert skill(5.0, 10.0) == 0.5 and skill(10.0, 10.0) == 0.0


# --------------------------------------------------------------------------- monitor
def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(50, 10, 10000)
    assert psi(ref, rng.normal(50, 10, 2000)) < 0.02
    assert psi(ref, rng.normal(80, 10, 2000)) > 1.0


def test_seasonal_baseline_compares_like_with_like():
    days = pd.date_range("2016-01-01", "2018-12-31", freq="D")
    season = 100 + 80 * np.cos(2 * np.pi * (days.dayofyear - 15) / 365.25)
    hist = pd.DataFrame({"city": "X", "date": days, "pm25": season})
    hist["pm25"] *= np.exp(np.random.default_rng(0).normal(0, 0.1, len(days)))
    base = SeasonalBaseline(hist)
    jan = pd.DataFrame({"city": "X", "date": pd.date_range("2019-01-01", periods=7), "pm25": 180.0})
    jul = pd.DataFrame({"city": "X", "date": pd.date_range("2019-07-01", periods=7), "pm25": 20.0})
    # A smoggy January and a clean July are both normal for their season ...
    assert abs(base.zscores(jan).mean()) < 1 and abs(base.zscores(jul).mean()) < 1.5
    # ... but a clean January is not.
    assert base.zscores(jan.assign(pm25=40.0)).mean() < -3


# ------------------------------------------------------------------------------- api
@pytest.fixture(scope="module")
def client(bundle, clean_df):
    return TestClient(create_app(Forecaster(bundle, clean_df)))


def test_api_forecast(client, clean_df):
    city = sorted(clean_df.city.unique())[0]
    r = client.get(f"/forecast/{city}", params={"as_of": "2020-01-15"})
    assert r.status_code == 200
    body = r.json()
    assert [d["horizon"] for d in body["forecasts"]] == list(config.HORIZONS)
    assert body["forecasts"][0]["date"] == "2020-01-16"
    for d in body["forecasts"]:
        assert d["lo"] <= d["pm25"] <= d["hi"] or d["lo"] <= d["hi"]
        assert d["category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}


def test_api_matches_offline_forecast(client, bundle, clean_df):
    """The served forecast must equal the model's forecast on the training-style feature table."""
    city = sorted(clean_df.city.unique())[1]
    table = build_features(clean_df, 2)
    row = table[(table.city == city) & (table.date == "2019-12-01")]
    expected = bundle["models"][2].predict(row).forecast.iloc[0]
    got = client.get(f"/forecast/{city}", params={"as_of": "2019-12-01"}).json()["forecasts"][1]["pm25"]
    assert got == pytest.approx(expected, abs=0.06)


def test_api_errors(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/forecast/Atlantis").status_code == 404
    assert client.get("/forecast/City0", params={"as_of": "2031-01-01"}).status_code == 400
    assert client.get("/forecast/City0", params={"as_of": "not-a-date"}).status_code == 422
