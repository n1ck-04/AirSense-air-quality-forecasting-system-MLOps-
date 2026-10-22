"""Central settings: paths, the time split, forecast horizons and AQI categories.

Every assumption a reviewer might question lives here, so changing one is a
one-line edit.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "city_day.csv"       # from Kaggle, see data/README.md
WEATHER_PATH = ROOT / "data" / "raw" / "weather.csv"     # optional, see scripts/fetch_weather.py
CLEAN_PATH = ROOT / "data" / "clean.parquet"
MODELS_DIR = ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "champion.joblib"             # what the API serves
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"            # experiment tracking + model registry
MLFLOW_EXPERIMENT = "airsense"

# ---------------------------------------------------------------------------
# Which cities. A city needs at least a year of history before validation
# starts, or the model has nothing to learn its seasonal pattern from.
# ---------------------------------------------------------------------------
MIN_HISTORY_START = pd.Timestamp("2018-01-01")

# ---------------------------------------------------------------------------
# Time split, by the date being FORECAST (the target date).
#   train:  2015-01-01 .. 2018-12-31
#   valid:  2019-01-01 .. 2019-06-30   (early stopping, model promotion gate)
#   test:   2019-07-01 .. 2020-07-01   (one full year, incl. the COVID-19 lockdown)
# ---------------------------------------------------------------------------
VALID_START = pd.Timestamp("2019-01-01")
TEST_START = pd.Timestamp("2019-07-01")
LOCKDOWN_START = pd.Timestamp("2020-03-25")   # India's national lockdown began

# Forecast 1, 2 and 3 days ahead. One model per horizon.
HORIZONS = (1, 2, 3)

# Prediction interval: the 10th and 90th percentiles, i.e. an 80% interval.
QUANTILES = (0.1, 0.9)

# ---------------------------------------------------------------------------
# CPCB AQI sub-index categories for PM2.5 (24-hour average, µg/m³).
# The official AQI is the WORST sub-index across pollutants; we forecast only
# PM2.5, so we report the PM2.5 category, and say so wherever it's shown.
# ---------------------------------------------------------------------------
PM25_CATEGORIES = [            # (upper bound inclusive, name)
    (30, "Good"),
    (60, "Satisfactory"),
    (90, "Moderate"),
    (120, "Poor"),
    (250, "Very Poor"),
    (float("inf"), "Severe"),
]
POOR_THRESHOLD = 90.0          # above this, the day is "Poor" or worse: worth a public alert
