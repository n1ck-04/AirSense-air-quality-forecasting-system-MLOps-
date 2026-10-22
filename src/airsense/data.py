"""Load and clean the CPCB city-day data.

What cleaning does, and why
    1. Rename columns to simple Python names (PM2.5 -> pm25).
    2. Keep cities with enough history (see config.MIN_HISTORY_START).
    3. Put every city on a COMPLETE daily calendar. Missing days become rows
       of NaN instead of disappearing. This matters: a "lag 1" feature must
       mean "yesterday", not "the previous row", which could be weeks earlier
       if the monitor was offline.
    4. Leave genuine extremes alone. Delhi really does exceed 500 µg/m³ around
       Diwali; clipping those would teach the model that such days don't exist.

Run:  python -m airsense.data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airsense import config

COLUMNS = {
    "City": "city", "Date": "date", "PM2.5": "pm25", "PM10": "pm10", "NO": "no", "NO2": "no2",
    "NOx": "nox", "NH3": "nh3", "CO": "co", "SO2": "so2", "O3": "o3",
}
POLLUTANTS = ["pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3"]


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle column names -> simple names, keep the columns we use, parse dates."""
    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()
    df["date"] = pd.to_datetime(df.date)
    return df


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to download city_day.csv.")
    return standardise(pd.read_csv(path))


def clean(raw: pd.DataFrame, min_history_start=config.MIN_HISTORY_START) -> pd.DataFrame:
    df = raw.drop_duplicates(["city", "date"]).copy()
    for col in POLLUTANTS:
        # Negative concentrations are sensor errors; zero is suspicious but possible.
        df.loc[df[col] < 0, col] = np.nan

    first_day = df.groupby("city").date.min()
    keep = first_day[first_day < min_history_start].index
    df = df[df.city.isin(keep)]

    end = df.date.max()
    parts = []
    for city, g in df.groupby("city"):
        days = pd.date_range(g.date.min(), end, freq="D", name="date")
        g = g.set_index("date").reindex(days)
        g["city"] = city
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out[["city", "date", *POLLUTANTS]].sort_values(["city", "date"]).reset_index(drop=True)


def pm25_category(value: float) -> str | None:
    """CPCB PM2.5 sub-index category for a 24-hour average."""
    if value is None or pd.isna(value):
        return None
    for upper, name in config.PM25_CATEGORIES:
        if value <= upper:
            return name
    return config.PM25_CATEGORIES[-1][1]


def main():
    df = clean(load_raw())
    config.CLEAN_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(config.CLEAN_PATH, index=False)
    print(f"{df.city.nunique()} cities, {len(df):,} city-days "
          f"({df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}), "
          f"PM2.5 missing on {df.pm25.isna().mean():.1%} of days")


if __name__ == "__main__":
    main()
