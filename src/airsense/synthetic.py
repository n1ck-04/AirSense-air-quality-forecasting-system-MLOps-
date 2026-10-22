"""A small fake city_day.csv with the same shape as the real one.

Used by the tests and by CI, so the whole pipeline can run on GitHub without
the real dataset (which is not stored in this repository). The numbers mean
nothing; they only have realistic structure: a winter peak, day-to-day
persistence, gaps, and cities that start at different dates.

Run:  python -m airsense.synthetic data/raw/city_day.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_city_day(n_cities: int = 4, start="2015-01-01", end="2020-07-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cities):
        city = f"City{i}"
        city_start = pd.Timestamp(start) + pd.Timedelta(days=int(rng.integers(0, 400)))
        days = pd.date_range(city_start, end, freq="D")
        level = rng.uniform(40, 90)
        season = 1 + 0.6 * np.cos(2 * np.pi * (np.asarray(days.dayofyear) - 15) / 365.25)    # peak in mid-January
        noise = np.zeros(len(days))
        for t in range(1, len(days)):                                           # AR(1): today resembles yesterday
            noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 0.15)
        pm25 = level * season * np.exp(noise)
        pm25[rng.random(len(days)) < 0.05] = np.nan                             # 5% missing days
        df = pd.DataFrame({"City": city, "Date": days.strftime("%Y-%m-%d"), "PM2.5": pm25.round(2)})
        df["PM10"] = (pm25 * rng.uniform(1.6, 2.2)).round(2)
        for col, scale in [("NO", 0.2), ("NO2", 0.4), ("NOx", 0.5), ("NH3", 0.3), ("CO", 0.02), ("SO2", 0.15),
                           ("O3", 0.5), ("Benzene", 0.05), ("Toluene", 0.1), ("Xylene", 0.05)]:
            df[col] = (pm25 * scale * rng.uniform(0.7, 1.3, len(days))).round(2)
        df["AQI"] = np.nan
        df["AQI_Bucket"] = np.nan
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/city_day.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_city_day().to_csv(out, index=False)
    print(f"wrote synthetic data -> {out}")


if __name__ == "__main__":
    main()
