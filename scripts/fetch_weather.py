"""Download daily weather for each city from Open-Meteo's historical archive (free, no API key).

Weather drives air pollution: low wind and cool, still winter nights trap
pollution near the ground; rain washes it out. The CPCB data has no weather,
so this script adds it. Run it on your own computer (it needs internet access):

    python scripts/fetch_weather.py

It writes data/raw/weather.csv. After that, `make train` automatically
includes weather features; see docs/EXERCISES.md for the experiment.

Data: Open-Meteo Historical Weather API (ERA5 reanalysis), https://open-meteo.com/
Licence: CC BY 4.0, attribution required: "Weather data by Open-Meteo.com".
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "weather.csv"
START, END = "2015-01-01", "2020-07-01"

# City-centre coordinates (approximate; one point stands in for each city).
CITIES = {
    "Ahmedabad": (23.0225, 72.5714), "Amaravati": (16.5150, 80.5160), "Amritsar": (31.6340, 74.8723),
    "Bengaluru": (12.9716, 77.5946), "Brajrajnagar": (21.8167, 83.9167), "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090), "Gurugram": (28.4595, 77.0266), "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873), "Jorapokhar": (23.7000, 86.4100), "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777), "Patna": (25.5941, 85.1376), "Talcher": (20.9500, 85.2300),
    "Thiruvananthapuram": (8.5241, 76.9366), "Visakhapatnam": (17.6868, 83.2185),
}
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]


def fetch_city(name: str, lat: float, lon: float) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
              "hourly": ",".join(HOURLY), "timezone": "Asia/Kolkata"}
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = json.load(r)["hourly"]
    h = pd.DataFrame(data)
    h["date"] = pd.to_datetime(h.pop("time")).dt.normalize()
    daily = h.groupby("date").agg(
        temp_mean=("temperature_2m", "mean"),
        rh_mean=("relative_humidity_2m", "mean"),
        wind_mean=("wind_speed_10m", "mean"),
        precip_sum=("precipitation", "sum"),
    ).round(2).reset_index()
    daily.insert(0, "city", name)
    return daily


def main():
    parts = []
    for name, (lat, lon) in CITIES.items():
        print(f"downloading {name} ...", flush=True)
        try:
            parts.append(fetch_city(name, lat, lon))
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}", file=sys.stderr)
        time.sleep(1)            # be polite to the free API
    if not parts:
        sys.exit("nothing downloaded; check your internet connection")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(parts)} cities)")


if __name__ == "__main__":
    main()
