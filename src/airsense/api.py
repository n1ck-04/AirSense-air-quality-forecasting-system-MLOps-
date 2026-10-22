"""Forecast API.

Endpoints
    GET /health                              service status and which model is loaded
    GET /cities                              cities the model covers, and the date range
    GET /forecast/{city}?as_of=2020-03-20    3-day PM2.5 outlook with 80% intervals

Errors are explicit: an unknown city is 404, a date outside the data is 400,
and a badly formatted date is 422 (rejected by FastAPI before our code runs).

Run:  uvicorn airsense.api:app --reload        then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from airsense.forecast import Forecaster

CATEGORY_NOTE = ("Category uses the CPCB PM2.5 sub-index only. The official AQI takes the worst of "
                 "several pollutants, so it can be higher.")


class DayOut(BaseModel):
    horizon: int
    date: str
    pm25: float
    lo: float
    hi: float
    category: str


class ForecastOut(BaseModel):
    city: str
    as_of: str
    observed_pm25: float | None
    forecasts: list[DayOut]
    note: str = CATEGORY_NOTE


def create_app(forecaster: Forecaster | None = None) -> FastAPI:
    app = FastAPI(title="AirSense forecast API", version="0.1.0")
    state = {"f": forecaster}

    def get() -> Forecaster:
        if state["f"] is None:          # loaded on first request, so tests can inject a small one
            state["f"] = Forecaster.from_disk()
        return state["f"]

    @app.get("/health")
    def health():
        f = get()
        return {"status": "ok", "horizons": sorted(f.models), "trained_until": f.bundle.get("trained_until"),
                "with_weather": f.bundle.get("with_weather", False)}

    @app.get("/cities")
    def cities():
        f = get()
        return {"cities": f.cities, "first_date": str(f.first_day.date()), "last_date": str(f.last_day.date())}

    @app.get("/forecast/{city}", response_model=ForecastOut)
    def forecast(city: str, as_of: date | None = None):
        f = get()
        try:
            return f.forecast(city, as_of)
        except KeyError:
            raise HTTPException(404, f"unknown city '{city}'. See /cities.") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    return app


app = create_app()
