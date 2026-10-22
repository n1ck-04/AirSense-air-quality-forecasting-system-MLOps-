# Code walkthrough

Read the files in this order, with each one open beside this guide. After each section, close both and explain the file out loud. If you can't, read it again; that's the gap an interviewer will find.

Total reading time: about 2 hours. Then do [EXERCISES.md](EXERCISES.md).

---

## 0. Run it first

```bash
make all && make test && make dashboard
```

Open the dashboard and click through all four tabs. Everything below explains how those numbers and charts were produced.

---

## 1. `config.py` (5 min)

The decisions a reviewer might question, in one place:
- **The split:** validation starts on 2019-01-01, testing on 2019-07-01. The lockdown date (2020-03-25) is used only for reporting.
- **`HORIZONS = (1, 2, 3)`:** one model per horizon.
- **`QUANTILES = (0.1, 0.9)`:** the 80% range.
- **`PM25_CATEGORIES`:** the CPCB breakpoints (30 / 60 / 90 / 120 / 250). `POOR_THRESHOLD = 90` is the alert level.

**Take-away:** the split is by *target* date, the date being forecast.

---

## 2. `data.py` (10 min)

- **`standardise`** renames Kaggle's columns (`PM2.5` → `pm25`) and parses dates.
- **`clean`:**
  1. Drop duplicates and negative readings (sensor errors).
  2. Keep cities whose data starts before 2018, which leaves 17.
  3. **Reindex each city to every calendar day.** `g.set_index("date").reindex(days)` inserts empty rows for missing days. This is what makes `shift(1)` mean "yesterday".
- **`pm25_category`:** walks the breakpoint list and returns the first category whose upper bound the value fits under.

**Check yourself:** why would forecasting go wrong if missing days were simply absent? (See `test_lags_are_calendar_days_not_rows`.)

---

## 3. `features.py`: the most important file (30 min)

**`_city_features(g, horizon)`** works on one city's daily series. Read it line by line:
- `pm.shift(k)`: PM2.5 *k* days before the forecast day. `shift` with k ≥ 0 only looks backwards.
- `pm.rolling(w, min_periods=...)`: averages over windows that **end** on the forecast day. `min_periods` lets the average exist when a few days are missing.
- `pm25_last_year`: the 15 days centred on (target date − 365). Read the comment: this was a real training/serving skew bug, and it's a good interview story (DECISIONS.md, #4).
- `target = pm.shift(-horizon)`: **the only line that looks forward.** It's the answer we train on, never an input.

**`build_features`** runs that for every city, then adds the season of the *target* date (sine and cosine of the day of year, so 31 Dec and 1 Jan are close), the weekday, and the city as a pandas category. The `cities` argument fixes the category list, so a one-city table (built by the API) encodes the city the same way as training.

**`split`** drops rows with no known target, then cuts by target date.

**Check yourself:**
- Why sine *and* cosine of the day of year? Sine alone gives the same value in spring and autumn.
- Read `test_features_do_not_see_the_future`. Why does deleting later data and comparing prove there's no leakage?

---

## 4. `baselines.py` and `evaluate.py` (15 min)

- **Baselines** are just columns of the feature table: `pm25_lag0` (persistence), `pm25_mean7`, `pm25_last_year`. Same rows, same information as the model.
- **`skill(model, reference) = 1 − model/reference`.** 0.13 means 13% of the reference's error removed.
- **`evaluate_horizon`:**
  - Scores only rows where persistence exists, so all methods face the same days.
  - Reports MAE, RMSE, bias, category accuracy, "Poor" alert precision/recall, and interval coverage (calibrated and raw).
  - Splits the results into before and during the lockdown.

**Check yourself:** at 3 days, why is "skill vs best baseline" (9%) lower than "skill vs persistence" (13%)? Which number would you quote, and why?

---

## 5. `models.py` (20 min)

- **`train_horizon`** fits three LightGBM models with early stopping on validation:
  - `point` on `log1p(target)` with squared error. `predict` undoes the log with `expm1`.
  - `lo` and `hi` with `objective="quantile"` at alpha 0.1 and 0.9, on the raw scale.
- **`conformal_margin`** (CQR):
  1. For each validation day, compute `max(lo − y, y − hi)`. That's positive when y fell outside the range, and negative (how far inside) when it fell within.
  2. Take its 80th percentile, with a small-sample correction: `ceil((n+1)·0.8)/n`.
  3. Widen every future range by that amount. `test_conformal_margin_widens_to_reach_coverage` shows a far-too-narrow range being fixed.
- **`HorizonModel.predict`** also handles the quantile models crossing (lo > hi happens) by sorting the pair.
- **`refit`** retrains on new data with the same tree counts and margin. It's used by monitoring, where there's no fresh validation set.

**Check yourself:** why is training on the log scale better for PM2.5? What does early stopping protect against?

---

## 6. `train.py` (20 min)

`main()` in order:
1. `train_all`: for each horizon, build features, split, train, predict the test year, collect baselines, evaluate.
2. Write `reports/` (metrics, forecasts, three charts).
3. **MLflow:**
   - `set_tracking_uri("sqlite:///mlflow.db")`, then start a run.
   - Log parameters and metrics, plus the reports as artifacts.
   - `mlflow.pyfunc.log_model` stores the whole bundle through the `AirSensePyfunc` wrapper.
   - `register_model` creates a new version of `airsense-pm25`.
4. **Promotion gate:** `should_promote(candidate, champion)`. If it passes, set the `champion` alias and write `models/champion.joblib`.

Try it: run `make train` twice, then `make mlflow-ui`. You'll see two runs, two versions, and the alias on the latest.

**Check yourself:** what stops a model trained on corrupted data from reaching the API?

---

## 7. `monitor.py` (25 min)

- **`psi`:** split the reference data into 10 equal-count bins, compare the share of the current week in each bin, and sum `(q − p) · ln(q/p)`. Identical distributions give about 0.
- **`SeasonalBaseline`:** the mean and standard deviation of log PM2.5 for each (city, month) in the training years. `zscores` says how unusual a day is *for that city and month*.
- **`replay(policy)`:** a weekly loop. At the start of each week, using only data from before that day:
  1. Compute the three signals: naive PSI of last week's PM2.5; seasonal z; and skill over the last 28 days of forecasts whose actual values are known.
  2. Decide whether to retrain: `monthly` retrains at the first check of each month; `triggered` retrains when a signal fires and 28 days have passed since the last retrain.
  3. Forecast the coming week's days.
- **`main`** runs all three policies and writes `monitoring.json`, the weekly CSV and the 3-panel chart.

**Check yourself:**
- Why did naive PSI ring 51 of 53 weeks?
- Why didn't the lockdown, a huge drift, hurt the model's skill?

---

## 8. `forecast.py`, `api.py`, `dashboard/app.py` (20 min)

- **`Forecaster.forecast(city, as_of)`:**
  1. Cut the city's history at `as_of`.
  2. For each horizon, build features (with the training city list) and take the row for `as_of`.
  3. Predict, then attach the category.
- **`api.py`:** three endpoints; pydantic response models; errors mapped to 404, 400 and 422. `create_app(forecaster)` lets the tests inject a small model.
- **Dashboard:** reads only `reports/`, so it can be hosted without data or model. The "Live forecast" tab imports `Forecaster` if it's available.

**Check yourself:** `test_api_matches_offline_forecast` compares the API with the model applied to the training feature table. Which bug would it catch?

---

## 9. Tests, CI and Docker (10 min)

- **`tests/conftest.py`** builds a fake 3-city dataset and trains a small model once.
- **26 tests:** cleaning, categories, calendar lags, target alignment, no-future features, serving consistency, split, weather join, conformal coverage, model sanity, refit, promotion gate, metrics, PSI, seasonal z, and the API, including error codes.
- **CI:** lint and tests (on the fake data); then the whole pipeline on the real data; then the Docker build and a real HTTP request to the container.

---

## Trace one forecast end to end

Delhi, as of 1 November 2019, a peak smog day: 388 µg/m³ measured.
1. History is cut at 2019-11-01. For horizon 1, `pm25_lag0 = 388.4`, the 3-day mean is high, and the target date's season is early November.
2. The point model predicts log(PM2.5); `expm1` turns that into **327**. The quantile models plus the conformal margin give **104–460**.
3. 327 > 250 → **"Severe"**. Horizons 2 and 3: 269 (Severe) and 239 (Very Poor). The model expects the episode to ease, as smog episodes usually do.
4. The API returns all three with the "PM2.5 sub-index only" note.
