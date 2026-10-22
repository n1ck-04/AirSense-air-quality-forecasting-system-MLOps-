# Design decisions

Each entry: what was decided, what else was considered, and why. If you change one, update its entry.

---

### 1. Forecast PM2.5, 1–3 days ahead

**Alternatives:** forecast the official AQI; forecast only tomorrow.

**Why PM2.5:** it's the pollutant that most often sets India's AQI and causes the most health harm. The official AQI is the *worst* sub-index across up to 8 pollutants. Forecasting it directly mixes several processes, and the data for some pollutants is missing 35–60% of the time. We forecast PM2.5 well and label its category honestly as the "PM2.5 sub-index".

**Why 1–3 days:** a one-day forecast was the first experiment, and the model beat persistence by only ~6%. Tomorrow's pollution is mostly today's. Persistence degrades quickly with the horizon, while a model that understands buildup, decay and seasons degrades more slowly, so the useful advantage appears at 2–3 days (10% and 13%). A 3-day outlook is also what people actually plan around.

### 2. A complete daily calendar per city

**Decision:** reindex every city to one row per day, with gaps as NaN.

**Why:** monitors go offline, sometimes for weeks. If missing days were simply absent, `shift(1)` would give "the previous row", which could be a month earlier, and the model would be told that was yesterday. A test (`test_lags_are_calendar_days_not_rows`) checks this.

### 3. Split by target date: train to 2018, validate on H1 2019, test on the year to July 2020

**Alternative:** random split.

**Why:** a forecast always looks into the future, so the evaluation must too. A random split puts days from the same week in both training and test sets, and since adjacent days are nearly identical, the test score becomes meaningless. The test year was also chosen because it contains a full seasonal cycle *and* the COVID-19 lockdown, a real-world shift the system has to survive.

### 4. Point-in-time features, one function for training and serving

**Decision:** every feature uses only data up to the forecast day. `build_features()` is used by both `train.py` and the API.

**How it's verified:**
- `test_features_do_not_see_the_future` rebuilds features with all later data deleted and requires identical values.
- `test_one_city_table_matches_full_table` checks that the API's single-city feature table matches training's, including the city's category code.

**A bug this caught.** The first version of the "same season last year" feature used a *centred* rolling window. It only ever averaged past pollution values, but it needed rows *after* the forecast day to exist in the table. In training they do; in live serving they don't. So the API computed a slightly different value from training (training/serving skew). In the case that exposed it, Delhi on 1 Nov 2019, the API's forecast was 0.6 µg/m³ off the evaluation's, and the top of the range was 6 µg/m³ off. Rewriting it as a trailing window over an extra-shifted series gives the same window without needing future rows. The leakage test fails on the old version and passes on the new one.

### 5. Three baselines, and "skill" as the headline metric

**Decision:** compare against persistence (today's value), the 7-day mean, and the same season last year. Report skill = share of the baseline's error removed. Report it against persistence *and* against whichever baseline was best at each horizon.

**Why:** in air-quality forecasting persistence is a strong baseline, and a model's raw error (e.g. 11 µg/m³) means nothing alone. At 3 days the 7-day mean beats persistence, so quoting only "vs persistence" (13%) would flatter the model. The best-baseline figure (9%) is the honest one.

### 6. LightGBM on log(PM2.5)

**Alternatives:** ARIMA/SARIMA per city, Prophet, LSTM, a separate model per city.

**Why:**
- **One global model across 17 cities, with the city as a categorical feature.** It learns shared patterns (winter buildup, weekend effects) from 26,000 city-days instead of ~1,500 per city, and still adjusts per city.
- **Gradient-boosted trees** handle missing values natively (16.8% of PM2.5 days are missing), mix many inputs and need no scaling. Classical time-series models handle one series at a time and struggle with gaps. An LSTM needs far more data to beat trees on tabular inputs like these.
- **Log target:** PM2.5 is heavily skewed (median 49, maximum 950). On the raw scale, a few Delhi smog days would dominate the squared error. On the log scale, errors are roughly proportional, which matches how pollution varies. Early tests showed the log target gave the lowest error at all three horizons.

### 7. An 80% range via quantile regression plus conformal calibration

**Alternatives:** no uncertainty; ± a fixed number; bootstrapped ensembles.

**Why:** "PM2.5 will be 140" is less useful than "probably 100–190". Two extra LightGBM models predict the 10th and 90th percentiles. Quantile models tend to be overconfident, and here the raw "80%" ranges caught only 72–75% of actual values. **Conformalized quantile regression (CQR)** measures, on the validation set, how far actual values fell outside the range, and widens every range by the 80th percentile of that distance. Coverage rose to 76–78%. It's still slightly short of 80% because the test year (with its downward trend and the lockdown) differs from the validation period, and conformal guarantees assume new data resembles the calibration data.

### 8. MLflow for tracking and a registry with a promotion gate

**Decision:**
- Every training run logs parameters, metrics, charts and the model to MLflow (SQLite backend, no server needed).
- The whole 3-horizon bundle is registered as a version of `airsense-pm25`.
- The `champion` alias moves to the new version only if its average validation MAE is no worse than the current champion's.

**Why:** without tracking, "which settings gave 11.19?" has no answer a week later. Without a gate, any training run, including one on broken data, would go live. The gate is a function (`should_promote`) with its own test.

**Why the API loads `models/champion.joblib` rather than calling MLflow:** the serving container stays small and doesn't depend on the tracking database. In a larger setup, deployment would pull the `champion` version from the registry, which is the step that file stands in for.

### 9. The served model is the evaluated model

**Alternative:** after evaluation, retrain on all data up to July 2020 and serve that.

**Why:** every number in the README describes exactly the model the API serves. A model refit on more data is usually a bit better, but its accuracy can't be reported honestly without new test data. The monitoring replay (decision 11) shows what periodic refitting would add: very little here.

### 10. Season-adjusted drift detection

**Decision:** compare each week's PM2.5 with the same city in the same calendar month of the training years (a z-score), rather than with all training data.

**Why:** the standard PSI check against the whole training distribution flagged 51 of 53 weeks, because every monsoon and winter looks different from the annual average. An alarm that always rings is ignored. Removing the expected seasonal swing cut alarms to 18 weeks. It still caught the lockdown, at the first weekly check after it began. It also exposed a real long-term change: pollution in 2019–20 averaged 0.68 standard deviations below 2015–18 levels for the same season, even before the lockdown.

### 11. Monitor model skill, and retrain on signals rather than on a calendar

**Decision:**
- Track the model's skill against persistence over the last 28 days of forecasts whose actual values are known.
- Retrain when skill drops below 0 or the seasonal shift exceeds 1 standard deviation, at most once every 28 days.

**Why:**
- *Drift is not degradation.* The lockdown was a huge shift in the data, but the model kept its skill: it uses recent measurements, so it follows the new level. What matters is whether the model still adds value over the simple alternative, and that's what skill measures.
- *Signal-triggered retraining* achieved the same accuracy as monthly retraining (MAE 11.16 vs 11.15) with 8 retrains instead of 11. With a model this cheap, that saving is small; with a large model, or with retraining that needs human review, it matters.

**Honest note:** retraining at all improved MAE by only 0.4% here. The replay's main value is showing *how to decide* whether retraining is worth it, with evidence.

### 12. Fake data for tests and CI; real data stays out of the repo

**Why:**
- *Licensing:* the Kaggle dataset's licence should be checked before redistribution.
- *Speed:* the tests and CI use a small synthetic dataset with the same columns, a winter peak, persistence and gaps. That's enough to exercise every code path, including training with MLflow, monitoring and building the Docker image, on GitHub's servers.
- *Limit:* synthetic data can't show that the model is *good*, only that the code works. The real results are in `reports/`, produced from the real data.

### 13. A dashboard that runs from `reports/` alone

**Why:** the dashboard can then be deployed for free (Streamlit Community Cloud) without the raw data or the model, so a reviewer can click a link and explore. The live-forecast tab appears only where the model exists.

### 14. Weather is optional, and labelled as an upper bound

**Decision:** `scripts/fetch_weather.py` downloads Open-Meteo reanalysis weather. If `data/raw/weather.csv` exists, training adds weather on the forecast day *and* on the target day.

**Why the label matters:** weather on the target day is a future value. In real use it would come from a weather *forecast*; here we use what was later observed, a perfect forecast. Results with it are an upper bound on what weather can add, and must be described that way.
