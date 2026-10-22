# Exercises: extend AirSense

Each one changes real behaviour and has a way to check it. Work on a branch, commit with clear messages, and update the README if numbers change.

---

## 1. Add weather and measure what it's worth (2–3 h) ⭐ most valuable

PM2.5 depends on wind, temperature, humidity and rain, which this dataset doesn't include.

1. Run `python scripts/fetch_weather.py` (needs internet; free, no key). It writes `data/raw/weather.csv`.
2. Run `make train monitor`. Training detects the file and adds 8 weather features.
3. Make a before/after table for each horizon: MAE, skill vs persistence, interval coverage. The promotion gate decides whether the new version becomes champion. Check `make mlflow-ui`.
4. **Then the honest version:** remove the `*_target` features (weather on the forecast date, which a real system wouldn't know) and train again with only `*_today` weather. Report both. The first is an upper bound; the second is what you could deploy.

**Explain:** why are the two results different, and which one belongs in the README headline?

## 2. Move a business decision (30 min)

Change `POOR_THRESHOLD` to 60 ("Moderate or worse") and run `make train`. How do alert precision and recall change for the model and for persistence? Who might prefer each threshold: a hospital, or a city deciding to close schools?

## 3. Forecast 7 days ahead (1 h)

Change `HORIZONS` to `(1, 2, 3, 5, 7)`. Check that `pm25_last_year` still only uses the past (read its comment), then run `make all test`.

**Plot** skill vs horizon. Where does it peak? At what horizon does "same season last year" start to win?

## 4. A per-city alert threshold (2 h)

The model is weakest in Hyderabad, Talcher and Amaravati. Add a check to `evaluate.per_city` that flags any city where skill < 0, and make the API return `"low_confidence": true` for those cities.

**Test:** add an API test with a forced low-skill city.

## 5. Replace PSI with a proper statistical test (2 h)

Add a Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) on the seasonal z-scores to `monitor.py`. Compare the weeks it flags with the seasonal threshold of 1.0. Which would you trust, and why does sample size matter here (7 days × 17 cities)?

## 6. Pull the champion from the registry (1 h)

Change `Forecaster.from_disk()` to load the model with `mlflow.pyfunc.load_model("models:/airsense-pm25@champion")` when an environment variable `AIRSENSE_FROM_REGISTRY=1` is set. What does the Docker image now need that it didn't before? Write the trade-off into DECISIONS.md.

## 7. Deploy the dashboard (20 min)

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app from this repo, with main file `dashboard/app.py`. It uses `dashboard/requirements.txt`.
3. Put the public link at the top of the README and on your resume.
