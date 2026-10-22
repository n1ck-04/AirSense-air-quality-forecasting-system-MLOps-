# Data

The raw data is **not stored in this repository**. Download it into `data/raw/`.

## Air quality (required)

**Air Quality Data in India (2015–2020)** on Kaggle, compiled by Vopani from the Central Pollution Control Board (CPCB) of India: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

You only need `city_day.csv`. Either:

- download it from the Kaggle page and save it as `data/raw/city_day.csv`, or
- with the Kaggle CLI configured (`~/.kaggle/kaggle.json`):
  ```bash
  kaggle datasets download -d rohanrao/air-quality-data-in-india -f city_day.csv -p data/raw --unzip
  ```

Check the licence on the Kaggle page before redistributing the data.

## Weather (optional)

`python scripts/fetch_weather.py` downloads daily temperature, humidity, wind and rainfall for each city from the [Open-Meteo](https://open-meteo.com/) historical archive (ERA5 reanalysis, CC BY 4.0) into `data/raw/weather.csv`. If that file exists, training adds weather features automatically. Weather data by Open-Meteo.com.

## No data at hand?

`python -m airsense.synthetic data/raw/city_day.csv` writes a small **fake** dataset with the same columns. The tests and CI use it. Its numbers mean nothing.
