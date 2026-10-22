# Data

## Air quality (included)

`data/raw/city_day.csv` comes from **Air Quality Data in India (2015–2020)** on Kaggle, compiled by Vopani from the Central Pollution Control Board (CPCB) of India: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

It's published under **CC0: Public Domain**, so it's included here and the pipeline runs straight after cloning. Credit to the CPCB for the measurements and to Vopani for compiling them.

To refresh it from Kaggle with the Kaggle CLI configured (`~/.kaggle/kaggle.json`):
```bash
kaggle datasets download -d rohanrao/air-quality-data-in-india -f city_day.csv -p data/raw --unzip
```

## Weather (optional, not included)

`python scripts/fetch_weather.py` downloads daily temperature, humidity, wind and rainfall for each city from the [Open-Meteo](https://open-meteo.com/) historical archive (ERA5 reanalysis, CC BY 4.0) into `data/raw/weather.csv`. If that file exists, training adds weather features automatically. Weather data by Open-Meteo.com.

## Fake data

`python -m airsense.synthetic <path>` writes a small **fake** dataset with the same columns. The unit tests use it, so they run in seconds; its numbers mean nothing.
