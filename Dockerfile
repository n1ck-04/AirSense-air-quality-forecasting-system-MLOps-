# The API in a container. Data is cleaned and the model trained at build time,
# so the image is self-contained. Put city_day.csv in data/raw/ first
# (or the fake one: python -m airsense.synthetic data/raw/city_day.csv).
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data/raw/city_day.csv ./data/raw/city_day.csv
RUN python -m airsense.data && python -m airsense.train

EXPOSE 8000
CMD ["uvicorn", "airsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
