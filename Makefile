# One command per pipeline step. `make all` rebuilds everything (~2 min).
# No Make on Windows? Each target is one line: run the command after the colon directly.

.PHONY: install data train monitor all test lint api dashboard mlflow-ui docker synthetic weather

install:          ## install the package, dashboard and dev tools
	pip install -e ".[dashboard,dev]" ruff

data:             ## clean data/raw/city_day.csv -> data/clean.parquet
	python -m airsense.data

train:            ## train 1-3 day models, log to MLflow, promote if better, write reports/
	python -m airsense.train

monitor:          ## replay the test year: drift signals + retraining policies
	python -m airsense.monitor

all: data train monitor

test:             ## run the tests (uses fake data; no download needed)
	pytest -q

lint:
	ruff check src tests dashboard scripts --select E,F,I,B --line-length 130

api:              ## forecast API at http://127.0.0.1:8000/docs
	uvicorn airsense.api:app --reload

dashboard:        ## dashboard at http://localhost:8501
	streamlit run dashboard/app.py

mlflow-ui:        ## browse experiment runs and the model registry at http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker:           ## build and run the API in a container
	docker build -t airsense . && docker run --rm -p 8000:8000 airsense

synthetic:        ## write FAKE data to data/raw/city_day.csv (for trying the pipeline without Kaggle)
	python -m airsense.synthetic data/raw/city_day.csv

weather:          ## download weather from Open-Meteo (needs internet)
	python scripts/fetch_weather.py
