"""Train, evaluate, track and (maybe) promote the 3-day forecaster.

Steps
    1. For each horizon (1, 2, 3 days): build features, split by time, train the
       point + interval models, forecast the test year, compare with baselines.
    2. Log everything to MLflow: parameters, metrics, charts, and the model
       itself, as a new version of the registered model "airsense-pm25".
    3. Promotion gate: the new version becomes "champion" only if its average
       validation MAE is no worse than the current champion's. The API serves
       the champion, so a bad training run can never reach users by accident.
    4. The champion is also exported to models/champion.joblib for the API,
       which then doesn't need MLflow installed to serve forecasts.

Run:  python -m airsense.train
      mlflow ui --backend-store-uri sqlite:///mlflow.db      (browse the runs)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from airsense import config
from airsense.baselines import baseline_forecasts
from airsense.evaluate import evaluate_horizon, per_city
from airsense.features import FEATURE_DESCRIPTIONS, build_features, feature_list, load_weather, split
from airsense.models import HorizonModel, train_horizon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REGISTERED_NAME = "airsense-pm25"


# ------------------------------------------------------------------ MLflow model wrapper
class AirSensePyfunc(mlflow.pyfunc.PythonModel):
    """Lets MLflow store and reload the whole 3-horizon bundle as one registered model."""

    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        h = int((params or {}).get("horizon", 1))
        return self.bundle["models"][h].predict(model_input)


# ------------------------------------------------------------------------- training
def train_all(clean: pd.DataFrame, weather: pd.DataFrame | None = None):
    feats = feature_list(with_weather=weather is not None)
    models: dict[int, HorizonModel] = {}
    metrics, forecasts, valid_mae = {}, [], {}
    for h in config.HORIZONS:
        table = build_features(clean, h, weather)
        train, valid, test = split(table)
        m = train_horizon(train, valid, h, feats)
        models[h] = m

        vp = m.predict(valid)
        valid_mae[h] = float(np.abs(vp.forecast - valid.target).mean())

        fc = pd.concat([test[["city", "date", "target_date"]], m.predict(test), baseline_forecasts(test)], axis=1)
        fc["actual"], fc["horizon"] = test.target, h
        forecasts.append(fc)
        metrics[f"h{h}"] = {**evaluate_horizon(fc), "trees": m.n_trees,
                            "conformal_margin": round(m.conformal_margin, 2), "valid_mae": round(valid_mae[h], 2),
                            "rows": {"train": len(train), "valid": len(valid), "test": len(test)}}
        print(f"h={h}: test MAE {metrics[f'h{h}']['model']['mae']} vs persistence "
              f"{metrics[f'h{h}']['persistence']['mae']} (skill {metrics[f'h{h}']['skill_vs_persistence']})")
    return models, metrics, pd.concat(forecasts, ignore_index=True), valid_mae


# --------------------------------------------------------------------------- charts
def plot_skill(metrics: dict, path: Path):
    hs = list(config.HORIZONS)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.2
    for i, (name, key) in enumerate([("Persistence", "persistence"), ("7-day mean", "mean7"),
                                     ("Same season last year", "last_year"), ("AirSense", "model")]):
        vals = [metrics[f"h{h}"][key]["mae"] for h in hs]
        ax.bar(np.arange(len(hs)) + (i - 1.5) * width, vals, width, label=name)
    ax.set_xticks(range(len(hs)), [f"{h} day{'s' if h > 1 else ''} ahead" for h in hs])
    ax.set(ylabel="Mean absolute error (µg/m³)", title="Test-year forecast error, lower is better")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(fc: pd.DataFrame, city: str, path: Path):
    d = fc[(fc.city == city) & (fc.horizon == 1)].sort_values("target_date")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(d.target_date, d.lo, d.hi, alpha=0.25, label="80% interval")
    ax.plot(d.target_date, d.actual, lw=1.2, color="black", label="Actual")
    ax.plot(d.target_date, d.forecast, lw=1.2, label="Next-day forecast")
    ax.axvline(config.LOCKDOWN_START, ls="--", color="grey")
    ax.text(config.LOCKDOWN_START, ax.get_ylim()[1] * 0.95, " lockdown", fontsize=8, color="grey")
    ax.set(ylabel="PM2.5 (µg/m³)", title=f"{city}: next-day forecasts over the test year")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(model: HorizonModel, path: Path):
    imp = pd.Series(model.point.booster_.feature_importance("gain"), index=model.features)
    imp = (imp / imp.sum()).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([FEATURE_DESCRIPTIONS.get(f, f) for f in imp.index], imp.values)
    ax.set(xlabel="Share of total gain", title=f"What drives the {model.horizon}-day forecast")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ registry + promotion
def should_promote(candidate_mae: float, champion_mae: float | None) -> bool:
    """Promotion gate: a new model replaces the champion only if it's at least as good on
    validation. Ties promote, so retraining on the same data refreshes the champion."""
    return champion_mae is None or candidate_mae <= champion_mae


def current_champion_valid_mae(client: MlflowClient) -> float | None:
    try:
        version = client.get_model_version_by_alias(REGISTERED_NAME, "champion")
    except Exception:
        return None
    run = client.get_run(version.run_id)
    return run.data.metrics.get("valid_mae_avg")


def main():
    clean = pd.read_parquet(config.CLEAN_PATH)
    weather = load_weather()
    if weather is not None:
        print("weather.csv found: training WITH weather features")

    models, metrics, fc, valid_mae = train_all(clean, weather)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)

    city_table = per_city(fc[fc.horizon == 1])
    metrics["per_city_h1"] = city_table.reset_index().to_dict(orient="records")
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    fc.to_parquet(config.REPORTS_DIR / "test_forecasts.parquet", index=False)
    plot_skill(metrics, config.REPORTS_DIR / "error_by_horizon.png")
    plot_example(fc, "Delhi" if "Delhi" in set(fc.city) else fc.city.iloc[0], config.REPORTS_DIR / "example_forecast.png")
    plot_importance(models[1], config.REPORTS_DIR / "feature_importance.png")

    bundle = {"models": models, "features": models[1].features, "cities": sorted(clean.city.unique()),
              "trained_until": str(config.VALID_START.date()), "with_weather": weather is not None}

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    client = MlflowClient()
    valid_avg = float(np.mean(list(valid_mae.values())))
    with mlflow.start_run() as run:
        mlflow.log_params({"horizons": str(config.HORIZONS), "n_features": len(bundle["features"]),
                           "with_weather": bundle["with_weather"], "valid_start": str(config.VALID_START.date()),
                           "test_start": str(config.TEST_START.date())})
        for h in config.HORIZONS:
            m = metrics[f"h{h}"]
            mlflow.log_metrics({f"h{h}_test_mae": m["model"]["mae"], f"h{h}_skill": m["skill_vs_persistence"],
                                f"h{h}_coverage": m["model"]["interval_coverage"], f"h{h}_valid_mae": m["valid_mae"]})
        mlflow.log_metric("valid_mae_avg", valid_avg)
        mlflow.log_artifacts(str(config.REPORTS_DIR), artifact_path="reports")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.joblib"
            joblib.dump(bundle, path)
            mlflow.pyfunc.log_model(name="model", python_model=AirSensePyfunc(), artifacts={"bundle": str(path)})
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_NAME)

    champion_mae = current_champion_valid_mae(client)
    if should_promote(valid_avg, champion_mae):
        client.set_registered_model_alias(REGISTERED_NAME, "champion", version.version)
        joblib.dump(bundle, config.BUNDLE_PATH)
        print(f"version {version.version} promoted to champion (valid MAE {valid_avg:.2f}"
              f"{'' if champion_mae is None else f' vs {champion_mae:.2f}'}) -> {config.BUNDLE_PATH.name}")
    else:
        print(f"version {version.version} NOT promoted: valid MAE {valid_avg:.2f} > champion {champion_mae:.2f}")


if __name__ == "__main__":
    main()
