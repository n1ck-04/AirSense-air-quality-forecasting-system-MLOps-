"""Simple forecasts every model must beat.

If a machine-learning model can't beat "tomorrow will be like today", it has
learned nothing useful. These baselines are computed from the same feature
table, so they see exactly the same information as the model.

    persistence     the last observed value (PM2.5 on the forecast day)
    mean7           the average of the last 7 days
    last_year       the same season one year earlier
"""

import numpy as np
import pandas as pd

BASELINES = {
    "persistence": "pm25_lag0",
    "mean7": "pm25_mean7",
    "last_year": "pm25_last_year",
}


def baseline_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: df[col].to_numpy(dtype=float) for name, col in BASELINES.items()}, index=df.index)


def skill(model_mae: float, reference_mae: float) -> float:
    """Skill score: the share of the reference's error the model removes.
    0 = no better than the reference, 1 = perfect, negative = worse."""
    return float(1 - model_mae / reference_mae) if reference_mae > 0 else np.nan
