"""Evaluation metrics for the bike-sharing model.

Provides both an **absolute** error metric (RMSE, in bikes) and a **relative**
one (MAPE, in %), plus a bootstrap confidence interval for the RMSE. The two
metrics tell different stories on a heavy-tailed target — see the README.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, root_mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (absolute error, in target units)."""
    return root_mean_squared_error(y_true, y_pred)


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (relative error, in %).

    Warning: undefined where ``y_true == 0`` and inflated by small values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse_confidence_interval(y_true, y_pred, confidence=0.95, random_state=42):
    """Bootstrap confidence interval for the RMSE (no normality assumption)."""
    squared_errors = (np.asarray(y_pred) - np.asarray(y_true)) ** 2
    boot = stats.bootstrap(
        [squared_errors],
        lambda s: np.sqrt(np.mean(s)),
        confidence_level=confidence,
        random_state=random_state,
    )
    return float(boot.confidence_interval.low), float(boot.confidence_interval.high)


def evaluate(model, X_test, y_test) -> dict:
    """Return the headline metrics for a fitted model on the test set."""
    y_pred = model.predict(X_test)
    low, high = rmse_confidence_interval(y_test, y_pred)
    return {
        "rmse": rmse(y_test, y_pred),
        "rmse_ci_95": (low, high),
        "mape": mape(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def error_by_segment(X_test, y_test, y_pred, column: str) -> pd.DataFrame:
    """RMSE / MAPE / mean demand broken down by the values of ``column``.

    Reveals *where* the model is strong or weak (e.g. peak vs off-peak hours,
    working days vs weekends) — a global metric hides this.
    """
    res = pd.DataFrame(
        {"segment": np.asarray(X_test[column]), "actual": np.asarray(y_test), "pred": np.asarray(y_pred)}
    )
    res["error"] = res["actual"] - res["pred"]

    def _metrics(g):
        return pd.Series(
            {
                "rmse": np.sqrt(np.mean(g["error"] ** 2)),
                "mape": np.mean(np.abs(g["error"] / g["actual"])) * 100,
                "mean_demand": g["actual"].mean(),
                "n": len(g),
            }
        )

    return res.groupby("segment").apply(_metrics, include_groups=False).round(1)
