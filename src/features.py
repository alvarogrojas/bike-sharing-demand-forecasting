"""Feature engineering and the preprocessing pipeline.

Two ideas drive the feature design:

1. **Cyclical encoding.** Hour, month and weekday are *circular* (hour 23 is as
   close to hour 0 as hour 1 is). Encoding them as ``sin``/``cos`` pairs lets the
   model treat them as the continuous cycles they are, instead of as arbitrary
   integers.
2. **Domain features.** A rush-hour flag and a temperature/humidity interaction
   give the model business-aware signals it cannot easily learn from raw columns.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# The bike-sharing system started on this date; used to measure long-term growth.
SYSTEM_START = pd.Timestamp("2011-01-01")
# Commuting peak hours on working days.
RUSH_HOURS = [8, 9, 17, 18]
MIDDAY_HOURS = [12, 13, 14]


def add_time_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encodings, a growth trend and interaction features.

    Stateless and side-effect free, so it is safe to wrap in a
    :class:`~sklearn.preprocessing.FunctionTransformer` inside a pipeline.
    """
    X = X.copy()

    # Cyclical encoding of circular time variables.
    X["hr_sin"] = np.sin(2 * np.pi * X["hr"] / 24)
    X["hr_cos"] = np.cos(2 * np.pi * X["hr"] / 24)
    X["mnth_sin"] = np.sin(2 * np.pi * X["mnth"] / 12)
    X["mnth_cos"] = np.cos(2 * np.pi * X["mnth"] / 12)
    X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 7)

    # Long-term growth: ridership rose steadily over the two years of data.
    X["days_since_start"] = (pd.to_datetime(X["dteday"]) - SYSTEM_START).dt.days

    # Domain features.
    X["is_workday_rush"] = (
        (X["workingday"] == 1) & (X["hr"].isin(RUSH_HOURS))
    ).astype(int)
    X["is_workday_midday"] = (
        (X["workingday"] == 1) & (X["hr"].isin(MIDDAY_HOURS))
    ).astype(int)
    # "Pleasantness": warm and dry weather drives rentals.
    X["temp_x_hum"] = X["temp"] * (1 - X["hum"])

    return X


# Categorical variables: the relationship with demand is non-linear, so one-hot.
ONEHOT_COLS = ["season", "weathersit"]
# Already-binary, normalized or engineered features passed through untouched.
PASSTHROUGH_COLS = [
    "yr", "holiday", "workingday",
    "hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos",
    "is_workday_rush", "is_workday_midday", "temp_x_hum",
    "temp", "hum", "windspeed",
]
# Unbounded numeric trend -> standardize.
SCALE_COLS = ["days_since_start"]


def build_preprocessing() -> Pipeline:
    """Build the full preprocessing pipeline (feature engineering + encoding).

    Fitting happens inside cross-validation/training, so each fold is preprocessed
    using only its own training data - no leakage from validation or test folds.
    """
    column_transformer = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS),
            ("num", StandardScaler(), SCALE_COLS),
        ]
    )
    return Pipeline(
        [
            ("add_features", FunctionTransformer(add_time_features)),
            ("column_transformer", column_transformer),
        ]
    )
