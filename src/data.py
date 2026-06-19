"""Data loading and the chronological train/test split.

The target ``cnt`` is the number of bikes rented in a given hour. Because this is
a **time series**, the test set must be the most recent slice of data; a random
split would let the model "see the future" and produce an over-optimistic score.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hour.csv"

TARGET = "cnt"
# Columns that leak the target or carry no signal:
#   casual + registered == cnt  -> direct target leakage
#   instant               -> row index, no predictive value
LEAK_COLUMNS = ["casual", "registered", "instant"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw hourly dataset."""
    return pd.read_csv(path)


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological split: the most recent ``test_size`` fraction is the test set.

    Rows are ordered by ``instant`` (a monotonic time index) before splitting, so
    the training set is always strictly in the past relative to the test set.
    """
    df = df.sort_values("instant").reset_index(drop=True)
    n_test = int(len(df) * test_size)
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    return train, test


def split_X_y(df: pd.DataFrame):
    """Split a dataframe into features ``X`` and target ``y``.

    ``dteday`` is intentionally kept in ``X`` because the feature pipeline uses it
    to derive a long-term trend (see :mod:`src.features`).
    """
    X = df.drop(columns=LEAK_COLUMNS + [TARGET])
    y = df[TARGET]
    return X, y
