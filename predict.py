"""Try the trained model on real data: predicted vs. actual demand for one day.

Run from the repository root (after `python reproduce.py` has saved the model):

    python predict.py
"""

import joblib

from src.data import load_data, temporal_train_test_split, split_X_y
from src.train import MODEL_PATH


def main() -> None:
    # Load the model saved by reproduce.py / src.train.train().
    model = joblib.load(MODEL_PATH)

    # Grab the held-out test set (data the model never saw during training).
    _, test_df = temporal_train_test_split(load_data())

    # Pick one full day from the test set to inspect hour by hour.
    day = test_df["dteday"].iloc[len(test_df) // 2]   # a day in the middle of the test period
    one_day = test_df[test_df["dteday"] == day].sort_values("hr")

    X_day, y_true = split_X_y(one_day)
    y_pred = model.predict(X_day)

    print(f"Predicted vs. actual hourly demand for {day}")
    print("-" * 44)
    print(f"{'hour':>4} | {'predicted':>9} | {'actual':>6} | {'error':>6}")
    print("-" * 44)
    for hr, pred, actual in zip(one_day["hr"], y_pred, y_true):
        print(f"{hr:>4} | {pred:>9.0f} | {actual:>6} | {pred - actual:>+6.0f}")

    total_pred, total_actual = y_pred.sum(), y_true.sum()
    print("-" * 44)
    print(f"Day total -> predicted: {total_pred:,.0f} | actual: {total_actual:,} bikes")


if __name__ == "__main__":
    main()
