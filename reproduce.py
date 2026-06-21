"""Reproduce the final model end to end: train, evaluate, report.

Run from the repository root:

    python reproduce.py
"""

from src.data import load_data, temporal_train_test_split, split_X_y
from src.train import train
from src.evaluate import evaluate


def main() -> None:
    train_df, test_df = temporal_train_test_split(load_data())
    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)

    print("Training the tuned Gradient Boosting model...")
    model = train(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    low, high = metrics["rmse_ci_95"]

    print("\nTest-set performance")
    print("-" * 32)
    print(f"RMSE      : {metrics['rmse']:.1f} bikes/hour")
    print(f"RMSE 95%CI: [{low:.1f}, {high:.1f}]")
    print(f"R^2       : {metrics['r2']:.3f}")
    print(f"MAPE      : {metrics['mape']:.0f}%")


if __name__ == "__main__":
    main()
