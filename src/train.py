"""Train and persist the final model.

The final estimator is a Gradient Boosting Regressor whose hyperparameters were
selected with ``GridSearchCV`` under a ``TimeSeriesSplit`` cross-validation
(see the README and the notebook). They are declared explicitly here so the model
is fully reproducible without re-running the search.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .features import build_preprocessing

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

# Best hyperparameters found via GridSearchCV (TimeSeriesSplit, n_splits=5).
BEST_PARAMS = dict(
    random_state=42,
    learning_rate=0.02,
    n_estimators=800,
    max_depth=5,
    min_samples_leaf=10,
)


def build_model() -> Pipeline:
    """Assemble the full pipeline: preprocessing + the tuned model."""
    return Pipeline(
        [
            ("preprocessing", build_preprocessing()),
            ("model", GradientBoostingRegressor(**BEST_PARAMS)),
        ]
    )


def train(X_train, y_train, save_path: Path = MODEL_PATH) -> Pipeline:
    """Fit the model on the training set and persist it to ``save_path``."""
    model = build_model()
    model.fit(X_train, y_train)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model
