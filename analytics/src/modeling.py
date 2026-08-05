"""
Trains and compares multiple classifiers for the high-performer
prediction task, with cross-validated hyperparameter tuning per model.

Three models are compared deliberately for different reasons:
  - LogisticRegression: interpretable linear baseline; coefficients
    are directly explainable to a non-technical stakeholder.
  - RandomForestClassifier: captures non-linear interactions
    (e.g. category x price) without manual feature crosses.
  - GradientBoostingClassifier: typically the strongest tabular
    performer here, at the cost of interpretability and speed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Everything downstream evaluation needs about one tuned model."""

    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_best_score: float
    X_test: pd.DataFrame
    y_test: pd.Series


MODEL_SEARCH_SPACE: dict[str, tuple] = {
    "LogisticRegression": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [4, 8, None]},
    ),
    "GradientBoosting": (
        Pipeline([("clf", GradientBoostingClassifier(random_state=42))]),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]},
    ),
}


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def tune_and_train(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> list[ModelResult]:
    """Run GridSearchCV (5-fold stratified) for every model in the search space."""
    results: list[ModelResult] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (pipeline, param_grid) in MODEL_SEARCH_SPACE.items():
        logger.info("Tuning %s ...", name)
        search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        search.fit(X_train, y_train)

        logger.info(
            "%s best params=%s | CV ROC-AUC=%.4f",
            name, search.best_params_, search.best_score_,
        )
        results.append(
            ModelResult(
                name=name,
                best_estimator=search.best_estimator_,
                best_params=search.best_params_,
                cv_best_score=search.best_score_,
                X_test=X_test,
                y_test=y_test,
            )
        )
    return results


def select_best_model(results: list[ModelResult]) -> ModelResult:
    best = max(results, key=lambda r: r.cv_best_score)
    logger.info("Selected best model: %s (CV ROC-AUC=%.4f)", best.name, best.cv_best_score)
    return best
