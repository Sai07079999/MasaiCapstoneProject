"""
Evaluation and reporting for trained models: metrics table, confusion
matrix, ROC curves (all models overlaid for comparison), feature
importance for the winning model, and serialization to disk.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: safe for scripts/CI, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from analytics.src.modeling import ModelResult

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def summarize_results(results: list[ModelResult]) -> pd.DataFrame:
    """Build a comparison table of test-set metrics for every model."""
    rows = []
    for result in results:
        y_pred = result.best_estimator.predict(result.X_test)
        y_proba = result.best_estimator.predict_proba(result.X_test)[:, 1]

        rows.append({
            "model": result.name,
            "cv_roc_auc": result.cv_best_score,
            "test_accuracy": accuracy_score(result.y_test, y_pred),
            "test_precision": precision_score(result.y_test, y_pred),
            "test_recall": recall_score(result.y_test, y_pred),
            "test_f1": f1_score(result.y_test, y_pred),
            "test_roc_auc": roc_auc_score(result.y_test, y_proba),
        })
    return pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(result: ModelResult, filename: str = "confusion_matrix.png") -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    y_pred = result.best_estimator.predict(result.X_test)
    ConfusionMatrixDisplay.from_predictions(
        result.y_test, y_pred, display_labels=["Standard", "High Rating"], ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {result.name}")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(results: list[ModelResult], filename: str = "roc_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        RocCurveDisplay.from_estimator(
            result.best_estimator, result.X_test, result.y_test, name=result.name, ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    result: ModelResult, feature_names: list[str], filename: str = "feature_importance.png"
) -> Path | None:
    """Plot feature importance/coefficients for tree or linear models. Returns None if unsupported."""
    clf = result.best_estimator.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        logger.warning("%s has no importances/coefficients to plot.", result.name)
        return None

    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(
        x=importances[order], y=[feature_names[i] for i in order], ax=ax, color="steelblue"
    )
    ax.set_title(f"Top Feature Importances — {result.name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_model(result: ModelResult, filename: str = "best_model.pkl") -> Path:
    path = OUTPUT_DIR / filename
    joblib.dump(result.best_estimator, path)
    logger.info("Serialized best model (%s) to %s", result.name, path)
    return path


def full_classification_report(result: ModelResult) -> str:
    y_pred = result.best_estimator.predict(result.X_test)
    return classification_report(
        result.y_test, y_pred, target_names=["Standard", "High Rating"]
    )
