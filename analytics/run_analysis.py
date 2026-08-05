"""
Run the full analytics case study end-to-end.

Run with:  python -m analytics.run_analysis

Produces (in analytics/outputs/):
  price_distribution.png, rating_by_category.png, correlation_heatmap.png,
  price_vs_stock.png, confusion_matrix.png, roc_curves.png,
  feature_importance.png, best_model.pkl, model_comparison.csv
"""
from __future__ import annotations

import logging
import sys

from analytics.src import eda
from analytics.src.data_loader import load_products
from analytics.src.evaluation import (
    OUTPUT_DIR,
    full_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    save_model,
    summarize_results,
)
from analytics.src.feature_engineering import (
    TARGET_COL,
    engineer_features,
    get_feature_columns,
)
from analytics.src.modeling import select_best_model, train_test_split_data, tune_and_train


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger("analytics.run_analysis")

    logger.info("Loading data...")
    raw_df = load_products()
    logger.info("Loaded %d products across %d categories.",
                len(raw_df), raw_df["category"].nunique())

    logger.info("Running EDA...")
    eda.plot_price_distribution(raw_df)
    eda.plot_rating_by_category(raw_df)
    eda.plot_correlation_heatmap(raw_df)
    eda.plot_price_vs_stock(raw_df)

    logger.info("Engineering features...")
    feature_df = engineer_features(raw_df)
    feature_cols = get_feature_columns(feature_df)
    X = feature_df[feature_cols]
    y = feature_df[TARGET_COL]
    logger.info("Feature matrix: %d rows x %d columns. Positive class rate: %.2f%%",
                len(X), len(feature_cols), 100 * y.mean())

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    logger.info("Training and tuning models (this may take a minute)...")
    results = tune_and_train(X_train, y_train, X_test, y_test)

    comparison = summarize_results(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))

    best = select_best_model(results)
    plot_confusion_matrix(best)
    plot_roc_curves(results)
    plot_feature_importance(best, feature_cols)
    save_model(best)

    logger.info("Classification report for best model (%s):\n%s",
                best.name, full_classification_report(best))
    logger.info("All outputs written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
