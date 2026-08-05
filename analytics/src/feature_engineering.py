"""
Feature engineering for the catalog dataset.

Business framing: predict whether a book is a **high performer**
(rating >= 4) from attributes known *before* reviews come in — price,
category, and stock behavior. A merchandising team could use this to
flag which new titles/categories are worth promoting or restocking
aggressively.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "high_rating"


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values with the median, categorical with mode."""
    df = df.copy()
    n_missing_before = df.isna().sum().sum()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    logger.info("Missing values before=%d after=%d", n_missing_before, df.isna().sum().sum())
    return df


def flag_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean Series marking IQR-based outliers in `column`."""
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ~df[column].between(lower, upper)


def remove_extreme_outliers(df: pd.DataFrame, column: str = "price") -> pd.DataFrame:
    """Drop rows flagged as outliers on `column` (IQR method)."""
    mask = flag_outliers_iqr(df, column)
    n_removed = int(mask.sum())
    if n_removed:
        logger.info("Removing %d outlier rows based on '%s'.", n_removed, column)
    return df.loc[~mask].reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature set from raw product columns."""
    df = handle_missing_values(df)
    df = remove_extreme_outliers(df, "price")

    df["stock_count"] = df["stock_count"].fillna(0)

    # Price relative to its own category's average — captures "premium
    # within genre" rather than raw price, which is dominated by genre.
    category_avg_price = df.groupby("category")["price"].transform("mean")
    df["price_vs_category_avg"] = df["price"] - category_avg_price

    df["low_stock_flag"] = (df["stock_count"] < df["stock_count"].median()).astype(int)

    # Target: binary high performer.
    df[TARGET_COL] = (df["rating"] >= 4).astype(int)

    df = pd.get_dummies(df, columns=["category"], prefix="cat", drop_first=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every engineered column except identifiers, target, and leakage-prone rating."""
    exclude = {"id", "title", "availability", "source_url", "scraped_at", "rating", TARGET_COL}
    return [c for c in df.columns if c not in exclude]
