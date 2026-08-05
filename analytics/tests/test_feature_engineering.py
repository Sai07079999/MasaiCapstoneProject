import numpy as np
import pandas as pd
import pytest

from analytics.src.feature_engineering import (
    TARGET_COL,
    engineer_features,
    flag_outliers_iqr,
    handle_missing_values,
    remove_extreme_outliers,
)


def _sample_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    categories = rng.choice(["Fiction", "Nonfiction", "Mystery"], size=n)
    return pd.DataFrame({
        "title": [f"Book {i}" for i in range(n)],
        "price": rng.uniform(5, 30, size=n),
        "rating": rng.integers(1, 6, size=n),
        "category": categories,
        "availability": "In stock",
        "stock_count": rng.integers(0, 50, size=n),
        "source_url": [f"https://example.com/{i}" for i in range(n)],
    })


def test_handle_missing_values_imputes_numeric_with_median():
    df = _sample_df(20)
    df.loc[0, "price"] = np.nan
    median_before = df["price"].median()
    result = handle_missing_values(df)
    assert result["price"].isna().sum() == 0
    assert result.loc[0, "price"] == pytest.approx(median_before)


def test_flag_outliers_iqr_detects_extreme_value():
    df = pd.DataFrame({"price": [10, 11, 12, 10, 11, 500]})
    flags = flag_outliers_iqr(df, "price")
    assert flags.iloc[-1] == True  # noqa: E712
    assert flags.iloc[:-1].sum() == 0


def test_remove_extreme_outliers_drops_flagged_rows():
    df = pd.DataFrame({"price": [10, 11, 12, 10, 11, 500]})
    result = remove_extreme_outliers(df, "price")
    assert 500 not in result["price"].values
    assert len(result) == 5


def test_engineer_features_creates_binary_target():
    df = _sample_df(50)
    result = engineer_features(df)
    assert TARGET_COL in result.columns
    assert set(result[TARGET_COL].unique()).issubset({0, 1})


def test_engineer_features_one_hot_encodes_category():
    df = _sample_df(50)
    result = engineer_features(df)
    dummy_cols = [c for c in result.columns if c.startswith("cat_")]
    assert len(dummy_cols) > 0
    assert "category" not in result.columns


def test_engineer_features_creates_price_vs_category_avg():
    df = _sample_df(50)
    result = engineer_features(df)
    assert "price_vs_category_avg" in result.columns
