"""
Exploratory data analysis visualizations, saved to `analytics/outputs/`.
Kept separate from feature engineering so EDA can run on raw data
before any transformation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def plot_price_distribution(df: pd.DataFrame, filename: str = "price_distribution.png") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["price"], bins=30, kde=True, ax=ax, color="steelblue")
    ax.set_title("Price Distribution")
    ax.set_xlabel("Price (£)")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rating_by_category(df: pd.DataFrame, filename: str = "rating_by_category.png") -> Path:
    order = df.groupby("category")["rating"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="category", y="rating", order=order, ax=ax, color="steelblue")
    ax.set_title("Rating Distribution by Category")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> Path:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix (Numeric Features)")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_price_vs_stock(df: pd.DataFrame, filename: str = "price_vs_stock.png") -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="price", y="stock_count", hue="rating", palette="viridis", ax=ax, alpha=0.7
    )
    ax.set_title("Price vs. Stock Count (colored by Rating)")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
