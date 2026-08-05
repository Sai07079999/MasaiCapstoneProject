"""
Loads product data straight from the SQLite database `data_pipeline`
writes to, via a plain read-only pandas/SQLAlchemy query — this module
never imports pipeline internals beyond the connection string, keeping
`analytics` independently runnable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[2] / "data_pipeline" / "products.db"
)


def load_products(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load the full `products` table into a DataFrame."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"No database found at {db_path}. Run the pipeline first: "
            "`python -m data_pipeline.main` (or "
            "`python -m data_pipeline.seed_synthetic_data` for sample data)."
        )
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql_table("products", engine)
    df.columns = [str(c) for c in df.columns]  # SQLAlchemy returns quoted_name, not plain str
    return df
