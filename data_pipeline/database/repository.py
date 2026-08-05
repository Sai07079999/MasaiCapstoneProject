"""
Repository pattern: the only place in the codebase that issues SQL.
`main.py` and future consumers depend on this interface, not on
SQLAlchemy directly — swapping ORMs later touches one file.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from data_pipeline.models import CleanProduct, Product

logger = logging.getLogger(__name__)


class ProductRepository:
    """CRUD + upsert operations for the `products` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, record: CleanProduct) -> Product:
        """Insert a product, or update it in place if `source_url` exists."""
        existing = self._session.execute(
            select(Product).where(Product.source_url == record.source_url)
        ).scalar_one_or_none()

        if existing is not None:
            for field_name in (
                "title", "price", "rating", "category",
                "availability", "stock_count",
            ):
                setattr(existing, field_name, getattr(record, field_name))
            return existing

        product = Product(**record.model_dump())
        self._session.add(product)
        return product

    def bulk_upsert(self, records: list[CleanProduct]) -> int:
        """Upsert many records in one transaction; returns count written."""
        for record in records:
            self.upsert(record)
        self._session.flush()
        logger.info("Upserted %d products.", len(records))
        return len(records)

    def count(self) -> int:
        return len(self._session.execute(select(Product)).scalars().all())

    def all(self) -> list[Product]:
        return list(self._session.execute(select(Product)).scalars().all())
