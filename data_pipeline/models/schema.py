"""
SQLAlchemy ORM schema for scraped product data.

One table, normalized, with the two columns analytics actually filters
on (`category`, `scraped_at`) indexed for query performance.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Product(Base):
    """A single product record scraped from the target catalog site."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    category: Mapped[str] = mapped_column(String(200), nullable=False)
    availability: Mapped[str] = mapped_column(String(50), nullable=False)
    stock_count: Mapped[int] = mapped_column(Integer, nullable=True)
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False, unique=True)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_products_category", "category"),
        Index("ix_products_scraped_at", "scraped_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Product id={self.id} title={self.title!r} price={self.price}>"
