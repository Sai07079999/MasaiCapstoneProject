"""
Data Transfer Objects: the schema contract between the scraper and the
database. Anything that fails validation here never reaches persistence.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RawProduct(BaseModel):
    """Unvalidated shape straight out of the HTML parser."""

    title: str
    price_text: str
    rating_text: str
    category: str
    availability_text: str
    source_url: str


class CleanProduct(BaseModel):
    """Validated, normalized record ready for the repository layer."""

    title: str = Field(min_length=1, max_length=500)
    price: float = Field(gt=0)
    rating: int = Field(ge=0, le=5)
    category: str = Field(min_length=1)
    availability: str
    stock_count: Optional[int] = None
    source_url: str

    @field_validator("title")
    @classmethod
    def strip_title(cls, v: str) -> str:
        return v.strip()

    @field_validator("category")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        return v.strip().title()
