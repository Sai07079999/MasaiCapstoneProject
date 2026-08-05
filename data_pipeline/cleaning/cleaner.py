"""
Transforms `RawProduct` (text straight from HTML) into `CleanProduct`
(typed, normalized, validated). Isolated from both scraping and
persistence so it can be unit tested with plain strings — no network,
no database required.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Iterator

from pydantic import ValidationError

from data_pipeline.models import CleanProduct, RawProduct

logger = logging.getLogger(__name__)

_PRICE_PATTERN = re.compile(r"[\d.]+")
_STOCK_PATTERN = re.compile(r"(\d+)\s+available")
_RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5, "Zero": 0}


def _parse_price(price_text: str) -> float:
    """Strip currency symbols/whitespace, e.g. 'Â£51.77' -> 51.77."""
    match = _PRICE_PATTERN.search(price_text)
    if not match:
        raise ValueError(f"Could not parse price from {price_text!r}")
    return float(match.group())


def _parse_rating(rating_text: str) -> int:
    return _RATING_WORDS.get(rating_text, 0)


def _parse_stock_count(availability_text: str) -> int | None:
    match = _STOCK_PATTERN.search(availability_text)
    return int(match.group(1)) if match else None


def clean_record(raw: RawProduct) -> CleanProduct | None:
    """Convert one raw record to a validated clean record, or None if invalid."""
    try:
        clean = CleanProduct(
            title=raw.title,
            price=_parse_price(raw.price_text),
            rating=_parse_rating(raw.rating_text),
            category=raw.category,
            availability=raw.availability_text.split("(")[0].strip() or "Unknown",
            stock_count=_parse_stock_count(raw.availability_text),
            source_url=raw.source_url,
        )
        return clean
    except (ValueError, ValidationError) as exc:
        logger.warning("Dropping invalid record (%s): %s", raw.source_url, exc)
        return None


def clean_batch(raw_records: Iterable[RawProduct]) -> Iterator[CleanProduct]:
    """Clean a stream of raw records, deduplicating by `source_url`."""
    seen_urls: set[str] = set()
    for raw in raw_records:
        if raw.source_url in seen_urls:
            logger.debug("Skipping duplicate: %s", raw.source_url)
            continue
        seen_urls.add(raw.source_url)

        clean = clean_record(raw)
        if clean is not None:
            yield clean
