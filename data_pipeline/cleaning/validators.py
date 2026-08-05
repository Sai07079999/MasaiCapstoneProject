"""
Batch-level schema/quality checks, run after per-record cleaning.
Per-record type/range validation lives in `models.dto.CleanProduct`;
this module checks properties of the *collection* as a whole.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from data_pipeline.models import CleanProduct

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Summary of a batch validation pass."""

    total_records: int
    unique_categories: int
    price_range: tuple[float, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.total_records > 0 and not self.warnings


def validate_batch(records: list[CleanProduct]) -> ValidationReport:
    """Run sanity checks on a cleaned batch and return a report."""
    warnings: list[str] = []

    if not records:
        warnings.append("Batch is empty.")
        return ValidationReport(0, 0, (0.0, 0.0), warnings)

    prices = [r.price for r in records]
    categories = {r.category for r in records}
    urls = [r.source_url for r in records]

    if len(urls) != len(set(urls)):
        warnings.append("Duplicate source_url values found in cleaned batch.")

    if min(prices) <= 0:
        warnings.append("Non-positive price found after cleaning.")

    if len(categories) == 1:
        warnings.append("All records share a single category — check parsing logic.")

    report = ValidationReport(
        total_records=len(records),
        unique_categories=len(categories),
        price_range=(min(prices), max(prices)),
        warnings=warnings,
    )

    if warnings:
        logger.warning("Validation warnings: %s", warnings)
    else:
        logger.info("Batch passed validation: %d records, %d categories.",
                     report.total_records, report.unique_categories)

    return report
