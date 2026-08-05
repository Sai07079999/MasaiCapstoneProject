"""
ETL pipeline entry point.

Run with:  python -m data_pipeline.main

Orchestrates: scrape -> clean -> validate -> persist. Each stage is a
plain function/class from another module; this file only wires them
together and logs progress, so the pipeline stays easy to test stage
by stage.
"""
from __future__ import annotations

import logging
import sys

from data_pipeline.cleaning import clean_batch, validate_batch
from data_pipeline.config import settings
from data_pipeline.database import ProductRepository, get_session, init_db
from data_pipeline.scraper import BookScraper


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def run_pipeline() -> int:
    """Execute one full pipeline run. Returns the number of records persisted."""
    logger = logging.getLogger("data_pipeline.main")
    logger.info("Starting pipeline run against %s (max_pages=%d)",
                settings.base_url, settings.max_pages)

    init_db()

    scraper = BookScraper()
    raw_records = scraper.scrape()

    clean_records = list(clean_batch(raw_records))
    logger.info("Cleaned %d records.", len(clean_records))

    report = validate_batch(clean_records)
    if not report.is_healthy:
        logger.warning("Validation reported issues: %s", report.warnings)

    with get_session() as session:
        repo = ProductRepository(session)
        written = repo.bulk_upsert(clean_records)

    logger.info("Pipeline run complete. %d records persisted to %s",
                written, settings.database_path)
    return written


if __name__ == "__main__":
    configure_logging()
    run_pipeline()
