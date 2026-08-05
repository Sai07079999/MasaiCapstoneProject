"""
Abstract scraper contract.

Any new site the pipeline needs to support (a different catalog, a job
board, a news site) implements this interface and plugs straight into
`main.py` without touching cleaning, validation, or persistence code.
This is the seam that keeps the pipeline reusable rather than a
one-off script tied to a single website.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterator

import requests

from data_pipeline.config import settings
from data_pipeline.models import RawProduct
from data_pipeline.scraper.retry import with_retry

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Template-method base class for paginated catalog scrapers."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": settings.user_agent})

    @with_retry
    def _get(self, url: str) -> requests.Response:
        response = self._session.get(url, timeout=settings.request_timeout_seconds)
        response.raise_for_status()
        return response

    def scrape(self) -> Iterator[RawProduct]:
        """Crawl paginated listing pages, yielding raw records as found."""
        page = 1
        while page <= settings.max_pages:
            url = self.build_page_url(page)
            if url is None:
                logger.info("No more pages after page %d.", page - 1)
                break

            logger.info("Fetching page %d: %s", page, url)
            try:
                response = self._get(url)
            except requests.exceptions.HTTPError as exc:
                logger.warning("Stopping pagination: %s", exc)
                break

            items = list(self.parse_page(response.text, url))
            if not items:
                logger.info("Page %d had no items; assuming end of catalog.", page)
                break

            yield from items
            page += 1

    @abstractmethod
    def build_page_url(self, page_number: int) -> str | None:
        """Return the URL for `page_number`, or None if out of range."""

    @abstractmethod
    def parse_page(self, html: str, page_url: str) -> Iterator[RawProduct]:
        """Parse a single listing page's HTML into raw product records."""
