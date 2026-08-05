"""
Concrete scraper for https://books.toscrape.com — a public catalog
site built for scraping practice, which is why it's used here rather
than a live commercial site.

Two-hop strategy per product:
  1. The paginated listing page gives us the product URL and a rating
     (encoded as a CSS class, e.g. "star-rating Three").
  2. The product detail page gives us the *authoritative* title,
     price, category (breadcrumb), and stock count — richer than what
     the listing page alone exposes.

This demonstrates handling pagination, multi-page enrichment, and
resilient parsing, not just a single flat page scrape.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Iterator, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from data_pipeline.config import settings
from data_pipeline.models import RawProduct
from data_pipeline.scraper.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

_RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


class BookScraper(BaseScraper):
    """Scrapes the full paginated book catalog at `settings.base_url`."""

    def build_page_url(self, page_number: int) -> Optional[str]:
        return urljoin(settings.base_url, f"catalogue/page-{page_number}.html")

    def parse_page(self, html: str, page_url: str) -> Iterator[RawProduct]:
        soup = BeautifulSoup(html, "lxml")
        articles = soup.select("article.product_pod")

        for article in articles:
            relative_link = article.select_one("h3 a")["href"]
            detail_url = urljoin(page_url, relative_link)

            rating_class = article.select_one("p.star-rating")["class"]
            rating_word = next((c for c in rating_class if c in _RATING_WORDS), "Zero")

            time.sleep(settings.request_delay_seconds)
            try:
                detail_response = self._get(detail_url)
            except Exception:  # noqa: BLE001 - one bad detail page shouldn't kill the run
                logger.exception("Failed to fetch detail page %s; skipping.", detail_url)
                continue

            product = self._parse_detail_page(detail_response.text, detail_url, rating_word)
            if product is not None:
                yield product

    @staticmethod
    def _parse_detail_page(
        html: str, detail_url: str, rating_word: str
    ) -> Optional[RawProduct]:
        soup = BeautifulSoup(html, "lxml")

        title_el = soup.select_one("div.product_main h1")
        price_el = soup.select_one("div.product_main p.price_color")
        availability_el = soup.select_one("div.product_main p.availability")
        breadcrumb = soup.select("ul.breadcrumb li a")

        if not (title_el and price_el and availability_el):
            logger.warning("Missing expected fields on %s; skipping.", detail_url)
            return None

        # Breadcrumb is: Home > Books > <Category> > <Title>
        category = breadcrumb[2].get_text(strip=True) if len(breadcrumb) >= 3 else "Unknown"

        return RawProduct(
            title=title_el.get_text(strip=True),
            price_text=price_el.get_text(strip=True),
            rating_text=rating_word,
            category=category,
            availability_text=availability_el.get_text(strip=True),
            source_url=detail_url,
        )
