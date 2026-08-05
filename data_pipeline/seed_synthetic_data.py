"""
Seed script: generates a realistic synthetic product catalog and writes
it through the SAME repository/model layer `data_pipeline.main` uses.

Why this exists: this project was built in a sandboxed environment
whose network allowlist does not include books.toscrape.com, so the
live scraper (fully implemented in `data_pipeline/scraper/`) could not
be executed here. To let `analytics/` run against real, schema-valid
data end-to-end, this script generates a plausible synthetic catalog
instead of fabricating "scraped" results.

Run with:  python -m data_pipeline.seed_synthetic_data
On an unrestricted network, prefer the real thing:
           python -m data_pipeline.main
"""
from __future__ import annotations

import logging
import random

from data_pipeline.config import settings
from data_pipeline.database import ProductRepository, get_session, init_db
from data_pipeline.models import CleanProduct

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Fiction", "Nonfiction", "Mystery", "Romance", "Fantasy",
    "Science Fiction", "Biography", "History", "Poetry", "Business",
    "Childrens", "Travel", "Cookbooks", "Self Help", "Young Adult",
]

# Each category gets a base price range and a base rating tendency,
# so the resulting dataset has real, learnable structure rather than
# pure noise (mirrors how genres actually differ in pricing/reception).
CATEGORY_PROFILE = {
    "Fiction": (8, 25, 3.4),
    "Nonfiction": (10, 35, 3.6),
    "Mystery": (7, 20, 3.5),
    "Romance": (6, 18, 3.2),
    "Fantasy": (9, 28, 3.8),
    "Science Fiction": (9, 30, 3.7),
    "Biography": (12, 32, 3.9),
    "History": (14, 40, 3.8),
    "Poetry": (6, 16, 3.3),
    "Business": (15, 45, 3.5),
    "Childrens": (5, 14, 4.0),
    "Travel": (10, 26, 3.4),
    "Cookbooks": (12, 38, 3.9),
    "Self Help": (9, 24, 3.3),
    "Young Adult": (8, 22, 3.9),
}

TITLE_WORDS_A = [
    "Shadow", "Silent", "Last", "Hidden", "Golden", "Broken", "Winter",
    "Secret", "Lost", "Distant", "Quiet", "Forgotten", "Endless", "First",
]
TITLE_WORDS_B = [
    "Garden", "Kingdom", "Letters", "River", "Promise", "Journey", "House",
    "Mirror", "Storm", "Harbor", "Circle", "Flame", "Voyage", "Echo",
]


def _generate_title(rng: random.Random) -> str:
    return f"The {rng.choice(TITLE_WORDS_A)} {rng.choice(TITLE_WORDS_B)}"


def generate_synthetic_catalog(n: int = 600, seed: int = 42) -> list[CleanProduct]:
    """Generate `n` schema-valid, category-correlated synthetic products."""
    rng = random.Random(seed)
    records: list[CleanProduct] = []

    for i in range(n):
        category = rng.choice(CATEGORIES)
        low_price, high_price, base_rating = CATEGORY_PROFILE[category]

        price = round(rng.uniform(low_price, high_price), 2)
        # Higher-priced items within a category skew slightly higher rated
        # (a mild, realistic signal for the model to find) plus noise.
        price_position = (price - low_price) / max(high_price - low_price, 1e-6)
        rating_float = base_rating + price_position * 0.6 + rng.gauss(0, 0.6)
        rating = min(5, max(1, round(rating_float)))

        stock_count = max(0, int(rng.gauss(35 - rating * 4, 15)))
        availability = "In stock" if stock_count > 0 else "Out of stock"

        records.append(
            CleanProduct(
                title=f"{_generate_title(rng)} #{i}",
                price=price,
                rating=rating,
                category=category,
                availability=availability,
                stock_count=stock_count,
                source_url=f"https://books.toscrape.com/synthetic/{i}",
            )
        )
    return records


def seed(n: int = 600) -> int:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(message)s")
    init_db()
    records = generate_synthetic_catalog(n=n)
    with get_session() as session:
        repo = ProductRepository(session)
        written = repo.bulk_upsert(records)
    logger.info("Seeded %d synthetic products into %s", written, settings.database_path)
    return written


if __name__ == "__main__":
    seed()
