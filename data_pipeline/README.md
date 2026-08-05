# Data Pipeline

A reusable, retry-safe ETL pipeline that scrapes a paginated product
catalog, cleans and validates it, and persists it to a relational
database.

## Architecture

```
BookScraper (BaseScraper)   →   clean_batch()   →   validate_batch()   →   ProductRepository
     scraper/                     cleaning/              cleaning/              database/
```

- **`scraper/`** — `BaseScraper` is an abstract template-method class
  (`scrape()` handles pagination + retry; subclasses implement
  `build_page_url` and `parse_page`). `BookScraper` targets
  [books.toscrape.com](https://books.toscrape.com), a site built
  specifically for scraping practice. Adding a new source means
  writing a new subclass — nothing else changes.
- **`cleaning/`** — pure functions, no I/O. Parses raw text (prices,
  ratings, stock counts) into typed `CleanProduct` records, drops
  invalid rows, deduplicates by URL, and runs batch-level sanity
  checks (`validators.py`).
- **`database/`** — repository pattern over SQLAlchemy. `main.py`
  never writes SQL directly; it calls `ProductRepository.bulk_upsert`.
- **`models/`** — `RawProduct`/`CleanProduct` (Pydantic DTOs, the
  schema contract) and `Product` (SQLAlchemy ORM row).
- **`config/`** — every tunable (URL, page limit, retry policy, DB
  path) is a `PipelineSettings` field, overridable via env vars or
  a `.env` file. Nothing is hardcoded in the logic modules.

## Running it

```bash
pip install -r requirements.txt
python -m data_pipeline.main
```

This creates `data_pipeline/products.db` (SQLite) and populates it
with up to `PIPELINE_MAX_PAGES` pages of products (default: 5 pages
≈ 100 books). Override any setting via env var, e.g.:

```bash
PIPELINE_MAX_PAGES=2 python -m data_pipeline.main
```

> **Sandbox note:** this response was generated in a sandboxed
> environment whose outbound network access is restricted to a small
> allowlist (PyPI, npm, GitHub) and does not include
> `books.toscrape.com`. The pipeline code itself is complete and unit
> tested end-to-end against fixture data (see `tests/`), but the live
> scrape was not executed here — run it on your own machine (or in
> GitHub Actions) to populate `products.db`, since normal
> unrestricted networks can reach the site fine.

## Tests

```bash
pytest data_pipeline/tests/ -v
```

15 tests covering price/rating/stock parsing, deduplication, batch
validation, and repository upsert semantics — all run against
in-memory fixtures, no network required.

## Design decisions

- **Two-hop scrape (listing → detail page)**: the listing page
  truncates titles and omits category; the detail page's breadcrumb
  gives the authoritative category and stock count. This trades a
  few extra requests for materially better data quality.
- **Upsert by `source_url`**: re-running the pipeline updates prices
  instead of creating duplicates, which is what a real re-scrape job
  needs.
- **Repository pattern**: keeps SQLAlchemy specifics out of
  `main.py`, so switching SQLite → PostgreSQL is a one-line change in
  `config/settings.py`.
