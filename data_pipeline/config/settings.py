"""
Centralized, environment-driven configuration for the data pipeline.

All tunables (scrape target, retry policy, DB location) live here so
no other module hardcodes a URL, path, or magic number. Values are
overridable via environment variables or a `.env` file at the repo root.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PipelineSettings(BaseSettings):
    """Runtime configuration for the scraping + ETL pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Scraper ---
    base_url: str = Field(
        default="https://books.toscrape.com/",
        description="Root URL of the site being scraped.",
    )
    max_pages: int = Field(
        default=5,
        description="Maximum number of listing pages to crawl (pagination cap).",
    )
    request_timeout_seconds: int = Field(default=10)
    request_delay_seconds: float = Field(
        default=0.5, description="Polite delay between requests."
    )
    user_agent: str = Field(
        default="EcommerceIntelBot/1.0 (+educational-project)"
    )

    # --- Retry policy ---
    max_retries: int = Field(default=3)
    retry_backoff_seconds: float = Field(default=1.0)

    # --- Database ---
    database_path: Path = Field(
        default=PROJECT_ROOT / "data_pipeline" / "products.db"
    )

    # --- Logging ---
    log_level: str = Field(default="INFO")


settings = PipelineSettings()
