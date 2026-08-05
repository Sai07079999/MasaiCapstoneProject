"""
Reusable retry policy for network calls.

Wraps `tenacity` so every scraper in this project (current and future)
gets the same exponential-backoff-with-jitter behavior from one place,
instead of each scraper re-implementing its own retry loop.
"""
from __future__ import annotations

import logging
from typing import Callable, TypeVar

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from data_pipeline.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator applying the pipeline's standard retry/backoff policy."""

    decorated = retry(
        reraise=True,
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential_jitter(initial=settings.retry_backoff_seconds),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying %s (attempt %d) after error: %s",
            func.__name__,
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        ),
    )(func)
    return decorated
