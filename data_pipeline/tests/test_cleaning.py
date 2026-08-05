import pytest

from data_pipeline.cleaning.cleaner import clean_batch, clean_record
from data_pipeline.models import RawProduct


def _make_raw(**overrides) -> RawProduct:
    defaults = dict(
        title="  A Great Book  ",
        price_text="£51.77",
        rating_text="Three",
        category=" fiction ",
        availability_text="In stock (22 available)",
        source_url="https://example.com/book-1",
    )
    defaults.update(overrides)
    return RawProduct(**defaults)


def test_clean_record_parses_price_and_strips_currency_symbol():
    result = clean_record(_make_raw())
    assert result is not None
    assert result.price == pytest.approx(51.77)


def test_clean_record_trims_and_title_cases_fields():
    result = clean_record(_make_raw())
    assert result.title == "A Great Book"
    assert result.category == "Fiction"


def test_clean_record_parses_rating_word_to_int():
    result = clean_record(_make_raw(rating_text="Five"))
    assert result.rating == 5


def test_clean_record_extracts_stock_count():
    result = clean_record(_make_raw(availability_text="In stock (7 available)"))
    assert result.stock_count == 7


def test_clean_record_handles_missing_stock_count():
    result = clean_record(_make_raw(availability_text="In stock"))
    assert result.stock_count is None


def test_clean_record_rejects_unparseable_price():
    result = clean_record(_make_raw(price_text="Contact us"))
    assert result is None


def test_clean_batch_deduplicates_by_source_url():
    raws = [_make_raw(), _make_raw()]  # identical source_url
    cleaned = list(clean_batch(raws))
    assert len(cleaned) == 1


def test_clean_batch_skips_invalid_and_keeps_valid():
    raws = [
        _make_raw(source_url="https://example.com/book-1"),
        _make_raw(source_url="https://example.com/book-2", price_text="n/a"),
    ]
    cleaned = list(clean_batch(raws))
    assert len(cleaned) == 1
    assert cleaned[0].source_url == "https://example.com/book-1"
