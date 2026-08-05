from data_pipeline.cleaning.validators import validate_batch
from data_pipeline.models import CleanProduct


def _make_clean(**overrides) -> CleanProduct:
    defaults = dict(
        title="Book",
        price=10.0,
        rating=4,
        category="Fiction",
        availability="In stock",
        stock_count=5,
        source_url="https://example.com/1",
    )
    defaults.update(overrides)
    return CleanProduct(**defaults)


def test_validate_batch_empty_list_is_unhealthy():
    report = validate_batch([])
    assert not report.is_healthy
    assert "empty" in report.warnings[0].lower()


def test_validate_batch_healthy_when_diverse_and_valid():
    records = [
        _make_clean(source_url="https://example.com/1", category="Fiction", price=10.0),
        _make_clean(source_url="https://example.com/2", category="Non Fiction", price=20.0),
    ]
    report = validate_batch(records)
    assert report.is_healthy
    assert report.total_records == 2
    assert report.unique_categories == 2
    assert report.price_range == (10.0, 20.0)


def test_validate_batch_flags_duplicate_urls():
    records = [
        _make_clean(source_url="https://example.com/1"),
        _make_clean(source_url="https://example.com/1"),
    ]
    report = validate_batch(records)
    assert any("duplicate" in w.lower() for w in report.warnings)


def test_validate_batch_flags_single_category():
    records = [
        _make_clean(source_url="https://example.com/1", category="Fiction"),
        _make_clean(source_url="https://example.com/2", category="Fiction"),
    ]
    report = validate_batch(records)
    assert any("single category" in w.lower() for w in report.warnings)
