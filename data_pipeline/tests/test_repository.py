import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data_pipeline.database.repository import ProductRepository
from data_pipeline.models import Base, CleanProduct


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


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


def test_upsert_inserts_new_record(session):
    repo = ProductRepository(session)
    repo.upsert(_make_clean())
    session.commit()

    assert repo.count() == 1


def test_upsert_updates_existing_record_by_source_url(session):
    repo = ProductRepository(session)
    repo.upsert(_make_clean(price=10.0))
    session.commit()

    repo.upsert(_make_clean(price=99.99))
    session.commit()

    products = repo.all()
    assert len(products) == 1
    assert products[0].price == pytest.approx(99.99)


def test_bulk_upsert_returns_count(session):
    repo = ProductRepository(session)
    records = [
        _make_clean(source_url="https://example.com/1"),
        _make_clean(source_url="https://example.com/2"),
    ]
    written = repo.bulk_upsert(records)
    session.commit()

    assert written == 2
    assert repo.count() == 2
