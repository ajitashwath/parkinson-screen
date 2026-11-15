from __future__ import annotations

from backend.database import normalize_url


def test_normalize_url_rewrites_render_and_heroku_postgres_urls() -> None:
    assert normalize_url("postgres://user:pw@host/db") == "postgresql+psycopg://user:pw@host/db"
    assert normalize_url("postgresql://user:pw@host/db") == "postgresql+psycopg://user:pw@host/db"


def test_normalize_url_leaves_non_postgres_urls_unchanged() -> None:
    sqlite_url = "sqlite:///./screening.db"
    assert normalize_url(sqlite_url) == sqlite_url
    assert normalize_url("sqlite://") == "sqlite://"
