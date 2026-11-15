from __future__ import annotations

from backend.settings import Settings


def test_origins_splits_trims_and_discards_empty_entries() -> None:
    settings = Settings(allowed_origins="https://a.example, http://localhost:3000, ,https://b.example,,")
    assert settings.origins() == [
        "https://a.example",
        "http://localhost:3000",
        "https://b.example",
    ]


def test_origins_returns_empty_list_when_nothing_configured() -> None:
    settings = Settings(allowed_origins=" ,  ,")
    assert settings.origins() == []
