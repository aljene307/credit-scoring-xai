"""Réglages communs aux tests."""

import pytest


@pytest.fixture(autouse=True)
def api_unreachable(monkeypatch):
    """Les tests n'appellent jamais une API réelle qui tournerait sur le PC : adresse injoignable."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("API_TIMEOUT", "1")
