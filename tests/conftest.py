"""Fixtures shared by every test."""

from __future__ import annotations

import pytest

from kubed.mcp import request
from kubed.mcp._config import _compose


@pytest.fixture(autouse=True)
def forget_warnings():
    """Warn-once is process-wide: each test starts having warned of nothing."""
    request._warned.clear()
    yield


@pytest.fixture(autouse=True)
def no_config_from_the_runner(monkeypatch):
    """A server built without ``environ=`` reads ``CONFIG_FILE`` and the built-in
    items' env vars from the process, and a CI runner may have set any of them: each
    test starts with none of them set."""
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    for item in _compose([]).items:
        if item.env is not None:
            monkeypatch.delenv(item.env, raising=False)
