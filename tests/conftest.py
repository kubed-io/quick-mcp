"""Fixtures shared by every test."""

from __future__ import annotations

import pytest

from kubed.mcp import request


@pytest.fixture(autouse=True)
def forget_warnings():
    """Warn-once is process-wide: each test starts having warned of nothing."""
    request._warned.clear()
    yield
