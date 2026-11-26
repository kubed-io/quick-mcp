"""Fixtures shared by every test."""

from __future__ import annotations

import logging

import pytest

import kubed.mcp._config
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
    sections' env vars from the process (``TRANSPORT``, ``HOST``, ``PORT``,
    ``LOG_LEVEL`` and the ``TELEMETRY_*`` and ``DELIVERY_*`` ones, which
    ``_compose([]).items`` yields), and a CI runner may have set any of them: each test
    starts with none of them set."""
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    for item in _compose([]).items:
        if item.env is not None:
            monkeypatch.delenv(item.env, raising=False)


@pytest.fixture
def no_builtins(monkeypatch):
    """No built-in sections, for tests of composition that name their own ``PORT`` or
    ``telemetry`` key, or assert a composition's exact keys. A ``QuickMCP`` can't be
    built under it: it reads the built-in sections by class."""
    monkeypatch.setattr(kubed.mcp._config, "_builtin_sections", lambda: ())


@pytest.fixture
def ran(monkeypatch, root_logging) -> list[dict]:
    """FastMCP's run, recorded. ``run()`` sets the root level, which is put back."""
    calls: list[dict] = []
    monkeypatch.setattr(
        "fastmcp.FastMCP.run", lambda self, **kwargs: calls.append(kwargs)
    )
    return calls


@pytest.fixture
def root_logging(monkeypatch):
    """The root logger's level and handlers, put back after the test.

    The level goes back through ``setLevel``, which clears every logger's cache of what
    it is enabled for.
    """
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    level = root.level
    yield root
    root.setLevel(level)
