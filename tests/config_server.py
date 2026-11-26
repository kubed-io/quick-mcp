"""A root section for tests that need a server with a config, and nothing more.

``Duck`` has the three serving settings ``QuickMCP.run()`` reads, and holds whatever a test
gives ``telemetry`` and ``delivery``, which ``QuickMCP`` reads by attribute until the
built-ins are components. It is a stand-in: a later change deletes it, and the telemetry and
delivery tests then move onto the real sections.
"""

from __future__ import annotations

from typing import Any

from kubed.mcp._config import Config, config


@config()
class Duck(Config):
    transport: str = "http"
    host: str = "0.0.0.0"
    port: int = 8000
    telemetry: Any = None
    delivery: Any = None
