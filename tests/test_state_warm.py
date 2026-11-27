"""APP-5: serving starts with FastMCP's session-state store already loaded."""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from contextlib import asynccontextmanager

from fastmcp import Client, FastMCP

from kubed.mcp import QuickMCP

# The modules FastMCP's _state_store imports on its first use.
STATE_MODULES = ("key_value.aio.stores.memory", "key_value.aio.adapters.pydantic")

# In a process of its own: another test may have imported the modules already. The
# lifespan is entered through the HTTP app, as uvicorn enters it, and not through
# fastmcp's Client: importing the client imports both modules.
_CHILD = textwrap.dedent(
    """
    import asyncio
    import sys

    from kubed.mcp import QuickMCP

    MODULES = {modules!r}

    mcp = QuickMCP("warm")


    @mcp.tool
    def ping() -> str:
        return "pong"


    app = mcp.server.http_app()
    before = [name for name in MODULES if name in sys.modules]
    assert not before, f"loaded before serving started: {{before}}"


    async def main() -> list[str]:
        async with app.router.lifespan_context(app):
            # No request has run: only the lifespan can have loaded them.
            return [name for name in MODULES if name not in sys.modules]


    missing = asyncio.run(main())
    assert not missing, f"not loaded while serving: {{missing}}"
    print("warm")
    """
).format(modules=STATE_MODULES)


def boot_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.config" and r.levelno == logging.WARNING
    ]


def test_serving_starts_with_the_state_store_loaded() -> None:
    done = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "warm"


# FastMCP reads the store on every request that applies session transforms, a tool
# call and the listings included, so with no working store those fail in FastMCP
# itself. What shows serving went on is the boot: the server's own lifespan ran and the
# client connected.


def served(entered: list[str]) -> QuickMCP:
    @asynccontextmanager
    async def lifespan(server):
        entered.append("lifespan")
        yield {}

    return QuickMCP("warm", lifespan=lifespan)


async def test_a_fastmcp_without_a_state_store_still_serves(
    monkeypatch, caplog
) -> None:
    monkeypatch.delattr(FastMCP, "_state_store", raising=True)
    caplog.set_level(logging.DEBUG)
    entered: list[str] = []
    async with Client(served(entered).server) as client:
        assert client.is_connected()
    assert entered == ["lifespan"]
    warnings = boot_warnings(caplog)
    assert len(warnings) == 1
    assert "session-state store" in warnings[0].getMessage()


async def test_a_warm_up_that_fails_is_one_warning_by_type(monkeypatch, caplog) -> None:
    def refuse(self) -> None:
        raise RuntimeError("secret-text")

    monkeypatch.setattr(FastMCP, "_state_store", property(refuse))
    caplog.set_level(logging.DEBUG)
    # What had been logged when the server's own lifespan was entered.
    seen: list[list[tuple[str, int, str]]] = []

    @asynccontextmanager
    async def lifespan(server):
        seen.append([(r.name, r.levelno, r.getMessage()) for r in caplog.records])
        yield {}

    async with Client(QuickMCP("warm", lifespan=lifespan).server) as client:
        assert client.is_connected()
    warnings = boot_warnings(caplog)
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].getMessage()
    # APP-5: the components start, then the warm-up, then the server's own lifespan.
    [before] = seen
    started = before.index(("kubed.mcp.telemetry", logging.INFO, "telemetry: off"))
    warned = before.index(
        ("kubed.mcp.config", logging.WARNING, warnings[0].getMessage())
    )
    assert started < warned
    formatter = logging.Formatter("%(message)s")
    for record in caplog.records:
        assert record.exc_info is None
        assert "secret-text" not in formatter.format(record)
