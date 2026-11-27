"""A server of its own for the shutdown tests: ``shutdown_server.py [ping]``.

It serves over HTTP as its env vars say (``TRANSPORT``, ``HOST``, ``PORT`` and the
rest), with one tool, ``slow``, that prints ``slow: started`` and sleeps the seconds it
is given. With ``ping``, the
SDK commits a listen's event stream and pings it every 0.2 s instead of every 15 s.
"""

from __future__ import annotations

import asyncio
import sys

from kubed.mcp import QuickMCP

if "ping" in sys.argv[1:]:
    import mcp.server._streamable_http_modern

    mcp.server._streamable_http_modern._SSE_PING_INTERVAL = 0.2

server = QuickMCP("shutdown")


@server.tool
async def slow(seconds: float) -> str:
    """Says it started, on stdout, then sleeps ``seconds`` and answers."""
    print("slow: started", flush=True)
    await asyncio.sleep(seconds)
    return f"slept {seconds}"


server.run()
