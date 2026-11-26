"""A ``subscriptions/listen`` stream is left out of the server spans.

FastMCP's ``FastMCPServerMiddleware`` opens a SERVER span around every request. A
listen stream lasts as long as the client's connection, so its span would last minutes
and swamp the latency views. ``_ListenUntraced`` runs just ahead of it and puts the
request under FastMCP's ``suppress_fastmcp_telemetry()``, so the seam opens no span,
while trace context still propagates. The call metrics never time a listen stream
anyway. Every other method is untouched.

It reaches FastMCP's low-level server through ``_mcp_server``, a private name. Without
it, its ``middleware`` list, or FastMCP's seam in that list, ``_leave_listen_untraced``
logs one warning and installs nothing: the server serves, and the stream is traced.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_LISTEN = "subscriptions/listen"


class _ListenUntraced:
    """A server middleware: a listen request runs with FastMCP tracing suppressed."""

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        if ctx.request_id is not None and ctx.method == _LISTEN:
            from fastmcp.telemetry import suppress_fastmcp_telemetry

            with suppress_fastmcp_telemetry():
                return await call_next(ctx)
        return await call_next(ctx)


def _leave_listen_untraced(server: Any) -> None:
    """Put ``_ListenUntraced`` just ahead of FastMCP's seam in the server's middleware.

    Never raises (TEL-6).
    """
    try:
        from fastmcp.server.low_level import FastMCPServerMiddleware

        middleware = server._mcp_server.middleware
        at = next(
            (
                i
                for i, found in enumerate(middleware)
                if isinstance(found, FastMCPServerMiddleware)
            ),
            None,
        )
        if at is None:
            _warn("no FastMCPServerMiddleware")
            return
        middleware.insert(at, _ListenUntraced())
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        _warn(type(error).__name__)


def _warn(why: str) -> None:
    log.warning(
        "subscriptions/listen streams may be traced as server spans: "
        "FastMCP's server has changed (%s)",
        why,
    )
