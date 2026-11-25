"""The handshake says what the request resolved, and the server's instructions are
written for it (DLV-11, DLV-12, GW-9).

FastMCP 4.2 runs ``initialize`` through ``Middleware.on_initialize`` and
``server/discover`` through ``Middleware.on_discover``, each handed the typed result on
its way back, so ``_Describe`` changes both with public hooks and touches no private.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, MutableMapping
from typing import Any

from fastmcp.server.middleware import Middleware

from kubed.mcp.delivery import Instructions, _echo, _resolved
from kubed.mcp.gateway import prefixed

# The delivery's logger: an instructions fn fails as a delivery_default one does.
log = logging.getLogger("kubed.mcp.delivery")

META_KEY = "io.kubed/delivery"


class _Describe(Middleware):
    """Echoes the resolution into every ``initialize`` and ``server/discover`` result,
    and writes the server's ``instructions`` fn into it.

    It sits inside ``_ResolveScope``, so ``_resolved()`` is the request's resolution.
    """

    def __init__(self, instructions: Callable[[], Instructions | None]) -> None:
        self._instructions = instructions

    async def on_initialize(self, context: Any, call_next: Any) -> Any:
        result = await call_next(context)
        _describe(result, self._instructions())
        return result

    async def on_discover(self, context: Any, call_next: Any) -> Any:
        result = await call_next(context)
        _describe(result, self._instructions())
        return result


def _written(fn: Instructions | None, resolved: Any) -> str | None:
    """``fn``'s instructions for ``resolved``, prefixed, or None to keep the built ones.

    An ``fn`` that raises, or returns anything but a string or None, is logged at error.
    """
    if fn is None:
        return None
    try:
        text = fn(resolved)
    except Exception:
        log.exception("instructions failed: the built instructions stay")
        return None
    if text is None:
        return None
    if not isinstance(text, str):
        log.error(
            "instructions gave %s, not a string: the built instructions stay",
            type(text).__name__,
        )
        return None
    return prefixed(text, resolved.prefix)


def _describe(result: Any, fn: Instructions | None) -> None:
    """Writes the echo into ``result``'s ``_meta``, beside what is there, and ``fn``'s
    instructions over the built ones (DLV-11, DLV-12). ``result`` is a typed result, or
    the mapping ``server/discover`` gives for a result type it does not know."""
    if result is None:
        return
    resolved = _resolved()
    text = _written(fn, resolved)
    echo = {META_KEY: _echo(resolved)}
    if isinstance(result, MutableMapping):
        result["_meta"] = {**(result.get("_meta") or {}), **echo}
        if text is not None:
            result["instructions"] = text
        return
    result.meta = {**(result.meta or {}), **echo}
    if text is not None:
        result.instructions = text
