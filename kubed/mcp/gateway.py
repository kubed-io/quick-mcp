"""What a gateway says about itself, and how its prefix is written into text.

A gateway in front of this server rewrites the structured fields of what it relays
(names, URIs), never the text of a skill or a prompt. So a skill that links to
``skill://lib/guide/SKILL.md`` would send the model to an address the gateway does
not serve. A client names the gateway's prefix, and ``prefixed`` writes it into
those addresses. ``_TakeThePrefix`` takes it off again from the addresses a client
sends back, before any provider is asked.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from fastmcp.exceptions import NotFoundError
from fastmcp.server.middleware import Middleware

from kubed.mcp.delivery import _INPUTS, _PREFIX, _resolved
from kubed.mcp.skills import FolderNotFound
from mcp.shared.exceptions import MCPError

SCHEMES = ("skill", "prompts", "agents")

# The start of an address of a served scheme, not the tail of a longer word or scheme.
_ADDRESS = re.compile(r"(?<![A-Za-z0-9+.-])(?=(?:skill|prompts|agents)://)")


def prefixed(text: str, prefix: str | None) -> str:
    """``text`` with ``prefix`` written before each address of a served scheme.

    With no prefix it is ``text`` itself. An address already carrying a prefix is left
    alone, so the result is idempotent. A prefix that is not valid is a ``ValueError``.
    """
    if prefix is None:
        return text
    if not _PREFIX.fullmatch(prefix):
        raise ValueError(
            f"{prefix[:64]!r} is not a valid prefix: {_INPUTS['Resource-Prefix'].rule}"
        )
    return _ADDRESS.sub(lambda _: prefix, text)


def prefix() -> str | None:
    """The current request's prefix, or None: the resolved one (DLV-3)."""
    return _resolved().prefix


# ---- incoming addresses (GW-4) -----------------------------------------------------

# An address of a served scheme, with or without a prefix before it.
_SERVED = re.compile(r"(?:[A-Za-z][A-Za-z0-9+.-]*\+)?(skill|prompts|agents)://")
# The methods besides resources/read that name a plugin address, matched as sent.
_AS_SENT = ("skills/get", "resources/directory/read")


def _plain(uri: str, prefix: str | None, *, dots: bool) -> str:
    """The address the providers are asked for ``uri``, by GW-4's steps.

    An address of another scheme, of another prefix, or prefixed on a request with
    none, is ``uri`` as sent. One that resolves to nothing is ``<scheme>://%2F``, an
    address no provider serves. ``dots`` resolves dot segments in the path, keeping a
    query or fragment as sent.
    """
    served = _SERVED.match(uri)
    if served is None:
        return uri
    scheme = f"{served[1]}://"
    nowhere = f"{scheme}%2F"
    if "%2f" in uri.lower():
        return nowhere
    if prefix and uri.startswith(prefix + scheme):
        uri = uri[len(prefix) :]
    elif not uri.startswith(scheme):
        return uri
    if not dots:
        return uri
    rest = uri[len(scheme) :]
    # The path ends at the first "?" or "#": what follows is kept as sent.
    cut = next((i for i, c in enumerate(rest) if c in "?#"), len(rest))
    path, tail = rest[:cut], rest[cut:]
    kept: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not kept:
                return nowhere
            kept.pop()
        else:
            kept.append(segment)
    # A trailing slash stays: it names no folder (SKL-14), so it is a plain miss.
    slash = "/" if kept and path.endswith("/") else ""
    return scheme + "/".join(kept) + slash + tail


def _as_sent(exc: Exception, passed: str, sent: str) -> Exception | None:
    """``exc``, raised for ``passed``, naming ``sent``; None to raise ``exc`` as is."""
    if isinstance(exc, FolderNotFound):
        return FolderNotFound(sent)
    if isinstance(exc, NotFoundError):
        return NotFoundError(str(exc).replace(passed, sent))
    if isinstance(exc, MCPError):
        data = exc.data
        if isinstance(data, dict) and data.get("uri") == passed:
            data = {**data, "uri": sent}
        return MCPError(
            code=exc.code, message=exc.message.replace(passed, sent), data=data
        )
    return None


class _TakeThePrefix(Middleware):
    """Asks every provider for the plain address of what a client sent (GW-4).

    It sits inside ``_PinScope``, so the prefix it takes off is the one the pinned
    build writes, and outside ``NameTheIndex`` and every provider, so none of them
    sees a prefix. A miss names the address as sent. A message with nothing to take
    off goes on as it came.
    """

    async def on_read_resource(self, context: Any, call_next: Any) -> Any:
        sent = str(context.message.uri)
        passed = _plain(sent, prefix(), dots=True)
        if passed == sent:
            return await call_next(context)
        message = context.message.model_copy(update={"uri": passed})
        return await self._ask(context.copy(message=message), call_next, passed, sent)

    async def on_request(self, context: Any, call_next: Any) -> Any:
        if context.method not in _AS_SENT:
            return await call_next(context)
        # FastMCP hands an extension method its raw params as a dict, and folds back
        # only a Mapping.
        message = context.message
        sent = message.get("uri") if isinstance(message, Mapping) else None
        if not isinstance(sent, str):
            return await call_next(context)
        passed = _plain(sent, prefix(), dots=False)
        if passed == sent:
            return await call_next(context)
        message = {**message, "uri": passed}
        return await self._ask(context.copy(message=message), call_next, passed, sent)

    @staticmethod
    async def _ask(context: Any, call_next: Any, passed: str, sent: str) -> Any:
        try:
            return await call_next(context)
        except Exception as exc:
            renamed = _as_sent(exc, passed, sent)
            if renamed is None:
                raise
            raise renamed from None
