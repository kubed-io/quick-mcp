"""What a request asks for besides its plugins, resolved once per request (DLV-2 to 5).

Each input is a header or a query parameter. It is resolved on its own by walking its
levels in order, and the first value the input takes answers. A value it does not take
is ignored with one warning, and the next level answers. With none, the built-in default
does. ``_ResolveScope`` makes the walk once per request, and ``_resolved()`` is what
the rest of the package reads.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Mapping
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol
from weakref import WeakKeyDictionary

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware

from kubed.mcp.request import Values, _warn_once, get_context, request_values

log = logging.getLogger(__name__)

__all__ = ["Delivery"]

# A letter, then letters, digits, +, - or ., ending in +: 64 characters at most.
_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{0,62}\+")
_KINDS = frozenset({"skills", "prompts", "agents"})
_ON = frozenset({"on", "true", "1", "yes"})
_OFF = frozenset({"off", "false", "0", "no", "none"})

Found = tuple[str, str]  # (raw value, source)
Level = Callable[[tuple[Values, Values] | None], Found | None]


class Delivery(Protocol):
    """The defaults a config may give the inputs (DLV-4): None is the built-in one."""

    @property
    def resources(self) -> str | None: ...
    @property
    def listed(self) -> str | Collection[str] | None: ...
    @property
    def via(self) -> str | None: ...


def _take_prefix(raw: str) -> str:
    value = raw.strip()
    if not _PREFIX.fullmatch(value):
        raise ValueError(raw)
    return value


def _take_resources(raw: str) -> str:
    value = raw.strip().lower()
    if value in _ON:
        return "on"
    if value in _OFF:
        return "off"
    raise ValueError(raw)


def _take_listed(raw: str) -> frozenset[str]:
    value = raw.strip().lower()
    if value in ("", "none"):
        return frozenset()
    named = {kind.strip() for kind in value.split(",")} & _KINDS
    if not named:
        raise ValueError(raw)
    return frozenset(named)


@dataclass(frozen=True)
class _Input:
    """One input: where it is read, what it takes, and what answers when none does."""

    name: str
    query: str
    attr: str | None  # the Delivery attribute, None when the config has none
    take: Callable[[str], object]  # raises ValueError for a value it does not take
    rule: str  # what a warning quotes
    default: object
    blank: bool  # an empty value is a value
    plus: bool  # a space in the query reads as +


_INPUTS: Mapping[str, _Input] = MappingProxyType(
    {
        entry.name: entry
        for entry in (
            _Input(
                "Resource-Prefix",
                "prefix",
                None,
                _take_prefix,
                "a prefix is a letter, then letters, digits, +, - or ., ending in +, "
                "at most 64 characters",
                None,
                blank=False,
                plus=True,
            ),
            _Input(
                "Resources",
                "resources",
                "resources",
                _take_resources,
                "it is on, true, 1, yes, or off, false, 0, no, none",
                "on",
                blank=False,
                plus=False,
            ),
            _Input(
                "Listed",
                "listed",
                "listed",
                _take_listed,
                "it is a comma-separated set of skills, prompts and agents, or empty "
                "or none for none",
                frozenset({"skills"}),
                blank=True,
                plus=False,
            ),
        )
    }
)


@dataclass(frozen=True)
class _Resolved:
    """What one request resolved its inputs to."""

    resources: str
    listed: frozenset[str]
    prefix: str | None


def _header(name: str) -> Level:
    """The header ``name``, first value."""
    key = name.lower()

    def level(values: tuple[Values, Values] | None) -> Found | None:
        if values is None or not values[1].get(key):
            return None
        return values[1][key][0], f"header {name}"

    return level


def _query(entry: _Input) -> Level:
    """The query parameter of ``entry``, first value."""

    def level(values: tuple[Values, Values] | None) -> Found | None:
        if values is None or not values[0].get(entry.query):
            return None
        raw = values[0][entry.query][0]
        if entry.plus:
            raw = raw.replace(" ", "+")
        return raw, f"query parameter {entry.query}"

    return level


def _config_level(entry: _Input, delivery: object | None) -> Level | None:
    """The level for what the config gives ``entry``, checked now, once.

    A value the input does not take is warned here, naming the attribute, and gives no
    level, so a bad config warns once and not once per session.
    """
    if delivery is None or entry.attr is None:
        return None
    value = getattr(delivery, entry.attr, None)
    if value is None:
        return None
    if isinstance(value, str):
        raw = value
    elif entry.name == "Listed" and isinstance(value, Collection):
        raw = ",".join(str(kind) for kind in value)
    else:
        raw = None
    source = f"config delivery.{entry.attr}"
    try:
        if raw is None:
            raise ValueError(value)
        if not raw.strip() and not entry.blank:
            return None
        entry.take(raw)
    except ValueError:
        _warning(source, raw.strip() if raw is not None else str(value), entry)
        return None
    found = (raw, source)
    return lambda values: found


def _warning(source: str, value: str, entry: _Input) -> None:
    log.warning(
        "%s %r is ignored, and the next source answers: %s",
        source,
        value[:64],
        entry.rule,
    )


class _Resolver:
    """Walks each input's levels, in order, for one request."""

    def __init__(
        self, header_prefix: str | None = None, delivery: object | None = None
    ) -> None:
        self.header_prefix = header_prefix
        self.levels: dict[str, list[Level]] = {}
        for entry in _INPUTS.values():
            levels = [_header(f"X-MCP-{entry.name}"), _query(entry)]
            config = _config_level(entry, delivery)
            if config is not None:
                levels.append(config)
            self.levels[entry.name] = levels

    def resolve(
        self, values: tuple[Values, Values] | None, *, warn: bool = True
    ) -> _Resolved:
        """What ``values`` resolve each input to; ``warn`` False walks silently."""
        got = {name: self._walk(_INPUTS[name], values, warn) for name in _INPUTS}
        return _Resolved(
            resources=got["Resources"],
            listed=got["Listed"],
            prefix=got["Resource-Prefix"],
        )

    def _walk(
        self, entry: _Input, values: tuple[Values, Values] | None, warn: bool
    ) -> Any:
        for level in self.levels[entry.name]:
            found = level(values)
            if found is None:
                continue
            raw, source = found
            if not raw.strip() and not entry.blank:
                continue
            try:
                return entry.take(raw)
            except ValueError:
                shown = raw.strip()[:64]
                if warn and _warn_once(source, shown):
                    _warning(source, shown, entry)
        return entry.default


_RESOLVED: ContextVar[tuple[_Resolver, _Resolved] | None] = ContextVar(
    "kubed_mcp_resolved", default=None
)
_RESOLVERS: WeakKeyDictionary[FastMCP, _Resolver] = WeakKeyDictionary()
_BARE = _Resolver()


class _ResolveScope(Middleware):
    """Resolves the inputs once for a request and keeps the answer for all of it.

    A request made inside another, a tool's ``ctx.read_resource`` say, finds the
    resolution open and shares it.
    """

    def __init__(self, resolver: _Resolver) -> None:
        self._resolver = resolver

    async def on_request(self, context: Any, call_next: Any) -> Any:
        current = _RESOLVED.get()
        if current is not None and current[0] is self._resolver:
            return await call_next(context)
        # The handshake's session has no id yet: it would warn under no session, and
        # again under the one it opens, so it resolves without warning.
        resolved = self._resolver.resolve(
            request_values(), warn=context.method != "initialize"
        )
        token = _RESOLVED.set((self._resolver, resolved))
        try:
            return await call_next(context)
        finally:
            _RESOLVED.reset(token)


def _resolved() -> _Resolved:
    """The current request's resolution.

    Off a request, as in a background task's worker, it is resolved afresh from what the
    context kept, by the resolver of the context's server when it has one.
    """
    current = _RESOLVED.get()
    if current is not None:
        return current[1]
    resolver = _BARE
    with suppress(RuntimeError):  # no context
        resolver = _RESOLVERS.get(get_context().fastmcp, _BARE)
    return resolver.resolve(request_values())
