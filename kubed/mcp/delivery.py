"""What a request asks for besides its plugins, resolved once per request (DLV-2 to 5).

Each input is a header or a query parameter. It is resolved on its own by walking its
levels in order, and the first value the input takes answers. A value it does not take
is ignored with one warning, and the next level answers. With none, the built-in default
does. ``Via`` is walked first, and ``Listed`` only as far as the level that answered it,
its preset answering past that (DLV-9). ``_ResolveScope`` makes the walk once per
request, and ``_resolved()`` is what the rest of the package reads.
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

from kubed.mcp.request import (
    Values,
    _warn_once,
    get_context,
    get_http_request,
    request_values,
)

log = logging.getLogger(__name__)

__all__ = ["BinaryPointer", "Delivery", "Resolved"]

# A letter, then letters, digits, +, - or ., ending in +: 64 characters at most.
_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{0,62}\+")
# A header-name token: letters and digits, with - only between them.
_HEADER_TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_KINDS = frozenset({"skills", "prompts", "agents"})
_VIAS = ("mcp", "extension", "local")
# What X-MCP-Listed defaults to under each Via (DLV-9).
_PRESET: Mapping[str, frozenset[str]] = MappingProxyType(
    {"mcp": frozenset({"skills"}), "extension": frozenset(), "local": frozenset()}
)
_ON = frozenset({"on", "true", "1", "yes"})
_OFF = frozenset({"off", "false", "0", "no", "none"})

Found = tuple[str, str]  # (raw value, source)
Level = Callable[[tuple[Values, Values] | None], Found | None]

# Where a binary's link is listed, given its URI as sent, or None (DLV-6).
BinaryPointer = Callable[[str], str | None]


class Delivery(Protocol):
    """The defaults a config may give the inputs (DLV-4): None is the built-in one."""

    @property
    def resources(self) -> str | None: ...
    @property
    def listed(self) -> str | Collection[str] | None: ...
    @property
    def via(self) -> str | None: ...


def _header_prefix(value: str | None) -> str | None:
    """``value`` as the server's header prefix (DLV-16), or a ``ValueError``."""
    if value is None:
        return None
    if not _HEADER_TOKEN.fullmatch(value):
        raise ValueError(
            f"header_prefix {value!r} is not a header-name token: letters and digits, "
            "with - only between them (Kb, Sf, My-Co)"
        )
    if value.lower() == "x-mcp":
        raise ValueError(
            f"header_prefix {value!r} is the shared X-MCP alias: name the server's own"
        )
    return value


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


def _take_via(raw: str) -> str:
    value = raw.strip().lower()
    if value not in _VIAS:
        raise ValueError(raw)
    return value


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
    default: object  # Listed's goes unused: its walk is given Via's preset
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
            _Input(
                "Via",
                "via",
                "via",
                _take_via,
                "it is mcp, extension or local",
                "mcp",
                blank=False,
                plus=False,
            ),
        )
    }
)


@dataclass(frozen=True)
class Resolved:
    """What one request resolved its inputs to (DLV-11).

    ``resources`` is ``"on"`` or ``"off"``, ``listed`` the kinds ``resources/list``
    shows, ``via`` ``"mcp"``, ``"extension"`` or ``"local"``, and ``prefix`` the
    gateway prefix or None. ``extensions`` is the extension ids the client declares,
    or None when the request does not say.
    """

    resources: str
    listed: frozenset[str]
    via: str
    prefix: str | None
    extensions: frozenset[str] | None = None


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


def _nothing(values: tuple[Values, Values] | None) -> Found | None:
    """A level that gives nothing: a config's, when the config gives no value."""
    return None


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
            if header_prefix is not None:
                levels.insert(0, _header(f"{header_prefix}-{entry.name}"))
            # Always a level, so Via's and Listed's levels line up by index (DLV-9).
            if entry.attr is not None:
                levels.append(_config_level(entry, delivery) or _nothing)
            self.levels[entry.name] = levels

    def resolve(
        self, values: tuple[Values, Values] | None, *, warn: bool = True
    ) -> Resolved:
        """What ``values`` resolve each input to; ``warn`` False walks silently.

        ``Via`` is walked first. ``Listed`` is walked up to and including the level
        that answered it, every level when the built-in did, and ``Via``'s preset
        answers past that: an ``X-MCP-Listed`` from a later level does not (DLV-9).
        """

        def walk(name: str) -> Any:
            entry = _INPUTS[name]
            return self._walk(entry, values, warn, entry.default)[0]

        via, at = self._walk(_INPUTS["Via"], values, warn, _INPUTS["Via"].default)
        upto = None if at == len(self.levels["Via"]) else at + 1
        listed = self._walk(_INPUTS["Listed"], values, warn, _PRESET[via], upto)[0]
        return Resolved(
            resources=walk("Resources"),
            listed=listed,
            via=via,
            prefix=walk("Resource-Prefix"),
        )

    def _walk(
        self,
        entry: _Input,
        values: tuple[Values, Values] | None,
        warn: bool,
        default: Any,
        upto: int | None = None,
    ) -> tuple[Any, int]:
        """What ``entry`` resolves to over its first ``upto`` levels (all when None),
        ``default`` answering when none does, and the index of the level that answered,
        or the number walked when ``default`` did."""
        levels = self.levels[entry.name][:upto]
        for index, level in enumerate(levels):
            found = level(values)
            if found is None:
                continue
            raw, source = found
            if not raw.strip() and not entry.blank:
                continue
            try:
                return entry.take(raw), index
            except ValueError:
                shown = raw.strip()[:64]
                if warn and _warn_once(source, shown):
                    _warning(source, shown, entry)
        return default, len(levels)


_RESOLVED: ContextVar[tuple[_Resolver, Resolved] | None] = ContextVar(
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


def _resolved() -> Resolved:
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
    if resolver is _BARE:
        # A custom route has a request, no context: FastMCP keeps its server on the app.
        with suppress(RuntimeError):  # no request
            served = getattr(get_http_request().app.state, "fastmcp_server", None)
            if served is not None:
                resolver = _RESOLVERS.get(served, _BARE)
    return resolver.resolve(request_values())
