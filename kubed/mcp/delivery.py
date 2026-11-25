"""What a request asks for besides its plugins, resolved once per request (DLV-2 to 5).

Each input is a header, a query parameter, or a default: the server's for the client
(``delivery_default``), the config's, or kubed-mcp's table of known clients. It is
resolved on its own by walking its levels in order, and the first value the input takes
answers. A value it does not take is ignored with one warning, and the next level
answers. With none, the built-in default does. ``Via`` is walked first, and ``Listed``
only as far as the level that answered it, its preset answering past that (DLV-9).
``_ResolveScope`` makes the walk once per request, and ``_resolved()`` is what the rest
of the package reads.
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
from mcp_types import Implementation

from kubed.mcp.request import (
    Values,
    _warn_once,
    get_context,
    get_http_request,
    request_values,
)

log = logging.getLogger(__name__)

__all__ = ["BinaryPointer", "Delivery", "DeliveryDefault", "Resolved"]

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

# Where a request's clientInfo sits in its _meta, on the 2026-07-28 protocol.
_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"

Found = tuple[str, str]  # (raw value, source)


@dataclass(frozen=True)
class _Asked:
    """What a level reads: the request's (query, headers), its client, and what
    ``delivery_default`` answered for it, asked once per resolution."""

    values: tuple[Values, Values] | None
    client: Implementation | None = None
    answer: object | None = None


Level = Callable[[_Asked], Found | None]

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


# A server's defaults for a client, given its clientInfo or None (DLV-14).
DeliveryDefault = Callable[[Implementation | None], Delivery | None]


@dataclass(frozen=True)
class _Partial:
    """A partial ``Delivery``: None leaves an input to the next level."""

    resources: str | None = None
    listed: str | None = None
    via: str | None = None


# kubed-mcp's known clients, by a prefix of clientInfo.name, the longest winning
# (DLV-15).
_KNOWN_CLIENTS: Mapping[str, _Partial] = MappingProxyType(
    {"Visual Studio Code": _Partial(resources="off")}
)


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
# The inputs a Delivery gives a default for, by attribute.
_ATTRS: Mapping[str, _Input] = MappingProxyType(
    {entry.attr: entry for entry in _INPUTS.values() if entry.attr is not None}
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

    def level(asked: _Asked) -> Found | None:
        values = asked.values
        if values is None or not values[1].get(key):
            return None
        return values[1][key][0], f"header {name}"

    return level


def _query(entry: _Input) -> Level:
    """The query parameter of ``entry``, first value."""

    def level(asked: _Asked) -> Found | None:
        values = asked.values
        if values is None or not values[0].get(entry.query):
            return None
        raw = values[0][entry.query][0]
        if entry.plus:
            raw = raw.replace(" ", "+")
        return raw, f"query parameter {entry.query}"

    return level


def _raw(entry: _Input, value: object) -> str | None:
    """``value`` as a raw value of ``entry``: a string, or for ``Listed`` a collection
    joined with ``,``. None when it is neither."""
    if isinstance(value, str):
        return value
    if entry.name == "Listed" and isinstance(value, Collection):
        return ",".join(str(kind) for kind in value)
    return None


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
    raw = _raw(entry, value)
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
    return lambda asked: found


def _nothing(asked: _Asked) -> Found | None:
    """A level that gives nothing: a config's, when the config gives no value."""
    return None


def _default_level(entry: _Input) -> Level:
    """What the server's ``delivery_default`` answered for ``entry`` (DLV-14)."""
    source = f"delivery_default {entry.attr}"

    def level(asked: _Asked) -> Found | None:
        # _answer has coerced it already: a raw value or None.
        raw = getattr(asked.answer, entry.attr or "", None)
        return None if raw is None else (raw, source)

    return level


def _known_level(entry: _Input) -> Level:
    """What kubed-mcp's table of known clients gives ``entry`` (DLV-15)."""

    def level(asked: _Asked) -> Found | None:
        name = getattr(asked.client, "name", None)
        if not isinstance(name, str):
            return None
        keys = [key for key in _KNOWN_CLIENTS if name.startswith(key)]
        if not keys:
            return None
        key = max(keys, key=len)
        raw = getattr(_KNOWN_CLIENTS[key], entry.attr or "", None)
        return None if raw is None else (raw, f"known client {key}")

    return level


def _valid_client(value: object) -> Implementation | None:
    """``value`` as a clientInfo, None for None; one that does not validate raises."""
    if value is None or isinstance(value, Implementation):
        return value
    return Implementation.model_validate(value)


def _session_client_info() -> Implementation | None:
    """The clientInfo of the current context's session, from its ``initialize``."""
    params = get_context().session.client_params
    return None if params is None else params.client_info


def _meta_client_info(context: Any) -> object:
    """The clientInfo in the request's ``_meta``. FastMCP lifts a built-in method's
    ``_meta`` onto its request context, and hands an extension method its raw params,
    ``_meta`` included."""
    request = getattr(context.fastmcp_context, "request_context", None)
    meta = getattr(request, "meta", None)
    if meta is None and isinstance(context.message, Mapping):
        meta = context.message.get("_meta")
    return meta.get(_CLIENT_INFO) if isinstance(meta, Mapping) else None


def _client_info(context: Any | None) -> Implementation | None:
    """The clientInfo of the request ``context`` is for, or None (DLV-14).

    On ``initialize`` it is its params' ``clientInfo``. On any other request it is its
    ``_meta``'s, then its session's, and off a request (``context`` None) its
    session's. A lookup that fails, or a value that is no clientInfo, counts as none,
    and the next source is tried.
    """
    lookups: list[Callable[[], object]] = []
    if context is not None and context.method == "initialize":
        lookups.append(lambda: context.message.params.client_info)
    elif context is not None:
        lookups += [lambda: _meta_client_info(context), _session_client_info]
    else:
        lookups.append(_session_client_info)
    for lookup in lookups:
        try:
            found = _valid_client(lookup())
        except Exception:  # noqa: BLE001 - a lookup that fails counts as none (DLV-14)
            continue
        if found is not None:
            return found
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
        self.default: DeliveryDefault | None = None
        self.levels: dict[str, list[Level]] = {}
        for entry in _INPUTS.values():
            levels = [_header(f"X-MCP-{entry.name}"), _query(entry)]
            if header_prefix is not None:
                levels.insert(0, _header(f"{header_prefix}-{entry.name}"))
            # Always these three, so Via's and Listed's levels line up by index (DLV-9).
            if entry.attr is not None:
                levels += [
                    _default_level(entry),
                    _config_level(entry, delivery) or _nothing,
                    _known_level(entry),
                ]
            self.levels[entry.name] = levels

    def resolve(
        self,
        values: tuple[Values, Values] | None,
        client: Implementation | None = None,
        extensions: frozenset[str] | None = None,
        *,
        warn: bool = True,
    ) -> Resolved:
        """What ``values``, from ``client``, resolve each input to; ``warn`` False walks
        silently. ``delivery_default`` is asked once, whatever the walks reach.

        ``Via`` is walked first. ``Listed`` is walked up to and including the level
        that answered it, every level when the built-in did, and ``Via``'s preset
        answers past that: an ``X-MCP-Listed`` from a later level does not (DLV-9).
        """

        asked = _Asked(values, client, self._answer(client, warn))

        def walk(name: str) -> Any:
            entry = _INPUTS[name]
            return self._walk(entry, asked, warn, entry.default)[0]

        via, at = self._walk(_INPUTS["Via"], asked, warn, _INPUTS["Via"].default)
        upto = None if at == len(self.levels["Via"]) else at + 1
        listed = self._walk(_INPUTS["Listed"], asked, warn, _PRESET[via], upto)[0]
        return Resolved(
            resources=walk("Resources"),
            listed=listed,
            via=via,
            prefix=walk("Resource-Prefix"),
            extensions=extensions,
        )

    def _answer(self, client: Implementation | None, warn: bool) -> _Partial | None:
        """``delivery_default``'s answer for ``client``, each value a raw one or None.

        An ``fn`` that raises, or answers with no ``Delivery``, is logged at error and
        gives nothing. A value of no type its input takes is warned once, as an untaken
        one is, and gives nothing.
        """
        if self.default is None:
            return None
        try:
            answer = self.default(client)
            if answer is None:
                return None
            if not any(hasattr(answer, attr) for attr in _ATTRS):
                log.error(
                    "delivery_default gave %s, not a Delivery: it gives nothing",
                    type(answer).__name__,
                )
                return None
            given = {attr: getattr(answer, attr, None) for attr in _ATTRS}
        except Exception:
            log.exception("delivery_default failed: it gives nothing")
            return None
        raws: dict[str, str | None] = {}
        for attr, entry in _ATTRS.items():
            value = given[attr]
            raws[attr] = raw = None if value is None else _raw(entry, value)
            if value is not None and raw is None:
                source, shown = f"delivery_default {attr}", str(value)[:64]
                if warn and _warn_once(source, shown):
                    _warning(source, shown, entry)
        return _Partial(**raws)

    def _walk(
        self,
        entry: _Input,
        asked: _Asked,
        warn: bool,
        default: Any,
        upto: int | None = None,
    ) -> tuple[Any, int]:
        """What ``entry`` resolves to over its first ``upto`` levels (all when None),
        ``default`` answering when none does, and the index of the level that answered,
        or the number walked when ``default`` did."""
        levels = self.levels[entry.name][:upto]
        for index, level in enumerate(levels):
            found = level(asked)
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
            request_values(),
            _client_info(context),
            warn=context.method != "initialize",
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
    return resolver.resolve(request_values(), _client_info(None))
