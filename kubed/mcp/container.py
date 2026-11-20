"""The container: recipes keyed by type, built on first use, kept for their lifetime.

A recipe is a factory and a lifetime. A factory's parameters annotated with a
registered type are built first and passed in; any other parameter must have a
default. A factory may be sync or async, and when what it returns is an async context
manager, the container enters it and exits it at teardown. Three lifetimes, longest
first:

- **singleton:** once per container, kept until the server shuts down;
- **session:** once per value of its ``by`` key, kept until the value goes unused
  for ``ttl`` seconds or is ended;
- **transient:** once per call, kept until the call ends.

A longer lifetime never depends on a shorter one, so a singleton that asks for a
session or a transient is refused when it is built.
"""

from __future__ import annotations

import inspect
import math
import typing
import weakref
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import anyio

SINGLETON, SESSION, TRANSIENT = "singleton", "session", "transient"
_RANK = {TRANSIENT: 0, SESSION: 1, SINGLETON: 2}


@dataclass(frozen=True)
class Recipe:
    factory: Any
    lifetime: str
    by: Any = None
    ttl: float = math.inf


@dataclass
class _Session:
    """Every session built for one value of one ``by`` key, ended together."""

    stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    built: dict[Any, Any] = field(default_factory=dict)
    locks: dict[Any, anyio.Lock] = field(default_factory=dict)
    ttl: float = math.inf
    used: float = field(default_factory=lambda: monotonic())


@dataclass
class _Call:
    """One call's transients, and the stack their context managers exit on."""

    container: Container
    built: dict[Any, Any] = field(default_factory=dict)
    stack: AsyncExitStack = field(default_factory=AsyncExitStack)


_CALL: ContextVar[_Call | None] = ContextVar("kubed_mcp_call", default=None)

# The container behind each FastMCP server, for a call that runs outside QuickMCP's
# middleware (a background task's worker): found through the request's Context.
SERVING: weakref.WeakKeyDictionary[Any, Container] = weakref.WeakKeyDictionary()


def name_of(key: Any) -> str:
    return getattr(key, "__qualname__", None) or repr(key)


def _key_of(annotation: Any) -> Any:
    """The key a parameter asks for: its type, or the type inside ``Annotated``."""
    if typing.get_origin(annotation) is typing.Annotated:
        return typing.get_args(annotation)[0]
    return annotation


def _parameters(factory: Any) -> list[inspect.Parameter]:
    try:
        return list(inspect.signature(factory, eval_str=True).parameters.values())
    except ValueError:  # a builtin with no signature takes nothing from us
        return []


class Container:
    def __init__(self) -> None:
        self.recipes: dict[Any, Recipe] = {}
        self.values: dict[Any, Any] = {}
        self.overrides: dict[Any, Any] = {}
        self.built: dict[Any, Any] = {}  # the singletons
        self.sessions: dict[tuple[Any, Any], _Session] = {}
        self.stack = AsyncExitStack()
        self.locks: dict[Any, anyio.Lock] = {}

    def register(self, key: Any, recipe: Recipe) -> None:
        self.recipes[key] = recipe

    def knows(self, key: Any) -> bool:
        try:
            return key in self.overrides or key in self.values or key in self.recipes
        except TypeError:  # an unhashable annotation is never a key
            return False

    async def resolve(self, key: Any, path: tuple[Any, ...] = ()) -> Any:
        if key in self.overrides:
            return self.overrides[key]
        if key in self.values:
            return self.values[key]
        if key not in self.recipes:
            raise LookupError(f"{name_of(key)} is not registered with this QuickMCP")
        if key in path:
            cycle = " -> ".join(name_of(k) for k in (*path, key))
            raise TypeError(f"{cycle} is a cycle")
        recipe, path = self.recipes[key], (*path, key)
        if recipe.lifetime == SINGLETON:
            return await self._once(self, key, recipe, path)
        if recipe.lifetime == SESSION:
            return await self._session(key, recipe, path)
        call = _CALL.get()
        if call is None or call.container is not self:  # no call: nothing to share
            return await self._build(key, recipe, path, self.stack)
        if key not in call.built:  # one call resolves its parameters one at a time
            call.built[key] = await self._build(key, recipe, path, call.stack)
        return call.built[key]

    async def _once(self, owner: Container | _Session, key, recipe, path) -> Any:
        """Built once into ``owner``, however many calls ask for it at once."""
        if key not in owner.built:
            async with owner.locks.setdefault(key, anyio.Lock()):
                if key not in owner.built:
                    built = await self._build(key, recipe, path, owner.stack)
                    owner.built[key] = built
        return owner.built[key]

    async def _session(self, key: Any, recipe: Recipe, path: tuple[Any, ...]) -> Any:
        slot = (recipe.by, await self.resolve(recipe.by, path))
        now = monotonic()
        stale = [
            s for s, open_ in self.sessions.items() if now - open_.used > open_.ttl
        ]
        for gone in stale:
            await self.sessions.pop(gone).stack.aclose()
        session = self.sessions.setdefault(slot, _Session())
        session.used, session.ttl = now, min(session.ttl, recipe.ttl)
        return await self._once(session, key, recipe, path)

    async def _build(self, key, recipe: Recipe, path, stack: AsyncExitStack) -> Any:
        kwargs = {}
        for parameter in _parameters(recipe.factory):
            wanted = _key_of(parameter.annotation)
            if self.knows(wanted):
                self._check_order(key, recipe, wanted)
                kwargs[parameter.name] = await self.resolve(wanted, path)
            elif parameter.default is parameter.empty and parameter.kind not in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"{name_of(recipe.factory)} needs {parameter.name!r}, "
                    "which has no registered type and no default"
                )
        built = recipe.factory(**kwargs)
        if inspect.isawaitable(built):
            built = await built
        if isinstance(built, AbstractAsyncContextManager):
            built = await stack.enter_async_context(built)
        return built

    def _check_order(self, key: Any, recipe: Recipe, wanted: Any) -> None:
        other = self.recipes.get(wanted)
        if other is None or wanted in self.overrides or wanted == recipe.by:
            return  # a value, an override or the session's own key: none expire
        if _RANK[other.lifetime] < _RANK[recipe.lifetime]:
            raise TypeError(
                f"{name_of(key)} is a {recipe.lifetime} and cannot depend on "
                f"{name_of(wanted)}, a {other.lifetime}: a longer lifetime never "
                "depends on a shorter one"
            )

    async def end_session(self, value: Any) -> None:
        for slot in [s for s in reversed(self.sessions) if s[1] == value]:
            await self.sessions.pop(slot).stack.aclose()

    async def aclose(self) -> None:
        """Every open session, then every singleton, each in reverse creation order."""
        try:
            for slot in reversed(list(self.sessions)):
                await self.sessions.pop(slot).stack.aclose()
        finally:
            stack, self.stack = self.stack, AsyncExitStack()
            self.built.clear()
            await stack.aclose()
