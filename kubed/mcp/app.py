"""``QuickMCP``: a FastMCP server, built by a factory, with a container beside it.

It wraps FastMCP and never subclasses it: ``mcp.server`` is the FastMCP instance, public
for what this class does not cover.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol, TypeVar

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware

from .container import (
    _CALL,
    SERVING,
    SESSION,
    SINGLETON,
    TRANSIENT,
    Container,
    Recipe,
    _Call,
)
from .plugins import Plugin, PluginLoader, Plugins

T = TypeVar("T")

# ``singleton``'s ``instance`` when none is given: ``None`` may be one.
_UNSET: Any = object()


class Serving(Protocol):
    """The three settings ``QuickMCP.run()`` reads: none of the rest of a config."""

    @property
    def transport(self) -> str: ...
    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...


class _CallScope(Middleware):
    """Opens the per-call slot transients are kept in, and exits what they entered."""

    def __init__(self, container: Container) -> None:
        self.container = container

    async def on_request(self, context: Any, call_next: Any) -> Any:
        call = _Call(self.container)
        token = _CALL.set(call)
        try:
            async with call.stack:
                return await call_next(context)
        finally:
            _CALL.reset(token)


def _with_teardown(
    container: Container,
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[Any]] | None,
) -> Callable[[FastMCP], AbstractAsyncContextManager[Any]]:
    """The server's lifespan, with the container torn down after it ends."""

    @asynccontextmanager
    async def run(server: FastMCP) -> AsyncIterator[Any]:
        try:
            if lifespan is None:
                yield {}
            else:
                async with lifespan(server) as state:
                    yield state
        finally:
            await container.aclose()

    return run


def quiet_fastmcp_logging() -> None:
    """FastMCP's own handlers go, and its records reach the server's handlers instead.

    FastMCP puts a Rich handler on its logger at import and stops it propagating, so
    whatever logging a server configures never sees it.
    """
    logger = logging.getLogger("fastmcp")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True


class QuickMCP:
    """A FastMCP server and a container, built by a factory and run by ``run()``."""

    def __init__(
        self, name: str, *, config: object | None = None, **fastmcp_options: Any
    ):
        self._container = Container()
        self._config = config
        lifespan = fastmcp_options.pop("lifespan", None)
        self.server = FastMCP(
            name, lifespan=_with_teardown(self._container, lifespan), **fastmcp_options
        )
        # Outermost, so every middleware and handler inside it shares one call's slot.
        self.server.middleware.insert(0, _CallScope(self._container))
        SERVING[self.server] = self._container
        if config is not None:
            self._container.values[type(config)] = config
        # FastMCP's own, unchanged.
        self.tool = self.server.tool
        self.resource = self.server.resource
        self.prompt = self.server.prompt
        self.add_middleware = self.server.add_middleware
        self.add_provider = self.server.add_provider
        self.add_extension = self.server.add_extension
        self.add_transform = self.server.add_transform
        self.custom_route = self.server.custom_route
        self._plugins = Plugins(self.server)

    def add_plugin(
        self, address: str | os.PathLike[str], *, namespace: str | None = None
    ) -> Plugin:
        """Serves the plugin at ``address``: its skills, its commands and its agents.

        A bare path or a ``file://`` URL is read in place; any other scheme needs a
        loader registered with ``add_loader``. ``namespace`` None is the server's name,
        so a skill at ``skills/<name>/`` is ``skill://<server>/<name>/SKILL.md``; given
        ``"x"``, it is ``skill://x/<name>/SKILL.md``, and given ``""``, the root,
        ``skill://<name>/SKILL.md``. Plugins under one namespace merge. A plugin that
        does not load, or that would serve something already served, is refused with a
        ``PluginError`` naming the reason, and nothing of it is served.
        """
        return self._plugins.add(address, namespace)

    def add_loader(self, loader: PluginLoader) -> None:
        """Registers ``loader`` for the address schemes it names.

        ``LocalLoader`` is always registered, for ``file``. A scheme that already has a
        loader is refused with a ``ValueError``.
        """
        self._plugins.add_loader(loader)

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        """The plugins this server serves, in the order they were added."""
        return tuple(self._plugins.served)

    def singleton(
        self,
        key: type[T],
        factory: Callable[..., Any] | None = None,
        *,
        instance: T = _UNSET,
    ) -> None:
        """Built once, on first use, and kept until the server shuts down.

        Given an ``instance``, built already, that is what ``key`` resolves to: the
        container never builds it, enters it or tears it down, as its caller owns it.
        """
        if instance is _UNSET:
            self._container.register(key, Recipe(factory or key, SINGLETON))
        elif factory is not None:
            raise TypeError("singleton() takes a factory or an instance, not both")
        else:
            self._container.provide(key, instance)

    def session(
        self,
        key: type[T],
        factory: Callable[..., Any] | None = None,
        *,
        by: type,
        ttl: float,
    ) -> None:
        """Built once per value of ``by``, kept until idle ``ttl`` seconds or ended."""
        self._container.register(key, Recipe(factory or key, SESSION, by=by, ttl=ttl))

    def transient(
        self, key: type[T], factory: Callable[..., Any] | None = None
    ) -> None:
        """Built once per call, and shared by everything in the call that asks."""
        self._container.register(key, Recipe(factory or key, TRANSIENT))

    def override(self, key: type[T], value: T) -> None:
        """What ``key`` resolves to from now on, whatever its recipe: for tests."""
        self._container.overrides[key] = value

    async def end_session(self, value: Any) -> None:
        """Ends every session built for ``value``, in reverse creation order."""
        await self._container.end_session(value)

    def run(self, serving: Serving | None = None, **http_options: Any) -> None:
        """Serves until the process stops, as ``serving`` says, or the config when None.

        ``http_options`` go to FastMCP's HTTP transport (``path``, ``middleware``,
        ``uvicorn_config``) and are not used on stdio. The log level is never
        passed on: given one, FastMCP's run puts its own handlers back on its logger
        and stops it propagating, for the life of the process.
        """
        settings: Any = serving if serving is not None else self._config
        if settings is None:
            raise TypeError("run() needs a Serving: pass one, or QuickMCP(config=...)")
        quiet_fastmcp_logging()
        if settings.transport == "stdio":
            self.server.run(transport="stdio", show_banner=False)
            return
        self.server.run(
            transport=settings.transport,
            host=settings.host,
            port=settings.port,
            show_banner=False,
            **http_options,
        )
