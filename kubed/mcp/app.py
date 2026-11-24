"""``QuickMCP``: a FastMCP server, built by a factory, with a container beside it.

It wraps FastMCP and never subclasses it: ``mcp.server`` is the FastMCP instance, public
for what this class does not cover.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol, TypeVar

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from starlette.routing import WebSocketRoute

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
from .plugins import Plugin, PluginLoader, Plugins, PluginSet, Site, View, _PinScope

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
        # The tool as registered, not as a caller is shown it: see the docstring of
        # QuickMCP.get_tool.
        self.get_tool = self.server.get_tool
        self._plugins = Plugins(self.server)
        # Right inside _CallScope, outside everything that reads plugins.
        self.server.middleware.insert(1, _PinScope(self._plugins))

    def websocket_route(
        self, path: str, endpoint: Callable[..., Any], *, name: str | None = None
    ) -> None:
        """Serves a WebSocket at ``path`` on the HTTP app, beside the custom routes.

        ``endpoint`` is a Starlette WebSocket endpoint: ``async def f(websocket)``.
        ``path`` is used as given, as ``custom_route`` does, so a server that mounts
        under a prefix puts the prefix in it. Nothing is served on stdio.
        """
        # FastMCP's ``custom_route`` makes only HTTP routes and it has no public way
        # to add a WebSocket one, though its app serves whatever Route is in this list.
        # This is the one place quick-mcp touches the private list.
        self.server._additional_http_routes.append(
            WebSocketRoute(path, endpoint, name=name)
        )

    def add_plugin(
        self,
        address: str | os.PathLike[str],
        *,
        namespace: str | None = None,
        manifest: Mapping[str, object] | None = None,
    ) -> Plugin:
        """Serves the plugin at ``address``: its skills, its commands and its agents.

        A bare path or a ``file://`` URL is read in place; any other scheme needs a
        loader registered with ``add_loader``. ``namespace`` None is the server's name,
        so a skill at ``skills/<name>/`` is ``skill://<server>/<name>/SKILL.md``; given
        ``"x"``, it is ``skill://x/<name>/SKILL.md``. An empty namespace is refused with
        a ``ValueError``: there is no root. Plugins under one namespace merge. A plugin
        that does not load, or that would serve something already served, is refused
        with a ``PluginError`` naming the reason, and nothing of it is served.

        ``manifest`` is the plugin's marketplace entry, merged with its ``plugin.json``
        as Claude Code merges them (``read_plugin``).
        """
        return self._plugins.add(address, namespace, manifest)

    def plugin_set(self) -> PluginSet:
        """A new, empty set of plugins, not served until ``serve(set)`` (PLUG-50).

        ``set.add(address, ...)`` reads a plugin as ``add_plugin`` does, and
        ``set.keep(plugin)`` carries one served now, unread.
        """
        return self._plugins.plugin_set()

    def serve(self, staging: PluginSet) -> None:
        """Serves ``staging`` whole, in place of what is served, and moves
        ``generation`` on by one (PLUG-51).

        A plugin left out of the set is no longer served. A set made by another
        server is refused with a ``ValueError``.
        """
        self._plugins.serve(staging)

    @property
    def generation(self) -> int:
        """How many sets this server has served: 0 until a set is first served."""
        return self._plugins.generation

    def plugin_view(self, fn: Callable[[], View]) -> Callable[[], View]:
        """Sets the server's view, and returns ``fn``, so it works as a decorator
        (PLUG-53).

        ``fn()`` runs once per MCP request, inside it, so it can read the request
        (``request_values()``). Its ``View`` says which plugins the request is served:
        one it does not admit is absent for that request. ``admits`` runs only when a
        build is made, once per generation and view key, so the key must decide what
        it admits. A ``fn()`` that raises fails the request with its error.
        """
        self._plugins.set_view(fn)
        return fn

    def add_rewriter(self, fn: Callable[[Site, str], str]) -> None:
        """Adds ``fn`` to the server's rewriters, run in the order added (PLUG-56).

        ``fn(site, text)`` returns the text to serve in place of ``text``: each text
        file a plugin serves, a skill's, a command's or an agent's, and each skill
        folder's ``_index.md`` and ``_index.json``. ``site`` says where the file is and
        who it is served to (``Site``). It runs when a build is made, once per
        generation and view key, never on a read, so a skill's digests match what is
        served. A rewriter that raises, returns anything but a ``str``, or leaves a
        file that no longer reads as what it was under its name fails that item for
        that view alone, logged at error (PLUG-59). It takes effect from the next
        request.
        """
        self._plugins.add_rewriter(fn)

    def add_loader(self, loader: PluginLoader) -> None:
        """Registers ``loader`` for the address schemes it names.

        ``LocalLoader`` is registered for ``file`` until the server adds its own, which
        replaces it. Any other scheme that already has a loader, or a second loader for
        ``file``, is refused with a ``ValueError``, and nothing is registered.
        """
        self._plugins.add_loader(loader)

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        """The plugins of the served set, in the order they were added to it."""
        return self._plugins.served

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
