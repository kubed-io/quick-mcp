"""``QuickMCP``: a FastMCP server, built by a factory, with a container beside it.

It wraps FastMCP and never subclasses it: ``mcp.server`` is the FastMCP instance, public
for what this class does not cover.
"""

from __future__ import annotations

import logging
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

T = TypeVar("T")


class Serving(Protocol):
    """The four settings ``QuickMCP.run()`` reads: none of the rest of a config."""

    @property
    def transport(self) -> str: ...
    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...
    @property
    def log_level(self) -> str: ...


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

    def singleton(
        self, key: type[T], factory: Callable[..., Any] | None = None
    ) -> None:
        """Built once, on first use, and kept until the server shuts down."""
        self._container.register(key, Recipe(factory or key, SINGLETON))

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
        ``uvicorn_config``) and are not used on stdio.
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
            log_level=settings.log_level,
            show_banner=False,
            **http_options,
        )
