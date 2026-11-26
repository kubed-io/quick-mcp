"""``QuickMCP``: a FastMCP server, built by a factory, with a container beside it.

It wraps FastMCP and never subclasses it: ``mcp.server`` is the FastMCP instance, public
for what this class does not cover.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from starlette.routing import WebSocketRoute

from ._handshake import _Describe
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
from .delivery import (
    _RESOLVERS,
    BinaryPointer,
    DeliveryDefault,
    Instructions,
    _header_prefix,
    _Resolver,
    _ResolveScope,
)
from .gateway import _TakeThePrefix
from .plugins import (
    AgentMeta,
    CommandReader,
    Plugin,
    PluginLoader,
    Plugins,
    PluginSet,
    Site,
    View,
    _PinScope,
)
from .telemetry import Telemetry, _call_metrics, _start_signals

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

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


def _signed_as(
    source: Callable[Concatenate[Any, P], R],
) -> Callable[[Callable[..., Any]], Callable[Concatenate[Any, P], R]]:
    """Gives a method that passes everything through to ``source`` its signature: for
    type checkers by its type, for ``inspect`` and ``help()`` by ``__wrapped__``, and
    its own docstring followed by ``source``'s."""

    def sign(fn: Callable[..., Any]) -> Callable[Concatenate[Any, P], R]:
        fn.__wrapped__ = source  # type: ignore[attr-defined]
        fn.__doc__ = "\n\n".join(
            inspect.cleandoc(doc) for doc in (fn.__doc__, source.__doc__) if doc
        )
        return fn

    return sign


def _with_teardown(
    container: Container,
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[Any]] | None,
    *,
    name: str,
    telemetry: Telemetry | None,
) -> Callable[[FastMCP], AbstractAsyncContextManager[Any]]:
    """The server's lifespan, with telemetry started before it and stopped after it.

    The container is torn down last, after telemetry has stopped.
    """

    @asynccontextmanager
    async def run(server: FastMCP) -> AsyncIterator[Any]:
        signals = _start_signals(name, telemetry)
        try:
            if lifespan is None:
                yield {}
            else:
                async with lifespan(server) as state:
                    yield state
        finally:
            if signals is not None:
                signals.stop()
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
    """A FastMCP server and a container, built by a factory and run by ``run()``.

    ``header_prefix`` names the server's own headers: with ``"Kb"``, every ``X-MCP-*``
    input is also read as ``Kb-*``, which wins (DLV-16). An invalid ``header_prefix``
    raises ``ValueError``.
    """

    def __init__(
        self,
        name: str,
        *,
        config: object | None = None,
        header_prefix: str | None = None,
        **fastmcp_options: Any,
    ):
        header_prefix = _header_prefix(header_prefix)
        self._container = Container()
        self._config = config
        telemetry = getattr(config, "telemetry", None)
        lifespan = fastmcp_options.pop("lifespan", None)
        self.server = FastMCP(
            name,
            lifespan=_with_teardown(
                self._container,
                lifespan,
                name=name,
                telemetry=telemetry,
            ),
            **fastmcp_options,
        )
        # _CallScope is outermost but for the call metrics, so every other middleware
        # and handler shares one call's slot (APP-6). The metrics go in last, below.
        self.server.middleware.insert(0, _CallScope(self._container))
        SERVING[self.server] = self._container
        if config is not None:
            self._container.values[type(config)] = config
        # FastMCP's own, unchanged.
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
        # The inputs resolve once per request, inside _CallScope and outside the pin, so
        # the build key's prefix is the one this request resolved.
        self._resolver = _Resolver(header_prefix, getattr(config, "delivery", None))
        _RESOLVERS[self.server] = self._resolver
        self.server.middleware.insert(1, _ResolveScope(self._resolver))
        # Inside the resolution, outside everything that reads plugins.
        self.server.middleware.insert(2, _PinScope(self._plugins))
        # Inside the pin, so the prefix it takes off is the one the pinned build
        # writes; outside NameTheIndex and every provider, so none sees a prefix.
        self.server.middleware.insert(3, _TakeThePrefix())
        # The call metrics go first, after the scopes above are placed by their index
        # from _CallScope, to record what the client received, the slot's teardown
        # included (TEL-16).
        if (metrics := _call_metrics(telemetry)) is not None:
            self.server.middleware.insert(0, metrics)
        # Inside the resolution, so the handshake echoes and is written for this
        # request's; appended, as no other middleware needs to see it first.
        self._instructions: Instructions | None = None
        self.add_middleware(_Describe(lambda: self._instructions))

    @_signed_as(FastMCP.resource)
    def resource(self, uri: Any, *args: Any, **kwargs: Any) -> Any:
        """FastMCP's ``resource``, with its signature, given what it is given (APP-4),
        except that once the server has a plugin a URI or template under ``prompts://``
        or ``agents://`` is refused with a ``ValueError``: those are the plugins'
        (PLUG-72). FastMCP's own documentation follows."""
        self._plugins.check_own(uri)
        return self.server.resource(uri, *args, **kwargs)

    @_signed_as(FastMCP.tool)
    def tool(self, name_or_fn: Any = None, *args: Any, **kwargs: Any) -> Any:
        """FastMCP's ``tool``, with its signature, given what it is given (APP-4),
        except that once the server has a plugin a tool named ``list_resources`` or
        ``read_resource`` is refused with a ``ValueError``: the plugin layer serves
        both itself (DLV-7). FastMCP's own documentation follows."""
        name = kwargs.get("name")
        if name is None:
            if isinstance(name_or_fn, str):
                name = name_or_fn
            elif inspect.isroutine(name_or_fn):
                name = name_or_fn.__name__
        if name is not None or name_or_fn is not None:
            self._plugins.check_own_tool(name)
            return self.server.tool(name_or_fn, *args, **kwargs)
        # @mcp.tool(), named by the function it is given.
        decorator = self.server.tool(name_or_fn, *args, **kwargs)

        def register(fn: Any) -> Any:
            self._plugins.check_own_tool(getattr(fn, "__name__", None))
            return decorator(fn)

        return register

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
        server is refused with a ``ValueError``, and a server's first plugins, when it
        has its own resource under ``prompts://`` or ``agents://``, with a
        ``PluginError`` (PLUG-72).
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
        it admits. A set ``fn()`` serves with ``serve()`` is the one that request
        reads. A ``fn()`` that raises fails the request with its error.
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

    def command_reader(self, fn: CommandReader) -> CommandReader:
        """Reads the commands of every plugin read from now on with ``fn``, and returns
        ``fn``, so it works as a decorator (PLUG-62).

        ``fn(text, path, name, plugin)`` is given the command file's text as served,
        its ``path`` from the plugin root, ``/``-separated, the ``name`` it is served
        under and the ``Plugin`` it is in, and returns a FastMCP ``Prompt`` to serve, or
        a string: the reason the command is skipped. It runs once when the plugin is
        read, where a string, a raise or any other result skips the command with the
        warning ``prompt <path> is skipped: <reason>``; and again in each build whose
        text for the command differs from the text read (a rewriter's, PLUG-56), where
        a skip fails the prompt for that build alone. With no rewriter, or text a build
        leaves unchanged, the first ``Prompt`` is served.

        A plugin is read and built by the reader set when it was read, and no build
        reads the disk. A ``Prompt`` named otherwise is served under the layer's name,
        and the name, namespace, clash and gateway prefix rules apply to it as to any
        command. The default reader is ``kubed.mcp.prompts.read_command``.
        """
        self._plugins.set_command_reader(fn)
        return fn

    def agent_meta(self, fn: AgentMeta) -> AgentMeta:
        """Adds ``fn``'s keys to each agent's ``agents/get`` ``_meta``, beside the
        layer's ``io.kubed/agent``, and returns ``fn``, so it works as a decorator
        (PLUG-73). A second call replaces the first.

        ``fn(name, plugin)`` runs once per agent when its plugin enters a set, after
        the agent's name is decided (PLUG-64), with that name and the ``Plugin`` as it
        enters, and never on a read; so it runs again on every set entry, each
        ``add_plugin`` included, and must be pure. It returns a dict, or None for
        nothing to add, and its JSON round trip is what is served. A returned
        ``io.kubed/agent`` key is dropped with a warning, as is a result that is not a
        dict or does not survive a trip through JSON. An ``fn`` that raises leaves
        that agent unserved in that set, logged at error and in the plugin's
        ``skipped`` as ``agent_meta failed: <reason>``. The ``agents/list`` card is
        unchanged.
        """
        self._plugins.set_agent_meta(fn)
        return fn

    def binary_pointer(self, fn: BinaryPointer) -> BinaryPointer:
        """Sets where ``read_resource`` says a binary's link is listed, and returns
        ``fn``, so it works as a decorator (DLV-6). A second call replaces the first.

        ``fn(uri)``, called with ``uri`` as the client sent it, returns where the
        blob's link is listed, or None. Its string follows, after a space, the
        sentence ``Not returned: binary content is not text.`` in the description
        ``read_resource`` gives of a blob that is neither an image nor text. An ``fn``
        that raises, or returns anything but a string, is logged at error and gives
        none.
        """
        self._plugins.set_binary_pointer(fn)
        return fn

    def delivery_default(self, fn: DeliveryDefault) -> DeliveryDefault:
        """Sets a per-client default for the delivery inputs, and returns ``fn``, so it
        works as a decorator (DLV-14). A second call replaces the first.

        ``fn(client_info)`` is called once each time a request's inputs are resolved,
        with the client's ``clientInfo``, an ``mcp_types.Implementation``, or None when
        the request names none: on ``initialize`` its ``params.clientInfo``, on any
        other request its ``_meta["io.modelcontextprotocol/clientInfo"]`` first, then
        the session's ``initialize``. A lookup that fails, or a ``clientInfo`` that is
        not one, counts as none. ``fn`` returns a partial ``Delivery``, None for any
        input it leaves alone, or None for all of them. Its answer beats the config's
        ``delivery`` and kubed-mcp's table of known clients, and a value its input does
        not take, or of a type ``Delivery`` does not allow it, falls through with one
        warning naming ``delivery_default <attribute>``. An ``fn`` that raises, or
        answers with anything but a partial ``Delivery`` or None, is logged at error
        and gives nothing.
        """
        self._resolver.default = fn
        return fn

    def instructions(self, fn: Instructions) -> Instructions:
        """Sets the server's instructions, and returns ``fn``, so it works as a
        decorator (DLV-11). A second call replaces the first.

        ``fn(delivery)`` is called once per ``initialize`` and once per
        ``server/discover``, with a ``kubed.mcp.delivery.Resolved``: what the request
        resolved, its ``resources``, ``listed``, ``via``, gateway ``prefix`` and the
        ``extensions`` the client declares. Its string is the result's
        ``instructions``, written with the prefix as served text is. None, a raise or
        anything but a string gives the ``instructions`` the server was built with, and
        the last two are logged at error.
        """
        self._instructions = fn
        return fn

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
