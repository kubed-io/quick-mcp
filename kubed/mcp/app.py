"""``QuickMCP``: a FastMCP server, built by a factory, with a container beside it.

It wraps FastMCP and never subclasses it: ``mcp.server`` is the FastMCP instance, public
for what this class does not cover.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider
from fastmcp.server.middleware import Middleware
from pydantic import BaseModel
from starlette.routing import WebSocketRoute

from ._builtins import DeliverySettings, ServingSettings, _builtin_components
from ._config import Config, LoadedConfig, Source, load_config
from ._handshake import _Describe
from .components import Component, _start_all, _stop_all, _what
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

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

# ``singleton``'s ``instance`` when none is given: ``None`` may be one.
_UNSET: Any = object()


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


_BOOT = logging.getLogger("kubed.mcp.config")

_AUTH_FIXED = (
    "auth is fixed once serving starts: set mcp.auth before run() or http_app()"
)


def _with_teardown(
    container: Container,
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[Any]] | None,
    *,
    parts: Sequence[tuple[Component, BaseModel | None]],
    boot_line: str,
    on_enter: Callable[[], None],
) -> Callable[[FastMCP], AbstractAsyncContextManager[Any]]:
    """The server's lifespan, with the components started before it and stopped after.

    On every entry: ``on_enter`` runs, the boot line is logged, each component starts
    in order, the server's ``lifespan`` runs, the started components stop in reverse,
    and the container is torn down last. A start that raises stops those already
    started, tears the container down and fails the boot with its error.
    """

    @asynccontextmanager
    async def run(server: FastMCP) -> AsyncIterator[Any]:
        try:
            on_enter()
            _BOOT.info("%s", boot_line)
            started = await _start_all(parts)
            try:
                if lifespan is None:
                    yield {}
                else:
                    async with lifespan(server) as state:
                        yield state
            finally:
                await _stop_all(started)
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


def _refuse_serving_options(
    serving: ServingSettings, http_options: Mapping[str, Any]
) -> None:
    """Refuse a serving setting passed to ``run()``: it is the config's to set.

    Given one, FastMCP would serve stdio with it unread, or take it twice. The error
    names the first such key and where it is set: the field's own env var and flag,
    read from the loaded class, so a subclass's names are the ones shown.
    """
    for key in http_options:
        if key == "show_banner":
            raise TypeError("run() takes no 'show_banner': no banner is ever printed")
        if key not in ("transport", "host", "port", "log_level"):
            continue
        extra = type(serving).model_fields[key].json_schema_extra
        names = [
            extra[k] for k in ("env", "arg") if isinstance(extra, dict) and k in extra
        ]
        raise TypeError(
            f"run() takes no {key!r}: set serving.{key} in the config "
            f"({', '.join([*names, 'a serving: file key'])}) or through "
            "hooks={'serving': ...}"
        )


class QuickMCP:
    """A FastMCP server and a container, built by a factory and run by ``run()``.

    ``header_prefix`` names the server's own headers: with ``"Kb"``, every ``X-MCP-*``
    input is also read as ``Kb-*``, which wins (DLV-16). An invalid ``header_prefix``
    raises ``ValueError``.

    ``sections`` are the config classes the server takes, each stamped with ``@config``,
    beside the built-in ``serving``, ``telemetry`` and ``delivery`` sections every
    server has: a subclass under one of their keys replaces it. ``components`` are
    ``Component`` instances: each one's ``section`` is composed with them, each
    registers at the end of construction, after the built-in components, and each
    starts when serving starts and stops when it ends (``Component``). ``config``
    gives their values in code: a mapping, a section instance, or an iterable of them.
    The config loads first thing in ``__init__`` (unless ``loaded`` gives it already
    loaded), so a bad one stops construction, and ``mcp.config`` is the loaded
    instance. Each section class resolves by ``Inject[...]``, and ``Inject[Config]`` is
    the whole config.

    ``argv`` is the server's flags, ``sys.argv[1:]`` from its entry point: None reads
    none, as quick-mcp never reads ``sys.argv``. ``environ`` is the env vars read, and
    ``os.environ`` when None. Neither goes with ``config``, which reads no file, env var
    or flag: passing either with it raises ``TypeError``. ``mcp.config_sources`` says
    where each item's value came from.

    ``hooks`` supplies a built-in's section from the server's own config, in place of
    that section's file entries, env vars and flags, none of which is then read. It maps
    ``"serving"``, ``"telemetry"`` or ``"delivery"`` to a callable. Once the config
    validates, with the hooked sections at their defaults (a hooked section's every
    field needs a default, since its own sources aren't read), each hook is called
    with it, in that order, and returns an instance of the class holding its key, the
    server's subclass when one replaces the built-in. That instance is validated, the
    config validated again with it in place, and its items' source is ``hook``. Any
    other key raises ``ValueError``, and a value that isn't callable, a ``hooks`` that
    isn't a mapping or a hook returning another type raises ``TypeError``. A hook that
    raises propagates as it is.

    ``loaded`` is a config ``load_config()`` loaded, and the server is built from it as
    it is: no source is read again, ``mcp.config_sources`` is its record, and its
    components register and start as ``components=`` do. Passing it with ``sections``,
    ``components``, ``hooks``, ``config``, ``argv`` or ``environ`` raises ``TypeError``,
    as does a ``loaded`` that isn't a ``LoadedConfig``. A ``LoadedConfig`` builds one
    server, so a second ``QuickMCP`` given the same one raises ``TypeError``.
    """

    def __init__(
        self,
        name: str,
        *,
        loaded: LoadedConfig | None = None,
        sections: Iterable[type[BaseModel]] | None = None,
        components: Iterable[Component] | None = None,
        hooks: Mapping[str, Callable[[Config], BaseModel]] | None = None,
        config: Mapping[str, Any] | BaseModel | Iterable[BaseModel] | None = None,
        argv: Sequence[str] | None = None,
        environ: Mapping[str, str] | None = None,
        header_prefix: str | None = None,
        **fastmcp_options: Any,
    ):
        if loaded is not None:
            if not isinstance(loaded, LoadedConfig):
                raise TypeError(
                    f"loaded= takes what load_config() returns, got {_what(loaded)}"
                )
            given = [
                f"{key}="
                for key, value in (
                    ("sections", sections),
                    ("components", components),
                    ("hooks", hooks),
                    ("config", config),
                    ("argv", argv),
                    ("environ", environ),
                )
                if value is not None
            ]
            if given:
                raise TypeError(
                    "loaded= is already loaded: pass "
                    f"{', '.join(given)} to load_config(), not with loaded="
                )
        else:
            loaded = load_config(
                sections=() if sections is None else sections,
                components=() if components is None else components,
                hooks=hooks,
                config=config,
                argv=argv,
                environ=environ,
            )
        checked = loaded._components
        loaded_config = loaded._loaded
        header_prefix = _header_prefix(header_prefix)
        self._serving = False
        self._container = Container()
        self._config = loaded_config.config
        self._config_sources = loaded_config.sources
        self._provided = loaded_config.provided
        parts = [
            (
                c,
                None
                if c.section is None
                else cast(BaseModel, loaded_config.provided[c.section]),
            )
            for c in (*_builtin_components(name), *checked)
        ]
        lifespan = fastmcp_options.pop("lifespan", None)
        self.server = FastMCP(
            name,
            lifespan=_with_teardown(
                self._container,
                lifespan,
                parts=parts,
                boot_line=loaded_config.boot_line,
                on_enter=self._serving_starts,
            ),
            **fastmcp_options,
        )
        # _CallScope is outermost but for the call metrics, so every other middleware
        # and handler shares one call's slot (APP-6). The telemetry component puts the
        # metrics in at registration, below.
        self.server.middleware.insert(0, _CallScope(self._container))
        SERVING[self.server] = self._container
        for key, value in loaded_config.provided.items():
            self._container.provide(key, value)
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
        self._resolver = _Resolver(
            header_prefix, cast(DeliverySettings, self._provided[DeliverySettings])
        )
        _RESOLVERS[self.server] = self._resolver
        self.server.middleware.insert(1, _ResolveScope(self._resolver))
        # Inside the resolution, outside everything that reads plugins.
        self.server.middleware.insert(2, _PinScope(self._plugins))
        # Inside the pin, so the prefix it takes off is the one the pinned build
        # writes; outside NameTheIndex and every provider, so none sees a prefix.
        self.server.middleware.insert(3, _TakeThePrefix())
        # Inside the resolution, so the handshake echoes and is written for this
        # request's; appended, as no other middleware needs to see it first.
        self._instructions: Instructions | None = None
        self.add_middleware(_Describe(lambda: self._instructions))
        # Claimed only once every argument has passed: from here on a component may
        # have registered, so the load stays claimed.
        loaded._claim()
        # Last, so a component's middleware goes in after the scopes above are placed
        # by their index: the built-ins first, then the server's, in order.
        for component, section in parts:
            component.register(self, section)
        # FastMCP reads auth when it builds the HTTP app, and its own HTTP run builds
        # it through this method: from then on auth is fixed. The instance's method is
        # wrapped, so QuickMCP still never subclasses FastMCP (APP-2).
        build = self.server.http_app

        @functools.wraps(build)
        def http_app(*args: Any, **kwargs: Any) -> Any:
            self._serving_starts()
            return build(*args, **kwargs)

        self.server.http_app = http_app  # type: ignore[method-assign]

    def _serving_starts(self) -> None:
        """Serving has started: ``run()`` was called, the HTTP app built
        (``mcp.server.http_app()``), or the lifespan entered."""
        self._serving = True

    @property
    def auth(self) -> AuthProvider | None:
        """The server's auth provider, ``mcp.server.auth``: None serves without auth.

        Set it before serving starts, from the server's build code or a component's
        ``register``: FastMCP reads it when it builds the HTTP app. Once ``run()`` is
        called, the HTTP app built (``mcp.server.http_app()``) or the lifespan
        entered, setting it raises ``RuntimeError``.
        """
        return self.server.auth

    @auth.setter
    def auth(self, provider: AuthProvider | None) -> None:
        if self._serving:
            raise RuntimeError(_AUTH_FIXED)
        self.server.auth = provider

    @property
    def config(self) -> Config:
        """The config loaded at construction: read-only, and never reloaded."""
        return self._config

    @property
    def config_sources(self) -> Mapping[str, Source]:
        """Where each leaf item's value came from, by its dotted path, in item order:
        read-only, as ``config`` is."""
        return self._config_sources

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

    def run(self, **http_options: Any) -> None:
        """Serves until the process stops, as the config's ``serving`` section says.

        Its ``transport``, ``host`` and ``port`` come from the loaded config, so from
        ``config=``, or the file, env vars and flags the server was built with. Its
        ``log_level`` is set on the root logger, which gets ``logging.basicConfig()``'s
        handler first only when it has none.

        ``http_options`` go to FastMCP's HTTP transport (``path``, ``middleware``,
        ``uvicorn_config``) and are not used on stdio. The log level is never
        passed on: given one, FastMCP's run puts its own handlers back on its logger
        and stops it propagating, for the life of the process.

        Serving starts here, so ``mcp.auth`` is fixed from this call on.

        Raises:
            TypeError: ``http_options`` holds a serving setting, ``transport``,
                ``host``, ``port``, ``log_level`` or ``show_banner``: those are the
                config's, and nothing runs.
        """
        serving = cast(ServingSettings, self._provided[ServingSettings])
        _refuse_serving_options(serving, http_options)
        self._serving_starts()
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig()
        root.setLevel(serving.log_level)
        quiet_fastmcp_logging()
        if serving.transport == "stdio":
            self.server.run(transport="stdio", show_banner=False)
            return
        self.server.run(
            transport=serving.transport,
            host=serving.host,
            port=serving.port,
            show_banner=False,
            **http_options,
        )
