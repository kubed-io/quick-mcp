"""The built-in sections every server has, and ``run()`` serving from them."""

from __future__ import annotations

import argparse
import ast
import asyncio
import inspect
import logging
import signal
import sys
import types
import typing
from pathlib import Path
from typing import Literal

import fastmcp.server.server
import pytest
import uvicorn
from fastmcp import Client, FastMCP
from pydantic import model_validator
from starlette.testclient import TestClient

import kubed.mcp
import kubed.mcp.app
import kubed.mcp.telemetry
from kubed.mcp import (
    Component,
    Config,
    DeliverySettings,
    Field,
    QuickMCP,
    ServingSettings,
    Source,
    TelemetrySettings,
    config,
    config_schema,
)
from kubed.mcp._config import _compose, _load
from kubed.mcp.auth import TokenVerifier
from kubed.mcp.components import _start_all, _stop_all
from kubed.mcp.container import Container

pytestmark = pytest.mark.unit


@config("serving")
class Pod(ServingSettings):
    transport: Literal["stdio", "http", "sse", "streamable-http"] = Field(
        "http", env="TRANSPORT", arg="--transport"
    )
    host: str = Field("0.0.0.0", env="HOST", arg="--host")


# -- built-ins


def test_every_server_has_the_three_built_ins_with_their_defaults() -> None:
    mcp = QuickMCP("t", environ={})
    assert set(type(mcp.config).model_fields) == {"serving", "telemetry", "delivery"}
    s = mcp.config.serving
    assert (s.transport, s.host, s.port, s.log_level) == (
        "stdio",
        "127.0.0.1",
        8000,
        "INFO",
    )
    assert mcp.config.telemetry == TelemetrySettings()
    assert mcp.config.delivery == DeliverySettings()
    assert set(config_schema()["properties"]) == {"serving", "telemetry", "delivery"}


@pytest.mark.parametrize(
    "path, env, arg",
    [
        ("serving.transport", "TRANSPORT", "--transport"),
        ("serving.host", "HOST", "--host"),
        ("serving.port", "PORT", "--port"),
        ("serving.log_level", "LOG_LEVEL", "--log-level"),
        ("serving.graceful_timeout", "GRACEFUL_TIMEOUT", "--graceful-timeout"),
        ("telemetry.profiles", "TELEMETRY_PROFILES", "--telemetry-profiles"),
        ("telemetry.profile_cpu", "TELEMETRY_PROFILE_CPU", "--telemetry-profile-cpu"),
        (
            "telemetry.profile_memory",
            "TELEMETRY_PROFILE_MEMORY",
            "--telemetry-profile-memory",
        ),
        ("telemetry.otel", "TELEMETRY_OTEL", "--telemetry-otel"),
        ("telemetry.tags", "TELEMETRY_TAGS", "--telemetry-tags"),
        ("delivery.resources", "DELIVERY_RESOURCES", "--delivery-resources"),
        ("delivery.listed", "DELIVERY_LISTED", "--delivery-listed"),
        ("delivery.via", "DELIVERY_VIA", "--delivery-via"),
        ("delivery.max", "DELIVERY_MAX", "--delivery-max"),
    ],
)
def test_each_built_in_item_has_its_names(path, env, arg) -> None:
    item = {i.path: i for i in _compose([]).items}[path]
    assert (item.env, item.arg) == (env, arg)


def test_off_is_a_value_and_the_env_turns_it_on() -> None:
    mcp = QuickMCP(
        "t",
        environ={
            "TELEMETRY_OTEL": "true",
            "TELEMETRY_TAGS": '{"team": "a"}',
            "TELEMETRY_PROFILES": "http://p:4040",
        },
    )
    t = mcp.config.telemetry
    assert t.otel is True and t.tags == {"team": "a"}
    assert t.profiles.get_secret_value() == "http://p:4040"
    assert mcp.config_sources["telemetry.profiles"] == Source(
        "env", "TELEMETRY_PROFILES"
    )
    assert "http://p:4040" not in repr(mcp.config)


def test_the_transports_are_fastmcps() -> None:
    # FastMCP.run's annotations name FastMCP and Transport, which its module imports
    # only for type checkers: they resolve in the module that defines them.
    hint = typing.get_type_hints(FastMCP.run, localns=vars(fastmcp.server.server))[
        "transport"
    ]
    theirs = {
        a
        for m in typing.get_args(hint) or (hint,)
        for a in typing.get_args(m) or (m,)
        if a is not type(None)
    }
    annotation = ServingSettings.model_fields["transport"].annotation
    assert set(typing.get_args(annotation)) == theirs


def test_a_serving_subclass_changes_the_defaults_and_keeps_the_names() -> None:
    mcp = QuickMCP("t", sections=[Pod], environ={})
    assert (mcp.config.serving.transport, mcp.config.serving.host) == (
        "http",
        "0.0.0.0",
    )
    hosted = QuickMCP("t", sections=[Pod], environ={"HOST": "10.0.0.1"})
    assert hosted.config.serving.host == "10.0.0.1"


def test_a_subclass_field_that_drops_its_names_is_refused() -> None:
    @config("serving")
    class Bare(ServingSettings):
        host: str = "0.0.0.0"

    with pytest.raises(TypeError) as caught:
        _compose([[Bare]])
    assert str(caught.value) == (
        "serving.host: redefines ServingSettings.host without its env='HOST' and "
        "arg='--host': restate Field(..., env='HOST', arg='--host') to keep them"
    )


def test_a_subclass_field_that_drops_one_name_is_refused_for_it() -> None:
    @config("serving")
    class Flagless(ServingSettings):
        port: int = Field(9000, env="PORT")

    with pytest.raises(TypeError) as caught:
        _compose([[Flagless]])
    assert str(caught.value) == (
        "serving.port: redefines ServingSettings.port without its arg='--port': "
        "restate Field(..., arg='--port') to keep it"
    )


@config("serving")
class NoHostFlag(ServingSettings):
    """Drops ``--host``: refused alone, and in every class below it."""

    host: str = Field("0.0.0.0", env="HOST")


class UnmarkedNoHostFlag(ServingSettings):
    """``NoHostFlag`` with no ``@config`` of its own."""

    host: str = Field("0.0.0.0", env="HOST")


@pytest.mark.parametrize(
    "middle, alongside",
    [(NoHostFlag, False), (NoHostFlag, True), (UnmarkedNoHostFlag, False)],
)
def test_a_name_dropped_up_the_chain_is_refused_below_it(middle, alongside) -> None:
    @config("serving")
    class Leaf(middle):
        port: int = Field(9000, env="PORT", arg="--port")

    with pytest.raises(TypeError) as caught:
        _compose([[middle, Leaf] if alongside else [Leaf]])
    assert str(caught.value) == (
        "serving.host: redefines ServingSettings.host without its arg='--host': "
        "restate Field(..., arg='--host') to keep it"
    )


def test_a_redefined_chain_names_the_nearest_base_with_a_dropped_name() -> None:
    @config("serving")
    class Leaf(NoHostFlag):
        host: str = "10.0.0.1"

    with pytest.raises(TypeError) as caught:
        _compose([[Leaf]])
    # env='HOST' is dropped from NoHostFlag, arg='--host' from ServingSettings.
    assert str(caught.value) == (
        "serving.host: redefines NoHostFlag.host without its env='HOST' and "
        "arg='--host': restate Field(..., env='HOST', arg='--host') to keep them"
    )


def test_a_chain_that_renames_its_names_is_accepted() -> None:
    @config("serving")
    class Renamed(ServingSettings):
        host: str = Field("0.0.0.0", env="APP_HOST", arg="--app-host")

    @config("serving")
    class Leaf(Renamed):
        port: int = Field(9000, env="PORT", arg="--port")

    mcp = QuickMCP("t", sections=[Leaf], environ={"APP_HOST": "10.0.0.2"})
    assert mcp.config.serving.host == "10.0.0.2"


def test_a_subclass_may_restate_a_name_under_another() -> None:
    @config("serving")
    class Renamed(ServingSettings):
        port: int = Field(9000, env="APP_PORT", arg="--app-port")

    mcp = QuickMCP("t", sections=[Renamed], environ={"APP_PORT": "9002"})
    assert mcp.config.serving.port == 9002


@config("telemetry")
class Traced(TelemetrySettings):
    sampler: str = "always"


def test_a_servers_telemetry_subclass_replaces_the_built_in() -> None:
    mcp = QuickMCP("t", sections=[Traced], config={"telemetry": {"sampler": "never"}})
    assert type(mcp.config.telemetry) is Traced
    assert mcp._provided[TelemetrySettings] is mcp.config.telemetry


# -- serving (moved here from test_config_sources.py's "a server's sources", onto the
# real serving)


def test_a_server_reads_only_the_flags_and_env_it_is_given(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "1")
    mcp = QuickMCP("t", argv=["--port", "2"], environ={})
    assert mcp.config.serving.port == 2
    assert mcp.config_sources["serving.port"] == Source("arg", "--port")
    assert QuickMCP("t", environ={"PORT": "3"}).config.serving.port == 3


def test_a_server_built_without_environ_reads_the_process_env(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "4")
    mcp = QuickMCP("t")
    assert mcp.config_sources["serving.port"] == Source("env", "PORT")


def test_config_with_argv_or_environ_is_a_type_error_through_the_server() -> None:
    with pytest.raises(TypeError, match="config= reads no flags or env vars"):
        QuickMCP("t", config={"serving": {"port": 1}}, argv=["--port", "2"])


def test_a_bad_flag_stops_construction_without_exiting() -> None:
    with pytest.raises(ValueError, match=r"the arguments \(arg --nope\)"):
        QuickMCP("t", argv=["--nope"], environ={})


def test_config_sources_is_read_only() -> None:
    mcp = QuickMCP("t", environ={})
    with pytest.raises(AttributeError):
        mcp.config_sources = {}
    with pytest.raises(TypeError):
        mcp.config_sources["serving.port"] = Source("code")


def test_run_reads_host_and_port_from_environ(ran) -> None:
    QuickMCP("t", environ={"TRANSPORT": "http", "HOST": "0.0.0.0", "PORT": "9001"}).run(
        path="/mcp"
    )
    [middleware] = ran[0].pop("middleware")
    assert middleware.cls.__module__ == "kubed.mcp._shutdown"
    assert ran == [
        {
            "transport": "http",
            "host": "0.0.0.0",
            "port": 9001,
            "show_banner": False,
            "path": "/mcp",
            "uvicorn_config": {"timeout_graceful_shutdown": 15},
        }
    ]


@pytest.mark.parametrize(
    "environ, argv, timeout",
    [
        ({}, None, 15),
        ({"GRACEFUL_TIMEOUT": "2.5"}, None, 2.5),
        ({}, ["--graceful-timeout", "3"], 3),
    ],
    ids=["default", "env", "flag"],
)
def test_run_gives_uvicorn_the_graceful_timeout(ran, environ, argv, timeout) -> None:
    QuickMCP("t", environ={"TRANSPORT": "http", **environ}, argv=argv).run()
    assert ran[0]["uvicorn_config"] == {"timeout_graceful_shutdown": timeout}


def test_the_graceful_timeout_is_never_negative() -> None:
    with pytest.raises(ValueError, match=r"serving\.graceful_timeout"):
        QuickMCP("t", environ={"GRACEFUL_TIMEOUT": "-1"})


def test_run_keeps_the_callers_other_uvicorn_config(ran) -> None:
    given = {"log_config": None, "ws": "none"}
    QuickMCP("t", environ={"TRANSPORT": "http"}).run(uvicorn_config=given)
    assert ran[0]["uvicorn_config"] == {
        "log_config": None,
        "ws": "none",
        "timeout_graceful_shutdown": 15,
    }
    assert given == {"log_config": None, "ws": "none"}


def test_the_listen_ender_comes_ahead_of_the_callers_middleware(ran) -> None:
    from starlette.middleware import Middleware

    class Own:
        def __init__(self, app) -> None:
            self.app = app

    own = Middleware(Own)
    QuickMCP("t", environ={"TRANSPORT": "http"}).run(middleware=[own])
    first, *rest = ran[0]["middleware"]
    assert first.cls.__module__ == "kubed.mcp._shutdown"
    assert rest == [own]


def test_run_hooks_uvicorns_stop_only_while_it_serves(
    monkeypatch, root_logging
) -> None:
    original = uvicorn.Server.handle_exit
    during: list[object] = []

    def serve(self, **kwargs) -> None:
        during.append(uvicorn.Server.handle_exit)

    monkeypatch.setattr("fastmcp.FastMCP.run", serve)
    QuickMCP("t", environ={"TRANSPORT": "http"}).run()
    assert during[0] is not original and during[0].__wrapped__ is original
    assert uvicorn.Server.handle_exit is original

    def fail(self, **kwargs) -> None:
        during.append(uvicorn.Server.handle_exit)
        raise OSError("address in use")

    monkeypatch.setattr("fastmcp.FastMCP.run", fail)
    with pytest.raises(OSError, match="address in use"):
        QuickMCP("t", environ={"TRANSPORT": "http"}).run()
    assert during[1] is not original and during[1].__wrapped__ is original
    assert uvicorn.Server.handle_exit is original


def test_run_hooks_nothing_on_stdio(monkeypatch, root_logging) -> None:
    original = uvicorn.Server.handle_exit
    during: list[object] = []
    monkeypatch.setattr(
        "fastmcp.FastMCP.run",
        lambda self, **kwargs: during.append(uvicorn.Server.handle_exit),
    )
    QuickMCP("t", environ={}).run()
    assert during == [original]


def test_uvicorns_stop_is_the_method_the_hook_wraps() -> None:
    """The stop hook replaces ``uvicorn.Server.handle_exit``, which uvicorn's signal
    handlers call as ``(self, sig, frame)``. sse-starlette replaces it on import with a
    function of ``*args, **kwargs`` that calls uvicorn's, so whichever is there must
    take those three."""
    handle_exit = inspect.getattr_static(uvicorn.Server, "handle_exit")
    assert inspect.isfunction(handle_exit)
    inspect.signature(handle_exit).bind(object(), signal.SIGTERM, None)
    own = getattr(sys.modules.get("sse_starlette.sse"), "AppStatus", None)
    uvicorns = getattr(own, "original_handler", None) or handle_exit
    assert list(inspect.signature(uvicorns).parameters) == ["self", "sig", "frame"]
    assert "signal.signal(sig, self.handle_exit)" in inspect.getsource(
        uvicorn.Server.capture_signals
    )


def test_run_on_stdio_passes_nothing_else(ran) -> None:
    QuickMCP("t", environ={}).run(path="/mcp")
    assert ran == [{"transport": "stdio", "show_banner": False}]


def test_run_takes_no_serving(ran) -> None:
    with pytest.raises(TypeError):
        QuickMCP("t", environ={}).run(  # type: ignore[call-arg]
            argparse.Namespace(transport="http", host="h", port=1)
        )
    assert ran == []
    assert not hasattr(kubed.mcp.app, "Serving")
    assert "Serving" not in kubed.mcp.__all__


@pytest.mark.parametrize(
    "key, hint",
    [
        ("transport", "set serving.transport in the config (TRANSPORT, --transport"),
        ("host", "set serving.host in the config (HOST, --host"),
        ("port", "set serving.port in the config (PORT, --port"),
        ("log_level", "set serving.log_level in the config (LOG_LEVEL, --log-level"),
        ("show_banner", "no banner is ever printed"),
    ],
)
@pytest.mark.parametrize("environ", [{}, {"TRANSPORT": "http"}], ids=["stdio", "http"])
def test_run_refuses_a_serving_setting_before_it_does_anything(
    ran, root_logging, key, hint, environ
) -> None:
    mcp = QuickMCP("t", environ={**environ, "LOG_LEVEL": "DEBUG"})
    level = root_logging.level
    with pytest.raises(TypeError) as caught:
        mcp.run(path="/mcp", **{key: "x"})
    assert str(caught.value).startswith(f"run() takes no {key!r}: {hint}")
    assert ran == [] and root_logging.level == level
    mcp.auth = Refuse()  # serving has not started


@pytest.mark.parametrize("environ", [{}, {"TRANSPORT": "http"}], ids=["stdio", "http"])
def test_run_refuses_uvicorns_graceful_timeout_before_it_does_anything(
    ran, root_logging, environ
) -> None:
    mcp = QuickMCP("t", environ={**environ, "LOG_LEVEL": "DEBUG"})
    level = root_logging.level
    with pytest.raises(TypeError) as caught:
        mcp.run(uvicorn_config={"log_config": None, "timeout_graceful_shutdown": 1})
    assert str(caught.value) == (
        "run() takes no uvicorn_config['timeout_graceful_shutdown']: set "
        "serving.graceful_timeout in the config (GRACEFUL_TIMEOUT, --graceful-timeout, "
        "a serving: file key) or through hooks={'serving': ...}"
    )
    assert ran == [] and root_logging.level == level
    mcp.auth = Refuse()  # serving has not started


def test_the_hint_for_a_serving_setting_names_the_config_and_the_hook(ran) -> None:
    with pytest.raises(TypeError) as caught:
        QuickMCP("t", environ={}).run(port=8080)
    assert str(caught.value) == (
        "run() takes no 'port': set serving.port in the config (PORT, --port, a "
        "serving: file key) or through hooks={'serving': ...}"
    )


@pytest.mark.parametrize("given", ["info", "Debug", "WARNING"])
def test_the_log_level_is_read_case_insensitively(given) -> None:
    @config("serving")
    class Loud(ServingSettings):
        log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = Field(
            "DEBUG", env="LOG_LEVEL", arg="--log-level"
        )

    for sections in ((), (Loud,)):
        mcp = QuickMCP("t", sections=sections, environ={"LOG_LEVEL": given})
        assert mcp.config.serving.log_level == given.upper()
    with pytest.raises(ValueError, match=r"serving\.log_level"):
        QuickMCP("t", environ={"LOG_LEVEL": "loud"})
    schema = config_schema()["$defs"]["ServingSettings"]["properties"]["log_level"]
    assert schema["enum"] == ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def test_run_sets_the_root_level_and_adds_a_handler_only_when_there_is_none(
    ran, root_logging, monkeypatch
) -> None:
    called: list[dict] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: called.append(kw))
    QuickMCP("t", environ={"LOG_LEVEL": "DEBUG"}).run()
    assert root_logging.level == logging.DEBUG and called == []  # pytest's are on root
    root_logging.handlers = []
    QuickMCP("t", environ={"LOG_LEVEL": "WARNING"}).run()
    assert root_logging.level == logging.WARNING and len(called) == 1


def test_construction_sets_no_level(root_logging) -> None:
    before = root_logging.level
    QuickMCP("t", environ={"LOG_LEVEL": "DEBUG"})
    assert root_logging.level == before


# -- no attribute reads


def test_no_section_is_found_by_attribute_name() -> None:
    """CMP-5: no getattr or hasattr on the config outside the module that loads it."""
    for path in Path(kubed.mcp.__file__).parent.glob("*.py"):
        if path.name == "_config.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
            ):
                first = ast.unparse(node.args[0]).lower()
                assert "config" not in first, f"{path.name}:{node.lineno}"


# -- a server's component

LOG = logging.getLogger("tests.components")
EVENTS: list[str] = []


@config("backend")
class Backend(Config):
    url: str = "mem://"


class Unmarked(Config):
    """A section with no ``@config`` of its own."""

    url: str = "mem://"


class StartFailed(RuntimeError):
    """A ``Recorder``'s start failing: none of FastMCP's own errors is one."""


class Recorder(Component):
    section = Backend

    def __init__(
        self, tag: str, *, fail_start: bool = False, fail_stop: bool = False
    ) -> None:
        self.tag, self.fail_start, self.fail_stop = tag, fail_start, fail_stop

    def register(self, mcp, section) -> None:
        EVENTS.append(f"register {self.tag} {section.url}")

    async def start(self, section) -> None:
        LOG.info("start %s", self.tag)
        EVENTS.append(f"start {self.tag} {section.url}")
        if self.fail_start:
            raise StartFailed("https://user:s3cret@x")

    async def stop(self) -> None:
        EVENTS.append(f"stop {self.tag}")
        if self.fail_stop:
            raise RuntimeError("https://user:s3cret@x")


def _holds(error: BaseException | None, kind: type[BaseException]) -> bool:
    """Whether ``error``, its cause or context, or a member of an exception group's
    ``exceptions``, at any depth, is a ``kind``: FastMCP may wrap a lifespan's error."""
    seen: set[int] = set()
    todo = [error]
    while todo:
        each = todo.pop()
        if each is None or id(each) in seen:
            continue
        seen.add(id(each))
        if isinstance(each, kind):
            return True
        todo += [each.__cause__, each.__context__]
        todo += list(getattr(each, "exceptions", ()))
    return False


async def test_a_component_gets_its_section_and_starts_and_stops_in_order() -> None:
    EVENTS.clear()
    mcp = QuickMCP(
        "t",
        components=[Recorder("a"), Recorder("b")],
        config={"backend": {"url": "redis://r"}},
    )
    assert EVENTS == ["register a redis://r", "register b redis://r"]
    async with Client(mcp.server):
        assert EVENTS[2:] == ["start a redis://r", "start b redis://r"]
    assert EVENTS[4:] == ["stop b", "stop a"]


async def test_a_raising_stop_is_logged_by_type_and_the_others_still_stop(
    caplog,
) -> None:
    EVENTS.clear()
    mcp = QuickMCP(
        "t", components=[Recorder("a"), Recorder("b", fail_stop=True)], environ={}
    )
    async with Client(mcp.server):
        pass
    assert EVENTS[-2:] == ["stop b", "stop a"]
    (record,) = [r for r in caplog.records if r.name == "kubed.mcp.components"]
    assert (record.levelno, record.getMessage()) == (
        logging.ERROR,
        "Recorder did not stop cleanly: RuntimeError",
    )
    assert "s3cret" not in caplog.text


async def test_a_raising_start_stops_those_started_and_fails_the_boot(
    monkeypatch, caplog
) -> None:
    EVENTS.clear()
    closed: list[bool] = []

    async def aclose(self) -> None:
        closed.append(True)

    monkeypatch.setattr(Container, "aclose", aclose)
    monkeypatch.setattr(kubed.mcp.telemetry, "_running", None)
    fake = types.ModuleType("pyroscope")
    fake.configure = lambda **kw: EVENTS.append("profiles started")
    fake.shutdown = lambda: EVENTS.append("profiles stopped")
    monkeypatch.setitem(sys.modules, "pyroscope", fake)
    mcp = QuickMCP(
        "t",
        components=[Recorder("a"), Recorder("b", fail_start=True), Recorder("c")],
        config={"telemetry": {"profiles": "http://p:4040"}},
    )
    with pytest.raises(BaseException) as caught:
        async with Client(mcp.server):
            pass
    assert _holds(caught.value, StartFailed)
    assert [e for e in EVENTS if not e.startswith("register")] == [
        "profiles started",
        "start a mem://",
        "start b mem://",
        "stop a",
        "profiles stopped",
    ]
    assert closed == [True]
    assert "s3cret" not in caplog.text


class Cancels(Component):
    """A component whose start or stop is cancelled, recording what ran."""

    def __init__(
        self, tag: str, events: list[str], *, on_start=False, on_stop=False
    ) -> None:
        self.tag, self.events = tag, events
        self.on_start, self.on_stop = on_start, on_stop
        self.cancelled = asyncio.CancelledError(tag)

    async def start(self, section) -> None:
        self.events.append(f"start {self.tag}")
        if self.on_start:
            raise self.cancelled

    async def stop(self) -> None:
        self.events.append(f"stop {self.tag}")
        if self.on_stop:
            raise self.cancelled


async def test_a_cancelled_stop_is_raised_once_every_stop_has_run(caplog) -> None:
    events: list[str] = []
    a, b, c = (
        Cancels("a", events, on_stop=True),
        Recorder("b", fail_stop=True),
        Cancels("c", events, on_stop=True),
    )
    EVENTS.clear()
    with pytest.raises(asyncio.CancelledError) as caught:
        await _stop_all([a, b, c])
    assert caught.value is c.cancelled  # the first raised, and only it
    assert events == ["stop c", "stop a"] and EVENTS == ["stop b"]
    assert "s3cret" not in caplog.text


async def test_a_cancelled_start_stops_those_started_and_is_raised() -> None:
    events: list[str] = []
    a, b, c = (
        Cancels("a", events),
        Cancels("b", events, on_start=True),
        Cancels("c", events),
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        await _start_all([(a, None), (b, None), (c, None)])
    assert caught.value is b.cancelled
    assert events == ["start a", "start b", "stop a"]


def test_components_takes_component_instances() -> None:
    for bad in (Recorder, object()):
        with pytest.raises(TypeError, match=r"components= takes Component instances"):
            QuickMCP("t", components=[bad], environ={})


def test_a_components_section_needs_its_own_config_mark() -> None:
    class Bare(Component):
        section = Unmarked

    with pytest.raises(TypeError, match=r"no @config of its own"):
        QuickMCP("t", components=[Bare()], environ={})


def test_a_component_with_no_section_gets_none() -> None:
    seen: list[object] = []

    class Plain(Component):
        def register(self, mcp, section) -> None:
            seen.append(section)

    QuickMCP("t", components=[Plain()], environ={})
    assert seen == [None]


# -- order


async def test_the_start_lines_are_the_boot_line_then_telemetry_then_the_components(
    caplog,
) -> None:
    with caplog.at_level(logging.INFO):
        mcp = QuickMCP("t", components=[Recorder("a")], environ={"PORT": "9000"})
        # Nothing at construction.
        assert not [r for r in caplog.records if r.name == "kubed.mcp.config"]
        for _ in range(2):
            async with Client(mcp.server):
                pass
    names = ("kubed.mcp.config", "kubed.mcp.telemetry", "tests.components")
    lines = [r.getMessage() for r in caplog.records if r.name in names]
    assert (
        lines
        == [
            "config: serving.port=9000 (env PORT)",
            "telemetry: off",
            "start a",
        ]
        * 2
    )


# -- auth


class Refuse(TokenVerifier):
    async def verify_token(self, token: str):
        return None


def test_auth_set_after_construction_is_enforced() -> None:
    mcp = QuickMCP("t", environ={})
    assert mcp.auth is None
    mcp.auth = Refuse()
    assert mcp.server.auth is mcp.auth
    client = TestClient(mcp.server.http_app())
    answer = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"accept": "application/json, text/event-stream"},
    )
    assert answer.status_code == 401


def test_a_component_can_set_auth_in_register() -> None:
    class Door(Component):
        def register(self, mcp, section) -> None:
            mcp.auth = Refuse()

    assert isinstance(QuickMCP("t", components=[Door()], environ={}).auth, Refuse)


def test_auth_is_fixed_once_run_is_called(ran) -> None:
    mcp = QuickMCP("t", environ={})
    mcp.run()
    with pytest.raises(RuntimeError, match=r"auth is fixed once serving starts"):
        mcp.auth = Refuse()


def test_auth_is_fixed_once_the_http_app_is_built() -> None:
    mcp = QuickMCP("t", environ={})
    mcp.server.http_app()
    with pytest.raises(RuntimeError, match=r"auth is fixed once serving starts"):
        mcp.auth = Refuse()
    assert mcp.auth is None


async def test_auth_is_fixed_once_fastmcps_own_http_run_builds_the_app(
    monkeypatch,
) -> None:
    import uvicorn

    async def serve(self, *args, **kwargs) -> None:
        return None

    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    mcp = QuickMCP("t", environ={})
    await mcp.server.run_http_async(show_banner=False)
    with pytest.raises(RuntimeError, match=r"auth is fixed once serving starts"):
        mcp.auth = Refuse()


def test_building_the_http_app_still_builds_fastmcps() -> None:
    mcp = QuickMCP("t", environ={})
    app = mcp.server.http_app(path="/elsewhere")
    assert any(getattr(r, "path", None) == "/elsewhere" for r in app.routes)


async def test_auth_is_fixed_inside_the_lifespan() -> None:
    mcp = QuickMCP("t", environ={})
    async with Client(mcp.server):
        with pytest.raises(RuntimeError, match=r"auth is fixed once serving starts"):
            mcp.auth = Refuse()


# -- hooks


@config()
class Kb(Config):
    port: int = Field(9100, env="PORT", arg="--port")
    bind: str = "0.0.0.0"


def from_root(config) -> ServingSettings:
    return ServingSettings(transport="http", host=config.bind, port=config.port)


def test_a_hook_builds_serving_from_the_servers_own_fields() -> None:
    mcp = QuickMCP(
        "kb", sections=[Kb], hooks={"serving": from_root}, environ={"PORT": "9200"}
    )
    s = mcp.config.serving
    assert (s.transport, s.host, s.port) == ("http", "0.0.0.0", 9200)
    assert mcp.config.port == 9200 and mcp.config_sources["port"] == Source(
        "env", "PORT"
    )
    assert {
        mcp.config_sources[f"serving.{n}"]
        for n in ("transport", "host", "port", "log_level")
    } == {Source("hook")}


def test_a_hooked_built_ins_env_and_flags_are_not_read() -> None:
    seen: list[object] = []

    def keep(config):
        seen.append(config.serving)
        return ServingSettings()

    mcp = QuickMCP(
        "t", hooks={"serving": keep}, environ={"HOST": "10.0.0.1", "PORT": "1"}
    )
    assert seen[0].host == "127.0.0.1" and mcp.config.serving.port == 8000
    with pytest.raises(
        ValueError, match=r"  the arguments \(arg --port\): no item takes this flag"
    ):
        QuickMCP("t", hooks={"serving": keep}, argv=["--port", "1"], environ={})


def test_a_hooked_sections_file_entries_are_not_read(tmp_path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("serving:\n  port: 1\n")
    mcp = QuickMCP(
        "t",
        hooks={"serving": lambda c: ServingSettings(port=2)},
        environ={"CONFIG_FILE": str(f)},
    )
    assert mcp.config.serving.port == 2 and mcp.config_sources[
        "serving.port"
    ] == Source("hook")


def test_a_servers_port_and_a_hooked_serving_do_not_clash() -> None:
    with pytest.raises(
        ValueError, match=r"'PORT' is declared by both port and serving\.port"
    ):
        QuickMCP("kb", sections=[Kb], environ={})
    QuickMCP("kb", sections=[Kb], hooks={"serving": from_root}, environ={})
    schema = config_schema(Kb, hooked=["serving"])
    serving = schema["$defs"]["ServingSettings"]["properties"]["port"]
    assert (
        "env" not in serving
        and "arg" not in serving
        and schema["properties"]["port"]["env"] == "PORT"
    )
    with pytest.raises(ValueError, match=r"PORT"):
        config_schema(Kb)


def test_a_clash_with_a_built_in_says_to_hook_it() -> None:
    hint = "; to keep your own, hook the built-in: hooks={'serving': ...}"
    with pytest.raises(ValueError) as caught:
        QuickMCP("kb", sections=[Kb], environ={})
    assert str(caught.value) == (
        f"The env var 'PORT' is declared by both port and serving.port{hint}"
    )

    @config("own")
    class Own(Config):
        port: int = Field(0, env="KB_PORT")

    @config()
    class Root(Config):
        port: int = Field(0, env="KB_PORT")

    with pytest.raises(ValueError) as caught:
        QuickMCP("kb", sections=[Root, Own], environ={})
    assert str(caught.value) == (
        "The env var 'KB_PORT' is declared by both port and own.port"
    )


def test_a_hooked_sections_class_is_still_checked_for_dropped_names() -> None:
    @config("serving")
    class Bare(ServingSettings):
        port: int = 9000

    with pytest.raises(TypeError, match=r"serving\.port: redefines ServingSettings"):
        QuickMCP(
            "t",
            sections=[Bare],
            hooks={"serving": lambda c: Bare()},
            environ={},
        )
    with pytest.raises(TypeError, match=r"serving\.port: redefines ServingSettings"):
        config_schema(Bare, hooked=["serving"])


def test_a_hook_returning_the_wrong_type_is_a_type_error() -> None:
    with pytest.raises(
        TypeError,
        match=r"hooks\['serving'\] returned a dict, not an instance of ServingSettings",
    ):
        QuickMCP("t", hooks={"serving": lambda c: {"port": 1}}, environ={})


def test_a_hook_may_return_the_servers_subclass_and_must_when_one_holds_the_key() -> (
    None
):
    mcp = QuickMCP(
        "t", sections=[Pod], hooks={"serving": lambda c: Pod(port=3)}, environ={}
    )
    assert type(mcp.config.serving) is Pod and mcp.config.serving.port == 3
    with pytest.raises(TypeError, match=r"not an instance of Pod"):
        QuickMCP(
            "t",
            sections=[Pod],
            hooks={"serving": lambda c: ServingSettings()},
            environ={},
        )


def test_a_hooks_result_is_validated_and_its_errors_name_the_hook() -> None:
    bad = ServingSettings.model_construct(port="not a port")
    with pytest.raises(
        ValueError, match=r"  serving\.port \(hook\): Input should be a valid integer"
    ):
        QuickMCP("t", hooks={"serving": lambda c: bad}, environ={})


@pytest.mark.parametrize(
    "hooks, error, match",
    [
        (
            {"recording": from_root},
            ValueError,
            r"hooks= takes a built-in's key, one of 'serving', 'telemetry', 'delivery'; got 'recording'",
        ),
        ({"serving": 1}, TypeError, r"hooks\['serving'\] must be callable"),
        ([("serving", from_root)], TypeError, r"hooks= takes a mapping"),
    ],
)
def test_a_bad_hooks_argument_is_refused(hooks, error, match) -> None:
    with pytest.raises(error, match=match):
        QuickMCP("t", hooks=hooks, environ={})


def test_every_built_in_takes_a_hook() -> None:
    mcp = QuickMCP(
        "t",
        environ={},
        hooks={
            "telemetry": lambda c: TelemetrySettings(otel=False, tags={"a": "b"}),
            "delivery": lambda c: DeliverySettings(via="local"),
        },
    )
    assert (
        mcp.config.telemetry.tags == {"a": "b"} and mcp.config.delivery.via == "local"
    )
    assert mcp.config_sources["delivery.via"] == Source("hook")


def test_the_boot_line_shows_hooked_items_as_hook() -> None:
    loaded = _load(
        [[]], None, environ={}, hooks={"serving": lambda c: ServingSettings(port=1)}
    )
    assert "serving.port=1 (hook)" in loaded.boot_line


def test_a_hooks_result_missing_a_field_is_a_load_error() -> None:
    bad = ServingSettings.model_construct()
    vars(bad).pop("port")
    with pytest.raises(ValueError, match=r"  serving\.port \(hook\): Field required"):
        QuickMCP("t", hooks={"serving": lambda c: bad}, environ={})


def test_config_entries_under_a_hooked_key_are_dropped() -> None:
    mcp = QuickMCP(
        "t",
        config={"serving": {"port": 5}},
        hooks={"serving": lambda c: ServingSettings(port=2)},
    )
    assert mcp.config.serving.port == 2
    assert mcp.config_sources["serving.port"] == Source("hook")


@config("remote")
class Remote(Config):
    url: str = Field("mem://", arg="--backend-url")


def test_the_flags_of_the_sections_after_a_hooked_one_line_up() -> None:
    mcp = QuickMCP(
        "t",
        sections=[Remote],
        hooks={"serving": lambda c: ServingSettings()},
        argv=["--backend-url", "y"],
        environ={},
    )
    assert mcp.config.remote.url == "y"
    assert mcp.config_sources["remote.url"] == Source("arg", "--backend-url")


class HookFailed(Exception):
    """A hook's own error, which the load lets through."""


def test_a_hook_that_raises_propagates_unchanged() -> None:
    def fail(config):
        raise HookFailed("boom")

    with pytest.raises(HookFailed, match=r"^boom$"):
        QuickMCP("t", hooks={"serving": fail}, environ={})


SEEN_PORTS: list[int] = []


@config()
class Watching(Config):
    @model_validator(mode="after")
    def watch(self):
        SEEN_PORTS.append(self.serving.port)
        return self


def test_a_root_validator_sees_the_hooked_values_on_the_second_validation() -> None:
    SEEN_PORTS.clear()
    QuickMCP(
        "t",
        sections=[Watching],
        hooks={"serving": lambda c: ServingSettings(port=9)},
        environ={},
    )
    assert SEEN_PORTS == [8000, 9]


@pytest.mark.parametrize(
    "hooked, error, match",
    [
        (
            ["recording"],
            ValueError,
            r"hooked= takes a built-in's key, one of 'serving', 'telemetry', "
            r"'delivery'; got 'recording'",
        ),
        ("serving", TypeError, r"hooked= takes an iterable of built-in keys"),
    ],
)
def test_a_bad_hooked_argument_is_refused(hooked, error, match) -> None:
    with pytest.raises(error, match=match):
        config_schema(hooked=hooked)
