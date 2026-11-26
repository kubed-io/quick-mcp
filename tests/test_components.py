"""The built-in sections every server has, and ``run()`` serving from them."""

from __future__ import annotations

import argparse
import ast
import logging
import typing
from pathlib import Path
from typing import Literal

import fastmcp.server.server
import pytest
from fastmcp import FastMCP

import kubed.mcp
import kubed.mcp.app
from kubed.mcp import (
    DeliverySettings,
    Field,
    QuickMCP,
    ServingSettings,
    Source,
    TelemetrySettings,
    config,
    config_schema,
)
from kubed.mcp._config import _compose

pytestmark = pytest.mark.unit


@config("serving")
class Pod(ServingSettings):
    transport: Literal["stdio", "http", "sse", "streamable-http"] = Field(
        "http", env="TRANSPORT", arg="--transport"
    )
    host: str = Field("0.0.0.0", env="HOST", arg="--host")


@pytest.fixture
def ran(monkeypatch, root_logging) -> list[dict]:
    """FastMCP's run, recorded. ``run()`` sets the root level, which is put back."""
    calls: list[dict] = []
    monkeypatch.setattr(
        "fastmcp.FastMCP.run", lambda self, **kwargs: calls.append(kwargs)
    )
    return calls


@pytest.fixture
def root_logging(monkeypatch):
    """The root logger's level and handlers, put back after the test.

    The level goes back through ``setLevel``, which clears every logger's cache of what
    it is enabled for.
    """
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    level = root.level
    yield root
    root.setLevel(level)


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
    assert ran == [
        {
            "transport": "http",
            "host": "0.0.0.0",
            "port": 9001,
            "show_banner": False,
            "path": "/mcp",
        }
    ]


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
