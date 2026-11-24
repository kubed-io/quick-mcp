"""Telemetry: the settings, and profiles that start and stop with the server."""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fastmcp import Client, FastMCP
from pydantic import BaseModel

import kubed.mcp.telemetry
from kubed.mcp import Inject, QuickMCP, TelemetrySettings
from kubed.mcp.app import _with_teardown
from kubed.mcp.container import Container

ADDRESS = "http://pyroscope.example.com:4040"
EVENTS: list[str] = []


@pytest.fixture(autouse=True)
def _clean(monkeypatch) -> None:
    EVENTS.clear()
    monkeypatch.setattr(kubed.mcp.telemetry, "_running", None)
    for name in ("OTEL_SERVICE_NAME", "POD_NAMESPACE", "POD_NAME"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def pyroscope(monkeypatch) -> list[dict]:
    """A fake pyroscope module; the fixture is the list of configure() calls."""
    calls: list[dict] = []
    module = types.ModuleType("pyroscope")

    def configure(**kwargs) -> None:
        calls.append(kwargs)
        EVENTS.append("profiles started")

    def shutdown() -> None:
        EVENTS.append("profiles stopped")

    module.configure = configure
    module.shutdown = shutdown
    monkeypatch.setitem(sys.modules, "pyroscope", module)
    return calls


@dataclass
class Config:
    transport: str = "http"
    host: str = "0.0.0.0"
    port: int = 8000
    telemetry: object = None


class Monitor:
    pass


@asynccontextmanager
async def make_monitor():
    yield Monitor()
    EVENTS.append("monitor stopped")


def served(name: str, telemetry: object, **options) -> QuickMCP:
    mcp = QuickMCP(name, config=Config(telemetry=telemetry), **options)
    mcp.singleton(Monitor, make_monitor)

    @mcp.tool
    def ping(monitor: Inject[Monitor]) -> str:
        return "pong"

    return mcp


def test_settings_default_to_every_signal_off() -> None:
    settings = TelemetrySettings()
    assert settings.profiles is None
    assert settings.profile_cpu is True
    assert settings.profile_memory is True
    assert settings.otel is False
    assert settings.tags == {}


def test_settings_embed_as_a_configs_telemetry_section() -> None:
    class Section(BaseModel):
        telemetry: TelemetrySettings = TelemetrySettings()

    config = Section.model_validate(
        {
            "telemetry": {
                "profiles": "http://pyroscope.example.com:4040",
                "tags": {"team": "a"},
            }
        }
    )
    assert config.telemetry.profiles == "http://pyroscope.example.com:4040"
    assert config.telemetry.tags == {"team": "a"}
    assert Section().telemetry.tags is not Section().telemetry.tags


async def test_profiles_start_before_the_lifespan_and_stop_before_the_container(
    pyroscope,
) -> None:
    @asynccontextmanager
    async def lifespan(server):
        EVENTS.append("lifespan up")
        yield {}
        EVENTS.append("lifespan down")

    mcp = served("profiled", TelemetrySettings(profiles=ADDRESS), lifespan=lifespan)
    async with Client(mcp.server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    assert pyroscope == [
        {
            "application_name": "profiled",
            "server_address": ADDRESS,
            "cpu_enabled": True,
            "mem_enabled": True,
            "tags": {},
        }
    ]
    assert EVENTS[:2] == ["profiles started", "lifespan up"]
    assert EVENTS[-3:] == ["lifespan down", "profiles stopped", "monitor stopped"]


async def test_profiles_stop_when_the_lifespan_fails(pyroscope) -> None:
    # Through _with_teardown directly, so the error is the lifespan's own.
    @asynccontextmanager
    async def lifespan(server):
        EVENTS.append("lifespan up")
        raise RuntimeError("boom")
        yield

    run = _with_teardown(
        Container(), lifespan, name="t", telemetry=TelemetrySettings(profiles=ADDRESS)
    )
    with pytest.raises(RuntimeError, match="boom"):
        async with run(FastMCP("t")):
            pass
    assert EVENTS == ["profiles started", "lifespan up", "profiles stopped"]


@pytest.mark.parametrize(
    "telemetry",
    [
        None,
        TelemetrySettings(),
        TelemetrySettings(profiles=""),
        TelemetrySettings(profiles=ADDRESS, profile_cpu=False, profile_memory=False),
    ],
)
async def test_profiles_off_import_nothing(monkeypatch, telemetry) -> None:
    monkeypatch.delitem(sys.modules, "pyroscope", raising=False)
    async with Client(served("off", telemetry).server) as client:
        await client.call_tool("ping", {})
    assert "pyroscope" not in sys.modules


async def test_a_config_without_telemetry_imports_nothing(monkeypatch) -> None:
    # A Serving with no telemetry attribute at all, as both servers' configs are today.
    monkeypatch.delitem(sys.modules, "pyroscope", raising=False)
    mcp = QuickMCP(
        "plain", config=argparse.Namespace(transport="http", host="0.0.0.0", port=8000)
    )
    async with Client(mcp.server):
        pass
    assert "pyroscope" not in sys.modules


def test_importing_kubed_mcp_imports_no_pyroscope() -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys, kubed.mcp; print('pyroscope' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


async def test_a_missing_package_is_one_warning_and_the_server_serves(
    monkeypatch, caplog
) -> None:
    monkeypatch.setitem(sys.modules, "pyroscope", None)
    async with Client(
        served("missing", TelemetrySettings(profiles=ADDRESS)).server
    ) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    warnings = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in warnings] == [logging.WARNING]
    assert "kubed-mcp[telemetry]" in warnings[0].getMessage()


async def test_a_profiler_that_fails_to_start_is_logged_and_the_server_serves(
    pyroscope, monkeypatch, caplog
) -> None:
    secret = "https://user:s3cret@pyroscope.example.com"

    def refuse(**kwargs):
        raise RuntimeError("agent refused")

    monkeypatch.setattr(sys.modules["pyroscope"], "configure", refuse)
    async with Client(
        served("failing", TelemetrySettings(profiles=secret)).server
    ) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    errors = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in errors] == [logging.ERROR]
    assert "s3cret" not in errors[0].getMessage()
    assert "profiles stopped" not in EVENTS


async def test_a_profiler_that_fails_to_stop_still_closes_the_container(
    pyroscope, monkeypatch, caplog
) -> None:
    def refuse() -> None:
        raise RuntimeError("agent stuck")

    monkeypatch.setattr(sys.modules["pyroscope"], "shutdown", refuse)
    async with Client(
        served("stuck", TelemetrySettings(profiles=ADDRESS)).server
    ) as client:
        await client.call_tool("ping", {})
    assert EVENTS[-1] == "monitor stopped"
    assert [r.levelno for r in caplog.records if r.name == "kubed.mcp.telemetry"] == [
        logging.ERROR
    ]
    # The slot is free, so a later start can run its own profiles.
    assert kubed.mcp.telemetry._running is None


async def test_a_profiler_already_running_is_one_error_and_never_shut_down(
    pyroscope, monkeypatch, caplog
) -> None:
    # pyroscope-io's configure() returns False when an agent already runs in the process.
    monkeypatch.setattr(sys.modules["pyroscope"], "configure", lambda **kwargs: False)
    secret = "https://user:s3cret@pyroscope.example.com"
    async with Client(
        served("taken", TelemetrySettings(profiles=secret)).server
    ) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "s3cret" not in records[0].getMessage()
    assert "profiles stopped" not in EVENTS


async def test_a_shutdown_that_returns_false_is_one_warning_and_the_container_closes(
    pyroscope, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(sys.modules["pyroscope"], "shutdown", lambda: False)
    async with Client(
        served("unclean", TelemetrySettings(profiles=ADDRESS)).server
    ) as client:
        await client.call_tool("ping", {})
    assert EVENTS[-1] == "monitor stopped"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert kubed.mcp.telemetry._running is None


async def test_the_first_server_owns_the_profiles(pyroscope) -> None:
    first = served("first", TelemetrySettings(profiles="http://a.example.com"))
    second = served("second", TelemetrySettings(profiles="http://b.example.com"))
    async with Client(first.server):
        async with Client(second.server):
            pass
        assert EVENTS.count("profiles stopped") == 0
    assert [c["server_address"] for c in pyroscope] == ["http://a.example.com"]
    assert EVENTS.count("profiles stopped") == 1


async def test_two_clients_on_one_server_start_one_profiler(pyroscope) -> None:
    # FastMCP 4 ref-counts lifespan entries, so the second client never reaches our
    # guard; test_the_first_server_owns_the_profiles covers it.
    mcp = served("shared", TelemetrySettings(profiles=ADDRESS))
    async with Client(mcp.server), Client(mcp.server):
        pass
    assert len(pyroscope) == 1
    assert EVENTS.count("profiles stopped") == 1


async def test_profiles_start_again_once_the_owner_has_stopped(pyroscope) -> None:
    first = served("first", TelemetrySettings(profiles="http://a.example.com"))
    async with Client(first.server):
        pass
    second = served("second", TelemetrySettings(profiles="http://b.example.com"))
    async with Client(second.server):
        pass
    assert [c["server_address"] for c in pyroscope] == [
        "http://a.example.com",
        "http://b.example.com",
    ]


async def test_a_servers_own_section_satisfies_telemetry(pyroscope) -> None:
    section = argparse.Namespace(
        profiles=ADDRESS, profile_cpu=True, profile_memory=False, otel=False, tags={}
    )
    async with Client(served("own", section).server):
        pass
    assert pyroscope[0]["mem_enabled"] is False


async def test_a_module_missing_inside_configure_is_a_failed_start_not_a_missing_package(
    pyroscope, monkeypatch, caplog
) -> None:
    def refuse(**kwargs):
        raise ModuleNotFoundError(
            "No module named 'something_else'", name="something_else"
        )

    monkeypatch.setattr(sys.modules["pyroscope"], "configure", refuse)
    async with Client(
        served("inner", TelemetrySettings(profiles=ADDRESS)).server
    ) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.ERROR]


async def test_a_section_missing_an_attribute_is_a_logged_error_and_the_server_serves(
    pyroscope, caplog
) -> None:
    mcp = served("partial", argparse.Namespace(profiles=ADDRESS))
    async with Client(mcp.server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "profiles stopped" not in EVENTS


async def test_otel_service_name_names_the_profiles(pyroscope, monkeypatch) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "selenium-flow")
    async with Client(served("built-name", TelemetrySettings(profiles=ADDRESS)).server):
        pass
    assert pyroscope[0]["application_name"] == "selenium-flow"


async def test_profiles_carry_the_pod_and_tags_win(pyroscope, monkeypatch) -> None:
    monkeypatch.setenv("POD_NAMESPACE", "tools")
    monkeypatch.setenv("POD_NAME", "sf-0")
    settings = TelemetrySettings(profiles=ADDRESS, tags={"pod": "pinned", "team": "a"})
    async with Client(served("labelled", settings).server):
        pass
    assert pyroscope[0]["tags"] == {"namespace": "tools", "pod": "pinned", "team": "a"}


async def test_an_empty_pod_variable_adds_no_label(pyroscope, monkeypatch) -> None:
    monkeypatch.setenv("POD_NAMESPACE", "")
    monkeypatch.setenv("POD_NAME", "sf-0")
    async with Client(served("blank", TelemetrySettings(profiles=ADDRESS)).server):
        pass
    assert pyroscope[0]["tags"] == {"pod": "sf-0"}


@pytest.mark.parametrize(("cpu", "memory"), [(True, False), (False, True)])
async def test_cpu_and_memory_switch_independently(pyroscope, cpu, memory) -> None:
    settings = TelemetrySettings(
        profiles=ADDRESS, profile_cpu=cpu, profile_memory=memory
    )
    async with Client(served("switched", settings).server):
        pass
    assert (pyroscope[0]["cpu_enabled"], pyroscope[0]["mem_enabled"]) == (cpu, memory)


async def test_every_keyword_passed_is_one_configure_takes(request) -> None:
    # A real import: CI installs the extra, so there it must not skip.
    real = (
        importlib.import_module("pyroscope")
        if os.environ.get("CI")
        else pytest.importorskip("pyroscope")
    )
    configure = inspect.signature(real.configure)
    calls = request.getfixturevalue(
        "pyroscope"
    )  # the fake goes in only after the real import
    settings = TelemetrySettings(profiles=ADDRESS, tags={"team": "a"})
    async with Client(served("checked", settings).server):
        pass
    assert set(calls[0]) <= set(configure.parameters)
    assert set(calls[0]) == {
        "application_name",
        "server_address",
        "cpu_enabled",
        "mem_enabled",
        "tags",
    }
