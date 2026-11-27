"""Telemetry: the settings, profiles, and OpenTelemetry, started and stopped with the server."""

from __future__ import annotations

import asyncio
import base64
import http.client
import http.server
import importlib
import inspect
import io
import json
import logging
import logging.handlers
import os
import queue
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP
from gateway_server import KIT_TREE, over_http
from plugin_server import write
from pydantic import BaseModel, ValidationError
from skill_server import request

import kubed.mcp
import kubed.mcp.telemetry
from kubed.mcp import Component, Inject, QuickMCP, TelemetrySettings, config
from kubed.mcp._builtins import _TelemetryComponent
from kubed.mcp._otel import _TraceText
from kubed.mcp.app import _with_teardown
from kubed.mcp.container import Container
from kubed.mcp.telemetry import _kind, _Probe, _start_signals

ADDRESS = "http://pyroscope.example.com:4040"
CHILD = Path(__file__).with_name("otel_child.py")
EVENTS: list[str | tuple[str, str]] = []
# Every provider initialize() sets, as _Otel's owned names them.
EVERY_KIND = frozenset({"tracer", "meter", "logger"})


def _reset_otel() -> None:
    if (otel := sys.modules.get("kubed.mcp._otel")) is not None:
        otel._stopped = False


@pytest.fixture(autouse=True)
def _clean(monkeypatch) -> Iterator[None]:
    EVENTS.clear()
    monkeypatch.setattr(kubed.mcp.telemetry, "_running", None)
    for name in [n for n in os.environ if n.startswith("OTEL_")]:
        monkeypatch.delenv(name)
    for name in ("POD_NAMESPACE", "POD_NAME"):
        monkeypatch.delenv(name, raising=False)
    # _set_defaults writes os.environ directly, so the whole of it is put back.
    environ = dict(os.environ)
    # A start whose stop the test never runs leaves its trace-text factory set.
    factory = logging.getLogRecordFactory()
    _reset_otel()
    yield
    _reset_otel()
    logging.setLogRecordFactory(factory)
    for name in [n for n in os.environ if n not in environ]:
        del os.environ[name]
    for name, value in environ.items():
        if os.environ.get(name) != value:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def _no_stray_initialize(request, monkeypatch) -> Iterator[None]:
    """Fail a test that ran the real initialize() in-process without saying so.

    Inside ``start`` a failed initialize() is caught and logged, so the test would pass.
    """
    if request.node.get_closest_marker("real_initialize"):
        yield
        return
    try:
        otel = importlib.import_module("kubed.mcp._otel")
    except ImportError:
        yield
        return
    calls: list[dict] = []
    monkeypatch.setattr(otel, "initialize", lambda **kwargs: calls.append(kwargs))
    yield
    if calls:
        pytest.fail(f"initialize() ran in-process, unmarked: {calls}")


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


class Monitor:
    pass


@asynccontextmanager
async def make_monitor():
    yield Monitor()
    EVENTS.append("monitor stopped")


def served(name: str, telemetry: object, **options) -> QuickMCP:
    mcp = QuickMCP(name, config={"telemetry": telemetry}, **options)
    mcp.singleton(Monitor, make_monitor)

    @mcp.tool
    def ping(monitor: Inject[Monitor]) -> str:
        return "pong"

    return mcp


def assert_safe(records: list[logging.LogRecord], secret: str) -> None:
    """No record carries a traceback, and none shows ``secret``."""
    formatter = logging.Formatter("%(message)s")
    for record in records:
        assert record.exc_info is None
        assert record.exc_text is None
        assert secret not in formatter.format(record)


def _json_objects(text: str) -> list[dict]:
    """Every JSON object in ``text``: the console exporter pretty-prints one per span."""
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    index = 0
    while (start := text.find("{", index)) != -1:
        try:
            found, index = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(found, dict):
            objects.append(found)
    return objects


def _child(mode: str, **env: str) -> subprocess.CompletedProcess[str]:
    """tests/otel_child.py in a process of its own: no OTEL_* but ``env``, and no network."""
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("OTEL_") and k not in ("POD_NAMESPACE", "POD_NAME")
    }
    clean |= {"OTEL_TRACES_EXPORTER": "console", "OTEL_METRICS_EXPORTER": "none"}
    return subprocess.run(
        [sys.executable, "-I", str(CHILD), mode],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=clean | env,
    )


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
    assert (
        config.telemetry.profiles.get_secret_value()
        == "http://pyroscope.example.com:4040"
    )
    assert config.telemetry.tags == {"team": "a"}
    assert Section().telemetry.tags is not Section().telemetry.tags


def test_every_setting_is_described_in_one_short_sentence() -> None:
    for name, field in TelemetrySettings.model_fields.items():
        assert field.description, name
        assert len(field.description) <= 100, name
        assert field.description.endswith("."), name


def test_a_typo_in_the_section_stops_the_boot() -> None:
    with pytest.raises(ValidationError, match="profile_cpus"):
        TelemetrySettings.model_validate({"profile_cpus": False})


def test_a_server_can_subclass_the_section() -> None:
    class Section(TelemetrySettings):
        exporter_port: int = 9464

    assert Section(exporter_port=1).exporter_port == 1
    with pytest.raises(ValidationError):
        Section.model_validate({"tag": {}})


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
            "upload_interval": 10,
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
        Container(),
        lifespan,
        parts=[(_TelemetryComponent("t"), TelemetrySettings(profiles=ADDRESS))],
        boot_line="config: all defaults",
        on_enter=lambda: None,
    )
    with pytest.raises(RuntimeError, match="boom"):
        async with run(FastMCP("t")):
            pass
    assert EVENTS == ["profiles started", "lifespan up", "profiles stopped"]


class Slow(Component):
    """Starts slowly, so a second lifespan entry arrives while the first starts."""

    async def start(self, section) -> None:
        await asyncio.sleep(0.05)


async def test_two_entries_at_once_start_and_stop_the_profiles_once(pyroscope) -> None:
    mcp = served("twice", TelemetrySettings(profiles=ADDRESS), components=[Slow()])

    async def enter() -> None:
        async with Client(mcp.server) as client:
            await client.call_tool("ping", {})

    await asyncio.gather(enter(), enter())
    assert [e for e in EVENTS if e.startswith("profiles")] == [
        "profiles started",
        "profiles stopped",
    ]
    assert kubed.mcp.telemetry._running is None


@pytest.mark.parametrize(
    "telemetry",
    [
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
    # A server given no telemetry values: the built-in section's defaults.
    monkeypatch.delitem(sys.modules, "pyroscope", raising=False)
    mcp = QuickMCP("plain")
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
        raise RuntimeError("agent refused at https://user:s3cret@pyroscope.example.com")

    monkeypatch.setattr(sys.modules["pyroscope"], "configure", refuse)
    async with Client(
        served("failing", TelemetrySettings(profiles=secret)).server
    ) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    errors = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in errors] == [logging.ERROR]
    assert "s3cret" not in errors[0].getMessage()
    assert "RuntimeError" in errors[0].getMessage()
    assert_safe(caplog.records, "s3cret")
    assert "profiles stopped" not in EVENTS


async def test_a_profiler_that_fails_to_stop_still_closes_the_container(
    pyroscope, monkeypatch, caplog
) -> None:
    def refuse() -> None:
        raise RuntimeError("agent stuck at https://user:s3cret@pyroscope.example.com")

    monkeypatch.setattr(sys.modules["pyroscope"], "shutdown", refuse)
    async with Client(
        served("stuck", TelemetrySettings(profiles=ADDRESS)).server
    ) as client:
        await client.call_tool("ping", {})
    assert EVENTS[-1] == "monitor stopped"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "RuntimeError" in records[0].getMessage()
    assert_safe(caplog.records, "s3cret")
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


@config("telemetry")
class OwnTelemetry(TelemetrySettings):
    sampler: str = "always"


async def test_a_servers_own_section_satisfies_telemetry(pyroscope) -> None:
    section = OwnTelemetry(profiles=ADDRESS, profile_memory=False)
    async with Client(served("own", section, sections=[OwnTelemetry]).server):
        pass
    assert pyroscope[0]["mem_enabled"] is False


async def test_a_module_missing_inside_configure_is_a_failed_start_not_a_missing_package(
    pyroscope, monkeypatch, caplog
) -> None:
    def refuse(**kwargs):
        raise ModuleNotFoundError(
            "No module named 'something_else' at "
            "https://user:s3cret@pyroscope.example.com",
            name="something_else",
        )

    monkeypatch.setattr(sys.modules["pyroscope"], "configure", refuse)
    async with Client(
        served("inner", TelemetrySettings(profiles=ADDRESS)).server
    ) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "ModuleNotFoundError" in records[0].getMessage()
    assert_safe(caplog.records, "s3cret")


async def test_a_start_that_raises_is_a_logged_error_and_the_server_serves(
    pyroscope, monkeypatch, caplog
) -> None:
    secret = "https://user:s3cret@pyroscope.example.com"

    def refuse(name, settings):
        raise AttributeError(f"no profiles at {secret}")

    monkeypatch.setattr(kubed.mcp.telemetry, "_start_profiles", refuse)
    mcp = served("partial", TelemetrySettings(profiles=secret))
    async with Client(mcp.server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "AttributeError" in records[0].getMessage()
    assert_safe(caplog.records, "s3cret")
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
        "upload_interval",
    }


# OpenTelemetry. The real configuration runs in tests/otel_child.py: its providers are
# set once per process, and pytest's must stay unconfigured for the tests after it.


def test_resource_attributes_merge_and_encode(monkeypatch) -> None:
    from kubed.mcp._otel import _resource_attributes

    monkeypatch.setenv("POD_NAMESPACE", "tools")
    monkeypatch.setenv("POD_NAME", "sf-0")
    merged = _resource_attributes(
        "team=preset", {"team": "a", "tier": "x y,z", "k8s.pod.name": "pinned"}
    )
    pairs = dict(p.split("=", 1) for p in merged.split(","))
    assert pairs == {
        "team": "preset",
        "k8s.namespace.name": "tools",
        "k8s.pod.name": "pinned",
        "tier": "x%20y%2Cz",
    }


def test_a_tag_key_the_sdk_would_split_is_left_off(monkeypatch, caplog) -> None:
    from kubed.mcp._otel import _resource_attributes

    tags = {
        "a,team=x": "1",
        "b=c": "2",
        " team ": "3",
        " tier ": "4",
        "": "5",
        "  ": "6",
    }
    merged = _resource_attributes("team=preset", tags)
    # A key with spaces is the stripped key: it loses to one already there.
    assert merged == "team=preset,tier=4"
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (
            logging.WARNING,
            "Tag left off traces: a resource attribute key can't hold ',' or '=': "
            + repr(key),
        )
        for key in ("a,team=x", "b=c", "", "  ")
    ]


def test_resource_attributes_add_nothing_when_there_is_nothing(monkeypatch) -> None:
    from kubed.mcp._otel import _resource_attributes, _set_defaults

    monkeypatch.setenv("POD_NAMESPACE", "")
    assert _resource_attributes("", {}) == ""
    _set_defaults("named", {})
    assert "OTEL_RESOURCE_ATTRIBUTES" not in os.environ


def test_defaults_fill_only_what_is_unset(monkeypatch) -> None:
    from kubed.mcp._otel import _set_defaults

    defaults = {
        "OTEL_SERVICE_NAME": "named",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_LOGS_EXPORTER": "none",
        "OTEL_PYTHON_LOG_CORRELATION": "true",
    }
    for variable in defaults:
        monkeypatch.delenv(variable, raising=False)
    _set_defaults("named", {})
    assert {v: os.environ[v] for v in defaults} == defaults
    preset = {
        "OTEL_SERVICE_NAME": "mine",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_PYTHON_LOG_CORRELATION": "false",
    }
    for variable, value in preset.items():
        monkeypatch.setenv(variable, value)
    _set_defaults("named", {})
    assert {v: os.environ[v] for v in defaults} == preset
    # Set but empty counts as unset, as _service_name reads it for profiles.
    for variable in defaults:
        monkeypatch.setenv(variable, "")
    _set_defaults("named", {})
    assert {v: os.environ[v] for v in defaults} == defaults


def test_the_real_configuration_exports_a_named_tagged_span() -> None:
    out = _child(
        "console",
        POD_NAMESPACE="tools",
        POD_NAME="sf-0",
        OTEL_RESOURCE_ATTRIBUTES="team=preset",
    )
    server = [
        o for o in _json_objects(out.stdout) if o.get("kind") == "SpanKind.SERVER"
    ]
    # FastMCP's server/discover and tools/list are SERVER spans too.
    calls = sorted(o["name"] for o in server if o["name"].startswith("tools/call"))
    assert calls == ["tools/call log", "tools/call ping"]
    # Only the first server traced: the second ran without OpenTelemetry.
    assert {o["resource"]["attributes"]["service.name"] for o in server} == {"traced"}
    attributes = server[0]["resource"]["attributes"]
    assert attributes["service.name"] == "traced"
    assert (attributes["team"], attributes["tier"]) == ("preset", "x y,z")
    assert (attributes["k8s.namespace.name"], attributes["k8s.pod.name"]) == (
        "tools",
        "sf-0",
    )
    # TEL-15 on the initialize() path: the record carries the server span's trace.
    log_span = next(o for o in server if o["name"] == "tools/call log")
    trace_id = log_span["context"]["trace_id"].removeprefix("0x")
    assert re.search(
        rf"^inside trace_id={trace_id} span_id=[0-9a-f]{{16}} trace={trace_id}$",
        out.stderr,
        re.M,
    )
    # The second server, otel on after the first stopped: served, and said why not.
    assert "second served pong" in out.stderr
    assert out.stderr.count("OpenTelemetry runs once per process") == 1


@pytest.mark.parametrize("mode", ["bare", "bare-configured"])
def test_otel_leaves_a_bare_root_logger_bare(mode) -> None:
    out = _child(mode)
    lines = [line for line in out.stdout.splitlines() if line.startswith("root=")]
    assert len(lines) == 2  # inside the tool, and after the server stopped
    for line in lines:
        seen = json.loads(line.removeprefix("root="))
        # OTEL_LOGS_EXPORTER defaults to none: OpenTelemetry's handler would drop records.
        assert seen["handlers"] == []
        assert seen["level"] == logging.WARNING
    # So logging.lastResort still prints the tool's WARNING.
    assert any(
        re.fullmatch(r"inside trace_id=[0-9a-f]{32} span_id=[0-9a-f]{16}", line)
        for line in out.stderr.splitlines()
    )
    if mode == "bare-configured":
        # The child's own provider traced: initialize() never replaced it.
        server = [
            o for o in _json_objects(out.stdout) if o.get("kind") == "SpanKind.SERVER"
        ]
        assert "tools/call log" in [o["name"] for o in server]
        assert {o["resource"]["attributes"]["service.name"] for o in server} == {
            "preset"
        }


def test_otel_off_imports_no_sdk() -> None:
    out = _child("off")
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def _raise_secret(*args, **kwargs) -> None:
    raise RuntimeError("https://user:s3cret@otlp.example.com")


@pytest.mark.real_initialize
@pytest.mark.parametrize("failing", ["distro", "configurator"])
async def test_a_failed_configure_logs_no_traceback_from_any_logger(
    monkeypatch, caplog, failing
) -> None:
    import opentelemetry.distro
    import opentelemetry.instrumentation.auto_instrumentation as auto

    import kubed.mcp._otel

    # The initialize() path, whatever a test before set as the global provider.
    monkeypatch.setattr(kubed.mcp._otel, "_sdk_configured", lambda: False)
    loggers = ["opentelemetry.instrumentation.auto_instrumentation"]
    if failing == "distro":
        # _initialize() logs it, with its traceback.
        distro = types.SimpleNamespace(configure=_raise_secret)
        monkeypatch.setattr(auto, "_load_distro", lambda: distro)
    else:
        # _load.py logs it, then _initialize() again, each with its traceback.
        monkeypatch.setattr(
            opentelemetry.distro.OpenTelemetryConfigurator, "_configure", _raise_secret
        )
        loggers.insert(0, "opentelemetry.instrumentation.auto_instrumentation._load")
    async with Client(served("failing", TelemetrySettings(otel=True)).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    # The filter keeps upstream's records, and drops only their tracebacks.
    upstream = [r.name for r in caplog.records if r.name.startswith("opentelemetry")]
    assert upstream == loggers
    assert_safe(caplog.records, "s3cret")
    assert "s3cret" not in caplog.text
    ours = [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.telemetry" or r.name.startswith("kubed.mcp._otel")
    ]
    assert [r.levelno for r in ours] == [logging.ERROR]
    assert "RuntimeError" in ours[0].getMessage()
    assert kubed.mcp._otel._stopped is False


async def test_otel_missing_is_one_warning_and_the_server_serves(
    monkeypatch, caplog
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    monkeypatch.delitem(sys.modules, "kubed.mcp._otel", raising=False)
    monkeypatch.delattr(kubed.mcp, "_otel", raising=False)
    async with Client(served("bare", TelemetrySettings(otel=True)).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "kubed-mcp[telemetry]" in records[0].getMessage()


async def test_distro_missing_is_one_warning_and_the_server_serves(
    monkeypatch, caplog
) -> None:
    import opentelemetry.instrumentation.auto_instrumentation as auto

    # Uninstalled, initialize() falls back to a distro that configures nothing.
    calls: list[dict] = []
    monkeypatch.setattr(auto, "initialize", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "opentelemetry.distro", None)
    monkeypatch.delitem(sys.modules, "kubed.mcp._otel", raising=False)
    monkeypatch.delattr(kubed.mcp, "_otel", raising=False)
    async with Client(served("bare", TelemetrySettings(otel=True)).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "kubed-mcp[telemetry]" in records[0].getMessage()
    assert calls == []


def test_the_sdk_handler_gone_still_imports(monkeypatch) -> None:
    import opentelemetry.sdk._logs
    from opentelemetry.instrumentation.logging.handler import LoggingHandler

    # Deprecated in the SDK, and the extra allows any opentelemetry-sdk<2.
    monkeypatch.delattr(opentelemetry.sdk._logs, "LoggingHandler")
    monkeypatch.delitem(sys.modules, "kubed.mcp._otel", raising=False)
    monkeypatch.delattr(kubed.mcp, "_otel", raising=False)
    otel = importlib.import_module("kubed.mcp._otel")
    assert (LoggingHandler,) == otel._OTEL_HANDLERS


async def test_otel_after_the_owner_stopped_is_one_warning(monkeypatch, caplog) -> None:
    import kubed.mcp._otel

    monkeypatch.setattr(kubed.mcp._otel, "_stopped", True)
    async with Client(served("late", TelemetrySettings(otel=True)).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "once per process" in records[0].getMessage()


def test_a_provider_that_hangs_on_shutdown_is_left_behind(monkeypatch, caplog) -> None:
    from opentelemetry import _logs, metrics, trace

    import kubed.mcp._otel
    from kubed.mcp._otel import _Otel

    release = threading.Event()
    stopped: list[object] = []

    class Hanging:
        def shutdown(self) -> None:
            release.wait(10)

    class Meter:
        def shutdown(self, timeout_millis: float = 30_000) -> None:
            stopped.append(timeout_millis)

    class Logger:
        def shutdown(self) -> None:
            stopped.append("logger")

    monkeypatch.setattr(kubed.mcp._otel, "_SHUTDOWN_SECONDS", 0.1)
    monkeypatch.setattr(trace, "get_tracer_provider", Hanging)
    monkeypatch.setattr(metrics, "get_meter_provider", Meter)
    monkeypatch.setattr(_logs, "get_logger_provider", Logger)
    began = time.monotonic()
    try:
        _Otel(owned=EVERY_KIND, logging_instrumented=False).stop()
        assert time.monotonic() - began < 2
    finally:
        release.set()
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.WARNING, "OpenTelemetry did not stop in time: tracer")
    ]
    assert stopped == [100, "logger"]
    assert kubed.mcp._otel._stopped is True


def test_a_provider_that_fails_to_stop_is_logged_by_type(monkeypatch, caplog) -> None:
    from opentelemetry import _logs, metrics, trace

    import kubed.mcp._otel
    from kubed.mcp._otel import _Otel

    class Failing:
        def shutdown(self, **kwargs) -> None:
            raise RuntimeError("https://user:s3cret@otlp.example.com")

    for module, getter in (
        (trace, "get_tracer_provider"),
        (metrics, "get_meter_provider"),
        (_logs, "get_logger_provider"),
    ):
        monkeypatch.setattr(module, getter, Failing)
    _Otel(owned=EVERY_KIND, logging_instrumented=False).stop()
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [r.getMessage() for r in records] == [
        "OpenTelemetry did not stop cleanly: RuntimeError"
    ] * 3
    assert_safe(caplog.records, "s3cret")
    assert kubed.mcp._otel._stopped is True


def test_every_package_the_extra_names_is_installed() -> None:
    # A real import: CI installs the extra, so there it must not skip.
    def real(name: str) -> types.ModuleType:
        if os.environ.get("CI"):
            return importlib.import_module(name)
        return pytest.importorskip(name)

    auto = real("opentelemetry.instrumentation.auto_instrumentation")
    assert "swallow_exceptions" in inspect.signature(auto.initialize).parameters
    real("opentelemetry.distro")
    real("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    real("opentelemetry.instrumentation.logging")
    sdk_trace = real("opentelemetry.sdk.trace")
    pyroscope_otel = real("pyroscope.otel")
    assert issubclass(pyroscope_otel.PyroscopeSpanProcessor, sdk_trace.SpanProcessor)
    pyroscope = real("pyroscope")
    for tag in (pyroscope.add_thread_tag, pyroscope.remove_thread_tag):
        assert list(inspect.signature(tag).parameters) == ["key", "value"]
    prometheus = real("opentelemetry.exporter.prometheus")
    assert hasattr(prometheus, "PrometheusMetricReader")
    real("prometheus_client")


def test_prometheus_serves_the_call_metrics_on_a_port_of_its_own() -> None:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    out = _child(
        "prometheus",
        OTEL_METRICS_EXPORTER="prometheus",
        OTEL_EXPORTER_PROMETHEUS_PORT=str(port),
    )
    lines = out.stdout.splitlines()
    # real initialize(), after the middleware was built in __init__: the proxy meter rebound
    assert any(
        line.startswith("mcp_server_operation_duration_seconds_count{")
        and 'gen_ai_tool_name="ping"' in line
        and line.endswith(" 1.0")
        for line in lines
    )
    assert any(
        line.startswith("tests_own_calls_total") and line.endswith(" 1.0")
        for line in lines
    )


def _fake_providers(monkeypatch) -> list[str]:
    """Fake global tracer, meter and logger providers; the list of those shut down."""
    from opentelemetry import _logs, metrics, trace

    stopped: list[str] = []

    def provider(kind: str) -> object:
        def shutdown(**kwargs) -> None:
            stopped.append(kind)

        return types.SimpleNamespace(shutdown=shutdown)

    for module, getter, kind in (
        (trace, "get_tracer_provider", "tracer"),
        (metrics, "get_meter_provider", "meter"),
        (_logs, "get_logger_provider", "logger"),
    ):
        monkeypatch.setattr(module, getter, lambda kind=kind: provider(kind))
    return stopped


def _stand_in_providers(
    monkeypatch, kept: frozenset[str] = frozenset()
) -> tuple[list[str], types.FunctionType]:
    """Global providers that count their shutdowns, and what sets them.

    The kinds in ``kept`` are set from the start. The rest are OpenTelemetry's proxies
    until the function returned is called, as initialize() sets them. The list is the
    kinds shut down, in order.
    """
    from opentelemetry import _logs, metrics, trace
    from opentelemetry._logs._internal import ProxyLoggerProvider
    from opentelemetry.metrics._internal import _ProxyMeterProvider

    stopped: list[str] = []

    class Provider:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def shutdown(self, **kwargs) -> None:
            stopped.append(self.kind)

    proxies = {
        "tracer": trace.ProxyTracerProvider(),
        "meter": _ProxyMeterProvider(),
        "logger": ProxyLoggerProvider(),
    }
    current = {
        kind: Provider(kind) if kind in kept else proxy
        for kind, proxy in proxies.items()
    }

    def install() -> None:
        for kind in EVERY_KIND - kept:
            current[kind] = Provider(kind)

    for module, getter, kind in (
        (trace, "get_tracer_provider", "tracer"),
        (metrics, "get_meter_provider", "meter"),
        (_logs, "get_logger_provider", "logger"),
    ):
        monkeypatch.setattr(module, getter, lambda kind=kind: current[kind])
    return stopped, install


def test_a_configure_that_fails_halfway_shuts_its_providers_down(
    monkeypatch, caplog
) -> None:
    import kubed.mcp._otel

    # Unset before initialize(), set by it before it raised.
    stopped, install = _stand_in_providers(monkeypatch)

    def initialize(**kwargs) -> None:
        install()
        _raise_secret()

    # Replaces the guard's recorder: this test means initialize() to run, and raise.
    monkeypatch.setattr(kubed.mcp._otel, "initialize", initialize)
    otel = kubed.mcp._otel.start("half", TelemetrySettings(otel=True), None)
    assert otel is None
    assert stopped == ["tracer", "meter", "logger"]
    assert kubed.mcp._otel._stopped is True
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.ERROR, "OpenTelemetry did not start; serving without it: RuntimeError")
    ]
    assert_safe(caplog.records, "s3cret")


def test_a_meter_provider_already_set_is_kept_and_never_owned(monkeypatch) -> None:
    import kubed.mcp._otel

    stopped, install = _stand_in_providers(monkeypatch, kept=frozenset({"meter"}))
    initialized: list[dict] = []

    def initialize(**kwargs) -> None:
        initialized.append(kwargs)
        install()

    monkeypatch.setattr(kubed.mcp._otel, "initialize", initialize)
    otel = kubed.mcp._otel.start("kept", TelemetrySettings(otel=True), None)
    # No tracer provider was set, so initialize() ran: it set the tracer and the logger.
    assert initialized == [{"swallow_exceptions": False}]
    assert otel._owned == frozenset({"tracer", "logger"})
    otel.stop()
    assert stopped == ["tracer", "logger"]
    assert kubed.mcp._otel._stopped is True


def test_a_configure_that_fails_halfway_keeps_a_provider_already_set(
    monkeypatch,
) -> None:
    import kubed.mcp._otel

    stopped, install = _stand_in_providers(monkeypatch, kept=frozenset({"meter"}))

    def initialize(**kwargs) -> None:
        install()
        _raise_secret()

    monkeypatch.setattr(kubed.mcp._otel, "initialize", initialize)
    assert kubed.mcp._otel.start("half", TelemetrySettings(otel=True), None) is None
    assert stopped == ["tracer", "logger"]


@pytest.mark.parametrize("exporter", [None, "prometheus"])
def test_a_kept_providers_exporter_is_off_for_initialize_and_put_back(
    monkeypatch, exporter
) -> None:
    import kubed.mcp._otel

    if exporter is not None:
        monkeypatch.setenv("OTEL_METRICS_EXPORTER", exporter)
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    _, install = _stand_in_providers(monkeypatch, kept=frozenset({"meter"}))
    seen: list[tuple[str | None, str | None]] = []

    def initialize(**kwargs) -> None:
        seen.append(
            (
                os.environ.get("OTEL_METRICS_EXPORTER"),
                os.environ.get("OTEL_TRACES_EXPORTER"),
            )
        )
        install()

    monkeypatch.setattr(kubed.mcp._otel, "initialize", initialize)
    otel = kubed.mcp._otel.start("kept", TelemetrySettings(otel=True), None)
    assert otel is not None
    # The kept meter's exporter is off; the tracer, which initialize() sets, keeps its own.
    assert seen == [("none", "console")]
    assert os.environ.get("OTEL_METRICS_EXPORTER") == exporter
    assert os.environ["OTEL_TRACES_EXPORTER"] == "console"


def test_a_meter_provider_the_process_set_is_kept_and_records_the_calls() -> None:
    out = _child("own_meter")
    result = json.loads(out.stdout.splitlines()[-1])
    assert result == {"kept": True, "points": 1, "tracer_is_sdk": True}


def test_an_initialize_that_sets_no_provider_owns_nothing(monkeypatch) -> None:
    import kubed.mcp._otel

    monkeypatch.setattr(kubed.mcp._otel, "_sdk_configured", lambda: False)
    monkeypatch.setattr(kubed.mcp._otel, "initialize", lambda **kwargs: None)
    otel = kubed.mcp._otel.start("empty", TelemetrySettings(otel=True), None)
    assert otel is not None
    stopped = _fake_providers(monkeypatch)
    otel.stop()
    # Nothing of ours to shut down, so a later server may still configure it.
    assert stopped == []
    assert kubed.mcp._otel._stopped is False


# An SDK someone configured before the server started, as opentelemetry-instrument does.
# A tracer provider is set once per process, so from the first test that asks for it,
# it stays set for the rest of the session: tests of the initialize() path patch
# _sdk_configured rather than rely on an unconfigured process.


@pytest.fixture(scope="session")
def traced() -> Iterator[tuple[object, object]]:
    """A tracer provider set globally, and the exporter its spans go to."""
    from opentelemetry import trace
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    exporter = InMemorySpanExporter()
    # Set up before _clean runs: the shell's OTEL_* (a sampler, OTEL_SDK_DISABLED) go.
    with pytest.MonkeyPatch.context() as env:
        for name in [n for n in os.environ if n.startswith("OTEL_")]:
            env.delenv(name)
        provider = TracerProvider(
            sampler=ALWAYS_ON,
            resource=Resource.create({"service.name": "preset"}),
        )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    # set_tracer_provider only warns when one was set before: it must be this one.
    assert trace.get_tracer_provider() is provider
    yield provider, exporter
    if LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().uninstrument()


@pytest.fixture
def spans(traced) -> object:
    """The session's exporter, emptied of the spans earlier tests left in it."""
    _, exporter = traced
    exporter.clear()
    return exporter


def server_spans(spans) -> list[str]:
    """The names of the finished tool-call spans the server side opened."""
    from opentelemetry.trace import SpanKind

    # FastMCP's server/discover and tools/list are SERVER spans too.
    return [
        s.name
        for s in spans.get_finished_spans()
        if s.kind is SpanKind.SERVER and s.name.startswith("tools/call")
    ]


async def test_a_configured_sdk_keeps_its_providers(traced, spans) -> None:
    from opentelemetry import trace

    import kubed.mcp._otel

    provider, _ = traced
    # The guard fails this test if initialize() runs: a configured SDK is kept as it is.
    async with Client(served("kept", TelemetrySettings(otel=True)).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    assert trace.get_tracer_provider() is provider
    assert server_spans(spans) == ["tools/call ping"]
    # Never shut down: the provider still exports after the server stopped.
    with provider.get_tracer("t").start_as_current_span("after"):
        pass
    assert "after" in [s.name for s in spans.get_finished_spans()]
    assert kubed.mcp._otel._stopped is False


class Keep(logging.Handler):
    """Keeps every record it handles."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def test_log_records_carry_the_trace_and_root_keeps_its_handlers(
    traced, spans
) -> None:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.trace import SpanKind

    from kubed.mcp._otel import _OTEL_HANDLERS

    # Else the start would find logging instrumented, and correctly leave it be.
    assert not LoggingInstrumentor().is_instrumented_by_opentelemetry
    root = logging.getLogger()
    factory, keep = logging.getLogRecordFactory(), Keep()
    inside: list[list[logging.Handler]] = []
    mcp = served("correlated", TelemetrySettings(otel=True))

    @mcp.tool
    def log() -> str:
        inside.append(list(root.handlers))
        logging.getLogger("tests.tool").warning("inside")
        return "logged"

    root.addHandler(keep)
    try:
        async with Client(mcp.server) as client:
            assert (await client.call_tool("log", {})).data == "logged"
        after = list(root.handlers)
    finally:
        root.removeHandler(keep)
        if LoggingInstrumentor().is_instrumented_by_opentelemetry:
            LoggingInstrumentor().uninstrument()
            pytest.fail("the server's stop left logging instrumented")
    calls = [
        s
        for s in spans.get_finished_spans()
        if s.kind is SpanKind.SERVER and s.name == "tools/call log"
    ]
    records = [r for r in keep.records if r.msg == "inside"]
    assert len(calls) == 1
    assert len(records) == 1
    assert records[0].otelTraceID == format(calls[0].context.trace_id, "032x")
    assert records[0].otelSpanID != "0"
    assert records[0].otelServiceName == "preset"
    # The server's handler stays, and the instrumentor's own is gone again. Under pytest
    # root has handlers, so basicConfig does nothing here and root's level and format
    # can't change: test_otel_leaves_a_bare_root_logger_bare pins those.
    for handlers in (inside[0], after):
        assert keep in handlers
        assert not any(isinstance(h, _OTEL_HANDLERS) for h in handlers)
    # Uninstrumented at stop: records are made as they were before the server.
    assert logging.getLogRecordFactory() is factory


async def test_logging_someone_else_instrumented_stays_instrumented(
    traced, spans
) -> None:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    # As opentelemetry-instrument leaves it: correlation on, before the server starts.
    instrumentor = LoggingInstrumentor()
    instrumentor.instrument(
        inject_trace_context=True, enable_log_auto_instrumentation=False
    )
    try:
        factory = logging.getLogRecordFactory()
        async with Client(
            served("borrowed", TelemetrySettings(otel=True)).server
        ) as client:
            assert (await client.call_tool("ping", {})).data == "pong"
        # Not ours, so our stop left it as it was.
        assert instrumentor.is_instrumented_by_opentelemetry
        assert logging.getLogRecordFactory() is factory
    finally:
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


async def test_logging_that_fails_to_instrument_still_keeps_the_trace(
    traced, spans, monkeypatch, caplog
) -> None:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    import kubed.mcp.telemetry

    def refuse(self, **kwargs) -> None:
        raise RuntimeError("https://user:s3cret@otlp.example.com")

    monkeypatch.setattr(LoggingInstrumentor, "instrument", refuse)
    handles: list[object] = []
    mcp = served("unlogged", TelemetrySettings(otel=True))

    @mcp.tool
    def held() -> str:
        running = kubed.mcp.telemetry._running
        handles.append(running and running._otel)
        return "held"

    async with Client(mcp.server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
        assert (await client.call_tool("held", {})).data == "held"
    # start returned a handle: only the records go without the trace.
    assert handles and handles[0] is not None
    assert server_spans(spans) == ["tools/call ping", "tools/call held"]
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.ERROR, "OpenTelemetry did not instrument logging: RuntimeError")
    ]
    assert_safe(caplog.records, "s3cret")
    assert not LoggingInstrumentor().is_instrumented_by_opentelemetry


# -- trace ids in the line (TEL-18) -------------------------------------------


class Lines(logging.Handler):
    """Keeps each record as its formatter writes it."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


class as_json(logging.Formatter):
    """A formatter that reads ``getMessage()``, as a JSON log formatter does."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"level": record.levelname, "message": record.getMessage()})


def iter_queue(source: queue.SimpleQueue) -> Iterator[logging.LogRecord]:
    while True:
        try:
            yield source.get_nowait()
        except queue.Empty:
            return


async def test_a_line_inside_a_span_ends_with_its_ids(traced, spans) -> None:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.trace import SpanKind

    text, data, keep = Lines(), Lines(), Keep()
    text.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    data.setFormatter(as_json())
    mcp = served("ids", TelemetrySettings(otel=True))

    @mcp.tool
    def log() -> str:
        logging.getLogger("tests.tool").warning("value %s", 5)
        return "logged"

    root = logging.getLogger()
    for handler in (text, data, keep):
        root.addHandler(handler)
    try:
        async with Client(mcp.server) as client:
            await client.call_tool("log", {})
            logging.getLogger("tests.tool").warning("outside")
    finally:
        for handler in (text, data, keep):
            root.removeHandler(handler)
        if LoggingInstrumentor().is_instrumented_by_opentelemetry:
            LoggingInstrumentor().uninstrument()
    (record,) = [r for r in keep.records if r.msg == "value %s"]
    ids = f" trace_id={record.otelTraceID} span_id={record.otelSpanID}"
    assert re.fullmatch(r" trace_id=[0-9a-f]{32} span_id=[0-9a-f]{16}", ids)
    (call,) = [
        s
        for s in spans.get_finished_spans()
        if s.kind is SpanKind.SERVER and s.name == "tools/call log"
    ]
    assert record.otelTraceID == format(call.context.trace_id, "032x")
    assert f"WARNING value 5{ids}" in text.lines
    assert "WARNING outside" in text.lines
    messages = [json.loads(x) for x in data.lines]
    assert {"level": "WARNING", "message": f"value 5{ids}"} in messages
    assert {"level": "WARNING", "message": "outside"} in messages
    assert (record.msg, record.args) == ("value %s", (5,))


async def test_a_queue_handler_shows_the_ids_once(traced, spans) -> None:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    queued: queue.SimpleQueue = queue.SimpleQueue()
    handler = logging.handlers.QueueHandler(queued)
    mcp = served("queued", TelemetrySettings(otel=True))

    @mcp.tool
    def log() -> str:
        logging.getLogger("tests.tool").warning("queued %s", 1)
        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("tests.tool").exception("failed %s", 2)
        return "logged"

    root = logging.getLogger()
    root.addHandler(handler)
    try:
        async with Client(mcp.server) as client:
            await client.call_tool("log", {})
    finally:
        root.removeHandler(handler)
        if LoggingInstrumentor().is_instrumented_by_opentelemetry:
            LoggingInstrumentor().uninstrument()
    recorded = list(iter_queue(queued))
    records = [r for r in recorded if r.getMessage().startswith("queued 1")]
    # As a QueueListener's handler does.
    formatted = logging.Formatter("%(message)s").format(records[0])
    assert formatted.count(" trace_id=") == 1
    assert re.fullmatch(
        r"queued 1 trace_id=[0-9a-f]{32} span_id=[0-9a-f]{16}", formatted
    )
    # The queued msg holds the ids, a newline and the traceback: still once.
    (failed,) = [r for r in recorded if r.getMessage().startswith("failed 2")]
    text = logging.Formatter("%(message)s").format(failed)
    assert text.count(" trace_id=") == 1
    assert "ValueError" in text


def test_remove_puts_the_previous_factory_back(traced) -> None:
    before = logging.getLogRecordFactory()
    text = _TraceText()
    text.install()
    try:
        assert logging.getLogRecordFactory() is not before
    finally:
        text.remove()
    assert logging.getLogRecordFactory() is before


def test_remove_after_another_factory_leaves_it_and_goes_quiet(traced) -> None:
    provider, _ = traced
    before = logging.getLogRecordFactory()
    text = _TraceText()
    text.install()
    ours = logging.getLogRecordFactory()
    other = lambda *a, **kw: ours(*a, **kw)  # noqa: E731
    logging.setLogRecordFactory(other)
    try:
        text.remove()
        assert logging.getLogRecordFactory() is other
        with provider.get_tracer("t").start_as_current_span("quiet"):
            record = logging.getLogger("tests").makeRecord(
                "tests", logging.INFO, "f", 1, "m", (), None
            )
        assert record.getMessage() == "m"
    finally:
        logging.setLogRecordFactory(before)


class Special(logging.LogRecord):
    """A record class some other library makes."""


def test_a_record_of_another_class_is_left_alone(traced) -> None:
    provider, _ = traced
    before = logging.getLogRecordFactory()
    logging.setLogRecordFactory(Special)
    text = _TraceText()
    text.install()
    try:
        with provider.get_tracer("t").start_as_current_span("special"):
            record = logging.getLogger("tests").makeRecord(
                "tests", logging.INFO, "f", 1, "m", (), None
            )
        assert type(record) is Special
        assert record.getMessage() == "m"
    finally:
        text.remove()
        logging.setLogRecordFactory(before)


async def test_otel_off_installs_no_factory(traced) -> None:
    provider, _ = traced
    before = logging.getLogRecordFactory()
    seen: list[object] = []
    mcp = served("no-ids", TelemetrySettings())

    @mcp.tool
    def look() -> str:
        seen.append(logging.getLogRecordFactory())
        with provider.get_tracer("t").start_as_current_span("own"):
            made = logging.getLogRecordFactory()(
                "tests", logging.INFO, "f", 1, "m", (), None
            )
            seen.append(made.getMessage())
        return "ok"

    async with Client(mcp.server) as client:
        await client.call_tool("look", {})
    assert seen == [before, "m"]


# Span profiles (TEL-14): with both signals on, each root span tags the profile samples
# taken while it runs. The SDK can't remove a processor, so every link a test adds stays
# on the session's provider, closed: none of these tests may see one an earlier test left.


@pytest.fixture
def span_profiles(pyroscope, monkeypatch) -> types.ModuleType:
    """A fake pyroscope.otel, whose processor records each root span's start and end."""
    from opentelemetry.sdk.trace import SpanProcessor

    module = types.ModuleType("pyroscope.otel")

    class PyroscopeSpanProcessor(SpanProcessor):
        # The real one's test: only a root span tags the thread. Recording every span
        # would hide that in-process the root is the client's span, not the server's.
        @staticmethod
        def _root(span) -> bool:
            return span.parent is None or span.parent.is_remote

        def on_start(self, span, parent_context=None) -> None:
            if self._root(span):
                EVENTS.append(("start", span.name))

        def on_end(self, span) -> None:
            if self._root(span):
                EVENTS.append(("end", span.name))

    module.PyroscopeSpanProcessor = PyroscopeSpanProcessor
    monkeypatch.setitem(sys.modules, "pyroscope.otel", module)
    return module


def _added(monkeypatch, provider) -> list[object]:
    """Each span processor added to ``provider`` from now on, then added as asked."""
    added: list[object] = []
    add = provider.add_span_processor

    def record(processor) -> None:
        added.append(processor)
        add(processor)

    monkeypatch.setattr(provider, "add_span_processor", record)
    return added


async def test_profiles_and_otel_link_each_root_span(
    pyroscope, span_profiles, traced, spans, monkeypatch
) -> None:
    from opentelemetry.context import Context

    provider, _ = traced
    profiles = sys.modules["pyroscope"]
    shutdown = profiles.shutdown

    def traced_shutdown() -> None:
        # A root span while profiles stop: OpenTelemetry stopped first, so it's untagged.
        provider.get_tracer("t").start_span(
            "profiles stopping", context=Context()
        ).end()
        shutdown()

    monkeypatch.setattr(profiles, "shutdown", traced_shutdown)
    settings = TelemetrySettings(profiles=ADDRESS, otel=True)
    async with Client(served("linked", settings).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    # In-process the root is the client's tools/call span, the server's is its child:
    # behind a real transport the server's span has a remote parent and is the root.
    assert ("start", "tools/call ping") in EVENTS
    assert ("end", "tools/call ping") in EVENTS
    assert ("start", "tools/call") not in EVENTS
    assert "profiles stopped" in EVENTS
    assert "profiles stopping" in [s.name for s in spans.get_finished_spans()]
    assert not [
        e for e in EVENTS if isinstance(e, tuple) and e[1] == "profiles stopping"
    ]
    assert server_spans(spans) == ["tools/call ping"]


async def test_the_link_goes_quiet_when_the_signals_stop(
    pyroscope, span_profiles, traced, spans, monkeypatch
) -> None:
    from kubed.mcp._otel import _ProfileLink

    provider, _ = traced
    added = _added(monkeypatch, provider)
    settings = TelemetrySettings(profiles=ADDRESS, otel=True)
    async with Client(served("quiet", settings).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    assert ("start", "tools/call ping") in EVENTS
    EVENTS.clear()
    # The provider isn't ours, so it serves on, and can't let go of the link.
    assert [type(p) for p in added] == [_ProfileLink]
    with provider.get_tracer("t").start_as_current_span("after"):
        pass
    assert not [e for e in EVENTS if isinstance(e, tuple)]
    assert "after" in [s.name for s in spans.get_finished_spans()]


async def test_a_failing_link_never_fails_a_call(
    pyroscope, span_profiles, traced, spans, monkeypatch, caplog
) -> None:
    def refuse(self, span, parent_context=None) -> None:
        raise RuntimeError("https://user:s3cret@pyroscope.example.com")

    monkeypatch.setattr(span_profiles.PyroscopeSpanProcessor, "on_start", refuse)
    settings = TelemetrySettings(profiles=ADDRESS, otel=True)
    async with Client(served("failing-link", settings).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
        assert (await client.call_tool("ping", {})).data == "pong"
    assert server_spans(spans) == ["tools/call ping", "tools/call ping"]
    assert_safe(caplog.records, "s3cret")
    # Logged once, though every root span failed.
    assert [
        r.getMessage() for r in caplog.records if "Span profiles" in r.getMessage()
    ] == ["Span profiles failed: RuntimeError"]
    # Still delegating after the first failure: the end of each root span was seen.
    assert ("end", "tools/call ping") in EVENTS


async def test_a_link_that_fails_to_be_added_is_logged_and_otel_runs(
    pyroscope, span_profiles, traced, spans, monkeypatch, caplog
) -> None:
    def refuse(self) -> None:
        raise RuntimeError("https://user:s3cret@pyroscope.example.com")

    monkeypatch.setattr(span_profiles.PyroscopeSpanProcessor, "__init__", refuse)
    settings = TelemetrySettings(profiles=ADDRESS, otel=True)
    async with Client(served("unlinked", settings).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    assert server_spans(spans) == ["tools/call ping"]
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.ERROR, "Span profiles did not start: RuntimeError")
    ]
    assert_safe(caplog.records, "s3cret")


async def test_otel_without_profiles_adds_no_link(traced, spans, monkeypatch) -> None:
    provider, _ = traced
    added = _added(monkeypatch, provider)
    monkeypatch.delitem(sys.modules, "pyroscope.otel", raising=False)
    async with Client(
        served("unprofiled", TelemetrySettings(otel=True)).server
    ) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    assert "pyroscope.otel" not in sys.modules
    assert added == []


async def test_pyroscope_otel_missing_is_one_warning(
    pyroscope, traced, spans, monkeypatch, caplog
) -> None:
    provider, _ = traced
    added = _added(monkeypatch, provider)
    monkeypatch.setitem(sys.modules, "pyroscope.otel", None)
    settings = TelemetrySettings(profiles=ADDRESS, otel=True)
    async with Client(served("half-linked", settings).server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == [
        "Span profiles are on, but pyroscope-otel is not installed: "
        "install kubed-mcp[telemetry]"
    ]
    assert server_spans(spans) == ["tools/call ping"]
    assert added == []


def test_the_initialize_path_links_the_provider_it_set(
    pyroscope, span_profiles, monkeypatch
) -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    import kubed.mcp._otel
    from kubed.mcp._otel import _ProfileLink

    configured = TracerProvider()
    added = _added(monkeypatch, configured)
    initialized: list[dict] = []
    unset = trace.ProxyTracerProvider()
    # Unconfigured until initialize() runs, which sets this provider: a link added any
    # earlier would find no SDK provider to add to.
    monkeypatch.setattr(
        kubed.mcp._otel, "initialize", lambda **kwargs: initialized.append(kwargs)
    )
    monkeypatch.setattr(
        trace, "get_tracer_provider", lambda: configured if initialized else unset
    )
    settings = TelemetrySettings(profiles=ADDRESS, otel=True)
    otel = kubed.mcp._otel.start("initialized", settings, sys.modules["pyroscope"])
    assert otel is not None
    assert initialized == [{"swallow_exceptions": False}]
    assert [type(p) for p in added] == [_ProfileLink]
    with configured.get_tracer("t").start_as_current_span("during"):
        pass
    assert EVENTS == [("start", "during"), ("end", "during")]
    _fake_providers(monkeypatch)
    otel.stop()
    EVENTS.clear()
    with configured.get_tracer("t").start_as_current_span("after"):
        pass
    assert EVENTS == []


@pytest.fixture
def pyroscope_tags(monkeypatch) -> dict[str, str]:
    """The thread's tags, live, as the real pyroscope.otel adds and removes them."""
    pyroscope_otel = importlib.import_module("pyroscope.otel")
    tags: dict[str, str] = {}

    def add(key: str, value: str) -> None:
        tags[key] = value

    def remove(key: str, value: str) -> None:
        # As pyroscope does: only the pair that was added goes.
        if tags.get(key) == value:
            del tags[key]

    # The module object the real processor calls, bound when pyroscope.otel was imported.
    monkeypatch.setattr(pyroscope_otel.pyroscope, "add_thread_tag", add)
    monkeypatch.setattr(pyroscope_otel.pyroscope, "remove_thread_tag", remove)
    return tags


def test_a_renamed_span_leaves_no_tag_behind(pyroscope_tags) -> None:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import TracerProvider
    from pyroscope import otel as pyroscope_otel

    from kubed.mcp._otel import _ProfileLink

    provider = TracerProvider()
    link = _ProfileLink(
        pyroscope_otel.PyroscopeSpanProcessor(), pyroscope_otel.pyroscope
    )
    provider.add_span_processor(link)
    # As FastMCP's server span: started as tools/call, renamed once the tool is known.
    span = provider.get_tracer("t").start_span("tools/call", context=Context())
    assert pyroscope_tags["span_name"] == "tools/call"
    span.update_name("tools/call ping")
    span.end()
    assert pyroscope_tags == {}
    # An unrenamed span is the inner processor's alone to untag.
    span = provider.get_tracer("t").start_span("tools/list", context=Context())
    assert set(pyroscope_tags) == {"span_id", "span_name", "trace_id"}
    span.end()
    assert pyroscope_tags == {}
    link.close()


# -- call metrics -------------------------------------------------------------


@pytest.fixture(scope="session")
def metered(traced) -> object:
    """A meter provider set globally, with one in-memory reader: the call metrics' sink.

    It needs ``traced``: _sdk_configured() reads the tracer provider, so the SDK counts as
    configured and initialize() never runs.
    """
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    # set_meter_provider only warns when one was set before: it must be this one.
    assert metrics.get_meter_provider() is provider
    return reader


def _middleware(mcp: QuickMCP) -> list[type]:
    """The server's middleware by class, less FastMCP's own default (it adds one)."""
    from fastmcp.server.middleware.dereference import DereferenceRefsMiddleware

    return [
        type(m)
        for m in mcp.server.middleware
        if not isinstance(m, DereferenceRefsMiddleware)
    ]


def _metric_points(reader, metric: str) -> list:
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for found in scope.metrics
        if found.name == metric
        for point in found.data.data_points
    ]


def duration_points(reader, server: str) -> list:
    """``mcp.server.operation.duration``'s points recorded by the server named ``server``."""
    return [
        point
        for point in _metric_points(reader, "mcp.server.operation.duration")
        if point.attributes.get("fastmcp.server.name") == server
    ]


def points_of(reader, server: str, method: str) -> list:
    """The server's duration points for ``method`` alone: a client lists and discovers too."""
    return [
        point
        for point in duration_points(reader, server)
        if point.attributes["mcp.method.name"] == method
    ]


async def test_each_tool_call_is_counted_under_the_tools_name(metered) -> None:
    mcp = served("counted", TelemetrySettings(otel=True))
    async with Client(mcp.server) as client:
        await client.call_tool("ping", {})
        await client.call_tool("ping", {})
    points = points_of(metered, "counted", "tools/call")
    assert [(p.attributes["gen_ai.tool.name"], p.count) for p in points] == [
        ("ping", 2)
    ]
    assert points[0].attributes["mcp.method.name"] == "tools/call"
    assert "error.type" not in points[0].attributes
    # user_id="omit": no salt asked for, and no user recorded
    assert not [k for p in points for k in p.attributes if k.startswith("enduser.")]


async def test_a_failed_tool_call_carries_error_type(metered) -> None:
    from fastmcp.exceptions import ToolError

    mcp = served("failing-calls", TelemetrySettings(otel=True))

    @mcp.tool
    def fail() -> str:
        raise ToolError("no")

    async with Client(mcp.server) as client:
        await client.call_tool("fail", {}, raise_on_error=False)
    (point,) = points_of(metered, "failing-calls", "tools/call")
    assert (point.attributes["gen_ai.tool.name"], point.attributes["error.type"]) == (
        "fail",
        "tool_error",
    )


async def test_an_unknown_tool_is_counted_without_its_name(metered) -> None:
    mcp = served("unknown-calls", TelemetrySettings(otel=True))
    async with Client(mcp.server) as client:
        await client.call_tool("nope", {}, raise_on_error=False)
    (point,) = points_of(metered, "unknown-calls", "tools/call")
    assert "gen_ai.tool.name" not in point.attributes
    assert point.attributes["error.type"] == "tool_error"


async def test_a_prompt_render_is_counted_under_the_prompts_name(metered) -> None:
    mcp = served("prompted", TelemetrySettings(otel=True))

    @mcp.prompt
    def greet() -> str:
        return "hello"

    async with Client(mcp.server) as client:
        await client.get_prompt("greet")
    (point,) = points_of(metered, "prompted", "prompts/get")
    assert (
        point.attributes["mcp.method.name"],
        point.attributes["gen_ai.prompt.name"],
    ) == ("prompts/get", "greet")


async def test_a_servers_own_counter_lands_beside_the_call_metrics(metered) -> None:
    from opentelemetry import metrics

    mcp = served("own-counter", TelemetrySettings(otel=True))

    @mcp.tool
    def count() -> str:
        metrics.get_meter("tests").create_counter("tests.own.calls").add(1)
        return "ok"

    async with Client(mcp.server) as client:
        await client.call_tool("count", {})
    (own,) = _metric_points(metered, "tests.own.calls")
    assert own.value == 1
    (point,) = points_of(metered, "own-counter", "tools/call")
    assert point.attributes["gen_ai.tool.name"] == "count"


class Leaky:
    pass


async def test_a_teardown_error_is_recorded_on_the_call(metered) -> None:
    from fastmcp.exceptions import MCPError

    @asynccontextmanager
    async def leaky():
        yield Leaky()
        raise RuntimeError("teardown")

    mcp = served("torn-down", TelemetrySettings(otel=True))
    mcp.transient(Leaky, leaky)

    @mcp.tool
    def leaks(leaky: Inject[Leaky]) -> str:
        return "ok"

    async with Client(mcp.server) as client:
        # The teardown fails outside the tool, so the client gets a protocol error.
        with pytest.raises(MCPError):
            await client.call_tool("leaks", {})
    (point,) = points_of(metered, "torn-down", "tools/call")
    # UsageMiddleware names an error that isn't FastMCP's by its class
    assert (point.attributes["gen_ai.tool.name"], point.attributes["error.type"]) == (
        "leaks",
        "RuntimeError",
    )


def test_call_metrics_are_outermost_with_otel_on() -> None:
    from fastmcp.server.middleware import Middleware
    from fastmcp.server.middleware.usage import UsageMiddleware

    from kubed.mcp._metrics import _CallMetrics, _ReadObserver
    from kubed.mcp.app import _CallScope
    from kubed.mcp.gateway import _TakeThePrefix

    class Given(Middleware):
        pass

    class Added(Middleware):
        pass

    mcp = QuickMCP(
        "ordered",
        config={"telemetry": TelemetrySettings(otel=True)},
        middleware=[Given()],
    )
    mcp.add_middleware(Added())
    assert isinstance(mcp.server.middleware[0], UsageMiddleware)
    kinds = _middleware(mcp)
    # The metrics, then the call's slot: every other layer, the plugin layer's
    # included, runs inside both, in the order it was given.
    assert kinds[:2] == [_CallMetrics, _CallScope]
    assert kinds.count(_CallMetrics) == 1
    assert kinds.index(Given) < kinds.index(Added)
    # The read observer sees the plain URI, under the pin.
    assert kinds[kinds.index(_TakeThePrefix) + 1] is _ReadObserver


def test_otel_off_adds_no_call_metrics() -> None:
    from fastmcp.server.middleware.usage import UsageMiddleware

    from kubed.mcp.app import _CallScope

    mcp = QuickMCP("plain", config={"telemetry": TelemetrySettings()})
    kinds = _middleware(mcp)
    assert UsageMiddleware not in kinds
    assert "_ReadObserver" not in [kind.__name__ for kind in kinds]
    assert kinds[0] is _CallScope


async def test_call_metrics_that_fail_to_build_are_one_error_and_the_server_serves(
    traced, monkeypatch, caplog
) -> None:
    import kubed.mcp._metrics as metrics
    from kubed.mcp.app import _CallScope

    real = metrics._CallMetrics

    def broken(**kwargs):
        raise RuntimeError("https://user:s3cret@otlp.example.com")

    monkeypatch.setattr(metrics, "_CallMetrics", broken)
    mcp = served("unbuilt", TelemetrySettings(otel=True))
    kinds = _middleware(mcp)
    assert real not in kinds
    assert kinds[0] is _CallScope
    async with Client(mcp.server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"
    records = [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (
            logging.ERROR,
            "Call metrics did not start; serving without them: RuntimeError",
        )
    ]
    assert_safe(caplog.records, "s3cret")


# -- subscriptions/listen (TEL-22) --------------------------------------------


def server_span_names(spans) -> list[str]:
    from opentelemetry.trace import SpanKind

    return [s.name for s in spans.get_finished_spans() if s.kind is SpanKind.SERVER]


async def listened_and_called(mcp: QuickMCP) -> None:
    async with Client(mcp.server) as client, client.listen(tools_list_changed=True):
        await client.call_tool("ping", {})


async def test_a_listen_stream_has_no_server_span_or_duration_point(
    metered, spans
) -> None:
    await listened_and_called(served("listened", TelemetrySettings(otel=True)))
    names = server_span_names(spans)
    assert "subscriptions/listen" not in names
    assert "tools/call ping" in names
    methods = [
        p.attributes.get("mcp.method.name")
        for p in duration_points(metered, "listened")
    ]
    # The call metrics never time listen today; this pins it against one that would.
    assert "subscriptions/listen" not in methods
    assert "tools/call" in methods


async def test_a_listen_stream_is_left_out_under_someone_elses_provider(
    traced, spans
) -> None:
    await listened_and_called(served("listened-off", TelemetrySettings()))
    names = server_span_names(spans)
    assert "subscriptions/listen" not in names
    assert "tools/call ping" in names


async def test_the_listen_middleware_stands_ahead_of_fastmcps_seam() -> None:
    from fastmcp.server.low_level import FastMCPServerMiddleware

    from kubed.mcp._listen import _ListenUntraced

    mcp = served("ordered", TelemetrySettings())
    kinds = [type(m) for m in mcp.server._mcp_server.middleware]
    assert kinds[:2] == [_ListenUntraced, FastMCPServerMiddleware]


@pytest.mark.parametrize(
    ("method", "request_id", "mode"),
    [
        ("subscriptions/listen", 1, "propagation_only"),
        ("tools/call", 2, "native"),
        ("subscriptions/listen", None, "native"),
    ],
)
async def test_only_a_listen_request_is_traced_by_propagation_alone(
    method, request_id, mode
) -> None:
    from fastmcp.telemetry import telemetry_mode

    from kubed.mcp._listen import _ListenUntraced

    seen = []

    async def call_next(ctx):
        seen.append(telemetry_mode())
        return "result"

    ctx = types.SimpleNamespace(method=method, request_id=request_id)
    assert await _ListenUntraced()(ctx, call_next) == "result"
    assert seen == [mode]


async def test_a_changed_fastmcp_server_warns_once_and_still_serves(caplog) -> None:
    from fastmcp.server.low_level import FastMCPServerMiddleware

    from kubed.mcp._listen import _leave_listen_untraced

    mcp = served("changed", TelemetrySettings())
    middleware = mcp.server._mcp_server.middleware
    middleware[:] = [
        m for m in middleware if not isinstance(m, FastMCPServerMiddleware)
    ]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="kubed.mcp._listen"):
        _leave_listen_untraced(mcp.server)
    warnings = [r for r in caplog.records if r.name == "kubed.mcp._listen"]
    assert [r.levelno for r in warnings] == [logging.WARNING]
    assert "no FastMCPServerMiddleware" in warnings[0].getMessage()
    async with Client(mcp.server) as client:
        assert (await client.call_tool("ping", {})).data == "pong"


def test_a_server_without_a_low_level_server_warns_once(caplog) -> None:
    from kubed.mcp._listen import _leave_listen_untraced

    with caplog.at_level(logging.WARNING, logger="kubed.mcp._listen"):
        _leave_listen_untraced(types.SimpleNamespace())
    warnings = [r for r in caplog.records if r.name == "kubed.mcp._listen"]
    assert [r.levelno for r in warnings] == [logging.WARNING]
    assert "AttributeError" in warnings[0].getMessage()


# -- resource reads (TEL-21) --------------------------------------------------


def reads(points: list) -> list[tuple[str | None, int]]:
    """Each ``resources/read`` point's template label, with its count."""
    return [
        (p.attributes.get("mcp.resource.template"), p.count)
        for p in points
        if p.attributes["mcp.method.name"] == "resources/read"
    ]


def labelled(points: list) -> list:
    """Each ``resources/read`` point's (template, library, plugin), with its count."""
    return [
        (
            (
                p.attributes.get("mcp.resource.template"),
                p.attributes.get("kubed.plugin.library"),
                p.attributes.get("kubed.plugin.name"),
            ),
            p.count,
        )
        for p in points
        if p.attributes["mcp.method.name"] == "resources/read"
    ]


def resourced(name: str) -> QuickMCP:
    mcp = QuickMCP(name, config={"telemetry": TelemetrySettings(otel=True)})

    @mcp.resource("data://items/{id}")
    def item(id: str) -> str:
        return id

    @mcp.resource("data://about")
    def about() -> str:
        return "about"

    return mcp


async def test_reads_of_one_template_are_one_series(metered) -> None:
    async with Client(resourced("templated").server) as client:
        await client.read_resource("data://items/1")
        await client.read_resource("data://items/2")
    assert reads(duration_points(metered, "templated")) == [("data://items/{id}", 2)]


async def test_a_static_read_is_labelled_by_its_uri(metered) -> None:
    async with Client(resourced("static").server) as client:
        await client.read_resource("data://about")
    assert reads(duration_points(metered, "static")) == [("data://about", 1)]


async def test_a_read_that_matches_nothing_has_only_the_error_label(metered) -> None:
    from fastmcp.exceptions import MCPError

    async with Client(resourced("missing").server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("data://nope")
    (point,) = [
        p
        for p in duration_points(metered, "missing")
        if p.attributes["mcp.method.name"] == "resources/read"
    ]
    assert "mcp.resource.template" not in point.attributes
    # Resource not found is invalid params, which the conventions don't count as an error.
    assert point.attributes["rpc.status_code"] == "-32602"
    assert "error.type" not in point.attributes


async def test_a_failed_template_read_is_labelled_by_its_template(metered) -> None:
    from fastmcp.exceptions import MCPError, ResourceError

    mcp = resourced("failed-read")

    @mcp.resource("data://broken/{id}")
    def broken(id: str) -> str:
        raise ResourceError("no")

    async with Client(mcp.server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("data://broken/1")
    (point,) = [
        p
        for p in duration_points(metered, "failed-read")
        if p.attributes["mcp.method.name"] == "resources/read"
    ]
    assert point.attributes["mcp.resource.template"] == "data://broken/{id}"
    assert (point.attributes["rpc.status_code"], point.attributes["error.type"]) == (
        "-32603",
        "-32603",
    )


async def test_a_failed_static_read_has_no_template_label(metered) -> None:
    from fastmcp.exceptions import MCPError, ResourceError

    mcp = resourced("failed-static")

    @mcp.resource("data://bad")
    def bad() -> str:
        raise ResourceError("no")

    async with Client(mcp.server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("data://bad")
    (point,) = [
        p
        for p in duration_points(metered, "failed-static")
        if p.attributes["mcp.method.name"] == "resources/read"
    ]
    assert "mcp.resource.template" not in point.attributes
    assert (point.attributes["rpc.status_code"], point.attributes["error.type"]) == (
        "-32603",
        "-32603",
    )


async def test_a_static_read_a_template_also_matches_is_labelled_by_it(
    metered,
) -> None:
    mcp = resourced("shadowed")

    @mcp.resource("data://items/fixed")
    def fixed() -> str:
        return "fixed"

    async with Client(mcp.server) as client:
        (content,) = await client.read_resource("data://items/fixed")
    # The static resource answered, and the template labels it.
    assert content.text == "fixed"
    assert reads(duration_points(metered, "shadowed")) == [("data://items/{id}", 1)]


async def test_a_plugin_skill_read_under_a_prefix_is_labelled_without_it(
    metered, tmp_path
) -> None:
    mcp = QuickMCP(
        "kit-reads",
        header_prefix="Kb",
        config={"telemetry": TelemetrySettings(otel=True)},
    )
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with over_http(mcp, headers={"Kb-Resource-Prefix": "kb+"}) as client:
        await client.read_resource("kb+skill://lib/guide/references/a.md")
        await client.read_resource("kb+skill://lib/guide/SKILL.md")
    assert sorted(labelled(duration_points(metered, "kit-reads"))) == [
        (("skill://{library}/{path}", "lib", "kit"), 1),
        (("skill://{library}/{path}/SKILL.md", "lib", "kit"), 1),
    ]


@pytest.mark.parametrize("uri", ["prompts://lib/ask.md", "agents://lib/scout.md"])
async def test_a_plugin_file_read_under_a_prefix_is_labelled_without_it(
    metered, tmp_path, uri
) -> None:
    name = f"kit-files-{uri.partition(':')[0]}"
    mcp = QuickMCP(
        name, header_prefix="Kb", config={"telemetry": TelemetrySettings(otel=True)}
    )
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with over_http(mcp, headers={"Kb-Resource-Prefix": "kb+"}) as client:
        await client.read_resource(f"kb+{uri}")
    scheme = uri.partition(":")[0]
    assert labelled(duration_points(metered, name)) == [
        ((f"{scheme}://{{library}}/{{path}}", "lib", "kit"), 1)
    ]


async def test_a_read_through_the_tool_is_a_call_and_a_read(metered, tmp_path) -> None:
    from kubed.mcp import DeliverySettings

    mcp = QuickMCP(
        "tool-reads",
        config={
            "telemetry": TelemetrySettings(otel=True),
            "delivery": DeliverySettings(resources="off"),
        },
    )
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")

    @mcp.resource("data://items/{id}")
    def item(id: str) -> str:
        return id

    async with Client(mcp.server) as client:
        assert "read_resource" in {tool.name for tool in await client.list_tools()}
        await client.call_tool("read_resource", {"uri": "data://items/1"})
        await client.call_tool("read_resource", {"uri": "skill://lib/guide/SKILL.md"})
    points = duration_points(metered, "tool-reads")
    assert sorted(
        (p.attributes["mcp.method.name"], p.attributes.get("gen_ai.tool.name"))
        for p in points
    ) == [
        ("resources/read", None),
        ("resources/read", None),
        ("server/discover", None),
        ("tools/call", "read_resource"),
        ("tools/list", None),
    ]
    assert sorted(labelled(points), key=str) == [
        (("data://items/{id}", None, None), 1),
        (("skill://{library}/{path}/SKILL.md", "lib", "kit"), 1),
    ]


async def test_overlapping_reads_keep_their_own_templates(metered) -> None:
    mcp = resourced("overlap")

    @mcp.resource("data://slow/{id}")
    async def slow(id: str) -> str:
        await asyncio.sleep(0.05)
        return id

    async with Client(mcp.server) as client:
        await asyncio.gather(
            client.read_resource("data://slow/1"),
            client.read_resource("data://items/1"),
        )
    assert sorted(reads(duration_points(metered, "overlap"))) == [
        ("data://items/{id}", 1),
        ("data://slow/{id}", 1),
    ]


async def test_tools_and_prompts_are_recorded_as_before(metered) -> None:
    mcp = resourced("mixed")
    mcp.tool(lambda: "pong", name="ping")
    mcp.prompt(lambda: "hello", name="greet")
    async with Client(mcp.server) as client:
        await client.call_tool("ping", {})
        await client.get_prompt("greet")
        await client.read_resource("data://about")
    points = duration_points(metered, "mixed")
    kinds = sorted(p.attributes["mcp.method.name"] for p in points)
    # The client discovers on connect, and lists the tools before it calls one.
    assert kinds == [
        "prompts/get",
        "resources/read",
        "server/discover",
        "tools/call",
        "tools/list",
    ]
    tool = next(p for p in points if p.attributes["mcp.method.name"] == "tools/call")
    assert "mcp.resource.template" not in tool.attributes


# -- listings (TEL-16) --------------------------------------------------------

MODES = ["auto", "legacy"]  # server/discover, initialize
LISTINGS = [
    "agents/list",
    "prompts/list",
    "resources/list",
    "resources/templates/list",
    "skills/list",
    "tools/list",
]
# What a listing's point may carry: its method, and what UsageMiddleware adds to any point.
LISTING_KEYS = {
    "mcp.method.name",
    "fastmcp.server.name",
    "mcp.protocol.version",
    "network.transport",
    "network.protocol.name",
    "network.protocol.version",
}


def listings_served(name: str, tmp_path: Path, **options) -> QuickMCP:
    """A server with something behind each of the seven listings."""
    mcp = QuickMCP(name, config={"telemetry": TelemetrySettings(otel=True)}, **options)
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    mcp.tool(lambda: "pong", name="ping")

    @mcp.resource("data://items/{id}")
    def item(id: str) -> str:
        return id

    return mcp


async def list_everything(client: Client) -> None:
    await client.list_tools()
    await client.list_resources()
    await client.list_resource_templates()
    await client.list_prompts()
    await request(client, "skills/list")
    await request(client, "agents/list")


def methods(points: list) -> list[tuple[str, int]]:
    """Each point's method, with its count, sorted."""
    return sorted((p.attributes["mcp.method.name"], p.count) for p in points)


def assert_unnamed(points: list) -> None:
    """No point names what it listed: no component, label, cursor or URI."""
    for point in points:
        assert set(point.attributes) <= LISTING_KEYS, point.attributes
        assert not [v for v in point.attributes.values() if "://" in str(v)]


@pytest.mark.parametrize("mode", MODES)
async def test_each_listing_is_one_point_under_its_method(
    metered, tmp_path, mode
) -> None:
    name = f"listings-{mode}"
    async with Client(listings_served(name, tmp_path).server, mode=mode) as client:
        await list_everything(client)
    points = duration_points(metered, name)
    discover = ["server/discover"] if mode == "auto" else []
    assert methods(points) == sorted((m, 1) for m in [*LISTINGS, *discover])
    assert_unnamed(points)


@pytest.mark.parametrize("mode", MODES)
async def test_a_second_listing_records_again_in_one_series(
    metered, tmp_path, mode
) -> None:
    name = f"listed-twice-{mode}"
    async with Client(listings_served(name, tmp_path).server, mode=mode) as client:
        for _ in range(2):
            await list_everything(client)
    points = [
        p
        for p in duration_points(metered, name)
        if p.attributes["mcp.method.name"] != "server/discover"
    ]
    assert methods(points) == [(m, 2) for m in LISTINGS]


@pytest.mark.parametrize("mode", MODES)
async def test_a_paged_listing_is_one_series(metered, mode) -> None:
    from kubed.mcp.skills import Skill, serve

    name = f"paged-{mode}"
    mcp = QuickMCP(
        name, list_page_size=1, config={"telemetry": TelemetrySettings(otel=True)}
    )
    serve(
        mcp.server,
        [
            Skill.from_files(
                {"SKILL.md": f"---\nname: {n}\ndescription: The {n} skill.\n---\n"}
            )
            for n in ("alpha", "beta", "gamma")
        ],
    )
    pages, cursor = 0, None
    async with Client(mcp.server, mode=mode) as client:
        while True:
            page = await request(
                client, "skills/list", **({} if cursor is None else {"cursor": cursor})
            )
            pages += 1
            if (cursor := page.get("nextCursor")) is None:
                break
    assert pages == 3
    points = duration_points(metered, name)
    assert [c for m, c in methods(points) if m == "skills/list"] == [3]
    assert_unnamed(points)


@pytest.mark.parametrize("mode", MODES)
async def test_a_listing_behind_a_prefix_is_one_unnamed_series(
    metered, tmp_path, mode
) -> None:
    name = f"prefixed-listings-{mode}"
    mcp = listings_served(name, tmp_path)
    async with over_http(mcp, header="kb+", mode=mode) as client:
        (entry,) = (await request(client, "skills/list"))["skills"]
        # The prefix is in force: the served text carries it.
        assert "kb+skill://" in entry["frontmatter"]["description"]
        await list_everything(client)
    points = duration_points(metered, name)
    listed = [(m, c) for m, c in methods(points) if m != "server/discover"]
    # skills/list twice, the rest once, each a single series.
    assert listed == [(m, 2 if m == "skills/list" else 1) for m in LISTINGS]
    assert_unnamed(points)


@pytest.mark.parametrize("mode", MODES)
async def test_a_get_and_a_directory_read_record_nothing(
    metered, tmp_path, mode
) -> None:
    name = f"gets-{mode}"
    async with Client(listings_served(name, tmp_path).server, mode=mode) as client:
        (agent,) = (await request(client, "agents/list"))["agents"]
        (entry,) = (await request(client, "skills/list"))["skills"]
        await request(client, "skills/get", uri=entry["uri"])
        await request(client, "agents/get", name=agent["name"])
        await request(client, "resources/directory/read", uri="skill://lib/guide")
    discover = [("server/discover", 1)] if mode == "auto" else []
    assert methods(duration_points(metered, name)) == sorted(
        [("agents/list", 1), ("skills/list", 1), *discover]
    )


async def test_a_listing_through_the_tool_is_a_call_and_a_listing(
    metered, tmp_path
) -> None:
    from kubed.mcp import DeliverySettings

    mcp = QuickMCP(
        "tool-lists",
        config={
            "telemetry": TelemetrySettings(otel=True),
            "delivery": DeliverySettings(resources="off"),
        },
    )
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with Client(mcp.server, mode="legacy") as client:
        await client.call_tool("list_resources", {})
    points = duration_points(metered, "tool-lists")
    # tools/list is the client's own, before the call; resources/list is the tool's.
    assert sorted(
        (p.attributes["mcp.method.name"], p.attributes.get("gen_ai.tool.name"), p.count)
        for p in points
    ) == [
        ("resources/list", None, 1),
        ("tools/call", "list_resources", 1),
        ("tools/list", None, 1),
    ]
    assert_unnamed(
        [p for p in points if p.attributes["mcp.method.name"] != "tools/call"]
    )


def kit_reads(name: str, tmp_path: Path) -> QuickMCP:
    mcp = QuickMCP(name, config={"telemetry": TelemetrySettings(otel=True)})
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    return mcp


async def test_many_files_of_one_skill_are_one_series(metered, tmp_path) -> None:
    tree = {**KIT_TREE, "skills/guide/references/b.md": "# B\n"}
    mcp = QuickMCP("series", config={"telemetry": TelemetrySettings(otel=True)})
    mcp.add_plugin(write(tmp_path / "kit", tree), namespace="lib")
    async with Client(mcp.server) as client:
        for name in ("a", "b", "a"):
            await client.read_resource(f"skill://lib/guide/references/{name}.md")
    assert labelled(duration_points(metered, "series")) == [
        (("skill://{library}/{path}", "lib", "kit"), 3)
    ]


READS = [
    "skill://lib/guide/_index.md",
    "skill://lib/guide/_manifest",
    "prompts://lib/_index.md",
]


async def test_an_index_and_a_manifest_read_are_labelled_by_the_layer(
    metered, tmp_path
) -> None:
    async with Client(kit_reads("indexes", tmp_path).server) as client:
        for uri in READS:
            await client.read_resource(uri)
    assert sorted(labelled(duration_points(metered, "indexes"))) == [
        (("prompts://{library}/{path}/_index.md", "lib", "kit"), 1),
        (("skill://{library}/{path}/_index.md", "lib", "kit"), 1),
        (("skill://{library}/{path}/_manifest", "lib", "kit"), 1),
    ]


async def test_no_point_carries_a_served_uri(metered, tmp_path) -> None:
    async with Client(kit_reads("no-uri", tmp_path).server) as client:
        for uri in [
            *READS,
            "skill://lib/guide/references/a.md",
            "agents://lib/scout.md",
        ]:
            await client.read_resource(uri)
    points = [
        p
        for p in duration_points(metered, "no-uri")
        if p.attributes["mcp.method.name"] == "resources/read"
    ]
    assert len(points) == 5
    for point in points:
        for value in point.attributes.values():
            assert not str(value).startswith(
                ("skill://lib/", "prompts://lib/", "agents://lib/")
            )


async def test_the_span_keeps_the_exact_uri(metered, spans, tmp_path) -> None:
    async with Client(kit_reads("span-uri", tmp_path).server) as client:
        await client.read_resource("skill://lib/guide/references/a.md")
    from opentelemetry.trace import SpanKind

    (span,) = [
        s
        for s in spans.get_finished_spans()
        if s.kind == SpanKind.SERVER
        and s.attributes.get("mcp.method.name") == "resources/read"
    ]
    assert span.attributes["mcp.resource.uri"] == "skill://lib/guide/references/a.md"


async def test_a_failed_read_of_a_missing_skill_file(metered, tmp_path) -> None:
    from fastmcp.exceptions import MCPError

    async with Client(kit_reads("missing-file", tmp_path).server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("skill://lib/guide/references/missing.md")
    (point,) = [
        p
        for p in duration_points(metered, "missing-file")
        if p.attributes["mcp.method.name"] == "resources/read"
    ]
    # A missing file is resource-not-found, a routing error: UsageMiddleware takes the
    # name attribute off the point, so none of the three labels reach it.
    for key in ("mcp.resource.template", "kubed.plugin.library", "kubed.plugin.name"):
        assert key not in point.attributes
    assert point.attributes["rpc.status_code"] == "-32602"
    assert "skill://lib/guide/references/missing.md" not in point.attributes.values()


async def test_a_template_read_beside_plugins_is_unchanged(metered, tmp_path) -> None:
    mcp = resourced("beside")
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with Client(mcp.server) as client:
        await client.read_resource("data://items/1")
    assert labelled(duration_points(metered, "beside")) == [
        (("data://items/{id}", None, None), 1)
    ]


async def test_a_server_without_plugins_never_pins_for_a_label(
    metered, monkeypatch
) -> None:
    from kubed.mcp.plugins import Plugins

    calls: list[None] = []

    async def pinned(self):
        calls.append(None)
        raise AssertionError("pinned")

    monkeypatch.setattr(Plugins, "pinned", pinned)
    async with Client(resourced("unpinned").server) as client:
        await client.read_resource("data://about")
        await client.read_resource("data://items/1")
    assert calls == []
    assert sorted(labelled(duration_points(metered, "unpinned"))) == [
        (("data://about", None, None), 1),
        (("data://items/{id}", None, None), 1),
    ]


async def test_a_read_whose_pin_fails_carries_no_label_and_no_error_of_its_own(
    metered, tmp_path
) -> None:
    from fastmcp.exceptions import MCPError

    mcp = kit_reads("pin-fails", tmp_path)

    def view():
        raise RuntimeError("view")

    mcp.plugin_view(view)
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("skill://lib/guide/SKILL.md")
    (point,) = [
        p
        for p in duration_points(metered, "pin-fails")
        if p.attributes["mcp.method.name"] == "resources/read"
    ]
    for key in ("mcp.resource.template", "kubed.plugin.library", "kubed.plugin.name"):
        assert key not in point.attributes


async def test_a_read_answered_without_a_provider_never_pins_for_a_label(
    metered, tmp_path
) -> None:
    from fastmcp.resources import ResourceResult
    from fastmcp.server.middleware import Middleware

    mcp = QuickMCP("answered", config={"telemetry": TelemetrySettings(otel=True)})
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    views: list[None] = []

    def view():
        views.append(None)
        raise RuntimeError("view")

    mcp.plugin_view(view)

    class Answers(Middleware):
        async def on_read_resource(self, context, call_next):
            return ResourceResult("about")

    # Added after the observer is placed, so it sits inside it.
    mcp.server.add_middleware(Answers())
    async with Client(mcp.server) as client:
        (content,) = await client.read_resource("data://about")
    assert content.text == "about"
    assert views == []
    assert labelled(duration_points(metered, "answered")) == [
        (("data://about", None, None), 1)
    ]


async def test_a_lookup_that_raises_falls_back_to_the_template(
    metered, caplog
) -> None:
    from kubed.mcp._metrics import _ReadObserver

    class Broken:
        def _read_labels(self, uri: str):
            raise RuntimeError("lookup")

    mcp = resourced("lookup-raises")
    (observer,) = [m for m in mcp.server.middleware if isinstance(m, _ReadObserver)]
    observer._plugins = Broken()
    with caplog.at_level(logging.WARNING):
        async with Client(mcp.server) as client:
            await client.read_resource("data://items/1")
    assert labelled(duration_points(metered, "lookup-raises")) == [
        (("data://items/{id}", None, None), 1)
    ]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_the_layer_is_asked_before_a_template(metered, tmp_path) -> None:
    mcp = kit_reads("layer-first", tmp_path)

    @mcp.server.resource("prompts://lib/extra/{id}")
    def extra(id: str) -> str:
        return id

    async with Client(mcp.server) as client:
        (content,) = await client.read_resource("prompts://lib/extra/1")
    assert content.text == "1"
    assert labelled(duration_points(metered, "layer-first")) == [
        (("prompts://{library}/{path}", "lib", "kit"), 1)
    ]


async def test_a_failed_template_read_below_the_layer_is_labelled_by_it(
    metered, tmp_path
) -> None:
    from fastmcp.exceptions import MCPError, ResourceError

    mcp = kit_reads("layer-first-fails", tmp_path)

    @mcp.server.resource("prompts://lib/extra/{id}")
    def extra(id: str) -> str:
        raise ResourceError("no")

    async with Client(mcp.server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("prompts://lib/extra/1")
    (point,) = [
        p
        for p in duration_points(metered, "layer-first-fails")
        if p.attributes["mcp.method.name"] == "resources/read"
    ]
    assert (
        point.attributes.get("mcp.resource.template"),
        point.attributes.get("kubed.plugin.library"),
        point.attributes.get("kubed.plugin.name"),
    ) == ("prompts://{library}/{path}", "lib", "kit")
    assert point.attributes["error.type"] == "-32603"


def test_usage_middleware_still_has_what_the_reads_use() -> None:
    from fastmcp.server.middleware.usage import UsageMiddleware

    from kubed.mcp._metrics import _CallMetrics, _Labelled

    assert callable(getattr(UsageMiddleware, "_record", None))
    # The reads call it by these keywords: a rename or a positional one would raise.
    inspect.signature(UsageMiddleware._record).bind(
        None, None, None, method="", name_attribute="", name=""
    )
    assert isinstance(_CallMetrics(user_id="omit")._duration, _Labelled)


class Recorded:
    """A histogram that keeps the attributes of each point."""

    def __init__(self) -> None:
        self.points: list[object] = []

    def record(self, amount: float, attributes: object = None) -> None:
        self.points.append(attributes)


def test_the_call_metrics_copy_and_still_label() -> None:
    import copy

    from opentelemetry.metrics import NoOpMeterProvider

    from kubed.mcp._metrics import (
        _LIBRARY,
        _MATCHED,
        _PLUGIN,
        _TEMPLATE,
        _CallMetrics,
        _Labelled,
        _Matched,
    )

    # An SDK histogram holds a lock, so a plain UsageMiddleware can't be deep-copied
    # under one either: a no-op provider keeps this about the wrapper alone.
    metrics = _CallMetrics(user_id="omit", meter_provider=NoOpMeterProvider())
    assert isinstance(copy.copy(metrics._duration), _Labelled)
    assert isinstance(copy.deepcopy(metrics)._duration, _Labelled)
    labelled = _Labelled(Recorded())
    for twin in (copy.copy(labelled), copy.deepcopy(labelled)):
        token = _MATCHED.set(_Matched("data://items/{id}"))
        try:
            twin.record(1.0, {_TEMPLATE: ""})
        finally:
            _MATCHED.reset(token)
        assert twin._histogram.points[-1] == {_TEMPLATE: "data://items/{id}"}
        token = _MATCHED.set(_Matched("skill://{library}/{path}", "lib", "kit"))
        try:
            twin.record(1.0, {_TEMPLATE: ""})
            twin.record(1.0, {"other": "x"})
        finally:
            _MATCHED.reset(token)
        assert twin._histogram.points[-2] == {
            _TEMPLATE: "skill://{library}/{path}",
            _LIBRARY: "lib",
            _PLUGIN: "kit",
        }
        assert twin._histogram.points[-1] == {"other": "x"}


async def test_a_listing_inside_a_read_takes_no_label_from_it() -> None:
    from opentelemetry.metrics import NoOpMeterProvider

    from kubed.mcp._metrics import (
        _LIBRARY,
        _MATCHED,
        _PLUGIN,
        _TEMPLATE,
        _CallMetrics,
        _Labelled,
        _Matched,
    )

    metrics = _CallMetrics(user_id="omit", meter_provider=NoOpMeterProvider())
    recorded = Recorded()
    metrics._duration = _Labelled(recorded)

    async def call_next(context: object) -> list:
        return []

    context = types.SimpleNamespace(method="skills/list", fastmcp_context=None)
    # The holder an outer read leaves, already labelled.
    outer = _Matched("data://items/{id}", "lib", "kit")
    token = _MATCHED.set(outer)
    try:
        await metrics.on_list_tools(context, call_next)
        await metrics.on_request(context, call_next)
        assert _MATCHED.get() is outer
    finally:
        _MATCHED.reset(token)
    assert [p["mcp.method.name"] for p in recorded.points] == [
        "tools/list",
        "skills/list",
    ]
    for point in recorded.points:
        assert not {_TEMPLATE, _LIBRARY, _PLUGIN} & set(point)


def _without_record(monkeypatch) -> None:
    from fastmcp.server.middleware.usage import UsageMiddleware

    monkeypatch.delattr(UsageMiddleware, "_record")


def _without_duration(monkeypatch) -> None:
    from fastmcp.server.middleware.usage import UsageMiddleware

    real = UsageMiddleware.__init__

    def init(self, **options) -> None:
        real(self, **options)
        del self._duration

    monkeypatch.setattr(UsageMiddleware, "__init__", init)


def _with_another_record(monkeypatch) -> None:
    from fastmcp.server.middleware.usage import UsageMiddleware

    async def record(self, context, call_next, *, method, attribute, label):
        return await call_next(context)

    monkeypatch.setattr(UsageMiddleware, "_record", record)


@pytest.mark.parametrize(
    "change", [_without_record, _without_duration, _with_another_record]
)
async def test_a_changed_usage_middleware_is_one_warning_and_reads_pass_through(
    change, monkeypatch, caplog
) -> None:
    from kubed.mcp._metrics import _CallMetrics, _Labelled

    change(monkeypatch)
    metrics = _CallMetrics(user_id="omit")
    assert not isinstance(getattr(metrics, "_duration", None), _Labelled)

    async def call_next(context: object) -> str:
        return "passed"

    # Nothing here could be recorded: a context with no server, and no histogram.
    context = types.SimpleNamespace(method="skills/list")
    for hook in (
        metrics.on_read_resource,
        metrics.on_list_tools,
        metrics.on_list_resources,
        metrics.on_list_resource_templates,
        metrics.on_list_prompts,
        metrics.on_discover,
        metrics.on_request,
    ):
        assert await hook(context, call_next) == "passed"
    records = [r for r in caplog.records if r.name == "kubed.mcp._metrics"]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (
            logging.WARNING,
            "Resource reads and listings are not in the call metrics: "
            "FastMCP's UsageMiddleware has changed",
        )
    ]


async def test_a_changed_usage_middleware_records_no_listing(
    metered, tmp_path, monkeypatch
) -> None:
    _with_another_record(monkeypatch)
    name = "listings-unrecorded"
    # Every listing is still answered: none calls the _record it can't call.
    async with Client(listings_served(name, tmp_path).server) as client:
        await list_everything(client)
    assert duration_points(metered, name) == []


# -- the start line -----------------------------------------------------------


def start_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "kubed.mcp.telemetry" and r.getMessage().startswith("telemetry: ")
    ]


async def test_the_start_line_says_off_with_nothing_on(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="kubed.mcp.telemetry"):
        async with Client(served("off", TelemetrySettings()).server):
            pass
    assert start_lines(caplog) == ["telemetry: off"]


async def test_the_start_line_names_profiles_and_the_host(pyroscope, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="kubed.mcp.telemetry"):
        async with Client(served("p", TelemetrySettings(profiles=ADDRESS)).server):
            pass
    assert start_lines(caplog) == ["telemetry: profiles → pyroscope.example.com:4040"]


async def test_the_start_line_names_otels_exporters(
    traced, pyroscope, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", " console ")
    settings = TelemetrySettings(profiles=ADDRESS, otel=True)
    with caplog.at_level(logging.INFO, logger="kubed.mcp.telemetry"):
        async with Client(served("o", settings).server):
            pass
    assert start_lines(caplog) == [
        "telemetry: profiles → pyroscope.example.com:4040; "
        "otel → traces console, metrics otlp (default), logs none"
    ]


def otel_line() -> str:
    signals = kubed.mcp.telemetry._Signals(None, otel=object())
    return kubed.mcp.telemetry._start_line(signals, None)


def test_the_start_line_shows_default_with_no_endpoint() -> None:
    assert otel_line() == "telemetry: otel → traces otlp (default), metrics otlp (default), logs none"


def test_the_start_line_names_the_shared_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example:4318")
    assert otel_line() == (
        "telemetry: otel → traces otlp (collector.example:4318), "
        "metrics otlp (collector.example:4318), logs none"
    )


def test_a_per_signal_endpoint_beats_the_shared_one(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://shared.example:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://traces.example:443/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "")  # empty counts as unset
    assert otel_line() == (
        "telemetry: otel → traces otlp (traces.example:443), "
        "metrics otlp (shared.example:4318), logs none"
    )


def test_an_endpoint_never_shows_its_credentials(monkeypatch) -> None:
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://user:s3cret@collector.example:4318/v1/traces?x=s3cret",
    )
    line = otel_line()
    assert line.startswith("telemetry: otel → traces otlp (collector.example:4318), metrics otlp (default)")
    assert "s3cret" not in line and "user" not in line and "/v1" not in line


def test_logs_over_otlp_show_their_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://logs.example:4318")
    assert otel_line().endswith(", logs none")
    monkeypatch.setenv("OTEL_LOGS_EXPORTER", "otlp")
    assert otel_line().endswith(", logs otlp (logs.example:4318)")


def test_only_the_otlp_entries_of_a_list_show_an_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", " otlp_proto_grpc , console ")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "prometheus")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "collector:4317")
    assert otel_line() == (
        "telemetry: otel → traces otlp_proto_grpc (collector:4317),console, "
        "metrics prometheus, logs none"
    )


@pytest.mark.parametrize("endpoint", ["s3cret", "   ", "http://"])
def test_an_endpoint_with_no_host_says_so(monkeypatch, endpoint) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
    assert otel_line() == (
        "telemetry: otel → traces otlp (an address with no host), "
        "metrics otlp (an address with no host), logs none"
    )


@pytest.mark.parametrize(
    "address, shown",
    [
        (
            "https://user:s3cret@pyroscope.example:4040/push?token=s3cret",
            "pyroscope.example:4040",
        ),
        ("user:s3cret@pyroscope:4040", "pyroscope:4040"),
        ("http://[::1]:4040", "[::1]:4040"),
        ("http://host:notaport", "host"),
        ("http://", "an address with no host"),
        ("s3cret", "an address with no host"),
        ("user:s3cret", "an address with no host"),
        ("a b c:1", "an address with no host"),
        # A password with an unencoded "/": the host is what follows the last "@".
        ("http://admin:4321/x@host:4040", "host:4040"),
        ("admin:4321/x@pyroscope.example", "pyroscope.example"),
        ("http://u:a@b@host:4040/push", "host:4040"),
        # An "@" in the query: the text before it may be a password holding an
        # unencoded "?", and the text after it is the query's.
        ("http://host:4040/x?u=a@b", "an address with no host"),
        ("http://u:s3cret?x@host:4040", "an address with no host"),
        ("http://host:4040/x#u=s3cret@b", "an address with no host"),
        # A Docker Compose service name.
        ("pyro_scope:4040", "pyro_scope:4040"),
        ("http://user:s3cret@pyro_scope:4040", "pyro_scope:4040"),
    ],
)
def test_the_start_line_never_shows_a_credential(address, shown) -> None:
    assert kubed.mcp.telemetry._where(address) == shown and "s3cret" not in shown


async def test_profiles_that_fail_to_start_are_not_named(monkeypatch, caplog) -> None:
    monkeypatch.setitem(sys.modules, "pyroscope", None)  # not installed
    with caplog.at_level(logging.INFO, logger="kubed.mcp.telemetry"):
        async with Client(
            served("missing", TelemetrySettings(profiles=ADDRESS)).server
        ):
            pass
    assert start_lines(caplog) == ["telemetry: off"]


# -- the probe (TEL-19) -------------------------------------------------------


class _Stub(http.server.ThreadingHTTPServer):
    daemon_threads = True
    status = 200
    delay = 0.0

    def __init__(self) -> None:
        self.seen: list[tuple[str, str | None]] = []
        super().__init__(("127.0.0.1", 0), _StubHandler)

    @property
    def port(self) -> int:
        return self.server_address[1]


class _StubHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.server.seen.append((self.path, self.headers.get("Authorization")))
        time.sleep(self.server.delay)
        self.send_response(self.server.status)
        if 300 <= self.server.status < 400:
            self.send_header("Location", "/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


@pytest.fixture
def stub() -> Iterator[_Stub]:
    server = _Stub()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture
def refused() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def no_proxies(monkeypatch, caplog) -> None:
    """No proxy variable, so a probe of localhost goes direct; the probe's lines are kept."""
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    caplog.set_level(logging.INFO, logger="kubed.mcp.telemetry")


def probe_lines(caplog) -> list[tuple[int, str]]:
    return [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.name == "kubed.mcp.telemetry" and "pyroscope at" in r.getMessage()
    ]


def probe_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "pyroscope-probe"]


def test_a_pyroscope_down_at_boot_is_one_warning_in_the_first_interval(
    refused, no_proxies, caplog
) -> None:
    probe = _Probe(f"http://127.0.0.1:{refused}", interval=0.05, timeout=1)
    probe.start()
    try:
        deadline = time.monotonic() + 2
        while not probe_lines(caplog) and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)  # several more intervals
    finally:
        probe.stop()
    assert probe_lines(caplog) == [
        (
            logging.WARNING,
            f"profiles: pyroscope at 127.0.0.1:{refused} unreachable: connection refused",
        )
    ]
    assert probe_threads() == []


def test_down_then_up_is_one_warning_then_one_info(stub, no_proxies, caplog) -> None:
    probe = _Probe(f"http://127.0.0.1:{stub.port}", timeout=1)
    stub.status = 503
    probe.check()
    probe.check()
    stub.status = 200
    probe.check()
    probe.check()
    host = f"127.0.0.1:{stub.port}"
    assert probe_lines(caplog) == [
        (
            logging.WARNING,
            f"profiles: pyroscope at {host} unreachable: server error (HTTP 503)",
        ),
        (logging.INFO, f"profiles: pyroscope at {host} reachable again"),
    ]


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_login_is_an_auth_warning(stub, no_proxies, caplog, status) -> None:
    stub.status = status
    _Probe(f"http://127.0.0.1:{stub.port}", timeout=1).check()
    assert probe_lines(caplog)[0][1].endswith(f"unreachable: auth (HTTP {status})")


def test_a_404_is_reachable(stub, no_proxies, caplog) -> None:
    stub.status = 404
    _Probe(f"http://127.0.0.1:{stub.port}", timeout=1).check()
    assert probe_lines(caplog) == []


def test_a_redirect_is_reachable_and_not_followed(stub, no_proxies, caplog) -> None:
    stub.status = 302
    _Probe(f"http://u:s3cret@127.0.0.1:{stub.port}", timeout=1).check()
    assert probe_lines(caplog) == []
    assert [path for path, _ in stub.seen] == ["/ready"]


def test_a_timeout_is_its_own_kind(stub, no_proxies, caplog) -> None:
    stub.delay = 1
    _Probe(f"http://127.0.0.1:{stub.port}", timeout=0.2).check()
    assert probe_lines(caplog)[0][1].endswith("unreachable: timeout")


def test_the_probe_sends_basic_auth_and_shows_none_of_it(
    stub, no_proxies, caplog
) -> None:
    stub.status = 401
    address = f"http://us%40er:s3cret@127.0.0.1:{stub.port}/base/?token=t0ken"
    _Probe(address, timeout=1).check()
    sent = "Basic " + base64.b64encode(b"us@er:s3cret").decode()
    assert stub.seen == [("/base/ready", sent)]
    for secret in ("s3cret", "us@er", "us%40er", "t0ken", "base"):
        assert_safe(caplog.records, secret)
    assert probe_lines(caplog) == [
        (
            logging.WARNING,
            f"profiles: pyroscope at 127.0.0.1:{stub.port} unreachable: auth (HTTP 401)",
        )
    ]


def test_a_probe_that_fails_oddly_never_raises(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="kubed.mcp.telemetry")

    def boom(*args, **kwargs):
        raise RuntimeError("https://u:s3cret@x")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    _Probe("http://127.0.0.1:1", timeout=1).check()
    assert probe_lines(caplog)[0][1].endswith("unreachable: RuntimeError")
    assert_safe(caplog.records, "s3cret")


def test_stop_ends_the_thread_within_the_timeout(stub, no_proxies) -> None:
    probe = _Probe(f"http://127.0.0.1:{stub.port}", interval=60, timeout=1)
    probe.start()
    started = time.monotonic()
    probe.stop()
    assert time.monotonic() - started < 1.5
    assert probe_threads() == []


@pytest.mark.probe
def test_profiles_start_the_probe_on_the_push_interval_and_stop_ends_it(
    pyroscope, stub, no_proxies
) -> None:
    signals = _start_signals(
        "t", TelemetrySettings(profiles=f"http://127.0.0.1:{stub.port}")
    )
    try:
        (thread,) = probe_threads()
        assert thread.daemon
        assert pyroscope[0]["upload_interval"] == 10
    finally:
        signals.stop()
    assert not thread.is_alive()


@pytest.mark.probe
def test_profiles_off_start_no_probe(monkeypatch, refused) -> None:
    assert _start_signals("t", TelemetrySettings()) is None
    assert probe_threads() == []
    monkeypatch.setitem(sys.modules, "pyroscope", None)  # not installed
    assert (
        _start_signals("t", TelemetrySettings(profiles=f"http://127.0.0.1:{refused}"))
        is None
    )
    assert probe_threads() == []


@pytest.mark.parametrize(
    "address", ["http://tok?en@host:4040", "http://u:p@host/?e=a@b"]
)
def test_an_ambiguous_userinfo_is_never_probed(monkeypatch, caplog, address) -> None:
    caplog.set_level(logging.INFO, logger="kubed.mcp.telemetry")
    opened: list[object] = []
    monkeypatch.setattr(
        urllib.request.OpenerDirector, "open", lambda *a, **kw: opened.append(a)
    )
    _Probe(address, timeout=1).check()
    assert opened == []
    assert [m for _, m in probe_lines(caplog)] == [
        "profiles: pyroscope at an address with no host unreachable: invalid address"
    ]
    for secret in ("tok", "en", "u:p", "a@b", "e=a"):
        assert_safe(caplog.records, secret)


class Body(io.BytesIO):
    """An error's response body that can fail to close."""

    fail = False

    def close(self) -> None:
        super().close()
        if self.fail:
            raise OSError("s3cret")


@pytest.mark.parametrize("fail", [False, True])
def test_an_http_error_s_response_is_closed(
    monkeypatch, no_proxies, caplog, fail
) -> None:
    body = Body()
    body.fail = fail
    # Held here, so the error's own finaliser can't be what closes it.
    raised: list[urllib.error.HTTPError] = []

    def answer(*args, **kwargs):
        raised.append(urllib.error.HTTPError("http://h/ready", 503, "no", {}, body))
        raise raised[-1]

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", answer)
    _Probe("http://127.0.0.1:1", timeout=1).check()
    assert body.closed
    assert probe_lines(caplog)[0][1].endswith("unreachable: server error (HTTP 503)")
    assert_safe(caplog.records, "s3cret")


@pytest.mark.parametrize(
    "address",
    ["http://u:p@host:4040/t@evil.example", "http://u:p@host:4040\\t@evil.example"],
)
def test_a_userinfo_with_a_slash_is_never_probed(
    monkeypatch, no_proxies, caplog, address
) -> None:
    # The agent ends an http host at the first / or \, so the credentials would go
    # elsewhere.
    opened: list[object] = []
    monkeypatch.setattr(
        urllib.request.OpenerDirector, "open", lambda *a, **kw: opened.append(a)
    )
    _Probe(address, timeout=1).check()
    assert opened == []
    ((_, line),) = probe_lines(caplog)
    assert line.endswith("unreachable: invalid address")
    assert_safe(caplog.records, "u:p")


def test_an_empty_userinfo_sends_no_authorization(stub, no_proxies, caplog) -> None:
    _Probe(f"http://@127.0.0.1:{stub.port}", timeout=1).check()
    assert stub.seen == [("/ready", None)]
    assert probe_lines(caplog) == []


def test_an_unknown_scheme_is_an_invalid_address(no_proxies, caplog) -> None:
    _Probe("htp://127.0.0.1:1", timeout=1).check()
    assert probe_lines(caplog)[0][1].endswith("unreachable: invalid address")


def test_a_port_that_is_not_a_number_is_an_invalid_address(no_proxies, caplog) -> None:
    _Probe("http://127.0.0.1:abc", timeout=1).check()
    assert probe_lines(caplog)[0][1].endswith("unreachable: invalid address")


def test_a_started_probe_is_stopped_when_the_start_then_fails(
    monkeypatch, pyroscope
) -> None:
    stopped: list[str] = []

    class Fake:
        def stop(self) -> None:
            stopped.append("probe")

    def fail(*args, **kwargs):
        raise RuntimeError("s3cret")

    monkeypatch.setattr(kubed.mcp.telemetry, "_start_probe", lambda address: Fake())
    monkeypatch.setattr(kubed.mcp.telemetry, "_Signals", fail)
    assert _start_signals("t", TelemetrySettings(profiles=ADDRESS)) is None
    assert stopped == ["probe"]


def test_the_probe_starts_after_otel(monkeypatch, pyroscope) -> None:
    # So an OTLP log handler is attached before the probe's first warning.
    started: list[str] = []

    def otel(*args, **kwargs):
        started.append("otel")

    def probe(address):
        started.append("probe")

    monkeypatch.setattr(kubed.mcp.telemetry, "_start_otel", otel)
    monkeypatch.setattr(kubed.mcp.telemetry, "_start_probe", probe)
    signals = _start_signals("t", TelemetrySettings(profiles=ADDRESS))
    try:
        assert started == ["otel", "probe"]
    finally:
        signals.stop()


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ConnectionRefusedError(), "connection refused"),
        (socket.gaierror(), "DNS"),
        (ssl.SSLCertVerificationError(), "TLS"),
        (TimeoutError(), "timeout"),
        (OSError(), "connection error"),
        (ValueError(), "invalid address"),
        (http.client.InvalidURL(), "invalid address"),
        (http.client.BadStatusLine(""), "connection error"),
        (urllib.error.URLError(socket.gaierror()), "DNS"),
        (urllib.error.URLError("unknown url type: htp"), "invalid address"),
        (KeyError(), "KeyError"),
    ],
)
def test_the_kind_of_a_failure(error, kind) -> None:
    assert _kind(error) == kind


def test_an_address_without_a_scheme_is_probed_over_http(
    stub, no_proxies, caplog
) -> None:
    _Probe(f"127.0.0.1:{stub.port}", timeout=1).check()
    assert [p for p, _ in stub.seen] == ["/ready"]
    assert probe_lines(caplog) == []


def test_fail_recover_fail_warns_twice(stub, no_proxies, caplog) -> None:
    probe = _Probe(f"http://127.0.0.1:{stub.port}", timeout=1)
    for status in (503, 200, 503):
        stub.status = status
        probe.check()
    assert [level for level, _ in probe_lines(caplog)] == [
        logging.WARNING,
        logging.INFO,
        logging.WARNING,
    ]
