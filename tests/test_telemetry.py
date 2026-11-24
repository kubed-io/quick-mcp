"""Telemetry: the settings, profiles, and OpenTelemetry, started and stopped with the server."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP
from pydantic import BaseModel, ValidationError

import kubed.mcp
import kubed.mcp.telemetry
from kubed.mcp import Inject, QuickMCP, TelemetrySettings
from kubed.mcp.app import _with_teardown
from kubed.mcp.container import Container

ADDRESS = "http://pyroscope.example.com:4040"
CHILD = Path(__file__).with_name("otel_child.py")
EVENTS: list[str] = []


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
    _reset_otel()
    yield
    _reset_otel()
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
    assert config.telemetry.profiles == "http://pyroscope.example.com:4040"
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


async def test_a_section_missing_an_attribute_is_a_logged_error_and_the_server_serves(
    pyroscope, caplog
) -> None:
    secret = "https://user:s3cret@pyroscope.example.com"
    mcp = served("partial", argparse.Namespace(profiles=secret))
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

    tags = {"a,team=x": "1", "b=c": "2", " team ": "3", " tier ": "4"}
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
        for key in ("a,team=x", "b=c")
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
    assert f"inside trace={trace_id}" in out.stderr
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
    assert "inside" in out.stderr.splitlines()
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
        _Otel(owned=True, logging_instrumented=False).stop()
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
    _Otel(owned=True, logging_instrumented=False).stop()
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


def test_a_configure_that_fails_halfway_shuts_its_providers_down(
    monkeypatch, caplog
) -> None:
    import kubed.mcp._otel

    # Unconfigured before initialize(), configured after it raised.
    answers = iter([False, True])
    monkeypatch.setattr(kubed.mcp._otel, "_sdk_configured", lambda: next(answers))
    # Replaces the guard's recorder: this test means initialize() to run, and raise.
    monkeypatch.setattr(kubed.mcp._otel, "initialize", _raise_secret)
    stopped = _fake_providers(monkeypatch)
    otel = kubed.mcp._otel.start("half", TelemetrySettings(otel=True), None)
    assert otel is None
    assert stopped == ["tracer", "meter", "logger"]
    assert kubed.mcp._otel._stopped is True
    records = [r for r in caplog.records if r.name.startswith("kubed.mcp")]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.ERROR, "OpenTelemetry did not start; serving without it: RuntimeError")
    ]
    assert_safe(caplog.records, "s3cret")


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
    records = [r for r in keep.records if r.getMessage() == "inside"]
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
