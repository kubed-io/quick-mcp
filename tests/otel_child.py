"""A process of its own for the telemetry tests that need one: ``otel_child.py <mode>``.

OpenTelemetry's providers are set once per process, so the real configuration runs here and
not in pytest's. The tests read what this writes: spans on stdout, log lines on stderr.

- ``console``: a traced server, then a second one with ``otel`` on after the first stopped.
- ``off``: servers with telemetry off, then the modules they loaded.
- ``bare``: a traced server in a process that configured no logging, and root as it saw it.
- ``bare-configured``: the same, with an SDK already configured, as opentelemetry-instrument does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from fastmcp import Client

from kubed.mcp import QuickMCP, TelemetrySettings

# What telemetry off must never load.
HEAVY = (
    "opentelemetry.sdk",
    "opentelemetry.instrumentation",
    "pyroscope.otel",
    "kubed.mcp._otel",
    "fastmcp.server.middleware.usage",
)


def configure_logging() -> None:
    """The server's own format, which names the trace a record was written in."""
    factory = logging.getLogRecordFactory()

    def record(*args, **kwargs) -> logging.LogRecord:
        made = factory(*args, **kwargs)
        # Outside the instrumentation too, so the format still applies.
        made.otelTraceID = "-"
        return made

    logging.setLogRecordFactory(record)
    logging.basicConfig(stream=sys.stderr, format="%(message)s trace=%(otelTraceID)s")


def root_seen() -> None:
    """Root's handlers and level, one line on stdout."""
    root = logging.getLogger()
    seen = {"handlers": [type(h).__name__ for h in root.handlers], "level": root.level}
    # One write, so a span exported meanwhile can't land inside the line.
    sys.stdout.write("root=" + json.dumps(seen) + "\n")
    sys.stdout.flush()


def server(name: str, telemetry: object, *, bare: bool = False) -> QuickMCP:
    config = argparse.Namespace(transport="http", host="0.0.0.0", port=8000)
    if telemetry is not None:
        config.telemetry = telemetry
    mcp = QuickMCP(name, config=config)

    @mcp.tool
    def ping() -> str:
        return "pong"

    @mcp.tool
    def log() -> str:
        if bare:
            root_seen()
        # In bare mode only logging.lastResort can print it.
        logging.getLogger("tests.tool").warning("inside")
        return "logged"

    return mcp


async def serve(mcp: QuickMCP, *tools: str) -> list[object]:
    async with Client(mcp.server) as client:
        return [(await client.call_tool(tool, {})).data for tool in tools]


async def console() -> None:
    settings = TelemetrySettings(otel=True, tags={"team": "a", "tier": "x y,z"})
    await serve(server("traced", settings), "ping", "log")
    (answer,) = await serve(server("second", TelemetrySettings(otel=True)), "ping")
    print(f"second served {answer}", file=sys.stderr)


async def off() -> None:
    for telemetry in (
        None,
        TelemetrySettings(),
        TelemetrySettings(otel=False, tags={"team": "a"}),
    ):
        await serve(server("off", telemetry), "ping")
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == heavy or name.startswith(heavy + ".") for heavy in HEAVY)
    )
    print(json.dumps(loaded))


async def bare() -> None:
    await serve(server("bare", TelemetrySettings(otel=True), bare=True), "log")
    root_seen()


async def bare_configured() -> None:
    """``bare``, after an SDK of its own was set: ``start`` takes the configured path."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "preset"}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    await bare()


async def prometheus() -> None:
    """The call metrics and a server's own counter, scraped from the exporter's port."""
    import os
    import urllib.request

    from opentelemetry import metrics

    config = argparse.Namespace(
        transport="http",
        host="0.0.0.0",
        port=8000,
        telemetry=TelemetrySettings(otel=True),
    )
    mcp = QuickMCP("prom", config=config)

    @mcp.tool
    def ping() -> str:
        metrics.get_meter("tests").create_counter("tests.own.calls").add(1)
        return "pong"

    url = f"http://localhost:{os.environ['OTEL_EXPORTER_PROMETHEUS_PORT']}/metrics"
    async with Client(mcp.server) as client:
        await client.call_tool("ping", {})
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode()
    for line in body.splitlines():
        if line.startswith(
            ("mcp_server_operation_duration_seconds_count", "tests_own_calls_total")
        ):
            print(line)


if __name__ == "__main__":
    mode = sys.argv[1]
    if not mode.startswith("bare"):
        configure_logging()
    modes = {
        "console": console,
        "off": off,
        "bare": bare,
        "bare-configured": bare_configured,
        "prometheus": prometheus,
    }
    asyncio.run(modes[mode]())
