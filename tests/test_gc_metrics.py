"""TEL-23: garbage-collection counts and pause times, per generation."""

from __future__ import annotations

import asyncio
import gc
import itertools
import logging
import sys
import threading
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from kubed.mcp._gc import _GcMonitor
from kubed.mcp._otel import _TraceText

BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
# Never drains on its own inside a test, but for the test of the periodic drain.
NEVER = 3600.0


def _monitors() -> list[object]:
    return [
        callback
        for callback in gc.callbacks
        if isinstance(getattr(callback, "__self__", None), _GcMonitor)
    ]


@pytest.fixture(autouse=True)
def _no_collections_but_ours() -> Iterator[None]:
    """Only the collections a test runs, and no monitor left behind by one."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        leaked = _monitors()
        for callback in leaked:
            gc.callbacks.remove(callback)
        assert not leaked, "a test left its monitor in gc.callbacks"


@pytest.fixture
def reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture
def meter(reader) -> Iterator[object]:
    """A meter of the test's own provider: the global one is set once per process."""
    provider = MeterProvider(metric_readers=[reader])
    yield provider.get_meter("test")
    provider.shutdown()


class Keep(logging.Handler):
    """Keeps every record it handles."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def long_passes():
    """A clock under which every collection pauses 0.6 s."""
    return itertools.count(0.0, 0.6).__next__


def points(reader: InMemoryMetricReader, name: str) -> list:
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def gc_points(reader: InMemoryMetricReader) -> list:
    return points(reader, "cpython.gc.collections") + points(
        reader, "cpython.gc.duration"
    )


async def test_the_callback_logs_nothing_and_the_drain_warns_with_the_trace(
    meter, root_logging
) -> None:
    keep = Keep()
    root_logging.addHandler(keep)
    root_logging.setLevel(logging.DEBUG)
    text = _TraceText()
    text.install()
    monitor = _GcMonitor(meter, clock=long_passes(), interval=NEVER)
    try:
        assert monitor.start() is True
        tracer = TracerProvider().get_tracer("test")
        with tracer.start_as_current_span("stalled") as span:
            gc.collect(2)
            # Before any loop turn: queued, and nothing logged.
            assert keep.records == []
            assert [(g, c) for g, _, c in monitor._passes] == [
                (2, span.get_span_context())
            ]
            context = span.get_span_context()
        monitor.drain()
    finally:
        monitor.stop()
        text.remove()
    warnings = [
        r
        for r in keep.records
        if r.name == "kubed.mcp.telemetry" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "generation 2" in message
    assert "0.60 s" in message
    # TEL-18's factory read the span current when the record was made.
    assert message.endswith(
        f" trace_id={context.trace_id:032x} span_id={context.span_id:016x}"
    )


async def test_a_warning_carries_only_the_trace_its_pass_stalled(
    meter, root_logging
) -> None:
    keep = Keep()
    root_logging.addHandler(keep)
    root_logging.setLevel(logging.DEBUG)
    text = _TraceText()
    text.install()
    tracer = TracerProvider().get_tracer("test")
    monitor = _GcMonitor(meter, clock=long_passes(), interval=0.01)
    try:
        # Started inside a span: the drain task copies the context it started in.
        with tracer.start_as_current_span("outer"):
            assert monitor.start() is True
        gc.collect(2)
        for _ in range(200):
            if keep.records:
                break
            await asyncio.sleep(0.01)
        with tracer.start_as_current_span("stalled") as stalled:
            gc.collect(1)
        gc.collect(0)
        # Stopped inside a span too, as a lifespan's exit can be: its last drain.
        with tracer.start_as_current_span("exit"):
            monitor.stop()
    finally:
        monitor.stop()
        text.remove()
    lines = {
        r.args[0]: r.getMessage()
        for r in keep.records
        if r.name == "kubed.mcp.telemetry" and r.levelno == logging.WARNING
    }
    assert sorted(lines) == [0, 1, 2]
    # Neither the periodic drain's pass nor the last drain's had a span of its own.
    assert "trace_id=" not in lines[2]
    assert "trace_id=" not in lines[0]
    # The last drain's pass in a span names its own span, not the exit's.
    context = stalled.get_span_context()
    assert lines[1].endswith(
        f" trace_id={context.trace_id:032x} span_id={context.span_id:016x}"
    )


async def test_a_long_pass_outside_a_span_still_warns(meter, reader, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    monitor = _GcMonitor(meter, clock=long_passes(), interval=NEVER)
    try:
        assert monitor.start() is True
        gc.collect(1)
        # No span was current: what it captured names no trace.
        assert [c.is_valid for _, _, c in monitor._passes] == [False]
        monitor.drain()
    finally:
        monitor.stop()
    warnings = [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.telemetry" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "generation 1" in warnings[0].getMessage()
    assert "trace_id=" not in warnings[0].getMessage()


async def test_a_short_pass_records_and_never_warns(meter, reader, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    monitor = _GcMonitor(meter, interval=NEVER)
    try:
        assert monitor.start() is True
        gc.collect(0)
        assert [c for _, _, c in monitor._passes] == [None]
        monitor.drain()
    finally:
        monitor.stop()
    assert [p.attributes for p in points(reader, "cpython.gc.collections")] == [
        {"cpython.gc.generation": 0}
    ]
    assert not [r for r in caplog.records if r.name == "kubed.mcp.telemetry"]


async def test_the_callback_records_nothing_and_the_drain_records(
    meter, reader
) -> None:
    monitor = _GcMonitor(meter, interval=NEVER)
    try:
        assert monitor.start() is True
        gc.collect(2)
        assert gc_points(reader) == []
        monitor.drain()
    finally:
        monitor.stop()
    collections = points(reader, "cpython.gc.collections")
    durations = points(reader, "cpython.gc.duration")
    assert [(p.attributes, p.value) for p in collections] == [
        ({"cpython.gc.generation": 2}, 1)
    ]
    assert [(p.attributes, p.count) for p in durations] == [
        ({"cpython.gc.generation": 2}, 1)
    ]
    assert list(durations[0].explicit_bounds) == BUCKETS
    data = reader.get_metrics_data()
    found = {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert found["cpython.gc.collections"].unit == "{collection}"
    assert found["cpython.gc.duration"].unit == "s"


async def test_stop_unregisters_cancels_and_empties(meter, reader) -> None:
    monitor = _GcMonitor(meter, interval=NEVER)
    try:
        assert monitor.start() is True
        assert _monitors() == [monitor._callback]
        task = monitor._task
        gc.collect(0)
    finally:
        monitor.stop()
    assert _monitors() == []
    await asyncio.sleep(0)
    assert task.cancelled()
    assert len(monitor._passes) == 0
    # The pass before the stop still reached the meter (ruling B).
    assert [p.value for p in points(reader, "cpython.gc.collections")] == [1]
    gc.collect()
    assert len(monitor._passes) == 0
    # Idempotent.
    monitor.stop()


def test_no_running_loop_captures_nothing(meter, reader) -> None:
    before = list(gc.callbacks)
    monitor = _GcMonitor(meter, interval=NEVER)
    assert monitor.start() is False
    assert gc.callbacks == before
    gc.collect()
    assert len(monitor._passes) == 0
    monitor.stop()


async def test_a_callback_that_fails_is_silent(
    meter, reader, caplog, monkeypatch
) -> None:
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    caplog.set_level(logging.DEBUG)

    def broken() -> float:
        raise RuntimeError("clock")

    monitor = _GcMonitor(meter, clock=broken, interval=NEVER)
    try:
        assert monitor.start() is True
        gc.collect()
        monitor.drain()
    finally:
        monitor.stop()
    assert unraisable == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_a_collection_in_another_thread_is_recorded(meter, reader) -> None:
    monitor = _GcMonitor(meter, interval=NEVER)
    try:
        assert monitor.start() is True
        worker = threading.Thread(target=gc.collect, args=(1,))
        worker.start()
        worker.join()
        monitor.drain()
    finally:
        monitor.stop()
    assert [
        (p.attributes, p.value) for p in points(reader, "cpython.gc.collections")
    ] == [({"cpython.gc.generation": 1}, 1)]


async def test_the_loop_drains_on_its_own(meter, reader) -> None:
    monitor = _GcMonitor(meter, interval=0.01)
    try:
        assert monitor.start() is True
        gc.collect(2)
        for _ in range(200):
            if not monitor._passes:
                break
            await asyncio.sleep(0.01)
        found = points(reader, "cpython.gc.collections")
    finally:
        monitor.stop()
    assert [p.attributes for p in found] == [{"cpython.gc.generation": 2}]
