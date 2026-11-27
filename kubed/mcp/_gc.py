"""Garbage-collection counts and pause times, per generation (TEL-23).

Imported only with ``otel`` on, by ``_otel``. A collection stops every thread, the
event loop's included, so a long one is a stall: this names it.

The callback that ``gc`` runs inside a collection does no I/O, no logging and no
recording on an instrument: it reads the clock and queues the pass. The SDK's
aggregation lock is a plain ``Lock``, which its metric reader holds while it allocates,
so a record from inside a collection the reader triggered would wait on that lock
forever. A task on the event loop drains the queue every second, recording each pass
and warning of a long one. With no running loop nothing is registered, and nothing is
captured.
"""

from __future__ import annotations

import asyncio
import collections
import gc
import logging
import time
from collections.abc import Callable
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanContext

log = logging.getLogger("kubed.mcp.telemetry")

# Seconds: the default buckets assume milliseconds.
_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
# Passes queued between drains: past it, the oldest go.
_QUEUED = 4096

_Pass = tuple[int, float, "SpanContext | None"]


class _GcMonitor:
    """Queues each collection's generation and pause, and records them from the loop.

    ``meter`` makes the two instruments, ``clock`` times a pass, a pass of
    ``threshold`` seconds or more also warns, and the loop drains every ``interval``
    seconds.
    """

    def __init__(
        self,
        meter: Any,
        *,
        clock: Callable[[], float] = time.perf_counter,
        threshold: float = 0.5,
        interval: float = 1.0,
    ) -> None:
        self._meter = meter
        self._clock = clock
        self._threshold = threshold
        self._interval = interval
        self._on = False
        # When the pass under way started; None outside one.
        self._started: float | None = None
        self._passes: collections.deque[_Pass] = collections.deque(maxlen=_QUEUED)
        self._task: asyncio.Task[None] | None = None
        self._collections: Any = None
        self._duration: Any = None
        self._failed = False
        # Stored once: what start appends to gc.callbacks is what stop removes.
        self._callback = self._on_gc

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        """Runs inside a collection: the clock and the queue, and nothing else."""
        try:
            if not self._on:
                return
            if phase == "start":
                self._started = self._clock()
            elif phase == "stop" and self._started is not None:
                duration = self._clock() - self._started
                self._started = None
                # A contextvar read: no lock, no I/O.
                context = (
                    trace.get_current_span().get_span_context()
                    if duration >= self._threshold
                    else None
                )
                self._passes.append((info["generation"], duration, context))
        except Exception:  # noqa: BLE001  # TEL-6: inside a collection, a failure is dropped silently
            pass

    def start(self) -> bool:
        """Register the callback and the drain, on the running loop. Never raises.

        With no running loop, returns False and registers nothing.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        try:
            self._collections = self._meter.create_counter(
                "cpython.gc.collections",
                unit="{collection}",
                description="Garbage collections run.",
            )
            self._duration = self._meter.create_histogram(
                "cpython.gc.duration",
                unit="s",
                description="How long each garbage collection paused the process.",
                explicit_bucket_boundaries_advisory=_BUCKETS,
            )
            # Loaded now, so its first use never happens inside a collection.
            trace.get_current_span()
            self._on = True
            gc.callbacks.append(self._callback)
            self._task = loop.create_task(self._drain_every(), name="kubed-gc-drain")
            return True
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("GC metrics did not start: %s", type(error).__name__)
            self.stop()
            return False

    async def _drain_every(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            self.drain()

    def drain(self) -> None:
        """Record each pass queued so far, and warn of each long one. Never raises.

        It runs under an empty context: the task's is a copy of the start's, and the
        stop's is the lifespan exit's, so either could hold a span that a pass, its
        warning or its points' exemplars would wrongly name.
        """
        try:
            token = otel_context.attach(otel_context.Context())
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            self._failure(error)
            return
        try:
            self._drain()
        finally:
            try:
                otel_context.detach(token)
            except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
                self._failure(error)

    def _drain(self) -> None:
        try:
            # The passes queued now: the drain's own allocations may queue more.
            for _ in range(len(self._passes)):
                generation, duration, context = self._passes.popleft()
                attributes = {"cpython.gc.generation": generation}
                if self._collections is not None:
                    self._collections.add(1, attributes)
                if self._duration is not None:
                    self._duration.record(duration, attributes)
                if duration >= self._threshold:
                    self._warn(generation, duration, context)
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            self._failure(error)

    def _failure(self, error: Exception) -> None:
        """Log the drain's first failure, by type only."""
        if not self._failed:
            self._failed = True
            log.error("GC metrics failed: %s", type(error).__name__)

    def _warn(
        self, generation: int, duration: float, context: SpanContext | None
    ) -> None:
        """One warning, in the span the pass stalled when one was current."""
        message = "gc: a generation %d collection paused the process for %.2f s"
        if context is not None and context.is_valid:
            with trace.use_span(trace.NonRecordingSpan(context)):
                log.warning(message, generation, duration)
        else:
            log.warning(message, generation, duration)

    def stop(self) -> None:
        """Unregister, cancel the drain, record what is left. Never raises.

        The last drain runs before the meter's provider shuts down, so the passes
        since the one before still reach it. Stopping again does nothing more.
        """
        self._on = False
        try:
            if self._callback in gc.callbacks:
                gc.callbacks.remove(self._callback)
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("GC metrics did not stop cleanly: %s", type(error).__name__)
        try:
            task, self._task = self._task, None
            if task is not None:
                task.cancel()
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("GC metrics did not stop cleanly: %s", type(error).__name__)
        self.drain()
        self._passes.clear()
