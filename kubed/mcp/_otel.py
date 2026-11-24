"""OpenTelemetry for a server that turns ``otel`` on: imported only then.

``start`` runs OpenTelemetry's standard configuration, ``initialize()``, read from
the ``OTEL_*`` variables, after filling a few defaults. Its providers are set once per
process, so once ours have stopped, no later start configures them again. An SDK
something else configured first is kept, and only log records are given its trace.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType
from urllib.parse import quote

# The SDK by name: without it a missing install is the one warning, not a failed start.
import opentelemetry.sdk.trace  # noqa: F401
from opentelemetry import _logs, metrics, trace
from opentelemetry.instrumentation.auto_instrumentation import initialize
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.logging.handler import (
    LoggingHandler as _InstrumentationHandler,
)
from opentelemetry.sdk._logs import LoggingHandler as _SdkHandler

from .telemetry import Telemetry

log = logging.getLogger(__name__)

# Set once our providers have shut down: OpenTelemetry can't be configured twice.
_stopped: bool = False

_SHUTDOWN_SECONDS = 5

# The loggers initialize() writes its failures to, each with a traceback.
_TRACEBACK_LOGGERS = (
    "opentelemetry.instrumentation.auto_instrumentation",
    "opentelemetry.instrumentation.auto_instrumentation._load",
)

# What sends records over OTLP: the logging instrumentor's handler, or the SDK's own.
_OTEL_HANDLERS = (_InstrumentationHandler, _SdkHandler)


def _resource_attributes(existing: str, tags: Mapping[str, str]) -> str:
    """``existing``, with the pod's keys and then ``tags`` added where it has none.

    The SDK splits on "," and then "=", strips keys and unquotes values, so a value is
    quoted and a key that holds either character is left off.
    """
    added: dict[str, str] = {}
    for key, variable in (
        ("k8s.namespace.name", "POD_NAMESPACE"),
        ("k8s.pod.name", "POD_NAME"),
    ):
        if value := os.environ.get(variable):
            added[key] = value
    for key, value in tags.items():
        if "," in key or "=" in key:
            log.warning(
                "Tag left off traces: "
                "a resource attribute key can't hold ',' or '=': %r",
                key,
            )
            continue
        added[key.strip()] = value
    pairs = [pair for pair in existing.split(",") if pair.strip()]
    written = {pair.split("=", 1)[0].strip() for pair in pairs}
    pairs += [
        f"{key}={quote(value, safe='')}"
        for key, value in added.items()
        if key not in written
    ]
    return ",".join(pairs)


def _set_defaults(name: str, tags: Mapping[str, str]) -> None:
    """Fill the variables a server leaves unset, before the distro fills its own.

    An empty variable counts as unset, as ``_service_name`` reads it for profiles.
    """
    for variable, default in (
        ("OTEL_SERVICE_NAME", name),
        # The HTTP exporter, the one the extra installs; the distro's default is gRPC.
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
        ("OTEL_LOGS_EXPORTER", "none"),
        ("OTEL_PYTHON_LOG_CORRELATION", "true"),
    ):
        if not os.environ.get(variable):
            os.environ[variable] = default
    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    if attributes := _resource_attributes(existing, tags):
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = attributes


def _sdk_configured() -> bool:
    """Whether something in the process already set a tracer provider."""
    return not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)


class _NoTraceback(logging.Filter):
    """Keeps the record and drops its traceback, which can carry a secret."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.exc_info = None
        record.exc_text = None
        return True


@contextmanager
def _without_tracebacks() -> Iterator[None]:
    """Inside the block, initialize()'s failures are logged without a traceback."""
    loggers = [logging.getLogger(name) for name in _TRACEBACK_LOGGERS]
    no_traceback = _NoTraceback()
    for logger in loggers:
        logger.addFilter(no_traceback)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(no_traceback)


@contextmanager
def _logging_kept() -> Iterator[None]:
    """Root's handlers and level as they were, but for OpenTelemetry's handler.

    The logging instrumentor's basicConfig gives a bare root a stderr handler and INFO.
    Its own handler stays only while ``OTEL_LOGS_EXPORTER`` sends records somewhere:
    otherwise it drops every record and keeps ``logging.lastResort`` from printing it.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        yield
    finally:
        exported = os.environ.get("OTEL_LOGS_EXPORTER", "none").strip().lower()
        keep = _OTEL_HANDLERS if exported != "none" else ()
        root.setLevel(level)
        for handler in list(root.handlers):
            if handler not in handlers and not isinstance(handler, keep):
                root.removeHandler(handler)
        for handler in handlers:
            if handler not in root.handlers:
                root.addHandler(handler)


def _instrument_logging() -> bool:
    """Put the current trace on log records, unless something already does.

    Returns whether this call did, so the stop undoes only that. The instrumentor
    reads ``OTEL_PYTHON_LOG_CORRELATION``, which ``_set_defaults`` turns on.
    """
    instrumentor = LoggingInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        return False
    with _logging_kept():
        instrumentor.instrument()
    # instrument() returns without instrumenting when its dependencies conflict.
    return instrumentor.is_instrumented_by_opentelemetry


def _shutdown(kind: str, provider: object) -> None:
    """Shut ``provider`` down, waiting at most ``_SHUTDOWN_SECONDS`` for it."""
    if kind == "meter":
        provider.shutdown(timeout_millis=_SHUTDOWN_SECONDS * 1000)
        return

    # The tracer and logger providers take no timeout, and against a collector that's
    # down their batch processors block for tens of seconds: a stop leaves them behind.
    def run() -> None:
        try:
            provider.shutdown()
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("OpenTelemetry did not stop cleanly: %s", type(error).__name__)

    thread = threading.Thread(target=run, name=f"otel-{kind}-shutdown", daemon=True)
    thread.start()
    thread.join(_SHUTDOWN_SECONDS)
    if thread.is_alive():
        log.warning("OpenTelemetry did not stop in time: %s", kind)


class _Otel:
    """What a start set up, so its stop undoes what it owns and nothing else."""

    def __init__(
        self,
        *,
        owned: bool,
        logging_instrumented: bool,
        link: object | None = None,
    ) -> None:
        self._owned = owned
        self._logging_instrumented = logging_instrumented
        self._link = link

    def stop(self) -> None:
        """Close the link, uninstrument logging, stop our providers. Never raises."""
        global _stopped
        try:
            if self._link is not None and hasattr(self._link, "close"):
                self._link.close()
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("OpenTelemetry did not stop cleanly: %s", type(error).__name__)
        try:
            if self._logging_instrumented:
                LoggingInstrumentor().uninstrument()
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("OpenTelemetry did not stop cleanly: %s", type(error).__name__)
        if not self._owned:
            return
        for kind, provided in (
            ("tracer", trace.get_tracer_provider),
            ("meter", metrics.get_meter_provider),
            ("logger", _logs.get_logger_provider),
        ):
            try:
                provider = provided()
                if hasattr(provider, "shutdown"):
                    _shutdown(kind, provider)
            except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
                log.error(
                    "OpenTelemetry did not stop cleanly: %s", type(error).__name__
                )
        _stopped = True


def start(name: str, settings: Telemetry, profiles: ModuleType | None) -> _Otel | None:
    """Configure OpenTelemetry for the server ``name``. Never raises.

    ``profiles`` is pyroscope when profiles started, else ``None``.
    """
    try:
        if _stopped:
            log.warning(
                "OpenTelemetry runs once per process, and it has stopped: "
                "serving without it"
            )
            return None
        _set_defaults(name, settings.tags)
        if _sdk_configured():
            # Someone else's providers: kept as they are, and never shut down by us.
            try:
                instrumented = _instrument_logging()
            except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
                # Their providers still trace: only the records go without the trace.
                log.error(
                    "OpenTelemetry did not instrument logging: %s", type(error).__name__
                )
                instrumented = False
            return _Otel(owned=False, logging_instrumented=instrumented)
        instrumentor = LoggingInstrumentor()
        before = instrumentor.is_instrumented_by_opentelemetry
        try:
            with _logging_kept(), _without_tracebacks():
                initialize(swallow_exceptions=False)
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error(
                "OpenTelemetry did not start; serving without it: %s",
                type(error).__name__,
            )
            # A configure that failed halfway may have set providers: they go too.
            _Otel(
                owned=_sdk_configured(),
                logging_instrumented=(
                    instrumentor.is_instrumented_by_opentelemetry and not before
                ),
            ).stop()
            return None
        # Ours only if initialize() set a provider: else a stop would claim one stopped.
        return _Otel(
            owned=_sdk_configured(),
            logging_instrumented=instrumentor.is_instrumented_by_opentelemetry
            and not before,
        )
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error(
            "OpenTelemetry did not start; serving without it: %s", type(error).__name__
        )
        return None
