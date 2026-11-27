"""The sections every server has, ``serving``, ``telemetry`` and ``delivery``, and the
components that take them.

``_config`` composes ``BUILTINS`` first in every config, so each server reads them, and
``config_schema()`` shows them, with no ``QuickMCP``. A server changes one by
subclassing it under the same key. ``_builtin_components`` makes each server's own
components for them, registered and started before the server's. This module imports
no FastMCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, field_validator

from ._config import Config, Field, config
from .components import Component
from .telemetry import (
    TelemetrySettings,
    _call_metrics,
    _Signals,
    _start_line,
    _start_signals,
    log,
)

if TYPE_CHECKING:
    from .app import QuickMCP


@config("serving")
class ServingSettings(Config):
    """How ``QuickMCP.run()`` serves, and the level it sets on the root logger.

    A server changes a default by subclassing it, marked ``@config("serving")``, and
    restates ``Field(default, env=..., arg=...)`` on each field it changes: a field
    redefined without its names is refused. The log level is read case-insensitively,
    in a subclass too.
    """

    transport: Literal["stdio", "http", "sse", "streamable-http"] = Field(
        "stdio",
        doc="The transport FastMCP serves on.",
        env="TRANSPORT",
        arg="--transport",
    )
    host: str = Field(
        "127.0.0.1",
        doc="The address an HTTP transport listens on.",
        env="HOST",
        arg="--host",
    )
    port: int = Field(
        8000,
        doc="The port an HTTP transport listens on.",
        env="PORT",
        arg="--port",
    )
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = Field(
        "INFO",
        doc="The root logger's level, set when the server runs.",
        env="LOG_LEVEL",
        arg="--log-level",
    )
    graceful_timeout: float = Field(
        15,
        ge=0,
        doc=(
            "Seconds in-flight HTTP requests get to finish when the server stops; "
            "stdio doesn't read it."
        ),
        env="GRACEFUL_TIMEOUT",
        arg="--graceful-timeout",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        """``info`` reads as ``INFO``, as FastMCP's own settings read it."""
        return value.upper() if isinstance(value, str) else value


@config("delivery")
class DeliverySettings(Config):
    """The config's defaults for the delivery inputs: None leaves the built-in one.

    Each is a lax string, checked when the server is built: a value its input doesn't
    take is warned once and ignored, and the next level answers.
    """

    resources: str | None = Field(
        None,
        doc="Whether the client reads resources, on, or takes them as tools, off.",
        env="DELIVERY_RESOURCES",
        arg="--delivery-resources",
    )
    listed: str | None = Field(
        None,
        doc="What resources/list shows, a comma string of skills, prompts and agents.",
        env="DELIVERY_LISTED",
        arg="--delivery-listed",
    )
    via: str | None = Field(
        None,
        doc="How the client takes the plugins: mcp, extension or local.",
        env="DELIVERY_VIA",
        arg="--delivery-via",
    )


BUILTINS: tuple[type[BaseModel], ...] = (
    ServingSettings,
    TelemetrySettings,
    DeliverySettings,
)


class _ServingComponent(Component):
    """Takes ``serving``, which ``QuickMCP.run()`` reads: it does nothing else."""

    section = ServingSettings


class _TelemetryComponent(Component):
    """Takes ``telemetry``: the call metrics at registration, the signals while serving.

    Off imports nothing: neither FastMCP's metrics nor a signal's package. Registration
    always keeps a ``subscriptions/listen`` stream out of the server spans (TEL-22).

    The lifespan may be entered again before an earlier entry has exited, so each
    start's handle is kept on a stack, None when another start owns the process's
    signals, and each stop takes one off. The owning handle is stopped by the last
    stop, whichever entry exits last.
    """

    section = TelemetrySettings

    def __init__(self, name: str) -> None:
        self._name = name
        self._signals: list[_Signals | None] = []

    def register(self, mcp: QuickMCP, section: BaseModel | None) -> None:
        # A listen stream is the client's connection, not an operation (TEL-22): left
        # out of the server spans whatever the settings, as FastMCP traces under any
        # tracer provider.
        from ._listen import _leave_listen_untraced

        _leave_listen_untraced(mcp.server)
        # Outermost, ahead of the scopes, to record what the client received, the
        # call slot's teardown included (TEL-16).
        metrics = _call_metrics(cast("TelemetrySettings | None", section))
        if metrics is not None:
            mcp.server.middleware.insert(0, metrics)
            # Inside the pin and the prefix, the one place a read's plain URI is seen.
            metrics.observe(mcp.server, mcp._plugins)

    async def start(self, section: BaseModel | None) -> None:
        # Never raises (TEL-6). None when another start owns the process's signals.
        settings = cast("TelemetrySettings | None", section)
        signals = _start_signals(self._name, settings)
        self._signals.append(signals)
        log.info("%s", _start_line(signals, settings))

    async def stop(self) -> None:
        signals = self._signals.pop() if self._signals else None
        if signals is not None:
            signals.stop()


class _DeliveryComponent(Component):
    """Takes ``delivery``, which the server's resolver reads: it does nothing else."""

    section = DeliverySettings


def _builtin_components(name: str) -> tuple[Component, Component, Component]:
    """The built-in components of the server named ``name``, made for it alone."""
    return _ServingComponent(), _TelemetryComponent(name), _DeliveryComponent()
