"""The sections every server has: ``serving``, ``telemetry`` and ``delivery``.

``_config`` composes ``BUILTINS`` first in every config, so each server reads them, and
``config_schema()`` shows them, with no ``QuickMCP``. A server changes one by
subclassing it under the same key. This module imports no FastMCP.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ._config import Config, Field, config
from .telemetry import TelemetrySettings


@config("serving")
class ServingSettings(Config):
    """How ``QuickMCP.run()`` serves, and the level it sets on the root logger.

    A server changes a default by subclassing it, marked ``@config("serving")``, and
    restates ``Field(default, env=..., arg=...)`` on each field it changes: a field
    redefined without its names is refused.
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
