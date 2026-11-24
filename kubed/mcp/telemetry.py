"""What a server reports about itself: profiles now, OpenTelemetry later.

Every signal is off until the server's config turns it on, and nothing a signal needs is
imported until then. The ``telemetry`` extra installs what the signals use.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class Telemetry(Protocol):
    """The settings telemetry reads."""

    @property
    def profiles(self) -> str | None: ...
    @property
    def profile_cpu(self) -> bool: ...
    @property
    def profile_memory(self) -> bool: ...
    @property
    def otel(self) -> bool: ...
    @property
    def tags(self) -> dict[str, str]: ...


class TelemetrySettings(BaseModel):
    """A ``telemetry`` section a server's config can embed.

    ``profiles`` is the address of the Pyroscope server. Setting it turns profiles
    on; there is no separate enabled flag. ``profile_cpu`` and ``profile_memory``
    choose which profiles are sent. ``otel`` is for OpenTelemetry, which a later
    release reads: nothing reads it yet. ``tags`` are labels added to every profile.
    """

    profiles: str | None = None
    profile_cpu: bool = True
    profile_memory: bool = True
    otel: bool = False
    tags: dict[str, str] = Field(default_factory=dict)
