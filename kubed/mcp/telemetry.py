"""What a server reports about itself: profiles now, OpenTelemetry later.

Every signal is off until the server's config turns it on, and nothing a signal needs is
imported until then. The ``telemetry`` extra installs what the signals use.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Protocol

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


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


class _Signals:
    """What a start began, so the same call can stop it."""

    def __init__(self, profiles: ModuleType | None) -> None:
        self._profiles = profiles

    def stop(self) -> None:
        """Stop what this start began. Never raises; frees the process's slot."""
        global _running
        try:
            if self._profiles is not None:
                self._profiles.shutdown()
        except Exception:  # TEL-6: telemetry never stops a server
            log.exception("Profiles did not stop cleanly")
        finally:
            _running = None


# Profilers are per-process, so one start owns the signals until it stops.
_running: _Signals | None = None


def _start_signals(name: str, settings: Telemetry | None) -> _Signals | None:
    """Start what ``settings`` turns on, unless a start already owns the process.

    Returns a handle only to the call that started something. Never raises.
    """
    global _running
    try:
        if _running is not None or settings is None:
            return None
        profiles = _start_profiles(name, settings)
        if profiles is None:
            return None
        _running = _Signals(profiles)
        return _running
    except Exception:  # TEL-6: telemetry never stops a server
        log.exception("Telemetry did not start; serving without it")
        return None


def _start_profiles(name: str, settings: Telemetry) -> ModuleType | None:
    """Configure pyroscope, imported only here so an install without it still works."""
    if not settings.profiles or not (settings.profile_cpu or settings.profile_memory):
        return None
    try:
        import pyroscope

        pyroscope.configure(
            application_name=name,
            server_address=settings.profiles,
            cpu_enabled=settings.profile_cpu,
            mem_enabled=settings.profile_memory,
            tags=dict(settings.tags),
        )
    except ModuleNotFoundError as error:
        if error.name != "pyroscope":
            # Some other module is missing: a failed start, not a missing package.
            log.exception("Profiles did not start; serving without them")
            return None
        log.warning(
            "Profiles are on, but pyroscope-io is not installed: "
            "install kubed-mcp[telemetry]"
        )
        return None
    except Exception:  # TEL-6: a profiler that fails never stops a server
        log.exception("Profiles did not start; serving without them")
        return None
    return pyroscope
