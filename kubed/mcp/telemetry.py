"""What a server reports about itself: profiles, and OpenTelemetry.

Every signal is off until the server's config turns it on, and nothing a signal needs is
imported until then. The ``telemetry`` extra installs what the signals use.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from types import ModuleType
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)


class Telemetry(Protocol):
    """What a config's ``telemetry`` section provides.

    ``TelemetrySettings`` satisfies it, and so does a server's own section.
    """

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
    choose which profiles are sent. ``otel`` turns OpenTelemetry on, configured by
    the standard ``OTEL_*`` variables. ``tags`` are labels added to every profile
    and resource attributes on every trace and metric. An unknown key is an error,
    so a typo stops the boot; a server's own section may subclass it to add keys.
    """

    model_config = ConfigDict(extra="forbid")

    profiles: str | None = Field(
        default=None,
        description="Pyroscope server address; setting it turns profiles on.",
    )
    profile_cpu: bool = Field(
        default=True, description="Sample CPU while profiles are on."
    )
    profile_memory: bool = Field(
        default=True, description="Sample allocations while profiles are on."
    )
    otel: bool = Field(
        default=False,
        description="Run OpenTelemetry, configured by the standard OTEL_* variables.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Labels on every profile, and resource attributes on every trace "
            "and metric."
        ),
    )


class _Signals:
    """What a start began, so the same call can stop it."""

    def __init__(self, profiles: ModuleType | None, otel: object | None = None) -> None:
        self._profiles = profiles
        self._otel = otel

    def stop(self) -> None:
        """Stop what this start began. Never raises; frees the process's slot.

        OpenTelemetry stops first, so no span still tags a thread once profiles stop.
        """
        global _running
        try:
            if self._otel is not None:
                self._otel.stop()
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("OpenTelemetry did not stop cleanly: %s", type(error).__name__)
        try:
            if self._profiles is not None and self._profiles.shutdown() is False:
                log.warning("Profiles did not stop cleanly")
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("Profiles did not stop cleanly: %s", type(error).__name__)
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
        otel = _start_otel(name, settings, profiles)
        if profiles is None and otel is None:
            return None
        _running = _Signals(profiles, otel)
        return _running
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error(
            "Telemetry did not start; serving without it: %s", type(error).__name__
        )
        return None


def _service_name(name: str) -> str:
    """The service's name: OTEL_SERVICE_NAME when set and not empty, else ``name``."""
    return os.environ.get("OTEL_SERVICE_NAME") or name


def _profile_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """The pod's labels from POD_NAMESPACE and POD_NAME, then ``tags``, which win."""
    labels: dict[str, str] = {}
    for label, variable in (("namespace", "POD_NAMESPACE"), ("pod", "POD_NAME")):
        if value := os.environ.get(variable):
            labels[label] = value
    labels.update(tags)
    return labels


def _start_profiles(name: str, settings: Telemetry) -> ModuleType | None:
    """Configure pyroscope, imported only here so an install without it still works."""
    if not settings.profiles or not (settings.profile_cpu or settings.profile_memory):
        return None
    try:
        import pyroscope

        started = pyroscope.configure(
            application_name=_service_name(name),
            server_address=settings.profiles,
            cpu_enabled=settings.profile_cpu,
            mem_enabled=settings.profile_memory,
            tags=_profile_tags(settings.tags),
        )
    except ModuleNotFoundError as error:
        if error.name != "pyroscope":
            # Some other module is missing: a failed start, not a missing package.
            log.error(
                "Profiles did not start; serving without them: %s",
                type(error).__name__,
            )
            return None
        log.warning(
            "Profiles are on, but pyroscope-io is not installed: "
            "install kubed-mcp[telemetry]"
        )
        return None
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error(
            "Profiles did not start; serving without them: %s", type(error).__name__
        )
        return None
    if started is False:
        # Not ours to stop: another caller's agent already runs in this process.
        log.error("Profiles did not start: a profiler already runs in this process")
        return None
    return pyroscope


def _start_otel(
    name: str, settings: Telemetry, profiles: ModuleType | None
) -> object | None:
    """Start OpenTelemetry when ``settings.otel``, imported only then. Never raises."""
    try:
        if not settings.otel:
            return None
        from . import _otel
    except ModuleNotFoundError as error:
        if not (error.name or "").startswith("opentelemetry"):
            # Some other module is missing: a failed start, not a missing package.
            log.error(
                "OpenTelemetry did not start; serving without it: %s",
                type(error).__name__,
            )
            return None
        log.warning(
            "OpenTelemetry is on, but its packages are not installed: "
            "install kubed-mcp[telemetry]"
        )
        return None
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error(
            "OpenTelemetry did not start; serving without it: %s", type(error).__name__
        )
        return None
    return _otel.start(name, settings, profiles)
