"""What a server reports about itself: profiles, and OpenTelemetry.

Every signal is off until the server's config turns it on, and nothing a signal needs is
imported until then. The ``telemetry`` extra installs what the signals use.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import SecretStr

from ._config import Config, Field, config

if TYPE_CHECKING:
    from fastmcp.server.middleware import Middleware

log = logging.getLogger(__name__)

_HOSTNAME = re.compile(r"[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?")


@config("telemetry")
class TelemetrySettings(Config):
    """The ``telemetry`` section, registered in every server.

    ``profiles`` is the address of the Pyroscope server. Setting it turns profiles
    on; there is no separate enabled flag. ``profile_cpu`` and ``profile_memory``
    choose which profiles are sent. ``otel`` turns OpenTelemetry on, configured by
    the standard ``OTEL_*`` variables. ``tags`` are labels added to every profile
    and resource attributes on every trace and metric. An unknown key is an error,
    so a typo stops the boot. A server's own subclass, marked ``@config("telemetry")``,
    replaces it to add keys.
    """

    profiles: SecretStr | None = Field(
        None,
        doc="Pyroscope server address; setting it turns profiles on.",
        env="TELEMETRY_PROFILES",
        arg="--telemetry-profiles",
    )
    profile_cpu: bool = Field(
        True,
        doc="Sample CPU while profiles are on.",
        env="TELEMETRY_PROFILE_CPU",
        arg="--telemetry-profile-cpu",
    )
    profile_memory: bool = Field(
        True,
        doc="Sample allocations while profiles are on.",
        env="TELEMETRY_PROFILE_MEMORY",
        arg="--telemetry-profile-memory",
    )
    otel: bool = Field(
        False,
        doc="Run OpenTelemetry, configured by the standard OTEL_* variables.",
        env="TELEMETRY_OTEL",
        arg="--telemetry-otel",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        doc=(
            "Labels on every profile, and resource attributes on every trace "
            "and metric."
        ),
        env="TELEMETRY_TAGS",
        arg="--telemetry-tags",
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


def _start_signals(name: str, settings: TelemetrySettings | None) -> _Signals | None:
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


def _start_profiles(name: str, settings: TelemetrySettings) -> ModuleType | None:
    """Configure pyroscope, imported only here so an install without it still works."""
    if not settings.profiles or not (settings.profile_cpu or settings.profile_memory):
        return None
    try:
        import pyroscope

        started = pyroscope.configure(
            application_name=_service_name(name),
            server_address=settings.profiles.get_secret_value(),
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
    name: str, settings: TelemetrySettings, profiles: ModuleType | None
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


def _call_metrics(settings: TelemetrySettings | None) -> Middleware | None:
    """FastMCP's call metrics when ``settings.otel``, imported only then.

    Never raises.
    """
    try:
        if settings is None or not settings.otel:
            return None
        from fastmcp.server.middleware.usage import UsageMiddleware

        return UsageMiddleware(user_id="omit")
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error(
            "Call metrics did not start; serving without them: %s",
            type(error).__name__,
        )
        return None


def _where(address: str) -> str:
    """The host a Pyroscope address names, with its port: never userinfo, path or query.

    An address without ``://`` is read as ``//<address>``. A host is shown only when
    it is an IP address or looks like a hostname (letters, digits, dots and hyphens,
    starting and ending with a letter or digit) and the address gave a scheme, a valid
    port, or a host with a dot. Otherwise a bare token, or ``user:pw`` with no ``@``,
    could print a credential, so it shows ``an address with no host``. Never raises.
    """
    nothing = "an address with no host"
    schemed = "://" in address
    try:
        parts = urlsplit(address if schemed else f"//{address}")
        host = parts.hostname
    except ValueError:
        return nothing
    if not host:
        return nothing
    try:
        port: int | None = parts.port
        port_ok = True
    except ValueError:
        port, port_ok = None, False
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    if not is_ip and not _HOSTNAME.fullmatch(host):
        return nothing
    if not (schemed or (port_ok and port is not None) or "." in host):
        return nothing
    shown = f"[{host}]" if ":" in host else host
    return f"{shown}:{port}" if port is not None else shown


def _start_line(signals: _Signals | None, settings: TelemetrySettings | None) -> str:
    """The one line a start logs: which signals started and where they go.

    Reads only the handle, the settings and ``os.environ``. Never shows a credential:
    profiles are named by host and port alone.
    """
    parts: list[str] = []
    if (
        signals is not None
        and signals._profiles is not None
        and settings
        and settings.profiles
    ):
        parts.append(f"profiles \u2192 {_where(settings.profiles.get_secret_value())}")
    if signals is not None and signals._otel is not None:
        exporters = ", ".join(
            f"{kind} {(os.environ.get(variable) or '').strip() or default}"
            for kind, variable, default in (
                ("traces", "OTEL_TRACES_EXPORTER", "otlp"),
                ("metrics", "OTEL_METRICS_EXPORTER", "otlp"),
                ("logs", "OTEL_LOGS_EXPORTER", "none"),
            )
        )
        parts.append(f"otel \u2192 {exporters}")
    return "telemetry: " + ("; ".join(parts) if parts else "off")
