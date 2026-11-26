"""What a server reports about itself: profiles, and OpenTelemetry.

Every signal is off until the server's config turns it on, and nothing a signal needs is
imported until then. The ``telemetry`` extra installs what the signals use.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import threading
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from pydantic import SecretStr

from ._config import Config, Field, config

if TYPE_CHECKING:
    from ._metrics import _CallMetrics

log = logging.getLogger(__name__)

_HOSTNAME = re.compile(r"[A-Za-z0-9_]([A-Za-z0-9._-]*[A-Za-z0-9_])?")

# Seconds between profile pushes, and so between probes.
_UPLOAD_INTERVAL = 10
# Seconds one probe waits for Pyroscope: under the interval.
_PROBE_TIMEOUT = 5.0


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

    def __init__(
        self,
        profiles: ModuleType | None,
        otel: object | None = None,
        probe: _Probe | None = None,
    ) -> None:
        self._profiles = profiles
        self._otel = otel
        self._probe = probe

    def stop(self) -> None:
        """Stop what this start began. Never raises; frees the process's slot.

        The probe stops first. OpenTelemetry stops next, so no span still tags a thread
        once profiles stop.
        """
        global _running
        try:
            if self._probe is not None:
                self._probe.stop()
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error(
                "The profiles probe did not stop cleanly: %s", type(error).__name__
            )
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
        probe = None
        try:
            # After OpenTelemetry, so its log handler sees the probe's first warning.
            if profiles is not None and settings.profiles:
                probe = _start_probe(settings.profiles.get_secret_value())
            if profiles is None and otel is None:
                return None
            _running = _Signals(profiles, otel, probe)
            return _running
        except BaseException:
            _stop_probe(probe)
            raise
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error(
            "Telemetry did not start; serving without it: %s", type(error).__name__
        )
        return None


def _stop_probe(probe: _Probe | None) -> None:
    """Stop a probe that no handle will stop. Never raises."""
    try:
        if probe is not None:
            probe.stop()
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error("The profiles probe did not stop cleanly: %s", type(error).__name__)


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
            upload_interval=_UPLOAD_INTERVAL,
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


def _call_metrics(settings: TelemetrySettings | None) -> _CallMetrics | None:
    """FastMCP's call metrics, with resource reads, when ``settings.otel``: imported
    only then.

    Never raises.
    """
    try:
        if settings is None or not settings.otel:
            return None
        from ._metrics import _CallMetrics

        return _CallMetrics(user_id="omit")
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error(
            "Call metrics did not start; serving without them: %s",
            type(error).__name__,
        )
        return None


def _split_userinfo(rest: str) -> tuple[str | None, str] | None:
    """Split an address without its scheme into userinfo and the rest, or ``None``.

    Userinfo ends at the last ``@``. An ``@`` after a ``?`` or ``#`` makes the split
    ambiguous, since the text before it may be a password holding an unencoded ``?``:
    that returns ``None``.
    """
    if "@" not in rest:
        return None, rest
    query = min((i for i in (rest.find("?"), rest.find("#")) if i >= 0), default=-1)
    if query >= 0 and "@" in rest[query:]:
        return None
    userinfo, _, host = rest.rpartition("@")
    return userinfo, host


def _where(address: str) -> str:
    """The host a Pyroscope address names, with its port: never userinfo, path or query.

    An address without ``://`` is read as ``//<address>``. Userinfo ends at the last
    ``@``, so a password with an unencoded ``/`` is never shown. An ``@`` after a ``?``
    or ``#`` shows nothing: the text before it may be a password holding an unencoded
    ``?``, and the text after it is the query's. A host is shown only when it is an IP
    address or looks like a hostname (letters, digits, dots, hyphens and underscores,
    starting and ending with a letter, digit or underscore) and the address gave a
    scheme, a valid port, or a host with a dot. Otherwise a bare token, or ``user:pw``
    with no ``@``, could print a credential, so it shows ``an address with no host``.
    Never raises.
    """
    nothing = "an address with no host"
    schemed = "://" in address
    scheme, _, rest = address.partition("://") if schemed else ("", "", address)
    split = _split_userinfo(rest)
    if split is None:
        return nothing
    rest = split[1]
    address = f"{scheme}://{rest}" if schemed else rest
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


def _ready(address: str) -> tuple[str, str | None]:
    """The URL to probe and the ``Authorization`` header to send, or ``None`` for none.

    An address without ``://`` gets ``http://``. Userinfo is split as ``_where`` splits
    it and sent as Basic auth; an empty one sends none. An address that can't be split
    that way raises ``ValueError``, so nothing is probed. So does a userinfo holding a
    ``/`` or ``\\``: the agent ends an http host at the first of either, so it would
    send the credentials to the host after the last ``@``. The URL is the host and port,
    the path without trailing slashes, then ``/ready``: the query and fragment are
    dropped.
    """
    scheme, _, rest = (
        address.partition("://") if "://" in address else ("http", "", address)
    )
    split = _split_userinfo(rest)
    if split is None:
        raise ValueError("ambiguous userinfo")
    userinfo, rest = split
    if userinfo is not None and ("/" in userinfo or "\\" in userinfo):
        raise ValueError("ambiguous userinfo")
    rest = re.split(r"[?#]", rest, maxsplit=1)[0]
    authority, slash, path = rest.partition("/")
    url = f"{scheme}://{authority}{(slash + path).rstrip('/')}/ready"
    if not userinfo:
        return url, None
    from base64 import b64encode

    user, _, password = userinfo.partition(":")
    token = b64encode(f"{unquote(user)}:{unquote(password)}".encode()).decode("ascii")
    return url, f"Basic {token}"


def _kind(error: BaseException) -> str:
    """What kind of failure ``error`` is, in words that carry no text of its own."""
    import http.client
    import socket
    import ssl
    from urllib.error import HTTPError, URLError

    if isinstance(error, HTTPError):
        code = error.code
        if code in (401, 403):
            return f"auth (HTTP {code})"
        if 500 <= code < 600:
            return f"server error (HTTP {code})"
        return ""
    if isinstance(error, URLError):
        # A reason that is not an exception is urllib's own, such as an unknown scheme.
        if isinstance(error.reason, Exception):
            return _kind(error.reason)
        return "invalid address"
    if isinstance(error, ConnectionRefusedError):
        return "connection refused"
    if isinstance(error, socket.gaierror):
        return "DNS"
    if isinstance(error, ssl.SSLError):
        return "TLS"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, http.client.InvalidURL):
        return "invalid address"
    if isinstance(error, OSError | http.client.HTTPException):
        return "connection error"
    if isinstance(error, ValueError):
        return "invalid address"
    return type(error).__name__


def _outcome(address: str, timeout: float) -> str | None:
    """Ask Pyroscope's ``/ready``: ``None`` when it answers, else the kind of failure.

    A redirect is not followed, so the credentials reach no other host, and counts as
    an answer. So does any status but 401, 403 and 5xx. Never raises.
    """
    try:
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        url, authorization = _ready(address)
        request = urllib.request.Request(url)
        if authorization is not None:
            request.add_header("Authorization", authorization)
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(request, timeout=timeout):
            pass
    except Exception as error:  # noqa: BLE001  # TEL-6: classified by type only, never its text
        _close(error)
        return _kind(error) or None
    return None


def _close(error: BaseException) -> None:
    """Close the response an ``HTTPError`` holds, freeing its socket. Never raises."""
    from urllib.error import HTTPError

    try:
        if isinstance(error, HTTPError):
            error.close()
    except Exception:  # noqa: BLE001  # a close that fails changes no outcome
        pass


class _Probe:
    """Checks that Pyroscope answers, on the push interval, and logs a change.

    It checks reachability, not the push itself: pyroscope-io 1.4 gives Python no status
    for a push. It logs one warning when Pyroscope first fails to answer, nothing more
    while it stays unreachable, and one line when it answers again. It runs on a daemon
    thread of its own, never tied to the event loop.
    """

    def __init__(
        self,
        address: str,
        *,
        interval: float = _UPLOAD_INTERVAL,
        timeout: float = _PROBE_TIMEOUT,
    ) -> None:
        self._address = address
        self._interval = interval
        self._timeout = timeout
        self._where = _where(address)
        self._down = False
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def check(self) -> None:
        """Probe once and log a change. Never raises."""
        try:
            kind = _outcome(self._address, self._timeout)
            if kind is not None and not self._down:
                log.warning(
                    "profiles: pyroscope at %s unreachable: %s", self._where, kind
                )
            elif kind is None and self._down:
                log.info("profiles: pyroscope at %s reachable again", self._where)
            self._down = kind is not None
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("The profiles probe failed: %s", type(error).__name__)

    def _run(self) -> None:
        while not self._stopping.is_set():
            self.check()
            self._stopping.wait(self._interval)

    def start(self) -> None:
        """Start the thread; the first probe runs at once."""
        self._thread = threading.Thread(
            target=self._run, name="pyroscope-probe", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread, waiting at most the timeout. Never raises."""
        try:
            self._stopping.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(self._timeout)
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error("The profiles probe did not stop: %s", type(error).__name__)


def _start_probe(address: str) -> _Probe | None:
    """Start a probe of ``address``. Never raises; tests replace it."""
    try:
        probe = _Probe(address)
        probe.start()
        return probe
    except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
        log.error("The profiles probe did not start: %s", type(error).__name__)
        return None


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
