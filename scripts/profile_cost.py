"""What profiling costs a bare server: its CPU idle and under load, profiles off and on.

``measure`` starts this script's own ``serve`` as a child, once per configuration and
run, and reads the child's CPU time from ``/proc``: Linux only. Each run waits out a
warm-up, measures an idle phase with no client, then a load phase of ``ping`` calls at a
fixed rate, and prints one Markdown table of millicores.

The Pyroscope address is the parent's own ``TELEMETRY_PROFILES``, the variable a server
reads, handed to the child in its environment. It is never printed.

    python scripts/profile_cost.py measure [--configs off,cpu,memory,both] [--runs 3]
        [--warmup 15] [--idle 60] [--load 30] [--rate 10] [--option KEY=VALUE ...]
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fastmcp import Client

from kubed.mcp import QuickMCP

# The telemetry variables each configuration sets in the child: None removes one, and a
# configuration without TELEMETRY_PROFILES keeps the parent's. Every other TELEMETRY_*
# variable is inherited as it is.
CONFIGS: dict[str, dict[str, str | None]] = {
    "off": {
        "TELEMETRY_PROFILES": None,
        "TELEMETRY_PROFILE_CPU": "true",
        "TELEMETRY_PROFILE_MEMORY": "true",
        "TELEMETRY_OTEL": "false",
    },
    "cpu": {
        "TELEMETRY_PROFILE_CPU": "true",
        "TELEMETRY_PROFILE_MEMORY": "false",
        "TELEMETRY_OTEL": "false",
    },
    "memory": {
        "TELEMETRY_PROFILE_CPU": "false",
        "TELEMETRY_PROFILE_MEMORY": "true",
        "TELEMETRY_OTEL": "false",
    },
    "both": {
        "TELEMETRY_PROFILE_CPU": "true",
        "TELEMETRY_PROFILE_MEMORY": "true",
        "TELEMETRY_OTEL": "false",
    },
    "otel": {
        "TELEMETRY_PROFILES": None,
        "TELEMETRY_PROFILE_CPU": "true",
        "TELEMETRY_PROFILE_MEMORY": "true",
        "TELEMETRY_OTEL": "true",
    },
}

# The pyroscope.configure keywords QuickMCP passes from its own settings, and app_name,
# configure's alias for application_name: an option may not override them.
# upload_interval is QuickMCP's too, but overriding it is the point.
OWNED = frozenset(
    {
        "application_name",
        "app_name",
        "server_address",
        "cpu_enabled",
        "mem_enabled",
        "tags",
    }
)

# The pyroscope.configure keywords that carry a credential: an option's value prints in
# its row's name, and a credential belongs in the TELEMETRY_PROFILES address instead.
SECRET = frozenset({"basic_auth_username", "basic_auth_password", "http_headers"})

# What QuickMCP logs when profiles are on but did not start; it serves on without them.
_NOT_PROFILED = (
    "Profiles did not start",
    "Profiles are on, but pyroscope-io is not installed",
)

# Seconds a child has to accept a connection, and then to stop once asked.
_BOOT_TIMEOUT = 60.0
_STOP_TIMEOUT = 15.0
_OPTIONS_VARIABLE = "PROFILE_COST_OPTIONS"


@dataclass(frozen=True)
class Sample:
    """One run of one configuration: CPU in millicores, and the calls the load made."""

    idle: float
    load: float
    calls: int


class ChildFailed(Exception):
    """The child exited, never accepted a connection, or served without its profiles.

    ``log`` holds its stderr.
    """

    def __init__(self, config: str, what: str, log: Path) -> None:
        super().__init__(f"the {config} server {what}; its stderr is in {log}")


def _profiled(config: str) -> bool:
    """Whether ``config`` keeps the parent's Pyroscope address."""
    return "TELEMETRY_PROFILES" not in CONFIGS[config]


def cpu_seconds(pid: int) -> float:
    """The CPU time ``pid`` has used, user and system, over all its threads."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    # The command name may hold spaces and parentheses: the fields start after the
    # last ")", at the third, so utime (the 14th) and stime (the 15th) are 11 and 12.
    fields = stat[stat.rindex(")") + 1 :].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def parse_option(text: str) -> tuple[str, object]:
    """``KEY=VALUE``, its value read as JSON when it parses, else as a string."""
    key, sep, raw = text.partition("=")
    if not sep or not key:
        raise ValueError(f"expected KEY=VALUE, not {text!r}")
    try:
        value: object = json.loads(raw)
    except ValueError:
        value = raw
    return key, value


def child_env(
    config: str, base: Mapping[str, str], port: int, options: Mapping[str, object]
) -> dict[str, str]:
    """The child's environment: ``base`` with ``config``'s variables and the serving."""
    env = dict(base)
    # A config file can set telemetry.profiles, which would profile the off rows.
    env.pop("CONFIG_FILE", None)
    for name, value in CONFIGS[config].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    env.update(TRANSPORT="http", HOST="127.0.0.1", PORT=str(port), LOG_LEVEL="WARNING")
    env.pop(_OPTIONS_VARIABLE, None)
    if options and _profiled(config):
        env[_OPTIONS_VARIABLE] = json.dumps(dict(options))
    return env


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_serving(
    child: subprocess.Popen[bytes], config: str, port: int, log: Path
) -> None:
    deadline = time.monotonic() + _BOOT_TIMEOUT
    while True:
        if child.poll() is not None:
            raise ChildFailed(config, "exited", log)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise ChildFailed(
                    config, "did not accept a connection within 60 s", log
                ) from None
            time.sleep(0.1)


def _millicores(seconds: float, started: float) -> float:
    """CPU ``seconds`` over the wall time since ``started``, in millicores."""
    return seconds / (time.monotonic() - started) * 1000


async def _load(port: int, pid: int, load: float, rate: float) -> tuple[float, int]:
    """Call ``ping`` ``rate * load`` times, call ``i`` at ``i / rate``: millicores."""
    calls = round(rate * load)
    async with Client(f"http://127.0.0.1:{port}/mcp") as client:
        before, started = cpu_seconds(pid), time.monotonic()
        for i in range(calls):
            # A late call goes at once, so every configuration makes the same calls.
            delay = started + i / rate - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await client.call_tool("ping", {})
        rest = started + load - time.monotonic()
        if rest > 0:
            await asyncio.sleep(rest)
        used = cpu_seconds(pid) - before
        return _millicores(used, started), calls


def measure_once(
    config: str,
    options: Mapping[str, object],
    *,
    warmup: float,
    idle: float,
    load: float,
    rate: float,
    log: Path | None = None,
) -> Sample:
    """Start a child in ``config``, measure its idle and load phases, and stop it.

    ``log`` holds the child's output, a new temporary file when None. Raises
    ``ChildFailed`` when the child exits, doesn't accept a connection in 60 s, or, in a
    profile configuration, logs that its profiles did not start.
    """
    if log is None:
        log = Path(tempfile.mkdtemp(prefix="profile-cost-")) / f"{config}.log"
    port = _free_port()
    env = child_env(config, os.environ, port, options)
    with log.open("wb") as output:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "serve"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
        )
        try:
            _wait_until_serving(child, config, port, log)
            time.sleep(warmup)
            before, started = cpu_seconds(child.pid), time.monotonic()
            time.sleep(idle)
            idle_cost = _millicores(cpu_seconds(child.pid) - before, started)
            if child.poll() is not None:
                raise ChildFailed(config, "exited", log)
            try:
                load_cost, calls = asyncio.run(_load(port, child.pid, load, rate))
            except Exception:
                if child.poll() is not None:
                    raise ChildFailed(config, "exited", log) from None
                raise
            if child.poll() is not None:
                raise ChildFailed(config, "exited", log)
            if _profiled(config) and _did_not_profile(log):
                raise ChildFailed(config, "served without its profiles", log)
            return Sample(idle=idle_cost, load=load_cost, calls=calls)
        finally:
            _stop(child)


def _did_not_profile(log: Path) -> bool:
    """Whether the child's ``log`` says its profiles did not start."""
    text = log.read_text(errors="replace")
    return any(message in text for message in _NOT_PROFILED)


def _stop(child: subprocess.Popen[bytes]) -> None:
    """SIGTERM, up to 15 s for pyroscope to flush, then SIGKILL."""
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _spread(values: Sequence[float]) -> str:
    return (
        f"{round(statistics.median(values))} "
        f"({round(min(values))}\u2013{round(max(values))})"
    )


def table(results: Mapping[str, Sequence[Sample]]) -> str:
    """A Markdown table, one row per configuration: medians, then lowest and highest."""
    lines = [
        "| config | idle (millicores) | load (millicores) | calls per run | runs |",
        "|---|---|---|---|---|",
    ]
    for name, samples in results.items():
        idle = _spread([s.idle for s in samples])
        load = _spread([s.load for s in samples])
        lines.append(
            f"| {name} | {idle} | {load} | {samples[0].calls} | {len(samples)} |"
        )
    return "\n".join(lines)


def serve() -> None:
    """The child: one ``ping`` tool, served as its environment's config says.

    ``PROFILE_COST_OPTIONS`` holds keywords for ``pyroscope.configure``, passed on top
    of QuickMCP's own.
    """
    raw = os.environ.get(_OPTIONS_VARIABLE)
    if raw:
        import pyroscope

        options = json.loads(raw)
        real = pyroscope.configure

        def configure(**kwargs: object) -> object:
            return real(**{**kwargs, **options})

        pyroscope.configure = configure

    mcp = QuickMCP("profile-cost")

    @mcp.tool
    def ping() -> str:
        return "pong"

    # LOG_LEVEL doesn't reach uvicorn's access log, which would write a line a call.
    mcp.run(uvicorn_config={"access_log": False})


def _refused(options: Mapping[str, object]) -> str | None:
    """Why ``options`` can't go to ``pyroscope.configure``, or None when they can."""
    if not options:
        return None
    try:
        import pyroscope
    except ModuleNotFoundError:
        return "--option needs pyroscope-io installed"
    taken = inspect.signature(pyroscope.configure).parameters
    for key in options:
        if key in OWNED:
            return f"--option {key}: QuickMCP sets it"
        if key in SECRET:
            return (
                f"--option {key}: it carries a credential, which belongs in the "
                "TELEMETRY_PROFILES address and is never printed"
            )
        if key not in taken:
            return f"--option {key}: pyroscope.configure takes no such keyword"
    return None


def _shown(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _no_proxy() -> None:
    """The load goes to 127.0.0.1, so a proxy in the environment must never see it."""
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = ",".join(
            part for part in (os.environ.get(name), "127.0.0.1,localhost") if part
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profile_cost.py", description=__doc__.split("\n\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure", help="measure, and print the table")
    measure.add_argument("--configs", default="off,cpu,memory,both")
    measure.add_argument("--runs", type=int, default=3)
    measure.add_argument("--warmup", type=float, default=15)
    measure.add_argument("--idle", type=float, default=60)
    measure.add_argument("--load", type=float, default=30)
    measure.add_argument("--rate", type=float, default=10)
    measure.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a pyroscope.configure keyword for the profile configurations",
    )
    commands.add_parser("serve", help="the child server the measurement runs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve()
        return 0
    configs = [name for name in args.configs.split(",") if name]
    if not configs or not set(configs) <= CONFIGS.keys():
        parser.error(f"--configs: choose from {', '.join(CONFIGS)}")
    if args.runs < 1 or args.warmup < 0 or min(args.idle, args.load, args.rate) <= 0:
        parser.error("--runs must be at least 1, and --idle, --load and --rate above 0")
    options: dict[str, object] = {}
    for text in args.option:
        try:
            key, value = parse_option(text)
        except ValueError:
            parser.error(f"--option {text}: expected KEY=VALUE")
        options[key] = value
    if (refusal := _refused(options)) is not None:
        parser.error(refusal)
    if any(_profiled(c) for c in configs) and not os.environ.get("TELEMETRY_PROFILES"):
        parser.error("a profile configuration needs TELEMETRY_PROFILES, the address")

    _no_proxy()
    logs = Path(tempfile.mkdtemp(prefix="profile-cost-"))
    suffix = "".join(f" {key}={_shown(value)}" for key, value in options.items())
    names = {c: c + suffix if _profiled(c) else c for c in configs}
    results: dict[str, list[Sample]] = {names[c]: [] for c in configs}
    try:
        for run in range(1, args.runs + 1):
            for config in configs:
                print(f"{names[config]}: run {run} of {args.runs}", file=sys.stderr)
                results[names[config]].append(
                    measure_once(
                        config,
                        options,
                        warmup=args.warmup,
                        idle=args.idle,
                        load=args.load,
                        rate=args.rate,
                        log=logs / f"{config}-{run}.log",
                    )
                )
    except ChildFailed as error:
        print(f"profile_cost.py: {error}", file=sys.stderr)
        return 1
    print(table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
