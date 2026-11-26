"""scripts/profile_cost.py: what profiling costs a bare server, measured on this machine."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "profile_cost", ROOT / "scripts/profile_cost.py"
)
assert _spec is not None and _spec.loader is not None
profile_cost = importlib.util.module_from_spec(_spec)
# Registered first, as importlib's recipe does: the dataclass looks its module up there.
sys.modules["profile_cost"] = profile_cost
_spec.loader.exec_module(profile_cost)

FAST = ("--warmup", "0", "--idle", "1", "--load", "1", "--runs", "1")

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="Linux /proc only"
)


def run(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """The script in a process of its own, with no TELEMETRY_* or OTEL_* but ``env``."""
    clean = {
        k: v for k, v in os.environ.items() if not k.startswith(("TELEMETRY_", "OTEL_"))
    }
    return subprocess.run(
        [sys.executable, "scripts/profile_cost.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=clean | env,
    )


def test_cpu_seconds_counts_this_process() -> None:
    before = profile_cost.cpu_seconds(os.getpid())
    deadline = time.process_time() + 0.2
    while time.process_time() < deadline:
        pass
    assert profile_cost.cpu_seconds(os.getpid()) - before >= 0.1


def test_options_are_read_as_json_first() -> None:
    assert profile_cost.parse_option("sample_rate=20") == ("sample_rate", 20)
    assert profile_cost.parse_option("gil_only=false") == ("gil_only", False)
    assert profile_cost.parse_option("tenant_id=team-a") == ("tenant_id", "team-a")
    with pytest.raises(ValueError):
        profile_cost.parse_option("sample_rate")


def test_each_config_sets_its_four_variables() -> None:
    base = {"TELEMETRY_PROFILES": "http://127.0.0.1:1", "TELEMETRY_TAGS": '{"a": "b"}'}
    off = profile_cost.child_env("off", base, 8123, {})
    assert "TELEMETRY_PROFILES" not in off and off["TELEMETRY_OTEL"] == "false"
    memory = profile_cost.child_env("memory", base, 8123, {"sample_rate": 20})
    assert (memory["TELEMETRY_PROFILE_CPU"], memory["TELEMETRY_PROFILE_MEMORY"]) == (
        "false",
        "true",
    )
    assert json.loads(memory["PROFILE_COST_OPTIONS"]) == {"sample_rate": 20}
    assert (
        memory["TRANSPORT"],
        memory["HOST"],
        memory["PORT"],
        memory["LOG_LEVEL"],
    ) == ("http", "127.0.0.1", "8123", "WARNING")
    assert "PROFILE_COST_OPTIONS" not in profile_cost.child_env(
        "off", base, 8123, {"sample_rate": 20}
    )
    assert memory["TELEMETRY_TAGS"] == '{"a": "b"}'


def test_an_inherited_config_file_reaches_no_child() -> None:
    # A config file can set telemetry.profiles, which would profile the off rows.
    base = {"TELEMETRY_PROFILES": "http://127.0.0.1:1", "CONFIG_FILE": "server.yaml"}
    for config in profile_cost.CONFIGS:
        assert "CONFIG_FILE" not in profile_cost.child_env(config, base, 8123, {})


def test_profiles_off_prints_one_row() -> None:
    out = run("measure", "--configs", "off", *FAST, env={})
    assert out.returncode == 0, out.stderr
    assert re.search(
        r"^\| off \| \d+ \(\d+\u2013\d+\) \| \d+ \(\d+\u2013\d+\) \| 10 \| 1 \|$",
        out.stdout,
        re.M,
    )


def test_the_load_writes_no_access_log(tmp_path) -> None:
    # The script keeps its logs in a directory tempfile makes, so under TMPDIR.
    out = run("measure", "--configs", "off", *FAST, env={"TMPDIR": str(tmp_path)})
    assert out.returncode == 0, out.stderr
    (log,) = tmp_path.glob("profile-cost-*/off-1.log")
    assert "POST /mcp" not in log.read_text()


def test_a_proxy_in_the_environment_never_sees_the_load() -> None:
    dead = "http://127.0.0.1:9"
    out = run(
        "measure",
        "--configs",
        "off",
        *FAST,
        env={"HTTP_PROXY": dead, "http_proxy": dead, "ALL_PROXY": dead},
    )
    assert out.returncode == 0, out.stderr


def test_a_profile_config_without_an_address_is_refused() -> None:
    out = run("measure", "--configs", "cpu", *FAST, env={})
    assert out.returncode == 2 and "TELEMETRY_PROFILES" in out.stderr


@pytest.mark.parametrize(
    "option",
    [
        "no_such_knob=1",
        "server_address=x",
        "tags={}",
        "app_name=x",
        "basic_auth_password=x",
        "http_headers={}",
        "auth_token=x",
    ],
)
def test_an_option_configure_does_not_take_or_quickmcp_owns_is_refused(option) -> None:
    out = run(
        "measure",
        "--configs",
        "cpu",
        "--option",
        option,
        *FAST,
        env={"TELEMETRY_PROFILES": "http://127.0.0.1:1"},
    )
    assert out.returncode == 2 and option.split("=")[0] in out.stderr
    # The refusal names the key, never its value.
    assert option not in out.stderr


@pytest.mark.parametrize(
    "option", ["basic_auth_password=x", "http_headers={}", "auth_token=x", "x_token=x"]
)
def test_an_option_named_like_a_credential_is_refused_as_one(option) -> None:
    # Refused by part of its name, so a credential keyword pyroscope-io adds later is too.
    out = run(
        "measure",
        "--configs",
        "cpu",
        "--option",
        option,
        *FAST,
        env={"TELEMETRY_PROFILES": "http://127.0.0.1:1"},
    )
    assert out.returncode == 2 and "credential" in out.stderr
    assert option not in out.stderr


def test_profiles_that_did_not_start_end_the_run() -> None:
    # configure needs an int: a string makes it raise, and QuickMCP serves on unprofiled.
    out = run(
        "measure",
        "--configs",
        "cpu",
        "--option",
        'sample_rate="x"',
        *FAST,
        env={"TELEMETRY_PROFILES": "http://127.0.0.1:1"},
    )
    assert out.returncode == 1, out.stdout + out.stderr
    (log,) = re.findall(r"(/\S+\.log)", out.stderr)
    assert Path(log).is_file()


def test_a_server_that_fails_at_boot_ends_the_run() -> None:
    started = time.monotonic()
    out = run("measure", "--configs", "off", *FAST, env={"TELEMETRY_TAGS": "{"})
    assert out.returncode == 1 and time.monotonic() - started < 30
    (log,) = re.findall(r"(/\S+\.log)", out.stderr)
    assert Path(log).is_file()


def test_the_address_never_shows() -> None:
    secret = "http://us3r:s3cret@127.0.0.1:9/p4th?t0ken=x"
    out = run(
        "measure",
        "--configs",
        "cpu",
        "--option",
        "sample_rate=20",
        *FAST,
        env={"TELEMETRY_PROFILES": secret},
    )
    assert out.returncode == 0, out.stderr
    for part in ("us3r", "s3cret", "p4th", "t0ken"):
        assert part not in out.stdout + out.stderr
    assert re.search(r"^\| cpu sample_rate=20 \|", out.stdout, re.M)
