"""Instructions per request, and the handshake says what it resolved (DLV-11, DLV-12,
DLV-13, GW-9)."""

from __future__ import annotations

import logging

import pytest
from gateway_server import kit, over_http
from mcp.client import advertise
from mcp_types import Implementation

from kubed.mcp import Middleware, QuickMCP
from kubed.mcp.delivery import Resolved, _resolved

pytestmark = pytest.mark.unit

META = "io.kubed/delivery"
MODES = ["legacy", "auto"]  # initialize, server/discover
VSC = Implementation(name="Visual Studio Code", version="1.99")
# FastMCP's client declares this itself on the 2026-07-28 protocol, with the tasks extra.
TASKS = "io.modelcontextprotocol/tasks"


def result(client):
    """The handshake result the client holds, whichever handshake it made."""
    return client.initialize_result or client.session.discover_result


def errors(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.delivery" and r.levelno == logging.ERROR
    ]


@pytest.mark.parametrize("mode", MODES)
async def test_instructions_get_the_resolution(mode):
    mcp = QuickMCP("t")
    got = []

    def fn(d):
        got.append(d)
        return f"via {d.via}, resources {d.resources}."

    assert mcp.instructions(fn) is fn
    headers = {"X-MCP-Via": "local", "X-MCP-Resources": "off"}
    async with over_http(mcp, mode=mode, headers=headers) as client:
        assert (client.initialize_result is None) is (mode == "auto")
        assert result(client).instructions == "via local, resources off."
        assert client.instructions == "via local, resources off."
    assert len(got) == 1  # once per handshake result
    assert isinstance(got[-1], Resolved)
    assert got[-1].listed == frozenset() and got[-1].prefix is None


@pytest.mark.parametrize("mode", MODES)
async def test_instructions_are_prefixed(tmp_path, mode):  # Review Focus 5, GW-7
    mcp = kit(tmp_path)
    mcp.instructions(lambda d: "Start at skill://lib/guide/SKILL.md.")
    async with over_http(mcp, header="kb+", mode=mode) as client:
        assert result(client).instructions == "Start at kb+skill://lib/guide/SKILL.md."
        assert result(client).meta[META]["prefix"] == "kb+"


@pytest.mark.parametrize("mode", MODES)
async def test_the_echo(mode):  # DLV-12, GW-9
    headers = {"X-MCP-Listed": "agents,skills,prompts"}
    async with over_http(QuickMCP("t"), mode=mode, headers=headers) as client:
        assert result(client).meta[META] == {
            "resources": "on",
            "listed": ["skills", "prompts", "agents"],
            "via": "mcp",
            "prefix": None,
        }


@pytest.mark.parametrize("mode", MODES)
async def test_the_echo_with_nothing_sent_is_the_built_ins(mode):
    async with over_http(QuickMCP("t"), mode=mode) as client:
        assert result(client).meta[META] == {
            "resources": "on",
            "listed": ["skills"],
            "via": "mcp",
            "prefix": None,
        }


class _SetsMeta(Middleware):
    """Sets the handshake result's ``_meta`` to ``{"x": 1}``, inside ``_Describe``."""

    async def on_initialize(self, context, call_next):
        got = await call_next(context)
        got.meta = {"x": 1}
        return got

    async def on_discover(self, context, call_next):
        got = await call_next(context)
        got.meta = {"x": 1}
        return got


@pytest.mark.parametrize("mode", MODES)
async def test_the_echo_keeps_other_meta(mode):
    mcp = QuickMCP("t")
    mcp.add_middleware(_SetsMeta())
    async with over_http(mcp, mode=mode) as client:
        meta = result(client).meta
    assert meta["x"] == 1
    assert meta[META]["via"] == "mcp"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (
            ["io.modelcontextprotocol/skills"],
            frozenset({"io.modelcontextprotocol/skills"}),
        ),
        ([], frozenset()),
    ],
)
@pytest.mark.parametrize("mode", MODES)
async def test_extensions_are_the_clients(declared, expected, mode):  # DLV-11
    # mcp-kb#87.
    mcp = QuickMCP("t")
    got = []
    mcp.instructions(lambda d: got.append(d))

    @mcp.tool
    def extensions() -> dict:
        return {"ids": sorted(_resolved().extensions - {TASKS})}

    sent = [advertise(identifier) for identifier in declared]
    async with over_http(mcp, mode=mode, extensions=sent) as client:
        assert got[-1].extensions - {TASKS} == expected
        called = await client.call_tool("extensions", {})
        assert called.data == {"ids": sorted(expected)}
        assert "extensions" not in result(client).meta[META]


@pytest.mark.parametrize("mode", MODES)
async def test_no_instructions_fn_keeps_the_built_ones(mode):
    async with over_http(QuickMCP("t", instructions="Fixed."), mode=mode) as client:
        assert result(client).instructions == "Fixed."


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("fn", "logged"),
    [(lambda d: None, False), (lambda d: 1 / 0, True), (lambda d: 3, True)],
    ids=["none", "raises", "not-a-string"],
)
async def test_a_failing_fn_keeps_the_built_ones(fn, logged, mode, caplog):
    mcp = QuickMCP("t", instructions="Fixed.")
    mcp.instructions(fn)
    with caplog.at_level(logging.ERROR, logger="kubed.mcp.delivery"):
        async with over_http(mcp, mode=mode) as client:
            assert result(client).instructions == "Fixed."
            assert result(client).meta[META]["via"] == "mcp"
    assert bool(errors(caplog)) is logged


async def test_vs_code_is_echoed_off():  # DLV-15 seen by a client
    async with over_http(QuickMCP("t"), mode="legacy", client_info=VSC) as client:
        assert result(client).meta[META]["resources"] == "off"


@pytest.mark.parametrize("mode", MODES)
async def test_a_second_instructions_call_replaces_the_first(mode):
    mcp = QuickMCP("t", instructions="Fixed.")
    mcp.instructions(lambda d: "first")
    mcp.instructions(lambda d: "second")
    async with over_http(mcp, mode=mode) as client:
        assert result(client).instructions == "second"
