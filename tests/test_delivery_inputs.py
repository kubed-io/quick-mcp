"""The delivery inputs: resolved by levels, each falling to the next (DLV-2 to DLV-5)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from config_server import Duck
from fastmcp import Client, Context
from gateway_server import over_http

from kubed.mcp import Middleware, QuickMCP
from kubed.mcp import request as request_module
from kubed.mcp.delivery import _resolved

pytestmark = pytest.mark.unit

KEYS = ("resources", "listed", "prefix")


def snapshot() -> dict:
    r = _resolved()
    return {
        "resources": r.resources,
        "listed": sorted(r.listed),
        "via": r.via,
        "prefix": r.prefix,
    }


def server(delivery=None, header_prefix=None) -> QuickMCP:
    mcp = QuickMCP(
        "t",
        sections=[Duck],
        config={"delivery": delivery},
        header_prefix=header_prefix,
    )

    @mcp.tool
    def seen() -> dict:
        return snapshot()

    @mcp.tool
    async def nested(ctx: Context) -> dict:
        read = await ctx.read_resource("state://resolved")
        return {"tool": snapshot(), "read": json.loads(read.contents[0].content)}

    @mcp.resource("state://resolved")
    def resolved() -> str:
        return json.dumps(snapshot())

    return mcp


def config(resources=None, listed=None):
    return SimpleNamespace(resources=resources, listed=listed, via=None)


async def seen(client: Client) -> dict:
    return (await client.call_tool("seen", {})).data


def warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.delivery" and r.levelno == logging.WARNING
    ]


@pytest.mark.parametrize("by", ["header", "query"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        *[(v, "on") for v in ["on", "true", "1", "yes", "ON", " Yes "]],
        *[(v, "off") for v in ["off", "false", "0", "no", "none", " NONE ", " OFF "]],
    ],
)
async def test_every_resources_spelling_resolves(by, raw, expected):
    where = (
        {"headers": {"X-MCP-Resources": raw}}
        if by == "header"
        else {"params": {"resources": raw}}
    )
    async with over_http(server(), **where) as client:
        assert (await seen(client))["resources"] == expected


@pytest.mark.parametrize("by", ["header", "query"])
@pytest.mark.parametrize(
    ("raw", "listed"),
    [
        ("skills,prompts", ["prompts", "skills"]),
        (" Skills , PROMPTS ", ["prompts", "skills"]),
        ("", []),
        ("none", []),
        (" None ", []),
        ("prompts,bogus", ["prompts"]),
    ],
)
async def test_listed_values(by, raw, listed):
    where = (
        {"headers": {"X-MCP-Listed": raw}}
        if by == "header"
        else {"params": {"listed": raw}}
    )
    async with over_http(server(), **where) as client:
        assert (await seen(client))["listed"] == listed


async def test_an_empty_resources_value_is_no_value(caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(
            server(),
            headers={"X-MCP-Resources": ""},
            params={"resources": "off"},
        ) as client:
            assert (await seen(client))["resources"] == "off"
    assert not warnings(caplog)


async def test_an_empty_listed_header_lists_none_and_wins():
    async with over_http(
        server(), headers={"X-MCP-Listed": ""}, params={"listed": "prompts"}
    ) as client:
        assert (await seen(client))["listed"] == []


async def test_header_beats_query_beats_config_beats_built_in():
    async with over_http(server()) as client:
        got = await seen(client)
        assert (got["resources"], got["listed"]) == ("on", ["skills"])
    mcp = server(config("off", "agents"))
    async with over_http(mcp) as client:
        got = await seen(client)
        assert (got["resources"], got["listed"]) == ("off", ["agents"])
    async with over_http(
        mcp, params={"resources": "on", "listed": "prompts"}
    ) as client:
        got = await seen(client)
        assert (got["resources"], got["listed"]) == ("on", ["prompts"])
    async with over_http(
        mcp,
        headers={"X-MCP-Resources": "off", "X-MCP-Listed": "skills"},
        params={"resources": "on", "listed": "prompts"},
    ) as client:
        got = await seen(client)
        assert (got["resources"], got["listed"]) == ("off", ["skills"])


@pytest.mark.parametrize("bad", ["header", "query"])
async def test_an_invalid_value_falls_to_the_next_source(bad, caplog):
    if bad == "header":
        where = {
            "headers": {"X-MCP-Resources": "maybe"},
            "params": {"resources": "off"},
        }  # the query differs from the config's "on"
    else:
        where = {"params": {"resources": "maybe"}}
    mcp = server(config(resources="on"))
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp, **where) as client:
            assert (await seen(client))["resources"] == (
                "off" if bad == "header" else "on"
            )
    (record,) = warnings(caplog)
    assert "'maybe' is ignored, and the next source answers" in record.getMessage()


async def test_an_invalid_config_value_warns_once_at_build(caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        mcp = server(config(resources="maybe", listed=["bogus"]))
    messages = [r.getMessage() for r in warnings(caplog)]
    assert len(messages) == 2
    assert messages[0].startswith(
        "config delivery.resources 'maybe' is ignored, and the next source answers: "
    )
    assert messages[1].startswith("config delivery.listed 'bogus' ")
    async with Client(mcp.server) as client:
        got = await seen(client)
    assert (got["resources"], got["listed"]) == ("on", ["skills"])
    assert len(warnings(caplog)) == 2


@pytest.mark.parametrize("listed", ["", [], (), "none"])
async def test_a_config_may_list_none(listed):
    async with Client(server(config(listed=listed)).server) as client:
        assert (await seen(client))["listed"] == []


@pytest.mark.parametrize(
    ("where", "source"),
    [
        ({"params": {"prefix": "kb"}}, "query parameter prefix"),
        ({"headers": {"X-MCP-Resource-Prefix": "kb"}}, "header X-MCP-Resource-Prefix"),
        ({"headers": {"X-MCP-Listed": "bogus"}}, "header X-MCP-Listed"),
        ({"params": {"resources": "maybe"}}, "query parameter resources"),
    ],
)
async def test_a_warning_names_its_true_source(where, source, caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(server(), **where) as client:
            await seen(client)
    (record,) = warnings(caplog)
    assert record.getMessage().startswith(f"{source} ")


async def test_warned_once_per_session_and_again_in_a_second(caplog):
    mcp = server()
    sent = {"headers": {"X-MCP-Resources": "maybe"}, "mode": "legacy"}
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp, **sent) as client:
            await seen(client)
            await seen(client)
        assert len(warnings(caplog)) == 1
        async with over_http(mcp, **sent) as client:
            await seen(client)
        assert len(warnings(caplog)) == 2


async def test_warned_once_per_value_without_a_session(caplog):
    mcp = server()
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp, headers={"X-MCP-Resources": "maybe"}) as client:
            await seen(client)
            await seen(client)
        assert len(warnings(caplog)) == 1
        async with over_http(mcp, headers={"X-MCP-Resources": "maybe"}) as client:
            await seen(client)
        assert len(warnings(caplog)) == 1
        async with over_http(mcp, headers={"X-MCP-Resources": "perhaps"}) as client:
            await seen(client)
        assert len(warnings(caplog)) == 2


async def test_the_warnings_remembered_are_bounded(monkeypatch, caplog):
    monkeypatch.setattr(request_module, "_WARNED_MAX", 2)
    mcp = server()
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        for value in ["a", "b", "a", "c", "a"]:
            async with over_http(mcp, headers={"X-MCP-Resources": value}) as client:
                await seen(client)
    # Least recently used goes: under first-in-first-out the last a warns again, and 4.
    assert len(warnings(caplog)) == 3


async def test_an_appended_level_answers_before_the_built_in():
    mcp = server()
    mcp._resolver.levels["Listed"].append(lambda values: ("prompts", "stand-in"))
    async with over_http(mcp) as client:
        assert (await seen(client))["listed"] == ["prompts"]
    mcp = server(config(listed="agents"))
    mcp._resolver.levels["Listed"].append(lambda values: ("prompts", "stand-in"))
    async with over_http(mcp) as client:
        assert (await seen(client))["listed"] == ["agents"]
    async with over_http(mcp, headers={"X-MCP-Listed": "skills"}) as client:
        assert (await seen(client))["listed"] == ["skills"]


async def test_on_stdio_the_defaults_answer():
    mcp = server(config(resources="off"))
    async with Client(mcp.server) as client:
        assert await seen(client) == {
            "resources": "off",
            "listed": ["skills"],
            "via": "mcp",
            "prefix": None,
        }


async def test_a_nested_request_shares_the_resolution():
    mcp = server()
    walks: list[int] = []

    def stand_in(values):
        walks.append(1)
        return ("agents" if len(walks) % 2 else "prompts"), "stand-in"

    mcp._resolver.levels["Listed"].append(stand_in)
    outer: list[str] = []

    class Count(Middleware):
        async def on_request(self, context, call_next):
            if context.method != "resources/read":  # the nested read is no outer one
                outer.append(context.method)
            return await call_next(context)

    mcp.add_middleware(Count())
    headers = {"X-MCP-Resource-Prefix": "kb+"}
    async with over_http(mcp, headers=headers) as client:
        walks.clear()
        outer.clear()
        got = (await client.call_tool("nested", {})).data
        assert "tools/call" in outer
        assert len(walks) == len(outer)  # one walk per outer request, none per read
    assert got["tool"] == got["read"]  # a fresh walk would have answered the other kind
    assert got["tool"]["prefix"] == "kb+"


def test_via_and_listed_out_of_line_fail_at_startup(monkeypatch):
    """An explicit check, not an assert, so python -O keeps it (DLV-9)."""
    from dataclasses import replace

    from kubed.mcp import delivery

    inputs = dict(delivery._INPUTS)
    inputs["Listed"] = replace(inputs["Listed"], attr=None)
    monkeypatch.setattr(delivery, "_INPUTS", inputs)
    with pytest.raises(RuntimeError, match=r"line up by index \(DLV-9\)"):
        QuickMCP("t")
