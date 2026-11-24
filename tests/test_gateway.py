"""The gateway prefix: read from the request (GW-2, GW-3), written into text (GW-5, GW-11)."""

from __future__ import annotations

import logging

import pytest
from fastmcp import Client
from fastmcp.server.dependencies import _background_task_headers
from gateway_server import kit, over_http

from kubed.mcp import QuickMCP
from kubed.mcp.gateway import prefix, prefixed

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see skill://a/SKILL.md", "see kb+skill://a/SKILL.md"),
        ("prompts://a/x.md agents://a/y.md", "kb+prompts://a/x.md kb+agents://a/y.md"),
        ("(skill://a) /skill://a", "(kb+skill://a) /kb+skill://a"),
        (
            "kb+skill://a xx+skill://a myskill://a my.skill://a my-skill://a 9skill://a",
            "kb+skill://a xx+skill://a myskill://a my.skill://a my-skill://a 9skill://a",
        ),
        ("https://example.com/x", "https://example.com/x"),
    ],
)
def test_prefixed_follows_gw5(text, expected):
    assert prefixed(text, "kb+") == expected
    assert prefixed(expected, "kb+") == expected  # idempotent


def test_prefixed_without_a_prefix_is_the_text():
    text = "skill://a"
    assert prefixed(text, None) is text


@pytest.mark.parametrize("bad", ["kb", "+kb+", "9kb+", "k b+", "", "kb+/"])
def test_prefixed_refuses_an_invalid_prefix(bad):
    with pytest.raises(ValueError):
        prefixed("skill://a", bad)


def server() -> QuickMCP:
    mcp = QuickMCP("gw")

    @mcp.tool
    def seen() -> str | None:
        return prefix()

    return mcp


async def seen(client: Client) -> str | None:
    return (await client.call_tool("seen", {})).data


async def test_the_header_names_the_prefix():
    async with over_http(server(), header="kb+") as client:
        assert await seen(client) == "kb+"


async def test_a_header_is_stripped():
    async with over_http(server(), header=" kb+ ") as client:
        assert await seen(client) == "kb+"


async def test_the_query_reads_a_space_as_plus():
    async with over_http(server(), query="kb+") as client:
        assert await seen(client) == "kb+"


async def test_the_header_wins_over_the_query():
    async with over_http(server(), header="kb+", query="xx+") as client:
        assert await seen(client) == "kb+"


async def test_an_empty_header_is_none():
    async with over_http(server(), header="", query="xx+") as client:
        assert await seen(client) == "xx+"


async def test_neither_is_none():
    mcp = server()
    async with over_http(mcp) as client:
        assert await seen(client) is None
    async with Client(mcp.server) as client:
        assert await seen(client) is None


def test_outside_a_request_it_is_none():
    assert prefix() is None


def test_a_background_task_keeps_the_header():
    token = _background_task_headers.set({"X-MCP-Resource-Prefix": "kb+"})
    try:
        assert prefix() == "kb+"
    finally:
        _background_task_headers.reset(token)


def warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.gateway" and r.levelno == logging.WARNING
    ]


async def test_an_invalid_prefix_is_none_and_warned_once_per_connection(caplog):
    mcp = server()
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.gateway"):
        async with over_http(mcp, header="kb", mode="legacy") as client:
            assert await seen(client) is None
            assert await seen(client) is None
        assert len(warnings(caplog)) == 1
        assert "X-MCP-Resource-Prefix" in warnings(caplog)[0].getMessage()
        async with over_http(mcp, header="kb", mode="legacy") as client:
            assert await seen(client) is None
        assert len(warnings(caplog)) == 2
        async with over_http(mcp, header="kb", query="xx+", mode="legacy") as client:
            assert await seen(client) is None


async def test_a_long_invalid_prefix_is_logged_cut(caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.gateway"):
        async with over_http(server(), header="x" * 1000) as client:
            assert await seen(client) is None
    (record,) = warnings(caplog)
    message = record.getMessage()
    assert repr("x" * 64) in message
    assert "x" * 65 not in message


async def test_the_kit_fixture_serves_its_plugin(tmp_path):
    async with Client(kit(tmp_path).server) as client:
        read = await client.read_resource("skill://lib/guide/SKILL.md")
        assert "# Guide" in read[0].text
        assert "lib_ask" in [p.name for p in await client.list_prompts()]
