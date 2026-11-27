"""A client's defaults: the server's, from its ``clientInfo`` (DLV-14), and kubed-mcp's
table of known clients (DLV-15), levels 4 and 6 of DLV-3."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from gateway_server import over_http
from mcp_types import Implementation, Request
from plugin_server import DEMO
from pydantic import TypeAdapter
from skill_server import ExtensionParams
from test_delivery_inputs import config, seen, server, warnings

from kubed.mcp import Middleware, delivery
from kubed.mcp.delivery import _Partial

pytestmark = pytest.mark.unit

VSC = Implementation(name="Visual Studio Code", version="1.99")
INSIDERS = "Visual Studio Code - Insiders"
CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
BUILT_INS = {"resources": "on", "listed": ["skills"], "via": "mcp", "prefix": None}


def named(name: str) -> Implementation:
    return Implementation(name=name, version="1")


async def seen_raw(client, meta: object = None) -> dict:
    """``seen``, called by one raw request, with ``meta`` as its ``_meta`` clientInfo
    when given: on the legacy era the client stamps nothing over it. ``call_tool``
    lists the tools after, a second request."""
    extra = {} if meta is None else {"_meta": {CLIENT_INFO: meta}}
    params = ExtensionParams(name="seen", arguments={}, **extra)
    result = await client.session.send_request(
        Request(method="tools/call", params=params), TypeAdapter(dict)
    )
    return json.loads(result["content"][0]["text"])


def errors(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.delivery" and r.levelno == logging.ERROR
    ]


@pytest.mark.parametrize("name", ["Visual Studio Code", INSIDERS])
@pytest.mark.parametrize("mode", ["legacy", "auto"])  # the session's, and the _meta's
async def test_vs_code_gets_resources_off(name, mode, monkeypatch):
    if mode == "auto":
        # The modern era builds a session from the envelope too: only _meta may answer.
        monkeypatch.setattr(delivery, "_session_client_info", lambda: None)
    mcp = server()
    mcp.add_plugin(DEMO)
    async with over_http(mcp, mode=mode, client_info=named(name)) as client:
        assert (await seen(client))["resources"] == "off"
        assert "list_resources" in [t.name for t in await client.list_tools()]


@pytest.mark.parametrize(
    "name", ["visual studio code", "VS Code", "Claude Code", "Code"]
)
async def test_other_names_match_no_row(name):
    async with over_http(server(), mode="legacy", client_info=named(name)) as client:
        assert (await seen(client))["resources"] == "on"


async def test_the_requests_meta_beats_the_sessions():
    async with over_http(server(), mode="legacy") as client:  # the session is fastmcp
        assert (await seen(client))["resources"] == "on"
        got = await seen_raw(client, VSC.model_dump(exclude_none=True))
        assert got["resources"] == "off"


def test_an_extension_methods_meta_is_read():
    # FastMCP hands an extension method its raw params, _meta included.
    context = SimpleNamespace(
        method="skills/list",
        message={"_meta": {CLIENT_INFO: {"name": INSIDERS, "version": "1"}}},
        fastmcp_context=None,
    )
    assert delivery._client_info(context).name == INSIDERS


def test_no_client_info_gives_the_built_ins():
    assert delivery._client_info(None) is None
    assert server()._resolver.resolve(None).resources == "on"


@pytest.mark.parametrize("beats", ["header", "query", "delivery_default", "config"])
async def test_a_servers_setting_beats_the_row(beats):
    mcp = server(config(resources="on") if beats == "config" else None)
    if beats == "delivery_default":
        mcp.delivery_default(lambda info: SimpleNamespace(resources="on"))
    where = {
        "header": {"headers": {"X-MCP-Resources": "on"}},
        "query": {"params": {"resources": "on"}},
    }.get(beats, {})
    async with over_http(mcp, mode="legacy", client_info=VSC, **where) as client:
        assert (await seen(client))["resources"] == "on"


async def test_the_longest_prefix_wins(monkeypatch):
    monkeypatch.setattr(
        delivery,
        "_KNOWN_CLIENTS",
        {
            "Visual": _Partial(resources="off"),
            "Visual Studio Code": _Partial(resources="on"),
        },
    )
    mcp = server()
    async with over_http(mcp, mode="legacy", client_info=VSC) as client:
        assert (await seen(client))["resources"] == "on"
    async with over_http(
        mcp, mode="legacy", client_info=named("Visual Basic")
    ) as client:
        assert (await seen(client))["resources"] == "off"


async def test_delivery_default_gets_the_client_once_per_request():
    mcp = server()
    calls: list = []

    @mcp.delivery_default
    def fn(info):
        calls.append(info)
        return SimpleNamespace(resources="off", listed=None, via="local")

    outer: list[str] = []

    class Count(Middleware):
        async def on_request(self, context, call_next):
            outer.append(context.method)
            return await call_next(context)

    mcp.add_middleware(Count())
    async with over_http(mcp, mode="legacy", client_info=VSC) as client:
        first = await seen(client)
        second = await seen(client)
    assert first["via"] == second["via"] == "local"
    assert outer.count("tools/call") == 2
    assert len(calls) == len(outer)
    assert [info.name for info in calls] == ["Visual Studio Code"] * len(calls)


async def test_delivery_default_returning_none_leaves_every_input():
    mcp = server()
    mcp.delivery_default(lambda info: None)
    async with over_http(mcp) as client:
        assert await seen(client) == BUILT_INS


async def test_delivery_default_returns_a_listed_collection():
    mcp = server()
    mcp.delivery_default(
        lambda info: SimpleNamespace(resources=None, listed=["prompts"], via=None)
    )
    async with over_http(mcp) as client:
        assert (await seen(client))["listed"] == ["prompts"]


async def test_a_default_via_presets_listed_at_its_level():
    # Level 4's local beats level 5's listed: the preset ranks where Via came from.
    mcp = server(config(listed="skills"))
    mcp.delivery_default(lambda info: SimpleNamespace(via="local"))
    async with over_http(mcp) as client:
        got = await seen(client)
    assert (got["via"], got["listed"]) == ("local", [])
    # At the same level, an explicit Listed wins over the preset.
    mcp = server()
    mcp.delivery_default(lambda info: SimpleNamespace(via="local", listed="skills"))
    async with over_http(mcp) as client:
        got = await seen(client)
    assert (got["via"], got["listed"]) == ("local", ["skills"])


async def test_an_invalid_default_falls_through_named(caplog):
    mcp = server(config(resources="off"))
    mcp.delivery_default(lambda info: SimpleNamespace(resources="maybe"))
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp) as client:
            assert (await seen(client))["resources"] == "off"
    (record,) = warnings(caplog)
    assert record.getMessage().startswith(
        "delivery_default resources 'maybe' is ignored"
    )


async def test_a_wrongly_typed_default_value_falls_through_named(caplog):
    mcp = server()
    mcp.delivery_default(lambda info: SimpleNamespace(resources=True, listed=3))
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp, mode="legacy") as client:
            got = await seen(client)
            await seen(client)
    assert (got["resources"], got["listed"]) == ("on", ["skills"])
    messages = sorted(r.getMessage() for r in warnings(caplog))
    assert len(messages) == 2
    assert messages[0].startswith("delivery_default listed '3' is ignored")
    assert messages[1].startswith("delivery_default resources 'True' is ignored")


@pytest.mark.parametrize("attr", ["via", "listed", "resources"])
async def test_a_huge_default_integer_falls_through_named(attr, caplog):
    mcp = server()
    mcp.delivery_default(lambda info: SimpleNamespace(**{attr: 10**5000}))
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp) as client:
            await seen(client)
    (record,) = warnings(caplog)
    rule = delivery._ATTRS[attr].rule
    assert record.getMessage() == (
        f"delivery_default {attr} 'a number of more than 10 digits' is ignored, "
        f"and the next source answers: {rule}"
    )


async def test_a_raising_delivery_default_gives_nothing(caplog):
    mcp = server()

    @mcp.delivery_default
    def fn(info):
        raise RuntimeError("boom")

    # Legacy: on the default era a tools/call makes an inner tools/list, two walks.
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp, mode="legacy") as client:
            caplog.clear()
            assert await seen_raw(client) == BUILT_INS
    (record,) = errors(caplog)
    assert "delivery_default" in record.getMessage()
    assert record.exc_info is not None


@pytest.mark.parametrize("answer", [{"resources": "off"}, "off", 3])
async def test_a_default_that_is_no_delivery_gives_nothing(answer, caplog):
    mcp = server()
    mcp.delivery_default(lambda info: answer)
    # Legacy: on the default era a tools/call makes an inner tools/list, two walks.
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp, mode="legacy") as client:
            caplog.clear()
            assert await seen_raw(client) == BUILT_INS
    (record,) = errors(caplog)
    assert "delivery_default" in record.getMessage()


@pytest.mark.parametrize("meta", ["Visual Studio Code", {"version": "1"}, 3])
async def test_a_malformed_meta_client_info_counts_as_none(meta, monkeypatch):
    async with over_http(server(), mode="legacy", client_info=VSC) as client:
        assert (await seen_raw(client, meta))["resources"] == "off"
        monkeypatch.setattr(delivery, "_session_client_info", lambda: None)
        assert (await seen_raw(client, meta))["resources"] == "on"


async def test_a_failed_session_lookup_counts_as_none(monkeypatch):
    def fail():
        raise RuntimeError("no session")

    monkeypatch.setattr(delivery, "_session_client_info", fail)
    async with over_http(server(), mode="legacy", client_info=VSC) as client:
        assert (await seen(client))["resources"] == "on"
