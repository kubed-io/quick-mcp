"""The server's own header prefix: every input is read under it too (DLV-16)."""

from __future__ import annotations

import logging

import pytest
from fastmcp import Context
from fastmcp.server.dependencies import _background_task_headers
from fastmcp.utilities.tests import asgi_server
from gateway_server import kit, over_http
from starlette.requests import Request
from starlette.responses import JSONResponse
from test_delivery_inputs import seen, server, warnings

from kubed.mcp import QuickMCP, gateway

pytestmark = pytest.mark.unit

INPUTS = [  # (header name, query, two valid values, the resolved key)
    ("Resource-Prefix", "prefix", "kb+", "xx+", "prefix"),
    ("Resources", "resources", "off", "on", "resources"),
    ("Listed", "listed", "prompts", "agents", "listed"),
]


def norm(key, value):
    return [value] if key == "listed" else value


@pytest.mark.parametrize(("name", "query", "a", "b", "key"), INPUTS)
async def test_the_servers_name_beats_the_alias_beats_the_query(
    name, query, a, b, key
):
    async with over_http(
        server(header_prefix="Kb"),
        headers={f"Kb-{name}": a, f"X-MCP-{name}": b},
    ) as client:
        assert (await seen(client))[key] == norm(key, a)
    async with over_http(
        server(header_prefix="Kb"),
        headers={f"X-MCP-{name}": a},
        params={query: b},
    ) as client:
        assert (await seen(client))[key] == norm(key, a)


async def test_kb_resource_prefix_serves_kb_plus(tmp_path):
    async with over_http(
        kit(tmp_path, header_prefix="Kb"), headers={"Kb-Resource-Prefix": "kb+"}
    ) as client:
        read = await client.read_resource("kb+skill://lib/guide/SKILL.md")
        assert "kb+skill://lib/guide/references/a.md" in read[0].text


async def test_a_bad_server_header_falls_to_the_alias_with_one_warning(caplog):
    async with over_http(
        server(header_prefix="Kb"),
        headers={"Kb-Listed": "bogus", "X-MCP-Listed": "prompts"},
    ) as client:
        assert (await seen(client))["listed"] == ["prompts"]
        assert (await seen(client))["listed"] == ["prompts"]
    found = warnings(caplog)
    assert len(found) == 1
    assert found[0].getMessage().startswith("header Kb-Listed 'bogus'")


async def test_without_a_header_prefix_kb_headers_are_ignored():
    async with over_http(
        server(), headers={"Kb-Listed": "prompts", "Kb-Resource-Prefix": "kb+"}
    ) as client:
        got = await seen(client)
    assert got["listed"] == ["skills"]
    assert got["prefix"] is None


async def test_header_names_match_in_any_case():
    async with over_http(
        server(header_prefix="KB"), headers={"kb-listed": "prompts"}
    ) as client:
        assert (await seen(client))["listed"] == ["prompts"]


async def test_a_background_task_reads_the_servers_header():
    mcp = server(header_prefix="Kb")
    async with Context(fastmcp=mcp.server):
        token = _background_task_headers.set({"Kb-Resource-Prefix": "kb+"})
        try:
            assert gateway.prefix() == "kb+"
        finally:
            _background_task_headers.reset(token)


@pytest.mark.parametrize(
    "bad",
    ["", " ", "-Kb", "Kb-", "Kb--Co", "K b", "Kb_", "Kb:", "X-MCP", "x-mcp"],
)
def test_an_invalid_header_prefix_is_refused(bad):
    with pytest.raises(ValueError, match="header_prefix"):
        QuickMCP("t", header_prefix=bad)


@pytest.mark.parametrize("good", ["Kb", "Sf", "My-Co", "kb", "X1"])
def test_a_valid_header_prefix_is_taken(good):
    QuickMCP("t", header_prefix=good)


async def test_each_source_warns_once_under_its_own_name(caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(
            server(header_prefix="Kb"),
            headers={"Kb-Listed": "bogus", "X-MCP-Listed": "bogus"},
        ) as client:
            await seen(client)
            await seen(client)
    found = [r.getMessage() for r in warnings(caplog)]
    assert len(found) == 2
    assert sum(m.startswith("header Kb-Listed ") for m in found) == 1
    assert sum(m.startswith("header X-MCP-Listed ") for m in found) == 1


async def test_a_custom_route_reads_the_servers_own_prefix_header():
    mcp = QuickMCP("t", header_prefix="Kb")

    @mcp.custom_route("/prefix", methods=["GET"], name="prefix")
    async def prefix(request: Request) -> JSONResponse:
        return JSONResponse({"prefix": gateway.prefix()})

    async with asgi_server(mcp.server) as running, running.http_client() as http:
        url = str(running.app.url_path_for("prefix"))
        answer = await http.get(url, headers={"Kb-Resource-Prefix": "kb+"})
        bare = await http.get(url)
    assert answer.json() == {"prefix": "kb+"}
    assert bare.json() == {"prefix": None}
