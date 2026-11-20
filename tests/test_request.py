"""request_values(): a request's query and headers, over a real HTTP stack."""

from __future__ import annotations

from fastmcp import Client
from fastmcp.server.dependencies import _background_task_headers
from fastmcp.utilities.tests import asgi_server

from kubed.mcp import QuickMCP
from kubed.mcp.request import request_values


def server() -> QuickMCP:
    mcp = QuickMCP("request")

    @mcp.tool
    def values() -> dict | None:
        found = request_values()
        if found is None:
            return None
        params, headers = found
        return {"params": params, "headers": headers}

    return mcp


async def test_query_and_headers_come_back_as_lists() -> None:
    async with asgi_server(server().server) as running:
        url = f"{running.url}?workspace=alpha&tag=a&tag=b"
        transport = running.transport(headers={"Sf-Workspace": "beta", "X-Two": "x"})
        transport.url = url
        async with Client(transport) as client:
            found = (await client.call_tool("values", {})).data
    assert found["params"] == {"workspace": ["alpha"], "tag": ["a", "b"]}
    assert found["headers"]["sf-workspace"] == ["beta"]
    assert found["headers"]["x-two"] == ["x"]
    assert all(name == name.lower() for name in found["headers"])


async def test_off_http_there_is_nothing() -> None:
    async with Client(server().server) as client:
        assert (await client.call_tool("values", {})).data is None


def test_a_background_tasks_snapshot_comes_back_with_no_query() -> None:
    # A task's worker has no request, only the headers FastMCP snapshotted at
    # submission, which it keeps in this context variable.
    token = _background_task_headers.set(
        {"Sf-Workspace": "alpha", "Host": "example.com"}
    )
    try:
        plain = request_values()
        rewritten = request_values(
            snapshot=lambda headers: {**headers, "sf-workspace": ["carried"]}
        )
    finally:
        _background_task_headers.reset(token)
    assert plain == ({}, {"sf-workspace": ["alpha"], "host": ["example.com"]})
    assert rewritten == ({}, {"sf-workspace": ["carried"], "host": ["example.com"]})


def test_outside_any_request_it_is_none() -> None:
    assert request_values() is None
