"""``X-MCP-Listed`` chooses what ``resources/list`` shows (DLV-8)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastmcp import Client
from gateway_server import over_http
from plugin_server import copy
from skill_server import load_skill, make_skill, read_directory, request
from test_delivery_inputs import config

from kubed.mcp import QuickMCP, skills

pytestmark = pytest.mark.unit

OWN = "state://own"


def make(tmp_path, delivery=None) -> QuickMCP:
    mcp = QuickMCP("t", config=SimpleNamespace(delivery=delivery))
    mcp.add_plugin(copy(tmp_path))

    @mcp.resource(OWN)
    def own() -> str:
        return "own"

    return mcp


async def rows(client: Client) -> list[dict]:
    return [
        r.model_dump(by_alias=True, exclude_none=True, mode="json")
        for r in await client.list_resources()
    ]


def uris(listed: list[dict]) -> list[str]:
    return [r["uri"] for r in listed]


def by_scheme(listed: list[dict], scheme: str) -> list[dict]:
    return [r for r in listed if r["uri"].startswith(f"{scheme}://")]


async def build_files(mcp: QuickMCP, scheme: str) -> list:
    build = (await mcp._plugins.pinned()).build
    return [m for u, m in build.files.items() if u.startswith(f"{scheme}://")]


async def test_the_default_listing_is_unchanged(tmp_path):
    mcp = make(tmp_path)
    async with Client(mcp.server) as client:
        listed = uris(await rows(client))
    assert OWN in listed
    assert any(u.startswith("skill://") for u in listed)
    assert all(u.endswith("/SKILL.md") for u in listed if u.startswith("skill://"))
    assert not any(u.startswith(("prompts://", "agents://")) for u in listed)


@pytest.mark.parametrize("scheme", ["prompts", "agents"])
async def test_listed_kinds_add_one_row_per_file(tmp_path, scheme):
    mcp = make(tmp_path)
    async with over_http(mcp, headers={"X-MCP-Listed": f"skills,{scheme}"}) as client:
        listed = await rows(client)
        kind_rows = by_scheme(listed, scheme)
        expected = [dict(m.row) for m in await build_files(mcp, scheme)]
        assert expected
        assert kind_rows == expected
        assert all("fastmcp" not in r["_meta"] for r in kind_rows)
        other = "agents" if scheme == "prompts" else "prompts"
        assert by_scheme(listed, other) == []
        if scheme == "prompts":
            assert len(kind_rows) == len(await client.list_prompts())


async def test_an_empty_listed_shows_only_the_servers_own(tmp_path):
    mcp = make(tmp_path)
    skills.serve(mcp.server, [load_skill(make_skill(tmp_path, {}))])
    async with over_http(mcp, headers={"X-MCP-Listed": ""}) as client:
        listed = uris(await rows(client))
    assert OWN in listed
    assert not any(u.startswith("skill://t/") for u in listed)
    assert not any(u.startswith(("prompts://", "agents://")) for u in listed)
    assert any(u.endswith("tmp-skill/SKILL.md") for u in listed)


async def test_unknown_kinds_are_dropped(tmp_path):
    mcp = make(tmp_path)
    async with over_http(mcp, headers={"X-MCP-Listed": "prompts,bogus"}) as client:
        listed = uris(await rows(client))
    assert any(u.startswith("prompts://") for u in listed)
    assert not any(u.startswith("skill://") for u in listed)


async def test_folders_and_indexes_are_never_listed(tmp_path):
    mcp = make(tmp_path)
    async with over_http(
        mcp, headers={"X-MCP-Listed": "skills,prompts,agents"}
    ) as client:
        listed = uris(await rows(client))
    build = (await mcp._plugins.pinned()).build
    folders = set(build.folders)
    assert not [u for u in listed if u.endswith(("_index.md", "_index.json"))]
    assert not [u for u in listed if u in folders]


async def test_the_other_lists_do_not_change(tmp_path):
    mcp = make(tmp_path)
    async with (
        over_http(mcp) as plain,
        over_http(mcp, headers={"X-MCP-Listed": ""}) as none,
    ):
        assert [p.name for p in await plain.list_prompts()] == [
            p.name for p in await none.list_prompts()
        ]
        assert await plain.list_prompts()
        for method, key in (("skills/list", "skills"), ("agents/list", "agents")):
            before = (await request(plain, method))[key]
            assert before
            assert (await request(none, method))[key] == before
        assert await plain.list_resource_templates() == (
            await none.list_resource_templates()
        )


async def test_a_folder_row_is_unchanged(tmp_path):
    mcp = make(tmp_path)
    build = (await mcp._plugins.pinned()).build
    async with over_http(mcp, headers={"X-MCP-Listed": "prompts"}) as client:
        folder = next(f for f in build.folders if f.startswith("prompts://"))
        read = (await read_directory(client, uri=folder))["resources"]
    files = [r for r in read if r["uri"] in build.files]
    assert files
    for r in files:
        assert r == dict(build.files[r["uri"]].row)


async def test_a_config_that_lists_none(tmp_path):
    mcp = make(tmp_path, config(listed=""))
    async with Client(mcp.server) as client:
        assert uris(await rows(client)) == [OWN]
