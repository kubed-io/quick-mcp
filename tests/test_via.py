"""``X-MCP-Via`` presets what each listing shows, and every read still answers
(DLV-2, DLV-3, DLV-4, DLV-9, DLV-10, DLV-16)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fastmcp import Client
from gateway_server import over_http
from plugin_server import ENTRY, copy
from skill_server import load_skill, make_skill, request
from test_delivery_inputs import seen, server, warnings

from kubed.mcp import QuickMCP, skills

pytestmark = pytest.mark.unit

OWN_PROMPT = "own_prompt"
OWN_STATE = "state://own"
# What each listing gives under no input, less the server's own.
PLUGIN_PROMPTS = ["greet"]
PLUGIN_SKILLS = [ENTRY]
PLUGIN_AGENTS = ["helper"]
PROMPT_FILE = "prompts://t/greet.md"
AGENT_FILE = "agents://t/helper.md"
PLUGIN_SCHEMES = ("skill://t/", "prompts://", "agents://")


def config(resources=None, listed=None, via=None):
    return SimpleNamespace(resources=resources, listed=listed, via=via)


def make(tmp_path, delivery=None, header_prefix=None) -> QuickMCP:
    """The demo plugin, beside a prompt, a resource and a catalogue of the server's own."""
    mcp = QuickMCP(
        "t", config=SimpleNamespace(delivery=delivery), header_prefix=header_prefix
    )
    mcp.add_plugin(copy(tmp_path))

    @mcp.prompt
    def own_prompt() -> str:
        return "own"

    @mcp.resource(OWN_STATE)
    def own() -> str:
        return "own"

    skills.serve(mcp.server, [load_skill(make_skill(tmp_path, {}))])
    return mcp


async def lists(client: Client) -> dict[str, list[str]]:
    """What each listing shows: prompt names, SKILL.md URIs, agent names, resource URIs."""
    return {
        "prompts": sorted(p.name for p in await client.list_prompts()),
        "skills": sorted(
            s["uri"] for s in (await request(client, "skills/list"))["skills"]
        ),
        "agents": sorted(
            a["name"] for a in (await request(client, "agents/list"))["agents"]
        ),
        "resources": sorted(str(r.uri) for r in await client.list_resources()),
    }


async def own_skill(mcp: QuickMCP) -> str:
    """The SKILL.md URI of the server's own catalogue, as skills/list gives it."""
    async with Client(mcp.server) as client:
        got = (await lists(client))["skills"]
    (own,) = [u for u in got if u not in PLUGIN_SKILLS]
    return own


def local_shape(own: str) -> dict[str, list[str]]:
    return {
        "prompts": [OWN_PROMPT],
        "skills": [own],
        "agents": [],
        "resources": sorted([own, OWN_STATE]),
    }


@pytest.mark.parametrize("by", ["header", "query", "kb-header"])
async def test_local_lists_none_of_the_plugins(tmp_path, by):
    mcp = make(tmp_path, header_prefix="Kb" if by == "kb-header" else None)
    own = await own_skill(mcp)
    where = {
        "header": {"headers": {"X-MCP-Via": "local"}},
        "query": {"params": {"via": "local"}},
        "kb-header": {"headers": {"Kb-Via": "local"}},
    }[by]
    async with over_http(mcp, **where) as client:
        got = await lists(client)
    assert got["prompts"] == [OWN_PROMPT]
    assert got["skills"] == [own]
    assert got["agents"] == []
    assert got["resources"] == sorted([own, OWN_STATE])


async def test_extension_drops_prompts_and_keeps_the_extensions(tmp_path):
    mcp = make(tmp_path)
    own = await own_skill(mcp)
    async with over_http(mcp, headers={"X-MCP-Via": "extension"}) as client:
        got = await lists(client)
    assert got["prompts"] == [OWN_PROMPT]
    assert got["skills"] == sorted([*PLUGIN_SKILLS, own])
    assert got["agents"] == PLUGIN_AGENTS
    assert not any(u.startswith(PLUGIN_SCHEMES) for u in got["resources"])
    assert OWN_STATE in got["resources"]


async def test_mcp_is_today(tmp_path):
    mcp = make(tmp_path)
    own = await own_skill(mcp)
    async with over_http(mcp) as client:
        plain = await lists(client)
    async with over_http(mcp, headers={"X-MCP-Via": "mcp"}) as client:
        header = await lists(client)
    async with over_http(mcp, params={"via": "mcp"}) as client:
        query = await lists(client)
    assert plain == header == query
    assert plain == {
        "prompts": sorted([*PLUGIN_PROMPTS, OWN_PROMPT]),
        "skills": sorted([*PLUGIN_SKILLS, own]),
        "agents": PLUGIN_AGENTS,
        "resources": sorted([*PLUGIN_SKILLS, own, OWN_STATE]),
    }


async def gets(client: Client) -> dict:
    """Every plugin get and read, as the client is answered."""
    prompts = {
        name: (await client.get_prompt(name, {"who": "Ada"})).model_dump(mode="json")
        for name in PLUGIN_PROMPTS
    }
    got_skills = {
        uri: await request(client, "skills/get", uri=uri) for uri in PLUGIN_SKILLS
    }
    agents = {
        name: await request(client, "agents/get", name=name) for name in PLUGIN_AGENTS
    }
    reads = {
        uri: [c.model_dump(mode="json") for c in await client.read_resource(uri)]
        for uri in (*PLUGIN_SKILLS, PROMPT_FILE, AGENT_FILE)
    }
    return {"prompts": prompts, "skills": got_skills, "agents": agents, "reads": reads}


@pytest.mark.parametrize("via", ["extension", "local"])
async def test_every_get_answers_in_every_mode(tmp_path, via):
    mcp = make(tmp_path)
    async with over_http(mcp) as client:
        today = await gets(client)
    assert all(today[kind] for kind in today)
    assert "Ada" in str(today["prompts"]["greet"])
    async with over_http(mcp, headers={"X-MCP-Via": via}) as client:
        assert await gets(client) == today


INDEXES = ("prompts://t/_index.json", "agents://t/_index.json")


@pytest.mark.parametrize("via", ["extension", "local"])
async def test_the_indexes_a_hook_installs_from_are_todays(tmp_path, via):  # DLV-10
    mcp = make(tmp_path)

    async def indexes(client: Client) -> dict:
        return {
            uri: [c.model_dump(mode="json") for c in await client.read_resource(uri)]
            for uri in INDEXES
        }

    async with over_http(mcp, headers={"X-MCP-Via": "mcp"}) as client:
        today = await indexes(client)
    assert all(today.values())
    assert "greet" in str(today[INDEXES[0]]) and "helper" in str(today[INDEXES[1]])
    async with over_http(mcp, headers={"X-MCP-Via": via}) as client:
        assert await indexes(client) == today


@pytest.mark.parametrize("via", ["extension", "local"])
async def test_an_explicit_listed_beats_the_preset(tmp_path, via):
    mcp = make(tmp_path)
    headers = {"X-MCP-Via": via, "X-MCP-Listed": "skills,agents"}
    async with over_http(mcp, headers=headers) as client:
        got = await lists(client)
    assert set(PLUGIN_SKILLS) <= set(got["resources"])
    assert AGENT_FILE in got["resources"]
    assert PROMPT_FILE not in got["resources"]
    assert got["prompts"] == [OWN_PROMPT]


async def test_a_client_via_beats_a_config_listed(tmp_path):
    mcp = make(tmp_path, config(listed="skills"))
    async with over_http(mcp, headers={"X-MCP-Via": "local"}) as client:
        got = await lists(client)
    assert not any(u.startswith("skill://t/") for u in got["resources"])
    mcp = make(tmp_path / "same", config(listed="skills", via="local"))
    async with over_http(mcp) as client:
        got = await lists(client)
    assert set(PLUGIN_SKILLS) <= set(got["resources"])
    assert got["prompts"] == [OWN_PROMPT]  # via is still local


async def test_a_kb_via_beats_a_shared_listed(tmp_path):
    mcp = make(tmp_path, header_prefix="Kb")
    headers = {"Kb-Via": "local", "X-MCP-Listed": "skills"}
    async with over_http(mcp, headers=headers) as client:
        got = await lists(client)
    assert not any(u.startswith("skill://t/") for u in got["resources"])
    headers = {"X-MCP-Via": "local", "Kb-Listed": "skills"}
    async with over_http(mcp, headers=headers) as client:
        got = await lists(client)
    assert set(PLUGIN_SKILLS) <= set(got["resources"])


async def test_a_config_sets_via(tmp_path):
    mcp = make(tmp_path, config(via="local"))
    own = await own_skill(mcp)
    async with over_http(mcp) as client:
        assert await lists(client) == local_shape(own)
    async with over_http(mcp, headers={"X-MCP-Via": "mcp"}) as client:
        got = await lists(client)
    assert got["prompts"] == sorted([*PLUGIN_PROMPTS, OWN_PROMPT])
    assert got["skills"] == sorted([*PLUGIN_SKILLS, own])
    assert got["agents"] == PLUGIN_AGENTS
    assert set(PLUGIN_SKILLS) <= set(got["resources"])


@pytest.mark.parametrize(
    ("raw", "via", "warned"),
    [(" LOCAL ", "local", 0), ("bogus", "mcp", 1), ("", "mcp", 0)],
)
async def test_via_values(raw, via, warned, caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(server(), headers={"X-MCP-Via": raw}) as client:
            assert (await seen(client))["via"] == via
    got = warnings(caplog)
    assert len(got) == warned
    assert all(r.getMessage().startswith("header X-MCP-Via ") for r in got)


async def test_an_empty_via_header_falls_to_the_query(caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(
            server(), headers={"X-MCP-Via": ""}, params={"via": "local"}
        ) as client:
            assert (await seen(client))["via"] == "local"
    assert not warnings(caplog)


async def test_an_invalid_config_via_warns_once_at_build(caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        mcp = server(config(via="bogus"))
    (record,) = warnings(caplog)
    assert record.getMessage().startswith("config delivery.via 'bogus' ")
    async with Client(mcp.server) as client:
        assert (await seen(client))["via"] == "mcp"
