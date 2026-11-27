"""``X-MCP-Via`` presets what each listing shows, and every read still answers
(DLV-2, DLV-3, DLV-4, DLV-9, DLV-10, DLV-16)."""

from __future__ import annotations

import logging

import pytest
from fastmcp import Client, FastMCP
from fastmcp.skills import SkillsClientExtension
from gateway_server import over_http
from plugin_server import ENTRY, copy
from skill_server import load_skill, make_skill, read_directory, request
from test_delivery_inputs import config, seen, server, warnings

from kubed.mcp import QuickMCP, delivery, skills
from kubed.mcp.skills import SkillsExtension

pytestmark = pytest.mark.unit

MODES = ["legacy", "auto"]

OWN_PROMPT = "own_prompt"
OWN_STATE = "state://own"
# What each listing gives under no input, less the server's own.
PLUGIN_PROMPTS = ["greet"]
PLUGIN_SKILLS = [ENTRY]
PLUGIN_AGENTS = ["helper"]
PROMPT_FILE = "prompts://t/greet.md"
AGENT_FILE = "agents://t/helper.md"


def make(tmp_path, delivery=None, header_prefix=None) -> QuickMCP:
    """The demo plugin, beside a prompt, a resource and a catalogue of the server's own."""
    mcp = QuickMCP(
        "t",
        config={} if delivery is None else {"delivery": delivery},
        header_prefix=header_prefix,
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


OWN_SKILL = "skill://tmp-skill/SKILL.md"
SHAPES = {  # DLV-9: each kind one way per mode
    "mcp": {
        "prompts": sorted([*PLUGIN_PROMPTS, OWN_PROMPT]),
        "skills": [],
        "agents": PLUGIN_AGENTS,
        "resources": sorted([*PLUGIN_SKILLS, OWN_SKILL, OWN_STATE]),
    },
    "extension": {
        "prompts": sorted([*PLUGIN_PROMPTS, OWN_PROMPT]),
        "skills": sorted([*PLUGIN_SKILLS, OWN_SKILL]),
        "agents": PLUGIN_AGENTS,
        "resources": [OWN_STATE],
    },
    "local": {
        "prompts": [OWN_PROMPT],
        "skills": [OWN_SKILL],
        "agents": [],
        "resources": [OWN_STATE],
    },
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("via", list(SHAPES))
@pytest.mark.parametrize("by", ["header", "query", "kb-header", "config"])
async def test_each_mode_lists_each_kind_one_way(tmp_path, mode, via, by):
    sent = {
        "header": {"headers": {"X-MCP-Via": via}},
        "query": {"params": {"via": via}},
        "kb-header": {"headers": {"Kb-Via": via}},
        "config": {},
    }[by]
    mcp = make(
        tmp_path,
        config(via=via) if by == "config" else None,
        header_prefix="Kb" if by == "kb-header" else None,
    )
    async with over_http(mcp, mode=mode, **sent) as client:
        assert await lists(client) == SHAPES[via]


@pytest.mark.parametrize("mode", MODES)
async def test_no_via_is_mcp_even_when_skills_list_declares_skills(tmp_path, mode):
    async with over_http(
        make(tmp_path), mode=mode, extensions=[SkillsClientExtension()]
    ) as client:
        assert await lists(client) == SHAPES["mcp"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("via", list(SHAPES))
async def test_the_skills_extension_is_declared_in_every_mode(tmp_path, mode, via):
    async with over_http(make(tmp_path), mode=mode, headers={"X-MCP-Via": via}) as client:
        declared = client.session.server_capabilities.extensions or {}
    assert declared[SkillsExtension.identifier] == {"directoryRead": True}


@pytest.mark.parametrize("mode", MODES)
async def test_mcp_listing_none_lists_skills_nowhere(tmp_path, mode):  # DLV-9 edge
    async with over_http(make(tmp_path), mode=mode, headers={"X-MCP-Listed": "none"}) as client:
        got = await lists(client)
        for uri in (*PLUGIN_SKILLS, OWN_SKILL):
            assert await client.read_resource(uri)
            assert (await request(client, "skills/get", uri=uri))["skill"]["uri"] == uri
    assert (got["skills"], got["resources"]) == ([], [OWN_STATE])


@pytest.mark.parametrize("mode", MODES)
async def test_an_explicit_listed_prompts_under_mcp_lists_them_twice(tmp_path, mode):
    async with over_http(
        make(tmp_path), mode=mode, headers={"X-MCP-Listed": "skills,prompts"}
    ) as client:
        got = await lists(client)
    assert PROMPT_FILE in got["resources"] and "greet" in got["prompts"]


@pytest.mark.parametrize("mode", MODES)
async def test_a_prefixed_extension_lists_one_way(tmp_path, mode):  # Review Focus 1
    async with over_http(
        make(tmp_path), header="kb+", mode=mode, headers={"X-MCP-Via": "extension"}
    ) as client:
        got = await lists(client)
    assert got["skills"] == sorted([*PLUGIN_SKILLS, OWN_SKILL])
    assert not [u for u in got["resources"] if "skill://" in u]


def test_delivered_is_none_off_quickmcp():  # Review Focus 2
    assert delivery._delivered() is None


@pytest.mark.parametrize("mode", MODES)
async def test_delivered_is_none_in_a_plain_servers_request(mode):
    plain = FastMCP("plain")

    @plain.tool
    def none() -> bool:
        return delivery._delivered() is None

    async with Client(plain, mode=mode) as client:
        assert (await client.call_tool("none", {})).data is True


async def gets(client: Client) -> dict:
    """Every plugin get and read, as the client is answered."""
    prompts = {
        name: (await client.get_prompt(name, {"who": "Ada"})).model_dump(mode="json")
        for name in PLUGIN_PROMPTS
    }
    got_skills = {
        uri: await request(client, "skills/get", uri=uri)
        for uri in (*PLUGIN_SKILLS, OWN_SKILL)
    }
    agents = {
        name: await request(client, "agents/get", name=name) for name in PLUGIN_AGENTS
    }
    reads = {
        uri: [c.model_dump(mode="json") for c in await client.read_resource(uri)]
        for uri in (*PLUGIN_SKILLS, OWN_SKILL, PROMPT_FILE, AGENT_FILE)
    }
    folders = {
        uri: await read_directory(client, uri=uri)
        for uri in ("skill://tmp-skill", "skill://t/demo-guide")
    }
    return {
        "prompts": prompts,
        "skills": got_skills,
        "agents": agents,
        "reads": reads,
        "folders": folders,
    }


@pytest.mark.parametrize("via", list(SHAPES))
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
    assert OWN_SKILL in got["resources"]
    assert AGENT_FILE in got["resources"]
    assert PROMPT_FILE not in got["resources"]
    assert got["prompts"] == (
        sorted([*PLUGIN_PROMPTS, OWN_PROMPT]) if via == "extension" else [OWN_PROMPT]
    )


async def test_a_client_via_beats_a_config_listed(tmp_path):
    mcp = make(tmp_path, config(listed="skills"))
    async with over_http(mcp, headers={"X-MCP-Via": "local"}) as client:
        got = await lists(client)
    assert not any(u.startswith("skill://t/") for u in got["resources"])
    assert OWN_SKILL not in got["resources"]
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
    async with over_http(mcp) as client:
        assert await lists(client) == SHAPES["local"]
    async with over_http(mcp, headers={"X-MCP-Via": "mcp"}) as client:
        assert await lists(client) == SHAPES["mcp"]


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
