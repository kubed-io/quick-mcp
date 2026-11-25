"""Plugin sets: built apart, served whole, counted by generation (PLUG-50 to PLUG-52)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastmcp import Client
from plugin_server import (
    DEMO,
    ENTRY,
    Folders,
    agent_md,
    command_md,
    copy,
    listed,
    served,
    skill,
    write,
)
from skill_server import request

from kubed.mcp import QuickMCP
from kubed.mcp.plugins import PluginError, read_plugin
from kubed.mcp.protocol import MCPError

pytestmark = pytest.mark.unit


def other(tmp_path: Path) -> Path:
    """A second plugin: a skill, a command and an agent."""
    return write(
        tmp_path / "other",
        {
            "skills/other-guide/SKILL.md": skill("other-guide"),
            "commands/wave.md": command_md(),
            "agents/scout.md": agent_md("scout"),
        },
    )


async def test_a_set_is_served_whole(tmp_path):
    mcp = served((DEMO, None))
    assert mcp.generation == 1
    staging = mcp.plugin_set()
    assert staging.keep(mcp.plugins[0]) is mcp.plugins[0]
    added = staging.add(other(tmp_path))
    assert staging.plugins == (mcp.plugins[0], added)
    assert await listed(mcp) == (["greet"], ["helper"], [ENTRY])  # not served yet
    mcp.serve(staging)
    assert mcp.generation == 2
    assert mcp.plugins == staging.plugins
    assert await listed(mcp) == (
        ["greet", "wave"],
        ["helper", "scout"],
        [ENTRY, "skill://t/other-guide/SKILL.md"],
    )


async def test_a_set_without_a_plugin_no_longer_serves_it(tmp_path):
    mcp = served((DEMO, None), (other(tmp_path), None))
    staging = mcp.plugin_set()
    staging.keep(mcp.plugins[1])
    mcp.serve(staging)
    assert await listed(mcp) == (
        ["wave"],
        ["scout"],
        ["skill://t/other-guide/SKILL.md"],
    )
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as caught:
            await request(client, "agents/get", name="helper")
    assert caught.value.error.message == "Agent not found"


def test_add_plugin_serves_a_new_set_and_a_refusal_moves_nothing(tmp_path):
    mcp = QuickMCP("t")
    assert mcp.generation == 0
    mcp.add_plugin(DEMO)
    with pytest.raises(PluginError, match="already served by plugin demo"):
        mcp.add_plugin(copy(tmp_path))
    assert mcp.generation == 1 and len(mcp.plugins) == 1


def test_a_clash_is_checked_against_the_set_alone(tmp_path):
    mcp = served((DEMO, None))
    fresh = mcp.plugin_set()
    fresh.add(copy(tmp_path))  # the served demo is not in this set
    kept = mcp.plugin_set()
    kept.keep(mcp.plugins[0])
    with pytest.raises(
        PluginError,
        match=r"skill://t/demo-guide/SKILL\.md is already served by plugin demo",
    ):
        kept.add(copy(tmp_path, "again"))
    assert kept.plugins == (mcp.plugins[0],)


def test_keep_takes_only_a_plugin_served_now():
    mcp = served((DEMO, None))
    with pytest.raises(ValueError, match="is not served now"):
        mcp.plugin_set().keep(read_plugin(DEMO))
    with pytest.raises(ValueError, match="is not served now"):
        mcp.plugin_set().keep(QuickMCP("u").add_plugin(DEMO))


def test_keeping_a_plugin_twice_is_a_clash():
    mcp = served((DEMO, None))
    staging = mcp.plugin_set()
    staging.keep(mcp.plugins[0])
    with pytest.raises(PluginError, match="already served by plugin demo"):
        staging.keep(mcp.plugins[0])


async def test_a_served_set_is_a_snapshot(tmp_path):
    mcp = QuickMCP("t")
    staging = mcp.plugin_set()
    staging.add(DEMO)
    mcp.serve(staging)
    staging.add(other(tmp_path))
    assert [p.name for p in mcp.plugins] == ["demo"]
    assert await listed(mcp) == (["greet"], ["helper"], [ENTRY])


def test_a_set_serves_only_on_its_own_server():
    with pytest.raises(ValueError, match="another server"):
        QuickMCP("t").serve(QuickMCP("u").plugin_set())


async def test_a_kept_plugin_is_not_read_again(tmp_path):
    copy(tmp_path)
    loader = Folders(tmp_path)
    mcp = QuickMCP("t")
    mcp.add_loader(loader)
    mcp.add_plugin("fixture://demo")
    shutil.rmtree(tmp_path / "demo")
    staging = mcp.plugin_set()
    staging.keep(mcp.plugins[0])
    mcp.serve(staging)
    assert loader.asked == ["fixture://demo"]
    async with Client(mcp.server) as client:
        text = (await client.read_resource(ENTRY))[0].text
    assert text == (DEMO / "skills/demo-guide/SKILL.md").read_text()
