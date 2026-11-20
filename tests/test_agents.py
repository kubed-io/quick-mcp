"""Agents through ``add_plugin``: what a plugin's ``agents/`` folder costs and serves.

Each test builds a plugin of its own and goes through a real FastMCP ``Client``. The
shipped demo plugin is covered in ``test_plugins.py``.
"""

from __future__ import annotations

import copy
import logging
import os
import stat
from pathlib import Path

import pytest
from fastmcp import Client
from mcp_types import INVALID_PARAMS, PaginatedRequestParams
from skill_server import request

from kubed.mcp import QuickMCP
from kubed.mcp.agents import META_KEY, AgentsExtension, GetAgentParams, parse_agent
from kubed.mcp.protocol import MCPError

pytestmark = pytest.mark.unit

BODY = "# Explore\n\nRead the skill first.\n"
GOOD = f"---\nname: explorer\ndescription: Explores.\n---\n{BODY}"


def agent(front: str, body: str = BODY) -> str:
    return f"---\n{front}\n---\n{body}"


def plugin(tmp_path: Path, files: dict[str, str | bytes]) -> QuickMCP:
    root = tmp_path / "p"
    for name, content in files.items():
        path = root / "agents" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content if isinstance(content, bytes) else content.encode()
        path.write_bytes(data)
    mcp = QuickMCP("t")
    mcp.add_plugin(root)
    return mcp


async def names(mcp: QuickMCP) -> list[str]:
    async with Client(mcp.server) as client:
        return [a["name"] for a in (await request(client, "agents/list"))["agents"]]


async def detail(mcp: QuickMCP, name: str = "explorer") -> dict:
    async with Client(mcp.server) as client:
        return await request(client, "agents/get", name=name)


@pytest.mark.parametrize(
    ("value", "role"),
    [
        ("opus", "large"),
        ("claude-opus-4-1", "large"),
        ("Sonnet", "medium"),
        ("claude-sonnet-4-5", "medium"),
        ("haiku", "small"),
        ("inherit", None),
        ("gpt-5", None),
        ("null", None),
        ("4", None),
    ],
)
async def test_the_routing_hint(tmp_path, value, role):
    front = f"name: explorer\ndescription: x\nmodel: {value}"
    got = await detail(plugin(tmp_path, {"x.md": agent(front)}))
    assert got.get("modelRoutingHint") == (
        {"role": role, "vision": False} if role else None
    )


async def test_a_file_that_is_not_utf8_is_skipped(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        mcp = plugin(
            tmp_path,
            {
                "explorer.md": GOOD,
                "bad.md": b"---\nname: caf\xe9\ndescription: x\n---\n",
            },
        )
    assert await names(mcp) == ["explorer"]
    assert "agent file" in caplog.text
    assert "bad.md" in caplog.text
    assert "is skipped" in caplog.text


async def test_crlf_line_endings_load(tmp_path):
    mcp = plugin(tmp_path, {"explorer.md": GOOD.replace("\n", "\r\n").encode()})
    assert (await detail(mcp))["instructions"] == "# Explore\n\nRead the skill first."


async def test_the_first_of_two_files_with_one_name_wins_and_both_are_named(
    tmp_path, caplog
):
    with caplog.at_level(logging.WARNING):
        mcp = plugin(
            tmp_path,
            {
                "a.md": agent("name: explorer\ndescription: First."),
                "b.md": agent("name: explorer\ndescription: Second."),
            },
        )
    assert (await detail(mcp))["agent"] == "explorer"
    async with Client(mcp.server) as client:
        listing = await request(client, "agents/list")
    assert [a["description"] for a in listing["agents"]] == ["First."]
    first = tmp_path / "p" / "agents" / "a.md"
    second = tmp_path / "p" / "agents" / "b.md"
    assert (
        f"agent explorer is already served by {first.resolve()}; "
        f"{second.resolve()} is skipped"
    ) in caplog.text


async def test_two_commands_with_one_name_name_both_files(tmp_path, caplog):
    root = tmp_path / "p"
    for folder in ("a", "b"):
        (root / "commands" / folder).mkdir(parents=True)
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "plugin.json").write_text(
        '{"commands": ["./commands/a/greet.md", "./commands/b/greet.md"]}'
    )
    for folder in ("a", "b"):
        (root / "commands" / folder / "greet.md").write_text(
            f"---\ndescription: {folder}.\n---\nHi.\n"
        )
    with caplog.at_level(logging.WARNING):
        QuickMCP("t").add_plugin(root)
    first = (root / "commands" / "a" / "greet.md").resolve()
    second = (root / "commands" / "b" / "greet.md").resolve()
    assert (
        f"prompt greet is already served by {first}; {second} is skipped"
    ) in caplog.text


async def test_a_cursor_is_refused(tmp_path):
    mcp = plugin(tmp_path, {"explorer.md": GOOD})
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as caught:
            await request(client, "agents/list", cursor="2")
    assert caught.value.error.code == INVALID_PARAMS
    assert caught.value.error.message == "Invalid agents cursor"


@pytest.mark.parametrize(
    ("params", "named"),
    [({"name": ""}, False), ({}, False), ({"name": "nope"}, True)],
)
async def test_a_bad_get_is_invalid_params(tmp_path, params, named):
    mcp = plugin(tmp_path, {"explorer.md": GOOD})
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as caught:
            await request(client, "agents/get", **params)
    assert caught.value.error.code == INVALID_PARAMS
    # An empty or missing name is refused by the schema, an unknown one by the lookup.
    assert (caught.value.error.message == "Agent not found") is named


async def test_a_caller_cannot_change_what_the_next_caller_is_served():
    """The handlers hand out copies, so a caller that edits a result changes nothing."""
    parsed = parse_agent(agent("name: zeta\ndescription: x\ncapabilities: [browse]"))
    assert not isinstance(parsed, str)
    extension = AgentsExtension([copy.deepcopy(parsed)])
    first = await extension.list_agents(None, PaginatedRequestParams())
    first["agents"][0]["capabilities"].append("hacked")
    got = await extension.get_agent(None, GetAgentParams(name="zeta"))
    got["_meta"][META_KEY]["frontmatter"]["name"] = "hacked"
    again = await extension.list_agents(None, PaginatedRequestParams())
    assert again["agents"][0]["capabilities"] == ["browse"]
    got = await extension.get_agent(None, GetAgentParams(name="zeta"))
    assert got["_meta"][META_KEY]["frontmatter"]["name"] == "zeta"


# ---- a folder that cannot be listed -------------------------------------------------


@pytest.fixture
def locked():
    """Folders made unreadable for the test, and restored so tmp_path can be cleaned."""
    folders: list[Path] = []
    yield folders
    for folder in folders:
        folder.chmod(stat.S_IRWXU)


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="a folder cannot be made unreadable to this user",
)
@pytest.mark.parametrize("kind", ["agents", "commands"])
async def test_a_folder_that_cannot_be_listed_is_a_note_and_the_rest_is_served(
    tmp_path, caplog, locked, kind
):
    root = tmp_path / "p"
    (root / "skills" / "s").mkdir(parents=True)
    (root / "skills" / "s" / "SKILL.md").write_text(
        "---\nname: s\ndescription: The s skill.\n---\n\n# s\n"
    )
    (root / kind).mkdir()
    (root / kind / "x.md").write_text(GOOD if kind == "agents" else "---\n---\nHi.\n")
    other = "commands" if kind == "agents" else "agents"
    (root / other).mkdir()
    (root / other / "y.md").write_text(
        "---\nname: scout\ndescription: Scouts.\n---\nGo.\n"
        if other == "agents"
        else "---\ndescription: Hi.\n---\nHi.\n"
    )
    (root / kind).chmod(0)
    locked.append(root / kind)
    mcp = QuickMCP("t")
    with caplog.at_level(logging.WARNING):
        added = mcp.add_plugin(root)
    note = next(n for n in added.components.notes if "cannot be listed" in n)
    assert f"'{kind}'" in note and "Permission denied" in note
    warnings = [r for r in caplog.records if "cannot be listed" in r.getMessage()]
    assert len(warnings) == 1
    assert "skill://t/s/SKILL.md" in added.skills
    if kind == "agents":
        assert added.agents == ()
        assert added.prompts == ("y",)
        assert await names(mcp) == []
    else:
        assert added.prompts == ()
        assert added.agents == ("scout",)
