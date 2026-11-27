"""A skill's hidden paths are never served, by any skill provider, and the rest of the skill is (SKL-24)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastmcp import Client
from mcp_types import INVALID_PARAMS
from plugin_server import copy
from plugin_server import served as plugin_served
from skill_server import GOOD, read_directory, read_json, request, served

from kubed.mcp import QuickMCP, skills
from kubed.mcp.protocol import MCPError
from kubed.mcp.skills import EntryOnlySkillProvider, Skill, serve

pytestmark = pytest.mark.unit

HIDDEN = {
    ".gitignore": "*.log\n",
    ".env": "SECRET=1\n",
    ".git/config": "[core]\n",
    ".git/_index.md": "reserved, and hidden\n",
    "assets/.gitkeep": "",
    "assets/recordings/har/.gitkeep": "",
}
ORDINARY = {"assets/a.txt": "a\n", "references/R.md": "# R\n"}
NO_FOLDER = [".git", "assets/recordings", "assets/recordings/har"]
SERVERS = ["folder", "catalogue", "per-request", "plugin", "rewritten"]


class PerRequest(EntryOnlySkillProvider):
    def __init__(self, skill: Skill) -> None:
        super().__init__()
        self.skill = skill

    async def load_skills(self):
        return {"tmp-skill": self.skill}


def build(kind: str, tmp_path: Path) -> tuple[QuickMCP, str, set[str]]:
    """The server, the skill's root URI and the files it serves."""
    everything = {**HIDDEN, **ORDINARY}
    if kind == "folder":
        return (
            served(tmp_path, everything),
            "skill://tmp-skill",
            {"SKILL.md", *ORDINARY},
        )
    if kind in ("catalogue", "per-request"):
        skill = Skill.from_files(
            {"SKILL.md": f"---\n{GOOD}\n---\n\n# Tmp\n", **everything}
        )
        mcp = QuickMCP("t")
        serve(mcp.server, [skill] if kind == "catalogue" else PerRequest(skill))
        return mcp, "skill://tmp-skill", {"SKILL.md", *ORDINARY}
    files = {f"skills/demo-guide/{p}": b for p, b in everything.items()}
    mcp = plugin_served((copy(tmp_path, files=files), None))
    if kind == "rewritten":
        # Skill files only: a rewriter is run on a folder's index too, and JSON
        # with a comment after it is no JSON.
        mcp.add_rewriter(
            lambda site, text: text + "\n<!-- r -->\n" if site.kind == "skill" else text
        )
    return (
        mcp,
        "skill://t/demo-guide",
        {"SKILL.md", "references/GUIDE.md", *ORDINARY},
    )


def hidden(uri: str, base: str) -> bool:
    return skills._is_hidden(uri.removeprefix(base + "/"))


async def miss(client, uri: str) -> tuple[int, str]:
    with pytest.raises(MCPError) as caught:
        await client.read_resource(uri)
    error = caught.value.error
    return error.code, error.message.replace(uri, "<uri>")


@pytest.mark.parametrize("kind", SERVERS)
async def test_every_other_file_reads(tmp_path, kind):
    mcp, base, files = build(kind, tmp_path)
    async with Client(mcp.server) as client:
        for path in files:
            assert await client.read_resource(f"{base}/{path}")
        whole = await client.read_skill(f"{base}/SKILL.md")
    assert set(whole.files) == files


@pytest.mark.parametrize("kind", SERVERS)
async def test_a_hidden_path_reads_as_a_missing_one(tmp_path, kind):
    mcp, base, _ = build(kind, tmp_path)
    folders = [
        *NO_FOLDER,
        *(f"{f}/{i}" for f in NO_FOLDER for i in ("_index.md", "_index.json")),
    ]
    async with Client(mcp.server) as client:
        missing = await miss(client, f"{base}/NOPE.md")
        for path in [*HIDDEN, *folders]:
            assert await miss(client, f"{base}/{path}") == missing, path
        for folder in NO_FOLDER:
            with pytest.raises(MCPError) as caught:
                await read_directory(client, uri=f"{base}/{folder}")
            assert caught.value.error.message == "Directory not found"
    assert missing[0] == INVALID_PARAMS


@pytest.mark.parametrize("kind", SERVERS)
async def test_no_listing_names_a_hidden_path(tmp_path, kind):
    mcp, base, files = build(kind, tmp_path)
    async with Client(mcp.server) as client:
        listed = [str(r.uri) for r in await client.list_resources()]
        folders = {
            f: (await read_directory(client, uri=f"{base}/{f}".rstrip("/")))[
                "resources"
            ]
            for f in ("", "assets", "references")
        }
        indexes = [
            await read_json(client, f"{base}/{f}_index.json")
            for f in ("", "assets/", "references/")
        ]
        index_md = (await client.read_resource(f"{base}/_index.md"))[0].text
        manifest = await read_json(client, f"{base}/_manifest")
        entries = (await request(client, "skills/list"))["skills"]
    named = [
        *listed,
        *(r["uri"] for rows in [*folders.values(), *indexes] for r in rows),
        *(line.split(": ", 1)[0] for line in index_md.splitlines()),
        *(f"{base}/{f['path']}" for f in manifest["files"]),
        *(r["uri"] for e in entries for r in e["resources"]),
    ]
    assert not [u for u in named if hidden(u, base)]
    assert f"{base}/SKILL.md" in [e["uri"] for e in entries]
    assert {f["path"] for f in manifest["files"]} == files
    assert [r["uri"] for r in folders["assets"]] == [f"{base}/assets/a.txt"]


@pytest.mark.parametrize("kind", SERVERS)
async def test_hidden_paths_are_left_out_quietly(tmp_path, caplog, kind):
    with caplog.at_level(logging.DEBUG, logger="kubed.mcp"):
        mcp, base, _ = build(kind, tmp_path)
        async with Client(mcp.server) as client:
            await client.list_resources()
            await read_directory(client, uri=base)
            await request(client, "skills/list")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "kubed.mcp.skills" and "hidden paths not served" in r.getMessage()
    ]
    assert len(lines) == (0 if kind == "per-request" else 1)
    assert all(".env" in line for line in lines)


async def test_a_rewriter_never_sees_a_hidden_path(tmp_path):
    files = {f"skills/demo-guide/{p}": b for p, b in {**HIDDEN, **ORDINARY}.items()}
    mcp = plugin_served((copy(tmp_path, files=files), None))
    seen = []

    def rewriter(site, text):
        seen.append(site.path)
        return text + "\n"

    mcp.add_rewriter(rewriter)
    async with Client(mcp.server) as client:
        await client.list_resources()
    skill_paths = [p for p in seen if p.startswith("skills/")]
    assert "skills/demo-guide/assets/a.txt" in skill_paths
    assert not [p for p in skill_paths if skills._is_hidden(p)]
