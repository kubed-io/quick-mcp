"""The Skills extension against SEP-2640 (Final), against built servers in both eras: what a
host acting on a skill checks, it checks here."""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from gateway_server import kit, over_http
from mcp_types import INVALID_PARAMS
from plugin_server import copy
from plugin_server import served as plugin_served
from skill_server import (
    DIRECTORY,
    ENTRY_URI,
    demo,
    read_bytes,
    read_directory,
    read_json,
    request,
    served,
)
from test_skills import description, header
from test_via import PLUGIN_SKILLS, make

from kubed.mcp import QuickMCP
from kubed.mcp.protocol import MCPError
from kubed.mcp.skills import Skill, serve

pytestmark = pytest.mark.unit

MODES = ["auto", "legacy"]
SERVERS = ["folder", "plugin", "rewritten", "prefixed"]

FILES = {
    "references/GUIDE.md": "# Guide\n\nRead references/deep/NOTE.md.\n",
    "references/deep/NOTE.md": "# Note\r\nline one\r\nline two\r\n",
    "scripts/run.sh": "#!/bin/sh\necho hi\n",
}


@asynccontextmanager
async def connect(
    kind: str, tmp_path: Path, mode: str
) -> AsyncIterator[tuple[Client, str]]:
    """An open client on a built server of ``kind``, and its skill's root URI."""
    if kind == "folder":
        mcp = served(tmp_path, {**FILES, "assets/logo.bin": b"\x00\xff\r\n"})
        async with Client(mcp.server, mode=mode) as client:
            yield client, "skill://tmp-skill"
    elif kind in ("plugin", "rewritten"):
        files = {f"skills/demo-guide/{p}": b for p, b in FILES.items()}
        mcp = plugin_served((copy(tmp_path, files=files), None))
        if kind == "rewritten":
            mcp.add_rewriter(lambda site, text: text + "\n<!-- r -->\n")
        async with Client(mcp.server, mode=mode) as client:
            yield client, "skill://t/demo-guide"
    else:
        async with over_http(kit(tmp_path), header="kb+", mode=mode) as client:
            yield client, "skill://lib/guide"


def three(**options: Any) -> QuickMCP:
    """A server named ``t`` with three catalogue skills, ``alpha``, ``beta`` and ``gamma``."""
    mcp = QuickMCP("t", **options)
    skills = [
        Skill.from_files(
            {
                "SKILL.md": f"---\nname: {n}\ndescription: The {n} skill.\n---\n",
                "a.md": "# A\n",
            }
        )
        for n in ("alpha", "beta", "gamma")
    ]
    serve(mcp.server, skills)
    return mcp


async def walk(client: Client, uri: str) -> set[str]:
    """Every file URI under folder ``uri``, by ``resources/directory/read`` alone."""
    found: set[str] = set()
    cursor = None
    while True:
        params = {"uri": uri} if cursor is None else {"uri": uri, "cursor": cursor}
        page = await read_directory(client, **params)
        for child in page["resources"]:
            if child.get("mimeType") == DIRECTORY:
                found |= await walk(client, child["uri"])
            else:
                found.add(child["uri"])
        cursor = page.get("nextCursor")
        if cursor is None:
            return found


async def entry_of(client: Client, root: str) -> dict:
    """The ``skills/list`` entry whose ``uri`` is ``root``'s SKILL.md."""
    cursor = None
    while True:
        params = {} if cursor is None else {"cursor": cursor}
        page = await request(client, "skills/list", **params)
        for entry in page["skills"]:
            if entry["uri"] == f"{root}/SKILL.md":
                return entry
        cursor = page.get("nextCursor")
        if cursor is None:
            raise AssertionError(f"no skills/list entry for {root}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", SERVERS)
async def test_a_directory_read_is_complete(tmp_path, kind, mode):
    async with connect(kind, tmp_path, mode) as (client, root):
        page = await read_directory(client, uri=root)
    assert page["resultType"] == "complete"
    assert page["resources"]


@pytest.mark.parametrize("mode", MODES)
async def test_a_prompts_folder_read_is_complete(tmp_path, mode):
    """The agents extension binds the method on a plugin server (PLUG-68)."""
    mcp = plugin_served((copy(tmp_path), None))
    async with Client(mcp.server, mode=mode) as client:
        page = await read_directory(client, uri="prompts://t")
    assert page["resultType"] == "complete"


DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
VIEWS = ("/_manifest", "/_index.md", "/_index.json")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", SERVERS)
async def test_every_file_verifies(tmp_path, kind, mode):
    async with connect(kind, tmp_path, mode) as (client, root):
        entry = await entry_of(client, root)
        reads = {
            r["uri"]: read_bytes((await client.read_resource(r["uri"]))[0])
            for r in entry["resources"]
        }
    for resource in entry["resources"]:
        data = reads[resource["uri"]]
        assert DIGEST.fullmatch(resource["digest"]), resource["uri"]
        assert resource["digest"] == f"sha256:{hashlib.sha256(data).hexdigest()}"
        assert resource["size"] == len(data), resource["uri"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", SERVERS)
async def test_every_file_is_listed_once_and_nothing_else(tmp_path, kind, mode):
    """Complete (SKL-25): a walk, the entry and the manifest agree; no view is a file (SKL-26)."""
    async with connect(kind, tmp_path, mode) as (client, root):
        entry = await entry_of(client, root)
        walked = await walk(client, root)
        manifest = await read_json(client, f"{root}/_manifest")
        index = (await client.read_resource(f"{root}/_index.md"))[0].text
        got = await request(client, "skills/get", uri=entry["uri"])
    assert isinstance(entry["resources"], list)
    named = [r["uri"] for r in entry["resources"]]
    assert len(named) == len(set(named))
    assert entry["uri"] in named
    assert set(named) == walked == {f"{root}/{f['path']}" for f in manifest["files"]}
    assert {
        (f"{root}/{f['path']}", f["hash"], f["size"]) for f in manifest["files"]
    } == {(r["uri"], r["digest"], r["size"]) for r in entry["resources"]}
    assert not [u for u in named if u.endswith(VIEWS)]
    assert index  # the views still read by exact URI
    assert got["skill"] == entry


@pytest.mark.parametrize("mode", MODES)
async def test_the_listing_pages_whole_entries(mode):
    async with Client(three().server, mode=mode) as client:
        whole = (await request(client, "skills/list"))["skills"]
    pages, cursor = [], None
    async with Client(three(list_page_size=1).server, mode=mode) as client:
        while True:
            page = await request(
                client, "skills/list", **({} if cursor is None else {"cursor": cursor})
            )
            pages.append(page["skills"])
            if (cursor := page.get("nextCursor")) is None:
                break
        got = [
            (await request(client, "skills/get", uri=e["uri"]))["skill"]
            for p in pages
            for e in p
        ]
    assert [len(p) for p in pages] == [1, 1, 1]
    assert [e for p in pages for e in p] == whole == got
    assert [e["uri"] for e in whole] == [
        f"skill://{n}/SKILL.md" for n in ("alpha", "beta", "gamma")
    ]


PAGED_CURSOR = INVALID_PARAMS  # a paged listing refuses it (SKL-28)


@pytest.mark.parametrize("paged", [False, True], ids=["one-page", "paged"])
async def test_an_invalid_list_cursor(paged):
    """SKL-28: FastMCP's handler answers it; the base protocol SHOULD say -32602."""
    mcp = three(list_page_size=1) if paged else three()
    async with Client(mcp.server) as client:
        first = await request(client, "skills/list")
        try:
            got = await request(client, "skills/list", cursor="not-a-cursor")
            answer = "ignored" if got["skills"] == first["skills"] else got
        except MCPError as exc:
            answer = exc.error.code
    assert answer == (PAGED_CURSOR if paged else "ignored")


FRONT = (
    "name: tmp-skill\n"
    "description: A skill for tests.\n"
    "license: Apache-2.0\n"
    "allowed-tools: Read Grep\n"
    "metadata:\n  version: '2.1.0'\n  owner: team-a\n"
    "x-unknown:\n  deep: [1, two]\n  flag: true\n"
)
EXPECTED = {
    "name": "tmp-skill",
    "description": "A skill for tests.",
    "license": "Apache-2.0",
    "allowed-tools": "Read Grep",
    "metadata": {"version": "2.1.0", "owner": "team-a"},
    "x-unknown": {"deep": [1, "two"], "flag": True},
}


@pytest.mark.parametrize("mode", MODES)
async def test_the_frontmatter_is_verbatim(tmp_path, mode):
    mcp = served(tmp_path, {}, front=FRONT)
    async with Client(mcp.server, mode=mode) as client:
        entry = (await request(client, "skills/list"))["skills"][0]
        got = (await request(client, "skills/get", uri=entry["uri"]))["skill"]
        text = (await client.read_resource(entry["uri"]))[0].text
    assert entry["frontmatter"] == got["frontmatter"] == header(text) == EXPECTED
    assert entry["uri"].rsplit("/", 2)[-2] == entry["frontmatter"]["name"]


ROW_NAME = "demo-skill/SKILL.md"  # FastMCP's row, kept (SKL-30)


async def test_the_skill_md_row():
    """SEP-2640 Resource Metadata: mimeType, description and name (SKL-8, SKL-30)."""
    async with Client(demo().server) as client:
        row = next(r for r in await client.list_resources() if str(r.uri) == ENTRY_URI)
    assert row.mime_type == "text/markdown"
    assert row.description == description()
    assert row.name == ROW_NAME


async def test_a_nested_skill_md_is_a_file_of_its_skill(tmp_path):
    inner = "---\nname: inner\ndescription: An inner skill.\n---\n\n# Inner\n"
    mcp = served(tmp_path, {"examples/inner/SKILL.md": inner})
    async with Client(mcp.server) as client:
        entries = (await request(client, "skills/list"))["skills"]
        with pytest.raises(MCPError) as caught:
            await request(
                client, "skills/get", uri="skill://tmp-skill/examples/inner/SKILL.md"
            )
    assert [e["uri"] for e in entries] == ["skill://tmp-skill/SKILL.md"]
    assert "skill://tmp-skill/examples/inner/SKILL.md" in [
        r["uri"] for r in entries[0]["resources"]
    ]
    assert caught.value.error.code == INVALID_PARAMS


async def test_an_unlisted_skill_is_still_got(tmp_path):
    """Under X-MCP-Via local the plugin's skills are unlisted, and skills/get answers each
    with the entry the listing gives under mcp (SKL-23, DLV-9, DLV-10)."""
    mcp = make(tmp_path)
    async with over_http(mcp) as client:
        listed = {e["uri"]: e for e in (await request(client, "skills/list"))["skills"]}
    async with over_http(mcp, headers={"X-MCP-Via": "local"}) as client:
        unlisted = [e["uri"] for e in (await request(client, "skills/list"))["skills"]]
        got = {
            u: (await request(client, "skills/get", uri=u))["skill"]
            for u in PLUGIN_SKILLS
        }
    assert not [u for u in unlisted if u in PLUGIN_SKILLS]
    assert got == {u: listed[u] for u in PLUGIN_SKILLS}
