"""The Skills extension against SEP-2640 (Final), against built servers in both eras: what a
host acting on a skill checks, it checks here."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastmcp import Client
from gateway_server import kit, over_http
from plugin_server import copy
from plugin_server import served as plugin_served
from skill_server import DIRECTORY, read_directory, request, served

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
