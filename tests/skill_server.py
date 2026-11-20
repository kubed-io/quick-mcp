"""Servers over a skill, and the requests no MCP client sends, for the skills tests."""

from __future__ import annotations

import json
from pathlib import Path

from mcp_types import Request, RequestParams
from pydantic import ConfigDict, TypeAdapter

from kubed.mcp import QuickMCP
from kubed.mcp.skills import load_skill, serve

# A small skill: SKILL.md, references/GUIDE.md (titled), references/NOTES.md (no
# heading) and scripts/run.sh.
DEMO = Path(__file__).parent / "skills" / "demo-skill"
ROOT = "skill://demo-skill"
REFERENCES = f"{ROOT}/references"
ENTRY_URI = f"{ROOT}/SKILL.md"
MANIFEST_URI = f"{ROOT}/_manifest"
DIRECTORY = "inode/directory"
GOOD = "name: tmp-skill\ndescription: A skill for tests."


class ExtensionParams(RequestParams):
    model_config = ConfigDict(extra="allow")


async def request(client, method: str, **params) -> dict:
    """Send ``method``, which no MCP client knows, with ``params``."""
    return await client.session.send_request(
        Request(method=method, params=ExtensionParams(**params)), TypeAdapter(dict)
    )


async def read_directory(client, **params) -> dict:
    return await request(client, "resources/directory/read", **params)


async def read_json(client, uri: str):
    contents = await client.read_resource(uri)
    assert contents[0].mime_type == "application/json"
    return json.loads(contents[0].text)


def demo() -> QuickMCP:
    """A server over the demo skill, loaded and added as a server would."""
    mcp = QuickMCP("t")
    serve(mcp.server, [load_skill(DEMO)])
    return mcp


def make_skill(
    tmp_path: Path,
    files: dict[str, str | bytes],
    folders: tuple[str, ...] = (),
    front: str = GOOD,
) -> Path:
    """A throwaway skill folder called ``tmp-skill``."""
    folder = tmp_path / "tmp-skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text(f"---\n{front}\n---\n\n# Tmp\n", encoding="utf-8")
    for path, body in files.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        data = body.encode("utf-8") if isinstance(body, str) else body
        (folder / path).write_bytes(data)
    for path in folders:
        (folder / path).mkdir(parents=True)
    return folder


def served(tmp_path: Path, files: dict[str, str | bytes], **options) -> QuickMCP:
    """A server over a throwaway skill called ``tmp-skill``."""
    mcp = QuickMCP("t")
    serve(mcp.server, [load_skill(make_skill(tmp_path, files, **options))])
    return mcp
