"""A skill's text files: JSON, YAML, TOML and scripts read as text, by their media type,
and rewriters and the gateway's prefix see them (PLUG-75, PLUG-58, SKL-9, GW-5)."""

from __future__ import annotations

import hashlib
import json

import pytest
import skill_server
from fastmcp import Client
from gateway_server import KIT_TREE, over_http
from plugin_server import ENTRY, copy, served, write
from skill_server import read_bytes, read_directory, read_json, request

from kubed.mcp import QuickMCP
from kubed.mcp._media import media_type, text_like, text_of
from kubed.mcp.plugins import to_mcp_form

pytestmark = pytest.mark.unit

SKILL = "skill://t/demo-guide"

# Under skills/demo-guide/ of the demo plugin: each file's bytes and media type.
TEXT = {
    "data.json": (b'{"a": 1}\n', "application/json"),
    "conf.yaml": (b"a: 1\r\n", "application/yaml"),  # CRLF kept
    "x.YML": (b"\xef\xbb\xbfa: 1\n", "application/yaml"),  # a BOM, upper case
    "CONFIG.JSON": (b"{}", "application/json"),
    "pyproject.toml": (b"[a]\n", "application/toml"),
    "run.sh": (b"echo hi\n", "application/x-sh"),
    "mod.ts": (b"export {}\n", "text/x-typescript"),
}

BLOBS = {
    "logo.png": (b"\x89PNG\r\n\x1a\n", "image/png"),
    "notes.txt": ("café".encode("latin-1"), "text/plain"),  # text-like, not UTF-8
    "bad.json": (b'{"a": "\xff"}', "application/json"),
    "x.bin": (b"\x00\x01", "application/octet-stream"),
    "Makefile": (b"all:\n", "application/octet-stream"),  # nothing guessed
}


def sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def with_files(tmp_path, files: dict[str, tuple[bytes, str]]) -> QuickMCP:
    """The demo plugin, with ``files`` written byte for byte into its skill."""
    root = copy(tmp_path)
    for name, (data, _) in files.items():
        (root / "skills/demo-guide" / name).write_bytes(data)
    return served((root, None))


async def entry_rows(client) -> dict[str, dict]:
    """The demo skill's ``skills/list`` rows, by URI: empty when it is not offered."""
    skills = (await request(client, "skills/list"))["skills"]
    return {
        row["uri"]: row
        for entry in skills
        if entry["uri"] == ENTRY
        for row in entry["resources"]
    }


async def manifest_rows(client) -> dict[str, dict]:
    manifest = json.loads((await client.read_resource(f"{SKILL}/_manifest"))[0].text)
    return {row["path"]: row for row in manifest["files"]}


async def test_a_skills_text_files_read_as_text(tmp_path):
    async with Client(with_files(tmp_path, TEXT).server) as client:
        rows = await entry_rows(client)
        manifest = await manifest_rows(client)
        for name, (data, media) in TEXT.items():
            content = (await client.read_resource(f"{SKILL}/{name}"))[0]
            assert content.text == data.decode("utf-8"), name
            assert content.mime_type == media, name
            assert rows[f"{SKILL}/{name}"]["digest"] == sha256(data), name
            assert rows[f"{SKILL}/{name}"]["size"] == len(data), name
            assert manifest[name]["hash"] == sha256(data), name
            assert manifest[name]["size"] == len(data), name


async def test_other_files_read_as_blobs(tmp_path):
    async with Client(with_files(tmp_path, BLOBS).server) as client:
        for name, (data, media) in BLOBS.items():
            content = (await client.read_resource(f"{SKILL}/{name}"))[0]
            assert getattr(content, "text", None) is None, name
            assert read_bytes(content) == data, name
            assert content.mime_type == media, name
        rows = await entry_rows(client)
        assert {f"{SKILL}/{name}" for name in BLOBS} <= set(rows)
        for uri, row in rows.items():
            served_bytes = read_bytes((await client.read_resource(uri))[0])
            assert row["digest"] == sha256(served_bytes), uri


async def test_the_index_rows_carry_the_media_type(tmp_path):
    async with Client(with_files(tmp_path, TEXT).server) as client:
        index = await read_json(client, f"{SKILL}/_index.json")
        directory = (await read_directory(client, uri=SKILL))["resources"]
    for rows in (index, directory):
        by_uri = {row["uri"]: row for row in rows}
        for name, (_, media) in TEXT.items():
            assert by_uri[f"{SKILL}/{name}"]["mimeType"] == media, name


async def test_a_catalogue_served_with_serve_reads_the_same(tmp_path):  # SKL-9
    mcp = skill_server.served(tmp_path, {"data.json": b"{}"})
    async with Client(mcp.server) as client:
        content = (await client.read_resource("skill://tmp-skill/data.json"))[0]
    assert content.text == "{}"
    assert content.mime_type == "application/json"


def test_the_media_types():
    assert media_type("a/B.YML") == "application/yaml"
    assert media_type("x.ts") == "text/x-typescript"
    assert media_type("Makefile") == "application/octet-stream"
    assert text_like("x.svg")  # image/svg+xml
    assert text_like("CONFIG.JSON")
    assert not text_like("x.png") and not text_like("Makefile")
    assert text_of("x.json", b"\xff") is None and text_of("x.json", b"") == ""
    assert text_of("x.png", b"plain") is None


async def test_a_rewriter_sees_a_skills_json_and_yaml(tmp_path):  # PLUG-58
    files = {
        "data.json": (b'{"root": "${CLAUDE_SKILL_DIR}/x"}', "application/json"),
        "conf.yaml": (b"see: ${CLAUDE_SKILL_DIR}/x\n", "application/yaml"),
    }
    mcp = with_files(tmp_path, files)
    mcp.add_rewriter(to_mcp_form)
    async with Client(mcp.server) as client:
        data = (await client.read_resource(f"{SKILL}/data.json"))[0].text
        conf = (await client.read_resource(f"{SKILL}/conf.yaml"))[0].text
        rows = await entry_rows(client)
    assert data == '{"root": "skill://t/demo-guide/x"}'
    assert conf == "see: skill://t/demo-guide/x\n"
    assert rows[f"{SKILL}/data.json"]["digest"] == sha256(data.encode())
    assert rows[f"{SKILL}/conf.yaml"]["digest"] == sha256(conf.encode())


async def test_the_prefix_reaches_a_skills_json(tmp_path):  # GW-5
    links = "skills/guide/references/links.json"
    tree = {**KIT_TREE, links: '{"see": "skill://lib/guide/SKILL.md"}'}
    mcp = QuickMCP("t")
    mcp.add_plugin(write(tmp_path / "kit", tree), namespace="lib")
    uri = "skill://lib/guide/references/links.json"
    async with over_http(mcp, header="kb+") as client:
        text = (await client.read_resource(uri))[0].text
        skills = (await request(client, "skills/list"))["skills"]
    assert '"kb+skill://lib/guide/SKILL.md"' in text
    rows = {row["uri"]: row for entry in skills for row in entry["resources"]}
    assert rows[uri]["digest"] == sha256(text.encode())
