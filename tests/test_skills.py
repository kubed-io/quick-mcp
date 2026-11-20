"""Serving skills: loading one strictly, the listing, the reads and the Skills extension.

Every test goes through a real FastMCP ``Client``, in both protocol eras where the era
matters. A Skills-aware host trusts nothing it cannot verify: it reads each file an entry
names and checks it against the digest, so the strongest tests here do that over the wire.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging

import pytest
import yaml
from fastmcp import Client
from fastmcp.skills import SkillsClientExtension
from mcp_types import INVALID_PARAMS, METHOD_NOT_FOUND
from skill_server import (
    DEMO,
    ENTRY_URI,
    GOOD,
    MANIFEST_URI,
    demo,
    make_skill,
    request,
    served,
)

from kubed.mcp import QuickMCP
from kubed.mcp.protocol import MCPError
from kubed.mcp.skills import (
    EntryOnlySkillProvider,
    Skill,
    SkillsExtension,
    load_skill,
    serve,
)

pytestmark = pytest.mark.unit

EXTENSION_ID = SkillsExtension.identifier
REFERENCES = ["references/GUIDE.md", "references/NOTES.md"]


def header(text: str) -> object:
    """The YAML front matter of a Markdown document."""
    lines = text.splitlines()
    end = next(i for i, line in enumerate(lines[1:], start=1) if line.rstrip() == "---")
    return yaml.safe_load("\n".join(lines[1:end]))


def description() -> str:
    return header((DEMO / "SKILL.md").read_text())["description"]


# ---- loading -------------------------------------------------------------------------


def test_load_skill_reads_a_valid_skill():
    skill = load_skill(DEMO)
    assert skill is not None
    assert skill.name == "demo-skill"
    assert set(skill.files) == {"SKILL.md", "scripts/run.sh", *REFERENCES}


def test_a_folder_with_no_skill_md_warns(tmp_path, caplog):
    folder = tmp_path / "tmp-skill"
    folder.mkdir()
    with caplog.at_level(logging.WARNING):
        assert load_skill(folder) is None
    assert [r.levelno for r in caplog.records if str(folder) in r.getMessage()] == [
        logging.WARNING
    ]


def invalid_skill(tmp_path, case: str):
    """A folder that breaks one Agent Skills rule."""
    fronts = {
        "bad-name": ("Odd_Skill", "name: Odd_Skill\ndescription: Does a thing."),
        "folder-mismatch": (
            "tmp-skill",
            "name: some-other-name\ndescription: Does a thing.",
        ),
        "no-description": ("tmp-skill", "name: tmp-skill"),
        "empty-description": ("tmp-skill", "name: tmp-skill\ndescription: ''"),
        "long-description": (
            "tmp-skill",
            "name: tmp-skill\ndescription: " + "x" * 1025,
        ),
        "not-a-mapping": ("tmp-skill", "- a list\n- not a mapping"),
        "date": ("tmp-skill", GOOD + "\nwhen: 2026-10-05"),
        "nan": ("tmp-skill", GOOD + "\nweight: .nan"),
        "int-key": ("tmp-skill", GOOD + "\nmetadata:\n  1: one"),
    }
    if case in fronts:
        name, front = fronts[case]
        folder = tmp_path / name
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            f"---\n{front}\n---\n\n# Body\n", encoding="utf-8"
        )
        return folder
    if case == "symlink":
        folder = make_skill(tmp_path, {})
        (tmp_path / "outside.md").write_text("# Outside\n", encoding="utf-8")
        (folder / "linked.md").symlink_to(tmp_path / "outside.md")
        return folder
    folder = tmp_path / "tmp-skill"  # not-utf8
    folder.mkdir()
    (folder / "SKILL.md").write_bytes(
        f"---\n{GOOD}\n---\n\n# Caf\xe9\n".encode("latin-1")
    )
    return folder


INVALID = {
    "bad-name": "lowercase letters",
    "folder-mismatch": "must match its SKILL.md name",
    "no-description": "description must contain",
    "empty-description": "description must contain",
    "long-description": "description must contain",
    "not-a-mapping": "frontmatter must be a mapping",
    "date": "finite JSON values",
    "nan": "finite JSON values",
    "int-key": "keys must be strings",
    "symlink": "symlink",
    "not-utf8": "utf-8",
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_an_invalid_skill_is_an_error_naming_why(tmp_path, caplog, case):
    """Strict: no skill, and one error naming the folder and the reason."""
    folder = invalid_skill(tmp_path, case)
    with caplog.at_level(logging.WARNING):
        assert load_skill(folder) is None
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert str(folder) in errors[0]
    assert INVALID[case] in errors[0]


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_a_server_that_adds_no_skills_has_no_extension(mode):
    """What a server that could not load its skill is left with."""
    mcp = QuickMCP("t")
    async with Client(mcp.server, mode=mode) as client:
        uris = [str(r.uri) for r in await client.list_resources()]
        extensions = client.session.server_capabilities.extensions or {}
        with pytest.raises(MCPError) as listed:
            await request(client, "skills/list")
        with pytest.raises(MCPError) as folder:
            await request(client, "resources/directory/read", uri="skill://demo-skill")
    assert not [u for u in uris if u.startswith("skill://")]
    assert EXTENSION_ID not in extensions
    assert listed.value.error.code == METHOD_NOT_FOUND
    assert folder.value.error.code == METHOD_NOT_FOUND


# ---- the listing and the reads --------------------------------------------------------


async def test_the_listing_names_the_skill_once():
    """One row, so a client scanning resources/list sees one skill, not its files."""
    async with Client(demo().server) as client:
        listed = await client.list_resources()
    rows = [r for r in listed if str(r.uri).startswith("skill://")]
    assert [str(r.uri) for r in rows] == [ENTRY_URI]
    assert rows[0].description == description()


async def test_no_skill_template_is_listed():
    async with Client(demo().server) as client:
        templates = await client.list_resource_templates()
    assert not [t for t in templates if t.uri_template.startswith("skill://")]


@pytest.mark.parametrize("path", ["_manifest", *REFERENCES])
async def test_every_file_still_reads_by_its_uri(path):
    async with Client(demo().server) as client:
        contents = await client.read_resource(f"skill://demo-skill/{path}")
    assert contents[0].text
    if path.endswith(".md"):
        assert contents[0].mime_type == "text/markdown"
    else:
        assert json.loads(contents[0].text)["skill"] == "demo-skill"


async def test_an_unknown_skill_file_is_not_found():
    async with Client(demo().server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("skill://demo-skill/NOPE.md")


async def test_the_served_text_keeps_its_frontmatter():
    async with Client(demo().server) as client:
        contents = await client.read_resource(ENTRY_URI)
    assert contents[0].text == (DEMO / "SKILL.md").read_text()


@pytest.mark.parametrize(
    "extensions", [[], [SkillsClientExtension()]], ids=["resources", "skills"]
)
async def test_a_fastmcp_client_reads_the_whole_skill(extensions):
    """Every file the manifest names, verified against its digest."""
    async with Client(demo().server, extensions=extensions) as client:
        manifest = json.loads((await client.read_resource(MANIFEST_URI))[0].text)
        read = await client.read_skill(ENTRY_URI)
    assert read.name == "demo-skill"
    assert set(read.files) == {f["path"] for f in manifest["files"]}


async def test_a_fastmcp_client_discovers_the_skill():
    async with Client(demo().server, extensions=[SkillsClientExtension()]) as client:
        found = await client.list_skills()
        listed = await client.list_skill_resources()
    assert [s.frontmatter.name for s in found] == ["demo-skill"]
    assert found[0].frontmatter.description == description()
    assert [str(r.uri) for r in listed] == [ENTRY_URI]


# ---- the Skills extension -------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_the_extension_is_declared_beside_resources(mode):
    async with Client(demo().server, mode=mode) as client:
        caps = client.session.server_capabilities
    assert caps.extensions[EXTENSION_ID] == {"directoryRead": True}
    assert caps.resources is not None


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_the_listing_is_the_one_skill(mode):
    async with Client(demo().server, mode=mode) as client:
        listing = await request(client, "skills/list")
    assert listing["resultType"] == "complete"
    assert listing["ttlMs"] == 0
    assert listing["cacheScope"] == "private"
    assert [entry["uri"] for entry in listing["skills"]] == [ENTRY_URI]


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_get_returns_the_listed_entry(mode):
    async with Client(demo().server, mode=mode) as client:
        listed = (await request(client, "skills/list"))["skills"][0]
        got = await request(client, "skills/get", uri=ENTRY_URI)
    assert got["skill"] == listed
    assert got["resultType"] == "complete"


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_every_file_verifies_against_its_digest(mode):
    async with Client(demo().server, mode=mode) as client:
        entry = (await request(client, "skills/list"))["skills"][0]
        for resource in entry["resources"]:
            # A blob or text by the host's media types: run.sh is either.
            data = read_bytes((await client.read_resource(resource["uri"]))[0])
            assert resource["size"] == len(data), resource["uri"]
            digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
            assert resource["digest"] == digest, resource["uri"]


async def test_the_entry_names_every_file_the_server_serves():
    """The listing shows SKILL.md alone, so the manifest is the served set."""
    async with Client(demo().server) as client:
        entry = (await request(client, "skills/list"))["skills"][0]
        manifest = json.loads((await client.read_resource(MANIFEST_URI))[0].text)
    served_uris = {f"skill://demo-skill/{f['path']}" for f in manifest["files"]}
    named = [resource["uri"] for resource in entry["resources"]]
    assert len(named) == len(set(named))
    assert set(named) == served_uris
    assert ENTRY_URI in named


async def test_the_frontmatter_is_the_served_header():
    async with Client(demo().server) as client:
        entry = (await request(client, "skills/list"))["skills"][0]
        text = (await client.read_resource(ENTRY_URI))[0].text
    assert entry["frontmatter"] == header(text)


async def test_a_fastmcp_client_gets_the_entry():
    async with Client(demo().server, extensions=[SkillsClientExtension()]) as client:
        listed = await client.list_skills()
        got = await client.get_skill(ENTRY_URI)
    assert [entry.uri for entry in listed] == [ENTRY_URI]
    assert got == listed[0]


@pytest.mark.parametrize(
    "uri",
    ["skill://nope/SKILL.md", MANIFEST_URI, "skill://demo-skill/references/GUIDE.md"],
)
async def test_get_of_any_other_uri_is_unknown(uri):
    async with Client(demo().server) as client:
        with pytest.raises(MCPError) as caught:
            await request(client, "skills/get", uri=uri)
    assert caught.value.error.code == INVALID_PARAMS
    assert caught.value.error.message == f"Unknown skill URI: {uri}"


async def test_a_list_cursor_is_ignored():
    """No list page size is set, so every entry is the one page."""
    async with Client(demo().server) as client:
        first = await request(client, "skills/list")
        again = await request(client, "skills/list", cursor="2")
    assert again["skills"] == first["skills"]
    assert "nextCursor" not in again


async def test_a_dash_run_inside_a_value_is_not_the_header_end(tmp_path):
    front = "name: tmp-skill\ndescription: before --- after"
    async with Client(served(tmp_path, {}, front=front).server) as client:
        entry = (await request(client, "skills/list"))["skills"][0]
    assert entry["frontmatter"]["description"] == "before --- after"


def read_bytes(content) -> bytes:
    text = getattr(content, "text", None)
    return text.encode() if text is not None else base64.b64decode(content.blob)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_a_crlf_file_is_served_byte_for_byte(tmp_path, mode):
    """The digest is of the bytes on disk, which are the bytes a read serves; a titled
    file keeps the skill in the listing."""
    body = b"# Crlf\r\nline one\r\nline two\r\n"
    mcp = served(tmp_path, {"references/CRLF.md": body, "data.bin": b"\x00\xff\r\n"})
    async with Client(mcp.server, mode=mode) as client:
        entry = (await request(client, "skills/list"))["skills"][0]
        manifest = json.loads(
            (await client.read_resource("skill://tmp-skill/_manifest"))[0].text
        )
        reads = {
            r["uri"]: read_bytes((await client.read_resource(r["uri"]))[0])
            for r in entry["resources"]
        }
    assert reads["skill://tmp-skill/references/CRLF.md"] == body
    hashes = {
        f"skill://tmp-skill/{f['path']}": (f["hash"], f["size"])
        for f in manifest["files"]
    }
    assert set(reads) == set(hashes)
    for resource in entry["resources"]:
        data = reads[resource["uri"]]
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        assert (resource["digest"], resource["size"]) == (digest, len(data))
        assert hashes[resource["uri"]] == (digest, len(data))


# ---- catalogues -----------------------------------------------------------------------


NESTED = "lib/folder/demo-skill"
NESTED_ROOT = f"skill://{NESTED}"


def nested() -> QuickMCP:
    mcp = QuickMCP("t")
    serve(mcp.server, {NESTED: load_skill(DEMO)})
    return mcp


async def test_a_nested_path_serves_the_skill_under_it():
    """The path's last segment is the name; the segments before it are the URI's."""
    async with Client(nested().server, extensions=[SkillsClientExtension()]) as client:
        uris = [str(r.uri) for r in await client.list_resources()]
        entry = (await request(client, "skills/list"))["skills"][0]
        manifest = json.loads(
            (await client.read_resource(f"{NESTED_ROOT}/_manifest"))[0].text
        )
        read = await client.read_skill(f"{NESTED_ROOT}/SKILL.md")
        guide = (await client.read_resource(f"{NESTED_ROOT}/references/GUIDE.md"))[0]
    assert uris == [f"{NESTED_ROOT}/SKILL.md"]
    assert entry["uri"] == f"{NESTED_ROOT}/SKILL.md"
    assert entry["frontmatter"]["name"] == "demo-skill"
    assert manifest["skill"] == "demo-skill"
    assert set(read.files) == {"SKILL.md", "scripts/run.sh", *REFERENCES}
    assert guide.text == (DEMO / "references/GUIDE.md").read_text()


async def test_a_catalogue_of_two_skills_lists_each_entry(tmp_path):
    other = Skill.from_files(
        {
            "SKILL.md": "---\nname: other\ndescription: Another skill.\n---\n",
            "a.md": "# A\n",
        }
    )
    mcp = QuickMCP("t")
    serve(mcp.server, {"demo-skill": load_skill(DEMO), "more/other": other})
    async with Client(mcp.server) as client:
        uris = sorted(str(r.uri) for r in await client.list_resources())
        entries = (await request(client, "skills/list"))["skills"]
    assert uris == [ENTRY_URI, "skill://more/other/SKILL.md"]
    assert [e["uri"] for e in entries] == uris


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_serving_twice_adds_a_provider_and_one_extension(mode):
    """A second call adds a provider, never a second extension or middleware."""
    other = Skill.from_files(
        {"SKILL.md": "---\nname: other\ndescription: Another skill.\n---\n"}
    )
    mcp = QuickMCP("t")
    first = serve(mcp.server, [load_skill(DEMO)])
    second = serve(mcp.server, EntryOnlySkillProvider([other]))
    async with Client(mcp.server, mode=mode) as client:
        extensions = client.session.server_capabilities.extensions
        entries = (await request(client, "skills/list"))["skills"]
    assert isinstance(first, EntryOnlySkillProvider)
    assert isinstance(second, EntryOnlySkillProvider)
    assert extensions[EXTENSION_ID] == {"directoryRead": True}
    assert [e["uri"] for e in entries] == [ENTRY_URI, "skill://other/SKILL.md"]


async def test_a_mounted_server_keeps_the_extension_it_brings():
    """The parent registers its own copy, still this extension: clone() keeps the class."""
    from fastmcp import FastMCP

    parent = FastMCP("parent")
    parent.mount(demo().server)
    async with Client(parent) as client:
        extensions = client.session.server_capabilities.extensions
        entries = (await request(client, "skills/list"))["skills"]
    assert extensions[EXTENSION_ID] == {"directoryRead": True}
    assert [e["uri"] for e in entries] == [ENTRY_URI]
    assert type(SkillsExtension().clone()) is SkillsExtension
