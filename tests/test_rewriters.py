"""Rewriters: the text a server serves, rewritten when a build is made (PLUG-56 to
PLUG-59)."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil

import pytest
from fastmcp import Client
from plugin_server import DEMO, ENTRY, GUIDE, copy, listed, served
from skill_server import read_bytes, request

from kubed.mcp.plugins import View
from kubed.mcp.protocol import MCPError

pytestmark = pytest.mark.unit


async def test_a_rewritten_skill_agrees_on_digest_and_size(tmp_path):
    mcp = served((DEMO, None))
    mcp.add_rewriter(lambda site, text: text.replace("Everything", "All"))  # GUIDE.md
    async with Client(mcp.server) as client:
        entry = (await request(client, "skills/list"))["skills"][0]
        manifest = json.loads(
            (await client.read_resource("skill://t/demo-guide/_manifest"))[0].text
        )
        for resource in entry["resources"]:
            data = (await client.read_resource(resource["uri"]))[0].text.encode()
            assert resource["size"] == len(data)
            assert resource["digest"] == f"sha256:{hashlib.sha256(data).hexdigest()}"
        guide = (await client.read_resource(GUIDE))[0].text
    by_path = {f["path"]: f for f in manifest["files"]}
    assert guide.startswith("# The guide\n\nAll a test needs")
    assert by_path["references/GUIDE.md"]["size"] == len(guide.encode())
    digest = hashlib.sha256(guide.encode()).hexdigest()
    assert by_path["references/GUIDE.md"]["hash"] == f"sha256:{digest}"


async def test_rewriters_run_in_the_order_added():
    mcp = served((DEMO, None))
    mcp.add_rewriter(lambda site, text: text.replace("Greet {{", "Hail {{"))
    mcp.add_rewriter(lambda site, text: text.replace("Hail", "Hello"))
    async with Client(mcp.server) as client:
        rendered = await client.get_prompt("greet", {"who": "x"})
    assert rendered.messages[0].content.text.startswith("Hello x warmly.")


async def test_each_file_is_shown_its_site():
    seen = []
    mcp = served((DEMO, None))
    mcp.add_rewriter(lambda site, text: seen.append(site) or text)
    async with Client(mcp.server) as client:
        await client.list_prompts()
    rows = {(s.kind, s.path, s.uri) for s in seen}
    assert {
        ("skill", "skills/demo-guide/SKILL.md", ENTRY),
        ("skill", "skills/demo-guide/references/GUIDE.md", GUIDE),
        ("command", "commands/greet.md", "prompts://t/greet.md"),
        ("agent", "agents/helper.md", "agents://t/helper.md"),
        ("index", "skills/demo-guide", "skill://t/demo-guide/_index.md"),
        ("index", "skills/demo-guide", "skill://t/demo-guide/_index.json"),
        (
            "index",
            "skills/demo-guide/references",
            "skill://t/demo-guide/references/_index.md",
        ),
    } <= rows
    assert {(s.view, s.admitted, s.namespace) for s in seen} == {
        (None, mcp.plugins, "t")
    }
    assert seen[0].skills == {"skills/demo-guide": "skill://t/demo-guide"}
    with pytest.raises(TypeError):
        seen[0].skills["x"] = "y"  # read-only: one build shares it


GREET = ("prompts/get", {"name": "greet", "arguments": {"who": "x"}})
READ_ENTRY = ("resources/read", {"uri": ENTRY})
GET_HELPER = ("agents/get", {"name": "helper"})


def at(path, kind=None):
    """True for the site at ``path``; an index site only when ``kind`` is "index"."""
    return lambda site: (
        site.path == path
        and (site.kind == "index") == (kind == "index")
        and (kind != "index" or site.uri.endswith("/_index.md"))
    )


def raising(site, text):
    raise ValueError("boom")


# (id, which site, the rewrite there, the read that must now fail, one that still
# works, logged)
BROKEN = [
    (
        "skill raises",
        at("skills/demo-guide/references/GUIDE.md"),
        raising,
        READ_ENTRY,
        GREET,
        "boom",
    ),
    ("command raises", at("commands/greet.md"), raising, GREET, READ_ENTRY, "boom"),
    ("agent raises", at("agents/helper.md"), raising, GET_HELPER, READ_ENTRY, "boom"),
    (
        "index raises",
        at("skills/demo-guide", "index"),
        raising,
        ("resources/read", {"uri": "skill://t/demo-guide/_index.md"}),
        READ_ENTRY,
        "boom",
    ),
    (
        "skill renamed",
        at("skills/demo-guide/SKILL.md"),
        lambda site, text: text.replace("name: demo-guide", "name: other-guide"),
        READ_ENTRY,
        GREET,
        "demo-guide",
    ),
    (
        "skill refused",
        at("skills/demo-guide/SKILL.md"),
        lambda site, text: text.replace("---\nname: demo-guide", "name: demo-guide", 1),
        READ_ENTRY,
        GREET,
        "SKILL.md",
    ),
    (
        "command broken",
        at("commands/greet.md"),
        lambda site, text: text.replace(
            "description: Greet someone by name.", "description: [", 1
        ),
        GREET,
        READ_ENTRY,
        "commands/greet.md",
    ),
    (
        "agent renamed",
        at("agents/helper.md"),
        lambda site, text: text.replace("name: helper", "name: other"),
        GET_HELPER,
        READ_ENTRY,
        "helper",
    ),
    (
        "agent non-str",
        at("agents/helper.md"),
        lambda site, text: None,
        GET_HELPER,
        READ_ENTRY,
        "str",
    ),
]


@pytest.mark.parametrize(
    ("where", "rewrite", "fails", "works", "logged"),
    [row[1:] for row in BROKEN],
    ids=[row[0] for row in BROKEN],
)
async def test_a_broken_rewrite_fails_only_its_item(  # Review Focus 3
    where, rewrite, fails, works, logged, caplog
):
    def scoped(site, text):
        return rewrite(site, text) if where(site) else text

    mcp = served((DEMO, None))
    mcp.add_rewriter(scoped)
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError):
            await request(client, fails[0], **fails[1])
        await request(client, works[0], **works[1])
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("scoped" in m and logged in m for m in errors)


async def test_binary_and_non_utf8_files_pass_untouched(tmp_path):
    png = b"\x89PNG\r\n${CLAUDE_PLUGIN_ROOT}"
    latin = "café ${CLAUDE_PLUGIN_ROOT}".encode("latin-1")
    root = copy(tmp_path)
    (root / "skills/demo-guide/logo.png").write_bytes(png)
    (root / "skills/demo-guide/notes.txt").write_bytes(latin)
    seen = []
    mcp = served((root, None))
    mcp.add_rewriter(
        lambda site, text: (
            seen.append(site.path) or text.replace("${CLAUDE_PLUGIN_ROOT}", "X")
        )
    )
    async with Client(mcp.server) as client:
        logo = (await client.read_resource("skill://t/demo-guide/logo.png"))[0]
        notes = (await client.read_resource("skill://t/demo-guide/notes.txt"))[0]
    assert not {"skills/demo-guide/logo.png", "skills/demo-guide/notes.txt"} & set(seen)
    assert read_bytes(logo) == png and read_bytes(notes) == latin


async def test_an_index_is_rewritten_once_per_build():
    key, calls = {"now": None}, []
    mcp = served((DEMO, None))
    mcp.plugin_view(lambda: View(key["now"], lambda plugin: True))

    def prefix(site, text):
        if site.kind == "index":
            calls.append(site.uri)
        return text.replace("skill://", f"{site.view}+skill://") if site.view else text

    mcp.add_rewriter(prefix)
    async with Client(mcp.server) as client:
        plain = (await client.read_resource("skill://t/demo-guide/_index.md"))[0].text
        key["now"] = "kb"
        first = (await client.read_resource("skill://t/demo-guide/_index.md"))[0].text
        again = (await client.read_resource("skill://t/demo-guide/_index.md"))[0].text
    assert "kb+skill://t/demo-guide/SKILL.md" in first and first == again
    assert "kb+" not in plain
    assert calls.count("skill://t/demo-guide/_index.md") == 2  # one per build


async def test_a_build_never_reads_the_disk(tmp_path):
    root = copy(tmp_path)
    mcp = served((root, None))
    shutil.rmtree(root)
    mcp.add_rewriter(
        lambda site, text: text.replace("Greet {{", "Hail {{").replace(
            "Read skill", "See skill"
        )
    )
    assert await listed(mcp) == (["greet"], ["helper"], [ENTRY])  # re-parsed in memory
    async with Client(mcp.server) as client:
        rendered = await client.get_prompt("greet", {"who": "x"})
    assert rendered.messages[0].content.text.startswith("Hail x")


async def test_no_rewriter_serves_the_objects_as_read():
    mcp = served((DEMO, None))
    # Internal on purpose: no copies is selenium-flow's byte parity.
    read = mcp._plugins.current[0]
    built = (await mcp._plugins.pinned()).build
    assert built.catalogue["t/demo-guide"] is read.catalogue["t/demo-guide"]
    assert built.prompts[0] is read.prompts[0].item
    assert built.agents[0] is read.agents[0].item
    assert read.prompts[0].data == (DEMO / "commands/greet.md").read_bytes()


async def test_a_rewriter_added_later_rebuilds():  # Review Focus 2
    mcp = served((DEMO, None))
    assert (await listed(mcp))[0] == ["greet"]
    mcp.add_rewriter(lambda site, text: None if site.kind == "command" else text)
    assert (await listed(mcp))[0] == []


async def test_a_build_begun_before_a_change_is_never_stored(
    monkeypatch,
):  # Review Focus 2
    mcp = served((DEMO, None))
    plain, changed = type(mcp._plugins)._make, []

    async def changing(self, state, view, key):
        build = await plain(self, state, view, key)
        if not changed:  # a rewriter arrives mid-build
            changed.append(1)
            mcp.add_rewriter(
                lambda site, text: None if site.kind == "command" else text
            )
        return build

    monkeypatch.setattr(type(mcp._plugins), "_make", changing)
    assert (await listed(mcp))[0] == ["greet"]  # its own request gets it
    assert (await listed(mcp))[0] == []  # nobody after does


async def test_the_chain_has_an_end_that_runs_last(monkeypatch):  # #28's seam (GW-5)
    mcp = served((DEMO, None))

    def last(site, text):
        return text.replace("skill://", "gw+skill://")

    plain = type(mcp._plugins)._chain
    monkeypatch.setattr(
        type(mcp._plugins), "_chain", lambda self, key: (*plain(self, key), last)
    )
    async with Client(mcp.server) as client:  # no server rewriter: the end alone
        index = (await client.read_resource("skill://t/demo-guide/_index.md"))[0].text
    assert "gw+skill://t/demo-guide/SKILL.md" in index
    mcp.add_rewriter(lambda site, text: text.replace("gw+", "early+"))  # runs first
    async with Client(mcp.server) as client:
        entry = (await client.read_resource(ENTRY))[0].text
    assert (
        "gw+skill://demo-guide/references/GUIDE.md" in entry and "early+" not in entry
    )
