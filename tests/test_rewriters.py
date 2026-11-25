"""Rewriters: the text a server serves, rewritten when a build is made (PLUG-56 to
PLUG-59)."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastmcp import Client
from plugin_server import DEMO, ENTRY, GUIDE, copy, listed, served, skill, write
from skill_server import read_bytes, request

from kubed.mcp import skills
from kubed.mcp.plugins import Components, Plugin, Site, View, to_mcp_form
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


async def test_binary_and_non_utf8_files_pass_untouched_and_json_is_text(tmp_path):
    png = b"\x89PNG\r\n${CLAUDE_PLUGIN_ROOT}"
    latin = "café ${CLAUDE_PLUGIN_ROOT}".encode("latin-1")
    data = b'{"root": "${CLAUDE_PLUGIN_ROOT}"}'  # UTF-8 and text-like (PLUG-75)
    root = copy(tmp_path)
    (root / "skills/demo-guide/logo.png").write_bytes(png)
    (root / "skills/demo-guide/notes.txt").write_bytes(latin)
    (root / "skills/demo-guide/data.json").write_bytes(data)
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
        config = (await client.read_resource("skill://t/demo-guide/data.json"))[0]
    assert "skills/demo-guide/SKILL.md" in seen  # the rewriter ran
    assert "skills/demo-guide/data.json" in seen
    untouched = {"logo.png", "notes.txt"}
    assert not {f"skills/demo-guide/{name}" for name in untouched} & set(seen)
    assert read_bytes(logo) == png and read_bytes(notes) == latin
    assert config.text == '{"root": "X"}'


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


KIT_TREE = {
    "skills/alpha/SKILL.md": skill("alpha"),
    "skills/alpha/references/a.md": "# A\n",
    "skills/alpha/references/deep/b.md": "# B\n",
    "skills/alpha/scripts/run.sh": "echo hi\n",
    "skills/alpha/refs/_index.md/kept.md": "# Kept\n",  # a folder at an index's URI
    "skills/beta/SKILL.md": skill("beta"),
    "skills/beta/docs/guide.md": "# Guide\n",
    "skills/beta/docs/more/c.md": "# C\n",
}
KIT_FOLDERS = [
    "skill://t/alpha",
    "skill://t/alpha/references",
    "skill://t/alpha/references/deep",
    "skill://t/alpha/refs",
    "skill://t/alpha/scripts",
    "skill://t/beta",
    "skill://t/beta/docs",
    "skill://t/beta/docs/more",
    "skill://lib/gamma",
]


async def test_a_rewritten_build_indexes_as_a_plain_server_does(tmp_path, caplog):
    kit = write(tmp_path / "kit", KIT_TREE)
    solo = write(tmp_path / "solo", {"skills/gamma/SKILL.md": skill("gamma")})
    plain = served((kit, None), (solo, "lib"))
    rewritten = served((kit, None), (solo, "lib"))
    rewritten.add_rewriter(lambda site, text: text)  # a chain, but no change
    caplog.clear()
    reads = {}
    for mcp in (plain, rewritten):
        async with Client(mcp.server) as client:
            for folder in KIT_FOLDERS:
                for name in ("_index.md", "_index.json"):
                    read = (await client.read_resource(f"{folder}/{name}"))[0]
                    reads.setdefault(f"{folder}/{name}", []).append(
                        (read.mime_type, read.text)
                    )
    assert all(first == second for first, second in reads.values())
    scripts = reads["skill://t/alpha/scripts/_index.md"]
    assert "skill://t/alpha/scripts/run.sh" in scripts[0][1]
    # The skill was trimmed, and warned of, when it was read: no build warns again.
    assert not [r for r in caplog.records if "refs/_index.md" in r.getMessage()]


async def test_a_build_lists_each_folder_from_its_own_skill_alone(
    tmp_path, monkeypatch
):
    plain, counts = skills._without_unserved, []
    for n in (4, 8):
        tree = {f"skills/s{i}/SKILL.md": skill(f"s{i}") for i in range(n)}
        tree |= {f"skills/s{i}/references/a.md": "# A\n" for i in range(n)}
        mcp = served((write(tmp_path / f"p{n}", tree), None))
        mcp.add_rewriter(lambda site, text: text)
        calls = []

        def counted(manifest, calls=calls):
            calls.append(manifest)
            return plain(manifest)

        monkeypatch.setattr(skills, "_without_unserved", counted)
        await mcp._plugins.pinned()  # one build: each manifest a listing loads
        monkeypatch.setattr(skills, "_without_unserved", plain)
        counts.append(len(calls))
    assert counts[0] <= 3 * 4  # a listing per folder (two a skill), a load a skill
    assert counts[1] <= 2 * counts[0]  # linear in the skills, not quadratic


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
    begun = mcp._plugins._memo  # the memo the first build begins against
    assert (await listed(mcp))[0] == ["greet"]  # its own request gets it
    assert not begun.builds  # not stored, not even in the memo it began against
    assert (await listed(mcp))[0] == []  # nobody after gets it


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


KIT = Plugin(
    name="kit",
    description="A kit.",
    version="1",
    root=Path("/kit"),
    components=Components(),
    namespace="kit",
    skills=("skill://kit/writing/SKILL.md", "skill://kit/writing-tools/SKILL.md"),
)
FOLDERS = {
    "skills/writing": "skill://kit/writing",
    "skills/writing-tools": "skill://kit/writing-tools",
}


def site(kind="skill", path="skills/writing/SKILL.md", admitted=(KIT,), skills=FOLDERS):
    return Site(
        KIT, "kit", kind, path, "skill://kit/writing/SKILL.md", None, admitted, skills
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "${CLAUDE_PLUGIN_ROOT}/skills/writing/refs/a.md",
            "skill://kit/writing/refs/a.md",
        ),
        (
            "${CLAUDE_PLUGIN_ROOT}/skills/writing-tools/x.md",
            "skill://kit/writing-tools/x.md",
        ),
        ("${CLAUDE_PLUGIN_ROOT}/skills/writing", "skill://kit/writing"),
        ("${CLAUDE_SKILL_DIR}/refs/a.md", "skill://kit/writing/refs/a.md"),
        ("Use kit:writing.", "Use skill://kit/writing/SKILL.md."),
        ("(kit:writing)", "(skill://kit/writing/SKILL.md)"),
        ("`kit:writing-tools`", "`skill://kit/writing-tools/SKILL.md`"),
    ],
)
def test_local_forms_become_addresses(text, expected):
    assert to_mcp_form(site(), text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "${CLAUDE_PLUGIN_ROOT}",
        "${CLAUDE_PLUGIN_ROOT}/skills/writing.md",
        "${CLAUDE_PLUGIN_ROOT}/commands/x.md",
        "/kit:writing",
        "https://kit:writing",
        "a/kit:writing",
        "kit:writing.md",
        "kit:writing/x",
        "kit:writing:x",
        "${input:topic}",
        "#tool:x",
        "kit:missing",
        "other:writing",
    ],
)
def test_what_names_nothing_served_stays_as_written(text):
    assert to_mcp_form(site(), text) == text


def test_the_skill_dir_is_only_a_skills():
    command = site(kind="command", path="commands/x.md")
    assert to_mcp_form(command, "${CLAUDE_SKILL_DIR}/a") == "${CLAUDE_SKILL_DIR}/a"


def test_a_name_resolves_only_among_the_admitted_the_first_winning():
    other = replace(KIT, namespace="kit2", skills=("skill://kit2/writing/SKILL.md",))
    assert to_mcp_form(site(admitted=()), "kit:writing") == "kit:writing"
    both = site(admitted=(other, KIT))
    assert to_mcp_form(both, "kit:writing") == "skill://kit2/writing/SKILL.md"


def test_a_skill_at_the_root_takes_every_path():
    solo = site(path="SKILL.md", skills={"": "skill://kit/solo"})
    assert (
        to_mcp_form(solo, "${CLAUDE_PLUGIN_ROOT}/refs/a.md")
        == "skill://kit/solo/refs/a.md"
    )


# A skill at the root, one below it and one inside that, the shallowest first.
NESTED = {
    "": "skill://kit/solo",
    "skills/a": "skill://kit/a",
    "skills/a/b": "skill://kit/b",
}


@pytest.mark.parametrize(
    ("path", "text", "expected"),
    [
        ("SKILL.md", "${CLAUDE_PLUGIN_ROOT}/skills/a/b/x.md", "skill://kit/b/x.md"),
        ("SKILL.md", "${CLAUDE_PLUGIN_ROOT}/skills/a/x.md", "skill://kit/a/x.md"),
        (
            "SKILL.md",
            "${CLAUDE_PLUGIN_ROOT}/skills/x.md",
            "skill://kit/solo/skills/x.md",
        ),
        ("skills/a/b/SKILL.md", "${CLAUDE_SKILL_DIR}/x.md", "skill://kit/b/x.md"),
        ("skills/a/b/refs/y.md", "${CLAUDE_SKILL_DIR}/x.md", "skill://kit/b/x.md"),
        ("skills/a/SKILL.md", "${CLAUDE_SKILL_DIR}/x.md", "skill://kit/a/x.md"),
        ("skills/ab/SKILL.md", "${CLAUDE_SKILL_DIR}/x.md", "skill://kit/solo/x.md"),
        ("SKILL.md", "${CLAUDE_SKILL_DIR}/x.md", "skill://kit/solo/x.md"),
    ],
)
def test_the_deepest_skill_folder_wins(path, text, expected):
    assert to_mcp_form(site(path=path, skills=NESTED), text) == expected


async def test_a_server_serves_its_plugin_in_mcp_form(tmp_path):
    root = copy(
        tmp_path,
        files={
            "commands/read.md": "---\ndescription: R.\n---\n"
            "Read ${CLAUDE_PLUGIN_ROOT}/skills/demo-guide/references/GUIDE.md"
            " and demo:demo-guide.\n"
        },
    )
    mcp = served((root, None))
    mcp.add_rewriter(to_mcp_form)
    async with Client(mcp.server) as client:
        rendered = await client.get_prompt("read")
    assert rendered.messages[0].content.text == f"Read {GUIDE} and {ENTRY}.\n"
