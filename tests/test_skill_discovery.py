"""Which folders are a plugin's skills: Claude Code's search, and nothing said of the rest (PLUG-19)."""

from __future__ import annotations

import logging

import pytest
from fastmcp import Client
from plugin_server import served, skill, write
from skill_server import read_json, request

from kubed.mcp import QuickMCP

pytestmark = pytest.mark.unit

# Claude Code 2.1.325's --plugin-dir probes (#75, 2026-11-27).
PROBE = {
    "skills/toplevel/SKILL.md": skill("toplevel"),
    "skills/toplevel/inner/SKILL.md": skill("inner"),
    "skills/category/deep/SKILL.md": skill("deep"),
}
TOPLEVEL = "skill://t/toplevel/SKILL.md"
DEEP = "skill://t/deep/SKILL.md"

# grafana/skills' shape: one root, source ./, a shared skills/<plugin>/<skill>.
GRAFANA = {
    "skills/grafana-k6/k6/SKILL.md": skill("k6"),
    "skills/grafana-k6/k6/inner/SKILL.md": skill("inner"),
    "skills/grafana-k6/k6-perf-test-website/SKILL.md": skill("k6-perf-test-website"),
    "skills/grafana-k6/k6-unlisted/SKILL.md": skill("k6-unlisted"),
    "skills/grafana-cloud/cloud-alerts/SKILL.md": skill("cloud-alerts"),
    "skills/grafana-cloud/cloud-logs/SKILL.md": skill("cloud-logs"),
}
ENTRIES = {
    "grafana/k6": {
        "name": "grafana-k6",
        "source": "./",
        "skills": [
            "./skills/grafana-k6/k6",
            "./skills/grafana-k6/k6-perf-test-website",
        ],
    },
    "grafana/cloud": {
        "name": "grafana-cloud",
        "source": "./",
        "skills": [
            "./skills/grafana-cloud/cloud-alerts",
            "./skills/grafana-cloud/cloud-logs",
        ],
    },
}
SERVED = {
    "grafana-k6": (
        "skill://grafana/k6/k6/SKILL.md",
        "skill://grafana/k6/k6-perf-test-website/SKILL.md",
    ),
    "grafana-cloud": (
        "skill://grafana/cloud/cloud-alerts/SKILL.md",
        "skill://grafana/cloud/cloud-logs/SKILL.md",
    ),
}
EVERY = sorted(uri for uris in SERVED.values() for uri in uris)


def grafana(tmp_path) -> QuickMCP:
    """A server with grafana/skills' one root added once per entry."""
    root = write(tmp_path / "grafana-skills", GRAFANA)
    mcp = QuickMCP("kb")
    for ns, entry in ENTRIES.items():
        mcp.add_plugin(root, namespace=ns, manifest=entry)
    return mcp


async def test_the_skills_listing_is_each_plugins_declared_skills(tmp_path):
    mcp = grafana(tmp_path)
    assert {p.name: p.skills for p in mcp.plugins} == SERVED
    async with Client(mcp.server) as client:
        listing = (await request(client, "skills/list"))["skills"]
        index = (await client.read_resource("skill://grafana/k6/k6/_index.md"))[0].text
    assert sorted(entry["uri"] for entry in listing) == EVERY
    assert [line.split(": ", 1)[0] for line in index.splitlines()] == [
        "skill://grafana/k6/k6/SKILL.md",
        "skill://grafana/k6/k6/inner",
    ]


async def test_resources_list_is_one_skill_md_row_per_skill(tmp_path):
    mcp = grafana(tmp_path)
    async with Client(mcp.server) as client:
        rows = sorted(
            str(r.uri)
            for r in await client.list_resources()
            if str(r.uri).startswith("skill://")
        )
        inner = (await client.read_resource("skill://grafana/k6/k6/inner/SKILL.md"))[
            0
        ].text
    assert rows == EVERY
    assert inner == skill("inner")


async def test_a_skill_md_inside_a_skill_is_one_of_its_files(tmp_path):
    mcp = served((write(tmp_path / "probe", PROBE), None))
    base = "skill://t/toplevel"
    async with Client(mcp.server) as client:
        rows = [str(r.uri) for r in await client.list_resources()]
        listing = (await request(client, "skills/list"))["skills"]
        manifest = await read_json(client, f"{base}/_manifest")
        inner = (await client.read_resource(f"{base}/inner/SKILL.md"))[0].text
    assert [u for u in rows if u.startswith("skill://")] == [TOPLEVEL]
    assert [entry["uri"] for entry in listing] == [TOPLEVEL]
    assert {f["path"] for f in manifest["files"]} == {"SKILL.md", "inner/SKILL.md"}
    assert inner == skill("inner")


def test_a_symlinked_or_dot_folder_is_not_searched(tmp_path, caplog):
    root = write(
        tmp_path / "p",
        {
            "skills/top/SKILL.md": skill("top"),
            "skills/.drafts/SKILL.md": skill("drafts"),
            "elsewhere/linked/SKILL.md": skill("linked"),
        },
    )
    (root / "skills/linked").symlink_to(
        root / "elsewhere/linked", target_is_directory=True
    )
    with caplog.at_level(logging.WARNING):
        mcp = served((root, None))
    assert mcp.plugins[0].skills == ("skill://t/top/SKILL.md",)
    assert not caplog.records


CLAUDE_CODE = [
    (None, (TOPLEVEL,)),
    (["./skills/category"], (TOPLEVEL, DEEP)),
    (["./skills/category/deep"], (TOPLEVEL, DEEP)),
]


@pytest.mark.parametrize(("declared", "found"), CLAUDE_CODE)
def test_skills_are_found_as_claude_code_finds_them(tmp_path, caplog, declared, found):
    root = write(tmp_path / "probe", PROBE)
    entry = (
        {"name": "probe"} if declared is None else {"name": "probe", "skills": declared}
    )
    with caplog.at_level(logging.DEBUG):
        plugin = QuickMCP("t").add_plugin(root, manifest=entry)
    assert plugin.skills == found
    assert plugin.components.notes == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    if declared is None:
        assert "category" not in caplog.text


def test_a_shared_skills_folder_is_passed_over_silently(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        mcp = grafana(tmp_path)
    assert [p.components.notes for p in mcp.plugins] == [(), ()]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "k6-unlisted" not in caplog.text


def test_a_declared_root_is_searched_one_level(tmp_path, caplog):
    root = write(
        tmp_path / "p",
        {
            "skills/x/SKILL.md": skill("x"),
            "z/SKILL.md": skill("z"),
            "tools/y/SKILL.md": skill("y"),
        },
    )
    with caplog.at_level(logging.WARNING):
        plugin = QuickMCP("t").add_plugin(root, manifest={"name": "p", "skills": "./"})
    assert plugin.skills == ("skill://t/x/SKILL.md", "skill://t/z/SKILL.md")
    assert plugin.components.notes == ()
    assert not caplog.records


def test_a_declared_skills_path_that_is_no_folder_is_still_noted(tmp_path, caplog):
    root = write(tmp_path / "p", {"skills/top/SKILL.md": skill("top")})
    entry = {"name": "p", "skills": ["./missing", "./skills/top/SKILL.md"]}
    with caplog.at_level(logging.WARNING):
        plugin = QuickMCP("t").add_plugin(root, manifest=entry)
    notes = (
        "skills entry 'missing' is not a folder in the plugin",
        "skills entry 'skills/top/SKILL.md' is not a folder in the plugin",
    )
    assert plugin.components.notes == notes
    assert plugin.skills == ("skill://t/top/SKILL.md",)
    for note in notes:
        assert f"plugin p: {note}" in caplog.text
