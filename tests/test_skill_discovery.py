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
