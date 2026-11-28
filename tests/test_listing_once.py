"""A build makes its skills' ``resources/list`` rows once, and each caller still gets its own
list (PLUG-55, DLV-8)."""

from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.skills.provider import SkillManifestProvider
from gateway_server import over_http
from plugin_server import DEMO, ENTRY, copy, listed, served, skill, write

from kubed.mcp import skills
from kubed.mcp.plugins import View
from kubed.mcp.skills import SkillFileResource

pytestmark = pytest.mark.unit


async def skill_rows(client) -> list[dict]:
    return [
        r.model_dump(by_alias=True, exclude_none=True, mode="json")
        for r in await client.list_resources()
        if str(r.uri).startswith("skill://")
    ]


async def build(mcp):
    return (await mcp._plugins.pinned()).build


@pytest.mark.parametrize("rewritten", [False, True])
async def test_listing_builds_no_row(monkeypatch, rewritten):
    made = []
    plain = skills._file_resource

    def counted(*args):
        made.append(args[3])
        return plain(*args)

    monkeypatch.setattr(skills, "_file_resource", counted)
    mcp = served((DEMO, None))
    if rewritten:
        mcp.add_rewriter(lambda site, text: text + "\n<!-- r -->\n")
    for _ in range(3):
        assert (await listed(mcp))[2] == [ENTRY]
    assert (await build(mcp)).rewritten is rewritten
    # Once, with the build, not once per listing. A rewritten build also makes each
    # file's row once, with it, for the indexes it renders (PLUG-56).
    indexed = ["SKILL.md", "references/GUIDE.md"] if rewritten else []
    assert made == [*indexed, "SKILL.md"]


@pytest.mark.parametrize("rewritten", [False, True])
async def test_the_rows_are_fastmcps(rewritten):
    mcp = served((DEMO, None))
    if rewritten:
        mcp.add_rewriter(lambda site, text: text + "\n<!-- r -->\n")
    async with Client(mcp.server) as client:
        seen = [await skill_rows(client) for _ in range(3)]
    oracle = [
        skills._wire(r)
        for r in await SkillManifestProvider._list_resources(mcp._plugins._skills)
        if isinstance(r, SkillFileResource) and r.file_path == skills.ENTRY
    ]
    assert oracle
    assert seen == [oracle] * 3
    assert [skills._wire(r) for r in (await build(mcp)).entries] == oracle  # untouched


async def test_a_new_set_lists_its_own_descriptions(tmp_path):
    root = copy(tmp_path)
    mcp = served((root, None))
    before = "How to use the demo plugin. Use when a test needs a skill."
    async with Client(mcp.server) as client:
        assert [r["description"] for r in await skill_rows(client)] == [before]
    write(
        root,
        {
            "skills/demo-guide/SKILL.md": "---\nname: demo-guide\ndescription: Changed.\n---\n"
        },
    )
    staging = mcp.plugin_set()
    staging.add(root)
    mcp.serve(staging)
    async with Client(mcp.server) as client:
        assert [r["description"] for r in await skill_rows(client)] == ["Changed."]
    mcp.serve(mcp.plugin_set())  # a set with no skills
    async with Client(mcp.server) as client:
        assert await skill_rows(client) == []


async def test_a_rewriter_added_later_changes_the_rows():
    mcp = served((DEMO, None))
    assert (await listed(mcp))[2] == [ENTRY]  # a build is made and kept
    mcp.add_rewriter(lambda site, text: text.replace("How to use", "How to run"))
    async with Client(mcp.server) as client:
        rows = await skill_rows(client)
    assert rows[0]["description"].startswith("How to run the demo plugin")


async def test_listed_kinds_are_per_caller_on_one_build():
    mcp = served((DEMO, None))
    async with (
        over_http(mcp) as plain,
        over_http(mcp, headers={"X-MCP-Listed": ""}) as none,
    ):
        assert [r["uri"] for r in await skill_rows(plain)] == [ENTRY]
        assert await skill_rows(none) == []
        assert [r["uri"] for r in await skill_rows(plain)] == [ENTRY]
    assert len(mcp._plugins._memo.builds) == 1


async def test_each_prefix_lists_its_own_rows(tmp_path):
    about = "Start at skill://t/demo-guide/references/GUIDE.md."
    root = copy(
        tmp_path,
        files={
            "skills/demo-guide/SKILL.md": f"---\nname: demo-guide\ndescription: {about}\n---\n"
        },
    )
    mcp = served((root, None))
    async with over_http(mcp) as plain, over_http(mcp, header="kb+") as kb:
        assert [r["description"] for r in await skill_rows(plain)] == [about]
        assert [r["description"] for r in await skill_rows(kb)] == [
            about.replace("skill://", "kb+skill://")
        ]
        assert [r["description"] for r in await skill_rows(plain)] == [about]
    assert len(mcp._plugins._memo.builds) == 2


async def test_each_view_lists_its_own_skills(tmp_path):
    roots = {
        name: write(
            tmp_path / name, {f"skills/{name}-guide/SKILL.md": skill(f"{name}-guide")}
        )
        for name in ("a", "b")
    }
    mcp = served((roots["a"], "a"), (roots["b"], "b"))
    current = ["a"]
    mcp.plugin_view(lambda: View(current[0], lambda p: p.namespace == current[0]))
    assert (await listed(mcp))[2] == ["skill://a/a-guide/SKILL.md"]
    current[0] = "b"
    assert (await listed(mcp))[2] == ["skill://b/b-guide/SKILL.md"]
    assert len(mcp._plugins._memo.builds) == 2
