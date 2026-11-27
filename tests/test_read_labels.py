"""The layer's lookup of a read's pattern, library and plugin: what a build says of a
skill, prompt or agent file's address, from the build alone."""

from __future__ import annotations

import json

import pytest
from gateway_server import KIT_TREE
from plugin_server import agent_md, command_md, write

from kubed.mcp import QuickMCP
from kubed.mcp.plugins import Plugins, _labels, _ReadLabels

pytestmark = pytest.mark.unit

TREE = {**KIT_TREE, "skills/guide/references/b.md": "# B\n"}


async def labels(mcp, uri):
    return _labels((await mcp._plugins.pinned()).build, uri)


@pytest.mark.parametrize(
    ("uri", "pattern"),
    [
        ("skill://lib/guide/SKILL.md", "skill://{library}/{path}/SKILL.md"),
        ("skill://lib/guide/_manifest", "skill://{library}/{path}/_manifest"),
        ("skill://lib/guide/_index.md", "skill://{library}/{path}/_index.md"),
        (
            "skill://lib/guide/references/_index.json",
            "skill://{library}/{path}/_index.json",
        ),
        ("skill://lib/guide/references/a.md", "skill://{library}/{path}"),
        ("skill://lib/guide/references/missing.md", "skill://{library}/{path}"),
        ("skill://lib/guide/references/SKILL.md", "skill://{library}/{path}"),
        ("prompts://lib/ask.md", "prompts://{library}/{path}"),
        ("agents://lib/scout.md", "agents://{library}/{path}"),
        ("prompts://lib/_index.md", "prompts://{library}/{path}/_index.md"),
        ("agents://lib/_index.json", "agents://{library}/{path}/_index.json"),
    ],
)
async def test_each_kind_of_read_has_its_pattern(tmp_path, uri, pattern):
    mcp = QuickMCP("labels", config={})
    mcp.add_plugin(write(tmp_path / "kit", TREE), namespace="lib")
    assert await labels(mcp, uri) == _ReadLabels(pattern, "lib", "kit")


async def test_two_files_of_one_skill_label_alike(tmp_path):
    mcp = QuickMCP("labels", config={})
    mcp.add_plugin(write(tmp_path / "kit", TREE), namespace="lib")
    a = await labels(mcp, "skill://lib/guide/references/a.md")
    b = await labels(mcp, "skill://lib/guide/references/b.md")
    assert a is not None
    assert a == b


async def test_the_library_is_the_namespaces_first_segment(tmp_path):
    mcp = QuickMCP("labels", config={})
    mcp.add_plugin(write(tmp_path / "grafana-k6", TREE), namespace="grafana/grafana-k6")
    assert await labels(mcp, "skill://grafana/grafana-k6/guide/SKILL.md") == (
        _ReadLabels("skill://{library}/{path}/SKILL.md", "grafana", "grafana-k6")
    )


@pytest.mark.parametrize(
    "uri",
    [
        "skill://lib/guide",
        "skill://lib/_index.md",
        "skill://other/x/SKILL.md",
        "prompts://elsewhere/ask.md",
        "data://items/1",
        "not a uri",
    ],
)
async def test_what_the_layer_doesnt_serve_is_none(tmp_path, uri):
    mcp = QuickMCP("labels", config={})
    mcp.add_plugin(write(tmp_path / "kit", TREE), namespace="lib")
    assert await labels(mcp, uri) is None


async def test_a_shared_folder_is_its_first_plugins(tmp_path):
    mcp = QuickMCP("labels", config={})
    mcp.add_plugin(
        write(tmp_path / "first", {"commands/a.md": command_md()}), namespace="grafana"
    )
    mcp.add_plugin(
        write(tmp_path / "second", {"commands/b.md": command_md()}),
        namespace="grafana",
    )
    index = await labels(mcp, "prompts://grafana/_index.md")
    assert index == _ReadLabels(
        "prompts://{library}/{path}/_index.md", "grafana", "first"
    )
    assert await labels(mcp, "prompts://grafana/b.md") == _ReadLabels(
        "prompts://{library}/{path}", "grafana", "second"
    )


async def test_a_renamed_agent_is_its_own_plugins(tmp_path):
    mcp = QuickMCP("labels", config={})
    for name in ("one", "two"):
        mcp.add_plugin(
            write(
                tmp_path / name,
                {
                    ".claude-plugin/plugin.json": json.dumps({"name": name}),
                    "agents/scout.md": agent_md("scout"),
                },
            ),
            namespace="lib",
        )
    files = (await mcp._plugins.pinned()).build.files
    assert "agents://lib/lib_scout@two.md" in files, sorted(files)
    pattern = "agents://{library}/{path}"
    assert await labels(mcp, "agents://lib/scout.md") == _ReadLabels(
        pattern, "lib", "one"
    )
    assert await labels(mcp, "agents://lib/lib_scout@two.md") == _ReadLabels(
        pattern, "lib", "two"
    )


@pytest.mark.parametrize("rewritten", [False, True])
async def test_an_index_labels_the_same_rewritten_or_not(tmp_path, rewritten):
    mcp = QuickMCP("labels", config={})
    if rewritten:
        mcp.add_rewriter(lambda site, text: text)
    mcp.add_plugin(write(tmp_path / "kit", TREE), namespace="lib")
    assert await labels(mcp, "skill://lib/guide/_index.md") == _ReadLabels(
        "skill://{library}/{path}/_index.md", "lib", "kit"
    )
    assert await labels(mcp, "prompts://lib/_index.md") == _ReadLabels(
        "prompts://{library}/{path}/_index.md", "lib", "kit"
    )


async def test_a_skill_a_rewriter_fails_is_not_the_layers(tmp_path):
    def fail(site, text):
        if site.path == "skills/guide/SKILL.md":
            raise ValueError("boom")
        return text

    mcp = QuickMCP("labels", config={})
    mcp.add_rewriter(fail)
    mcp.add_plugin(write(tmp_path / "kit", TREE), namespace="lib")
    build = (await mcp._plugins.pinned()).build
    assert "lib/guide" not in build.catalogue
    assert await labels(mcp, "skill://lib/guide/SKILL.md") is None
    assert await labels(mcp, "prompts://lib/ask.md") == _ReadLabels(
        "prompts://{library}/{path}", "lib", "kit"
    )


async def test_a_server_without_plugins_has_no_labels_and_never_pins(monkeypatch):
    mcp = QuickMCP("bare", config={})

    async def boom(self):
        raise AssertionError("pinned")

    monkeypatch.setattr(Plugins, "pinned", boom)
    assert mcp._plugins._read_labels("skill://lib/guide/SKILL.md") is None


async def test_a_read_inside_a_pin_is_labelled_from_it_and_outside_one_is_not(tmp_path):
    from kubed.mcp.plugins import _PINS, _Slot

    mcp = QuickMCP("labels", config={})
    mcp.add_plugin(write(tmp_path / "kit", TREE), namespace="lib")
    uri = "skill://lib/guide/SKILL.md"
    assert mcp._plugins._read_labels(uri) is None
    slot = _Slot()
    token = _PINS.set({mcp._plugins: slot})
    try:
        assert mcp._plugins._read_labels(uri) is None
        slot.pin = await mcp._plugins.pinned()
        assert mcp._plugins._read_labels(uri) == _ReadLabels(
            "skill://{library}/{path}/SKILL.md", "lib", "kit"
        )
    finally:
        _PINS.reset(token)
