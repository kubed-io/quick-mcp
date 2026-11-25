"""The nesting guard: a plugin's namespace never sits inside another plugin's skill
(PLUG-76)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from plugin_server import command_md, listed, served, skill, write

from kubed.mcp.plugins import PluginError

pytestmark = pytest.mark.unit

ALPHA = {"skills/tidy/SKILL.md": skill("tidy")}
BETA = {"skills/x/SKILL.md": skill("x"), "commands/go.md": command_md()}
GAMMA = {"commands/go.md": command_md()}


def plugin(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    """A plugin folder named ``name`` in its manifest, holding ``files``."""
    manifest = {".claude-plugin/plugin.json": json.dumps({"name": name})}
    return write(tmp_path / name, {**manifest, **files})


@pytest.fixture
def alpha(tmp_path):
    return plugin(tmp_path, "alpha", ALPHA)


@pytest.fixture
def beta(tmp_path):
    return plugin(tmp_path, "beta", BETA)


@pytest.fixture
def gamma(tmp_path):
    return plugin(tmp_path, "gamma", GAMMA)


@pytest.mark.parametrize(
    "inner", ["superpowers/tidy", "superpowers/tidy/", "superpowers/tidy/more"]
)
def test_a_namespace_inside_a_skill_is_refused(alpha, beta, inner):  # Review Focus 5
    mcp = served((alpha, "superpowers"))
    ns = inner.strip("/")
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {beta} is not served: namespace {ns} is inside skill "
            "skill://superpowers/tidy of plugin alpha"
        ),
    ):
        mcp.add_plugin(beta, namespace=inner)
    assert [p.name for p in mcp.plugins] == ["alpha"]
    assert mcp.generation == 1


def test_a_skill_holding_a_namespace_is_refused(alpha, beta):
    mcp = served((beta, "superpowers/tidy"))
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {alpha} is not served: skill skill://superpowers/tidy holds "
            "namespace superpowers/tidy of plugin beta"
        ),
    ):
        mcp.add_plugin(alpha, namespace="superpowers")
    assert [p.name for p in mcp.plugins] == ["beta"]
    assert mcp.generation == 1


def test_a_plugin_with_no_skill_is_refused_too(alpha, gamma):
    mcp = served((alpha, "superpowers"))
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {gamma} is not served: namespace superpowers/tidy is inside "
            "skill skill://superpowers/tidy of plugin alpha"
        ),
    ):
        mcp.add_plugin(gamma, namespace="superpowers/tidy")
    assert [p.name for p in mcp.plugins] == ["alpha"]


def test_the_servers_own_namespace_is_checked(alpha, beta):  # Review Focus 5
    mcp = served((alpha, None), name="t")  # skill://t/tidy
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {beta} is not served: namespace t/tidy is inside skill "
            "skill://t/tidy of plugin alpha"
        ),
    ):
        mcp.add_plugin(beta, namespace="t/tidy")
    assert [p.name for p in mcp.plugins] == ["alpha"]


def test_the_servers_own_namespace_is_checked_inside(alpha, beta):  # Review Focus 5
    mcp = served((beta, "t/tidy"), name="t")
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {alpha} is not served: skill skill://t/tidy holds namespace "
            "t/tidy of plugin beta"
        ),
    ):
        mcp.add_plugin(alpha)
    assert [p.name for p in mcp.plugins] == ["beta"]


@pytest.mark.parametrize(
    "other",
    ["superpowers/tidying", "superpowers/tid", "superpowers", "elsewhere/tidy"],
)
async def test_only_whole_segments_clash(alpha, beta, other):
    mcp = served((alpha, "superpowers"), (beta, other))
    assert [p.name for p in mcp.plugins] == ["alpha", "beta"]
    assert f"skill://{other}/x/SKILL.md" in (await listed(mcp))[2]


def test_keep_checks_again(alpha, beta):
    mcp = served((beta, "superpowers/tidy"))
    staging = mcp.plugin_set()
    staging.add(alpha, namespace="superpowers")
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {beta} is not served: namespace superpowers/tidy is inside "
            "skill skill://superpowers/tidy of plugin alpha"
        ),
    ):
        staging.keep(mcp.plugins[0])
    assert [p.name for p in staging.plugins] == ["alpha"]


def test_a_refused_plugin_claims_nothing(alpha, beta, gamma):  # Review Focus 4
    mcp = served()
    staging = mcp.plugin_set()
    staging.add(alpha, namespace="superpowers")
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {beta} is not served: namespace superpowers/tidy is inside "
            "skill skill://superpowers/tidy of plugin alpha"
        ),
    ):
        staging.add(beta, namespace="superpowers/tidy")
    # beta would have claimed superpowers_go; refused, it claims nothing.
    entered = staging.add(gamma, namespace="superpowers/other")
    assert entered.prompts == ("superpowers_go",)
    assert [p.name for p in staging.plugins] == ["alpha", "gamma"]


async def test_a_set_without_the_outer_plugin_serves_the_inner(alpha, beta):
    mcp = served((alpha, "superpowers"))
    staging = mcp.plugin_set()
    staging.add(beta, namespace="superpowers/tidy")
    mcp.serve(staging)
    assert [p.name for p in mcp.plugins] == ["beta"]
    assert (await listed(mcp))[2] == ["skill://superpowers/tidy/x/SKILL.md"]


def test_the_nested_skill_of_before_is_refused_now(tmp_path):
    outer = write(tmp_path / "outer", {"skills/a/SKILL.md": skill("a")})
    inner = write(tmp_path / "inner", {"skills/x/SKILL.md": skill("x")})
    with pytest.raises(
        PluginError,
        match="^"
        + re.escape(
            f"plugin {inner} is not served: namespace lib/a is inside skill "
            "skill://lib/a of plugin outer"
        ),
    ):
        served((outer, "lib"), (inner, "lib/a"))
