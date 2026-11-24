"""Readers and names: how a plugin's prompts and agents are named (PLUG-63)."""

from __future__ import annotations

import pytest
from plugin_server import DEMO, command_md, listed, served, write

from kubed.mcp.plugins import PluginError

pytestmark = pytest.mark.unit


async def test_the_prefix_is_the_namespaces_first_segment():
    mcp = served((DEMO, "kit/tools"))
    assert await listed(mcp) == (
        ["kit_greet"],
        ["kit_helper"],
        ["skill://kit/tools/demo-guide/SKILL.md"],
    )


def test_plugins_under_one_first_segment_clash_on_a_prompt(tmp_path):
    loki = write(tmp_path / "loki", {"commands/greet.md": command_md()})
    k6 = write(tmp_path / "k6", {"commands/greet.md": command_md()})
    mcp = served((loki, "grafana/loki"))
    with pytest.raises(
        PluginError, match="prompt grafana_greet is already served by plugin loki"
    ):
        mcp.add_plugin(k6, namespace="grafana/k6")


async def test_a_namespace_below_the_servers_name_is_prefixed():
    assert (await listed(served((DEMO, "kb/x"), name="kb")))[0] == ["kb_greet"]
