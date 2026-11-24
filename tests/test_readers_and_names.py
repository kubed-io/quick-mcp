"""Readers and names: how a plugin's commands are read (PLUG-62) and its prompts and
agents named (PLUG-63)."""

from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.prompts import Message, Prompt, PromptResult
from gateway_server import KIT_TREE, over_http
from mcp_types import ImageContent
from plugin_server import DEMO, ENTRY, command_md, listed, served, write

from kubed.mcp import QuickMCP
from kubed.mcp.plugins import PluginError
from kubed.mcp.prompts import read_command

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


# ---- command_reader (PLUG-62) ------------------------------------------------------


class Echo(Prompt):
    """A server's own prompt: it renders a fixed text, whatever its file says."""

    async def render(self, arguments=None):
        return "see skill://lib/guide/SKILL.md"


class EchoList(Prompt):
    """A server's own prompt that renders strings and messages, not one string."""

    async def render(self, arguments=None):
        return [
            "see skill://lib/guide/SKILL.md",
            Message("see skill://lib/guide/SKILL.md", role="assistant"),
        ]


class EchoResult(Prompt):
    """A server's own prompt that renders a ``PromptResult``."""

    async def render(self, arguments=None):
        return PromptResult([Message("see skill://lib/guide/SKILL.md")])


async def test_the_reader_is_given_text_path_name_and_plugin():
    calls = []
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        calls.append((text, path, name, plugin.name, plugin.namespace))
        return read_command(text, path, name, plugin)

    mcp.add_plugin(DEMO, namespace="kit")
    assert calls == [
        (
            (DEMO / "commands/greet.md").read_text(),
            "commands/greet.md",
            "kit_greet",
            "demo",
            "kit",
        )
    ]


def test_command_reader_returns_fn():
    mcp = QuickMCP("t")
    assert mcp.command_reader(read_command) is read_command


async def test_a_readers_reason_skips_the_command(tmp_path, caplog):
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        if path.endswith(".prompt.md"):
            return "not my dialect"
        return read_command(text, path, name, plugin)

    root = write(
        tmp_path / "p",
        {"commands/a.md": command_md(), "commands/b.prompt.md": command_md()},
    )
    plugin = mcp.add_plugin(root)
    assert plugin.prompts == ("a",)
    assert ("commands/b.prompt.md", "not my dialect") in plugin.skipped
    assert "is skipped: not my dialect" in caplog.text


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (ValueError("bad"), "bad"),
        (ValueError(), "ValueError"),
        (None, "command_reader returned NoneType, not a Prompt or str"),
    ],
)
def test_a_reader_that_raises_or_returns_neither_skips(result, reason):
    # Review Focus 1: the command is skipped, the rest of the plugin served.
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        if isinstance(result, Exception):
            raise result
        return result

    plugin = mcp.add_plugin(DEMO)
    assert plugin.prompts == () and plugin.agents == ("helper",)
    assert ("commands/greet.md", reason) in plugin.skipped


async def test_a_readers_prompt_is_served_under_the_layers_name():
    mcp = QuickMCP("t")
    mcp.command_reader(lambda text, path, name, plugin: Echo(name="other"))
    mcp.add_plugin(DEMO)
    async with Client(mcp.server) as client:
        assert [p.name for p in await client.list_prompts()] == ["greet"]
        said = (await client.get_prompt("greet", {})).messages[0].content.text
    assert said == "see skill://lib/guide/SKILL.md"


async def test_a_build_reads_the_rewritten_text():
    texts = []
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        texts.append(text)
        return read_command(text, path, name, plugin)

    mcp.add_plugin(DEMO)
    mcp.add_rewriter(lambda site, text: text.replace("Greet {{", "Hail {{"))
    await listed(mcp)
    async with Client(mcp.server) as client:
        rendered = (await client.get_prompt("greet", {"who": "x"})).messages[0]
    assert any(text.startswith("---") and "Hail {{" in text for text in texts)
    assert rendered.content.text.startswith("Hail x")


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        ("no", "no"),
        (ValueError("bad"), "bad"),
        (None, "command_reader returned NoneType, not a Prompt or str"),
    ],
)
async def test_a_failed_build_read_fails_only_that_prompt(caplog, result, reason):
    # Review Focus 1: a reason, a raise or a None in a build fails the prompt for that
    # build, never the request, and the rest is served.
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        if "Hail" not in text:
            return read_command(text, path, name, plugin)
        if isinstance(result, Exception):
            raise result
        return result

    plugin = mcp.add_plugin(DEMO)
    assert plugin.prompts == ("greet",)
    mcp.add_rewriter(lambda site, text: text.replace("Greet {{", "Hail {{"))
    assert await listed(mcp) == ([], ["helper"], [ENTRY])
    assert "prompt greet is not served for view" in caplog.text
    assert caplog.text.rstrip().endswith(reason)


async def test_a_build_given_the_plugin_as_served():
    seen = []
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        seen.append(plugin)
        return read_command(text, path, name, plugin)

    plugin = mcp.add_plugin(DEMO)
    mcp.add_rewriter(lambda site, text: text.replace("Greet {{", "Hail {{"))
    await listed(mcp)
    assert seen[-1] is plugin


async def test_no_rewriter_reuses_the_first_prompt():
    count = 0
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        nonlocal count
        count += 1
        return read_command(text, path, name, plugin)

    mcp.add_plugin(DEMO)
    await listed(mcp)
    await listed(mcp)
    assert count == 1
    assert (await mcp._plugins.pinned()).build.prompts[0] is mcp._plugins.current[
        0
    ].prompts[0].item


async def test_a_plugin_keeps_the_reader_it_was_read_with():
    mcp = QuickMCP("t")
    mcp.command_reader(lambda text, path, name, plugin: Echo(name=name))
    mcp.add_plugin(DEMO)
    mcp.command_reader(read_command)
    mcp.add_rewriter(lambda site, text: text.replace("Greet {{", "Hail {{"))
    async with Client(mcp.server) as client:
        said = (await client.get_prompt("greet", {})).messages[0].content.text
    assert said == "see skill://lib/guide/SKILL.md"


@pytest.mark.parametrize("kind", [Echo, EchoList, EchoResult])
async def test_a_readers_prompt_is_prefixed_when_rendered(tmp_path, kind):
    # Review Focus 5: every text a custom prompt renders is prefixed.
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        if path.endswith("ask.md"):
            return kind(name="other", description="Echo.")
        return read_command(text, path, name, plugin)

    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with over_http(mcp, header="kb+") as client:
        messages = (await client.get_prompt("lib_ask", {})).messages
    said = [message.content.text for message in messages]
    assert said and all(text == "see kb+skill://lib/guide/SKILL.md" for text in said)
    if kind is EchoList:
        assert [message.role for message in messages] == ["user", "assistant"]


async def test_a_prefixed_render_passes_other_content_as_it_is(tmp_path):
    image = ImageContent(type="image", data="AAAA", mimeType="image/png")

    class Pictured(Prompt):
        async def render(self, arguments=None):
            return [Message(image), "see skill://lib/guide/SKILL.md"]

    mcp = QuickMCP("t")
    mcp.command_reader(lambda text, path, name, plugin: Pictured(name=name))
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with over_http(mcp, header="kb+") as client:
        messages = (await client.get_prompt("lib_ask", {})).messages
    assert messages[0].content.data == "AAAA"
    assert messages[1].content.text == "see kb+skill://lib/guide/SKILL.md"


class Answered(Prompt):
    """A server's own prompt whose ``convert_result`` serves its text as the
    assistant's."""

    async def render(self, arguments=None):
        return "see skill://lib/guide/SKILL.md"

    def convert_result(self, raw_value):
        return PromptResult([Message(raw_value, role="assistant")])


class Dispatched(Prompt):
    """A server's own prompt that serves by its ``_render``, not its ``render``."""

    async def render(self, arguments=None):
        return "render is not what is served"

    async def _render(self, arguments=None):
        return PromptResult("see skill://lib/guide/SKILL.md")


async def test_a_readers_convert_result_is_served_behind_a_prefix(tmp_path):
    mcp = QuickMCP("t")
    mcp.command_reader(lambda text, path, name, plugin: Answered(name=name))
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with over_http(mcp, header="kb+") as client:
        messages = (await client.get_prompt("lib_ask", {})).messages
    assert [m.role for m in messages] == ["assistant"]
    assert messages[0].content.text == "see kb+skill://lib/guide/SKILL.md"


async def test_a_readers_render_override_is_served_behind_a_prefix(tmp_path):
    mcp = QuickMCP("t")
    mcp.command_reader(lambda text, path, name, plugin: Dispatched(name=name))
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    async with over_http(mcp, header="kb+") as client:
        said = (await client.get_prompt("lib_ask", {})).messages[0].content.text
    assert said == "see kb+skill://lib/guide/SKILL.md"
