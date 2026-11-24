"""Readers and names: how a plugin's commands are read (PLUG-62), its prompts and agents
named (PLUG-63), and a clashing agent renamed when its plugin enters a set (PLUG-64,
PLUG-65, PLUG-74)."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client
from fastmcp.prompts import Message, Prompt, PromptResult
from gateway_server import KIT_TREE, over_http
from mcp_types import ImageContent
from plugin_server import DEMO, ENTRY, agent_md, command_md, listed, served, write
from skill_server import request

from kubed.mcp import QuickMCP
from kubed.mcp.agents import META_KEY, parse_agent, served_as
from kubed.mcp.plugins import AgentFile, PluginError
from kubed.mcp.prompts import read_command
from kubed.mcp.protocol import MCPError

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


async def test_a_build_given_the_plugin_as_served(tmp_path):
    # beta is served with review@beta, read with review: a build must give its reader
    # the plugin as its set serves it, not as read.
    seen = []
    mcp = QuickMCP("t")

    @mcp.command_reader
    def reader(text, path, name, plugin):
        if "Done it." in text:
            seen.append((name, plugin.agents))
        return read_command(text, path, name, plugin)

    for name in ("alpha", "beta"):
        mcp.add_plugin(
            write(
                tmp_path / name,
                {
                    ".claude-plugin/plugin.json": json.dumps({"name": name}),
                    "agents/review.md": agent_md("review"),
                    f"commands/{name}-cmd.md": command_md(),
                },
            )
        )
    mcp.add_rewriter(lambda site, text: text.replace("Do it.", "Done it."))
    assert (await listed(mcp))[0] == ["alpha-cmd", "beta-cmd"]
    assert ("beta-cmd", ("review@beta",)) in seen


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


# ---- agents renamed on set entry (PLUG-64, PLUG-65, PLUG-74) -----------------------


def alpha_beta(tmp_path, *more):
    """Plugins alpha and beta, named by manifest, each with agents/review.md, both
    under the server's name; then ``more``, more (root, namespace) pairs."""
    return [
        (
            write(
                tmp_path / name,
                {
                    ".claude-plugin/plugin.json": json.dumps({"name": name}),
                    "agents/review.md": agent_md("review"),
                },
            ),
            None,
        )
        for name in ("alpha", "beta")
    ] + list(more)


async def test_a_second_plugins_agent_is_served_renamed(tmp_path, caplog):
    mcp = served(*alpha_beta(tmp_path))
    assert (await listed(mcp))[1] == ["review", "review@beta"]
    assert mcp.plugins[1].agents == ("review@beta",)
    assert mcp.plugins[1].agent_files == (
        AgentFile("review@beta", "agents/review.md", True),
    )
    assert mcp.plugins[0].agent_files == (
        AgentFile("review", "agents/review.md", False),
    )
    async with Client(mcp.server) as client:
        detail = await request(client, "agents/get", name="review@beta")
    assert detail["agent"] == "review@beta"
    mcp.add_plugin(write(tmp_path / "gamma", {"commands/x.md": command_md()}))
    warned = [r for r in caplog.records if "serves it as review@beta" in r.getMessage()]
    assert [r.getMessage() for r in warned] == [
        "agent review is already served by plugin alpha, so plugin beta serves it as "
        "review@beta"
    ]


def test_a_renamed_plugin_stays_the_one_served(tmp_path):
    mcp = served(*alpha_beta(tmp_path))
    beta = mcp.plugins[1]
    mcp.add_plugin(write(tmp_path / "gamma", {"commands/x.md": command_md()}))
    assert mcp.plugins[1] is beta  # beta re-entered as it was, so keep() still finds it
    staging = mcp.plugin_set()
    staging.keep(mcp.plugins[0])
    assert staging.keep(beta) is beta


async def test_a_kept_plugin_gets_its_plain_name_back(tmp_path):
    mcp = served(*alpha_beta(tmp_path))
    staging = mcp.plugin_set()
    kept = staging.keep(mcp.plugins[1])
    mcp.serve(staging)
    assert kept.agents == ("review",)
    assert kept.agent_files == (AgentFile("review", "agents/review.md", False),)
    assert mcp.plugins == (kept,)
    assert (await listed(mcp))[1] == ["review"]
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as caught:
            await request(client, "agents/get", name="review@beta")
        assert (await request(client, "agents/get", name="review"))["agent"] == (
            "review"
        )
    assert caught.value.error.message == "Agent not found"


async def test_a_third_clash_skips_just_that_agent(tmp_path, caplog):
    mcp = served(*alpha_beta(tmp_path))
    again = write(
        tmp_path / "again",
        {
            ".claude-plugin/plugin.json": json.dumps({"name": "beta"}),
            "agents/review.md": agent_md("review"),
            "agents/other.md": agent_md("other"),
        },
    )
    third = mcp.add_plugin(again)
    assert third.agents == ("other",)
    assert third.agent_files == (AgentFile("other", "agents/other.md", False),)
    reason = "agent review@beta is already served by plugin beta"
    assert ("agents/review.md", reason) in third.skipped
    assert f"plugin beta: agent file agents/review.md is skipped: {reason}" in (
        caplog.text
    )
    assert (await listed(mcp))[1] == ["other", "review", "review@beta"]


def test_a_rename_needs_a_usable_plugin_name(tmp_path):  # Review Focus 3
    mcp = served(*alpha_beta(tmp_path)[:1])
    late = mcp.add_plugin(
        write(
            tmp_path / "my plugin",
            {"agents/review.md": agent_md("review"), "commands/x.md": command_md()},
        )
    )
    assert late.name == "my plugin"
    assert late.agents == () and late.prompts == ("x",)
    assert late.agent_files == ()
    assert (
        "agents/review.md",
        "agent name 'review@my plugin' is not a usable name",
    ) in late.skipped


def test_a_skip_is_warned_of_once(tmp_path, caplog):
    mcp = served(*alpha_beta(tmp_path)[:1])
    mcp.add_plugin(
        write(tmp_path / "my plugin", {"agents/review.md": agent_md("review")})
    )
    mcp.add_plugin(write(tmp_path / "gamma", {"commands/x.md": command_md()}))
    skips = [r for r in caplog.records if "is not a usable name" in r.getMessage()]
    assert len(skips) == 1


async def test_a_rewritten_build_serves_the_renamed_agent(tmp_path):  # Review Focus 2
    mcp = served(*alpha_beta(tmp_path))
    mcp.add_rewriter(lambda site, text: text.replace("Go.", "Went."))
    assert (await listed(mcp))[1] == ["review", "review@beta"]
    async with Client(mcp.server) as client:
        detail = await request(client, "agents/get", name="review@beta")
    assert detail["agent"] == "review@beta"
    assert detail["instructions"] == "Went."


def test_a_read_plugin_names_each_agent_file():
    plugin = served((DEMO, None)).plugins[0]
    assert plugin.agent_files == (AgentFile("helper", "agents/helper.md", False),)


def test_served_as_leaves_the_agent_given_untouched():
    agent = parse_agent(agent_md("review"))
    card, detail = json.dumps(agent.card), json.dumps(agent.detail)
    renamed = served_as(agent, "review@beta", {"io.example/x": {"a": 1}})
    assert renamed.name == renamed.card["name"] == renamed.detail["agent"]
    assert renamed.detail["_meta"]["io.example/x"] == {"a": 1}
    assert (json.dumps(agent.card), json.dumps(agent.detail)) == (card, detail)
    assert served_as(agent, "review", {}) is agent


# ---- agent_meta (PLUG-73) ----------------------------------------------------------


async def kit_helper(mcp):
    """The definition and the card of the demo's agent, served under ``kit``."""
    async with Client(mcp.server) as client:
        detail = await request(client, "agents/get", name="kit_helper")
        (card,) = (await request(client, "agents/list"))["agents"]
    return detail, card


async def plain_kit_helper():
    """``kit_helper`` as a server with no ``agent_meta`` serves it: the ``_meta`` the
    layer and FastMCP give, and the card."""
    plain = QuickMCP("t")
    plain.add_plugin(DEMO, namespace="kit")
    return await kit_helper(plain)


async def test_agent_meta_adds_keys_beside_the_layers():  # kb#83
    layers, plain_card = await plain_kit_helper()
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        return {"io.kubed.mcp-kb/agent": {"library": plugin.namespace, "name": name}}

    mcp.add_plugin(DEMO, namespace="kit")
    detail, card = await kit_helper(mcp)
    assert detail["_meta"]["io.kubed.mcp-kb/agent"] == {
        "library": "kit",
        "name": "kit_helper",
    }
    assert "frontmatter" in detail["_meta"][META_KEY]
    assert detail["_meta"][META_KEY] == layers["_meta"][META_KEY]
    assert set(detail["_meta"]) == {*layers["_meta"], "io.kubed.mcp-kb/agent"}
    assert set(card) == {"name", "description", "capabilities"}
    assert card == plain_card


def test_agent_meta_returns_fn():
    mcp = QuickMCP("t")

    def fn(name, plugin):
        return None

    assert mcp.agent_meta(fn) is fn


def test_agent_meta_gets_the_served_name(tmp_path):
    calls = []
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        calls.append((name, plugin.name))

    for root, namespace in alpha_beta(tmp_path):
        mcp.add_plugin(root, namespace=namespace)
    assert calls[-1] == ("review@beta", "beta")


def test_agent_meta_is_given_the_plugin_as_it_enters(tmp_path):
    seen = []
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        seen.append((plugin.agents, plugin.agent_files))

    for root, namespace in alpha_beta(tmp_path):
        mcp.add_plugin(root, namespace=namespace)
    assert seen[-1] == (
        ("review@beta",),
        (AgentFile("review@beta", "agents/review.md", True),),
    )


async def test_a_later_agent_meta_replaces_the_first():
    mcp = QuickMCP("t")
    mcp.agent_meta(lambda name, plugin: {"first": 1})
    mcp.agent_meta(lambda name, plugin: {"second": 2})
    mcp.add_plugin(DEMO, namespace="kit")
    detail, _ = await kit_helper(mcp)
    layers, _ = await plain_kit_helper()
    assert set(detail["_meta"]) == {*layers["_meta"], "second"}


async def test_the_layers_key_wins(caplog):
    mcp = QuickMCP("t")
    mcp.agent_meta(lambda name, plugin: {META_KEY: {"x": 1}, "k": 1})
    mcp.add_plugin(DEMO, namespace="kit")
    detail, _ = await kit_helper(mcp)
    assert "frontmatter" in detail["_meta"][META_KEY] and detail["_meta"]["k"] == 1
    assert "x" not in detail["_meta"][META_KEY]
    assert (
        "plugin demo: agent_meta for agent kit_helper returned io.kubed/agent, which "
        "is the layer's; it is dropped"
    ) in caplog.text


@pytest.mark.parametrize(
    ("result", "warning"),
    [
        (["k"], "returned list, not a dict; it is dropped"),
        ("k", "returned str, not a dict; it is dropped"),
        ({"k": {1, 2}}, "is not representable as JSON; it is dropped"),
        ({"k": float("nan")}, "is not representable as JSON; it is dropped"),
    ],
)
async def test_a_result_that_is_not_a_json_dict_is_dropped(
    result, warning, caplog
):  # Review Focus 4
    mcp = QuickMCP("t")
    mcp.agent_meta(lambda name, plugin: result)
    mcp.add_plugin(DEMO, namespace="kit")
    detail, _ = await kit_helper(mcp)
    layers, _ = await plain_kit_helper()
    assert detail["_meta"] == layers["_meta"]  # the layer's, and FastMCP's serverInfo
    assert "it is dropped" in caplog.text
    assert f"plugin demo: agent_meta for agent kit_helper {warning}" in caplog.text


async def test_a_raising_agent_meta_leaves_that_agent_unserved(tmp_path, caplog):
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        if name == "a":
            raise RuntimeError("boom")

    root = write(
        tmp_path / "p", {"agents/a.md": agent_md("a"), "agents/b.md": agent_md("b")}
    )
    plugin = mcp.add_plugin(root)
    assert plugin.agents == ("b",)
    assert plugin.agent_files == (AgentFile("b", "agents/b.md", False),)
    assert ("agents/a.md", "agent_meta failed: boom") in plugin.skipped
    assert (await listed(mcp))[1] == ["b"]
    assert (
        "plugin p: agent_meta failed on agent a, so it is not served: boom"
        in caplog.text
    )


def test_a_raise_without_a_message_gives_its_type(tmp_path):
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        raise ValueError

    plugin = mcp.add_plugin(DEMO)
    assert plugin.agents == ()
    assert ("agents/helper.md", "agent_meta failed: ValueError") in plugin.skipped


def test_a_raising_agent_claims_no_name(tmp_path):
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        if plugin.name == "alpha":
            raise RuntimeError("boom")

    for root, namespace in alpha_beta(tmp_path):
        mcp.add_plugin(root, namespace=namespace)
    assert mcp.plugins[0].agents == ()
    assert mcp.plugins[1].agents == ("review",)


def test_a_raise_on_re_entry_is_logged_once(tmp_path, caplog):
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        raise RuntimeError("boom")

    mcp.add_plugin(DEMO)
    mcp.add_plugin(write(tmp_path / "gamma", {"commands/x.md": command_md()}))
    failed = [r for r in caplog.records if "agent_meta failed on" in r.getMessage()]
    assert len(failed) == 1
    assert (
        mcp.plugins[0].skipped.count(("agents/helper.md", "agent_meta failed: boom"))
        == 1
    )


async def test_agent_meta_stores_a_copy():
    given = {"k": {"v": 1}}
    mcp = QuickMCP("t")
    mcp.agent_meta(lambda name, plugin: given)
    mcp.add_plugin(DEMO, namespace="kit")
    given["k"]["v"] = 2
    detail, _ = await kit_helper(mcp)
    assert detail["_meta"]["k"] == {"v": 1}


def test_a_plugin_with_agent_meta_stays_the_one_served(tmp_path):
    mcp = QuickMCP("t")
    mcp.agent_meta(lambda name, plugin: {"k": name})
    demo = mcp.add_plugin(DEMO)
    mcp.add_plugin(write(tmp_path / "gamma", {"commands/x.md": command_md()}))
    assert mcp.plugins[0] is demo
    assert mcp.plugin_set().keep(demo) is demo


async def test_agent_meta_returning_none_serves_the_objects_as_read():
    mcp = QuickMCP("t")
    mcp.agent_meta(lambda name, plugin: None)
    mcp.add_plugin(DEMO)
    # Internal on purpose: no copies is selenium-flow's byte parity.
    read = mcp._plugins.current[0]
    assert read.plugin is read.read
    assert read.agents is read.read_agents
    built = (await mcp._plugins.pinned()).build
    assert built.agents[0] is read.read_agents[0].item


async def test_a_rewritten_build_keeps_the_agents_meta():
    mcp = QuickMCP("t")
    mcp.agent_meta(lambda name, plugin: {"io.kubed.mcp-kb/agent": {"name": name}})
    mcp.add_plugin(DEMO, namespace="kit")
    mcp.add_rewriter(lambda site, text: text.replace("first.", "before all."))
    detail, _ = await kit_helper(mcp)
    assert detail["instructions"].endswith("before all.")
    assert detail["_meta"]["io.kubed.mcp-kb/agent"] == {"name": "kit_helper"}
    assert "frontmatter" in detail["_meta"][META_KEY]


def test_a_rename_that_agent_meta_fails_is_not_announced(tmp_path, caplog):
    mcp = QuickMCP("t")

    @mcp.agent_meta
    def meta(name, plugin):
        if name == "review@beta":
            raise RuntimeError("boom")

    for root, namespace in alpha_beta(tmp_path):
        mcp.add_plugin(root, namespace=namespace)
    assert mcp.plugins[1].agents == ()
    assert "serves it as review@beta" not in caplog.text


async def test_an_agent_meta_set_later_keeps_the_plugin_held(tmp_path):
    mcp = QuickMCP("t")
    demo = mcp.add_plugin(DEMO, namespace="kit")
    mcp.agent_meta(lambda name, plugin: {"io.kubed.mcp-kb/agent": {"name": name}})
    mcp.add_plugin(write(tmp_path / "gamma", {"commands/x.md": command_md()}))
    assert mcp.plugins[0] is demo
    assert mcp.plugin_set().keep(demo) is demo
    detail, _ = await kit_helper(mcp)
    assert detail["_meta"]["io.kubed.mcp-kb/agent"] == {"name": "kit_helper"}
