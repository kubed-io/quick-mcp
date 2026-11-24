"""Prompts and agents as resources (PLUG-66): each file's address, and its claim."""

from __future__ import annotations

import json
import logging

import pytest
from plugin_server import agent_md, command_md, listed, served, write

from kubed.mcp import QuickMCP
from kubed.mcp.plugins import PluginError

pytestmark = pytest.mark.unit


def alpha_beta(tmp_path):
    """Plugins alpha and beta, named by manifest, each with agents/review.md, both
    under the server's name (a copy of test_readers_and_names.py's)."""
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
    ]


def addresses(mcp: QuickMCP) -> list[tuple[str, str]]:
    """What a rewriter is shown as each site's kind and address, as builds run."""
    seen: list[tuple[str, str]] = []

    def record(site, text):
        seen.append((site.kind, site.uri))
        return text

    mcp.add_rewriter(record)
    return seen


async def addressed(
    mcp: QuickMCP, seen: list[tuple[str, str]]
) -> tuple[list[str], list[str]]:
    """The command and agent addresses one build shows a rewriter. Builds are
    memoised, so the next build is made only once what is served changes."""
    seen.clear()
    await listed(mcp)
    return (
        [uri for kind, uri in seen if kind == "command"],
        [uri for kind, uri in seen if kind == "agent"],
    )


def manifest(name: str, **fields) -> str:
    return json.dumps({"name": name, **fields})


async def test_a_file_is_at_its_path_below_its_folder(tmp_path):
    root = write(
        tmp_path / "p",
        {
            "commands/db/migrate.md": command_md(),
            "commands/x.prompt.md": command_md(),
            "agents/review/security.md": agent_md("security"),
        },
    )
    mcp = served((root, "kit"))
    seen = addresses(mcp)
    commands, agents = await addressed(mcp, seen)
    assert commands == ["prompts://kit/db/migrate.md", "prompts://kit/x.prompt.md"]
    assert agents == ["agents://kit/review/security.md"]
    assert mcp.plugins[0].prompts == ("kit_db_migrate", "kit_x")


async def test_a_listed_file_outside_its_folder_is_at_its_root_path(tmp_path):
    root = write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": manifest(
                "p",
                commands=["./extra/x.md", "./commands/db/y.md"],
                agents=["./more/scout.md", "./agents/sub/helper.md"],
            ),
            "extra/x.md": command_md(),
            "commands/db/y.md": command_md(),
            "more/scout.md": agent_md("scout"),
            "agents/sub/helper.md": agent_md("helper"),
        },
    )
    mcp = served((root, "kit"))
    seen = addresses(mcp)
    commands, agents = await addressed(mcp, seen)
    assert sorted(commands) == ["prompts://kit/db/y.md", "prompts://kit/extra/x.md"]
    assert mcp.plugins[0].prompts == ("kit_x", "kit_y")  # names stay the stem
    assert sorted(agents) == [
        "agents://kit/more/scout.md",
        "agents://kit/sub/helper.md",
    ]


async def test_a_renamed_agent_is_at_its_served_name(tmp_path):
    mcp = served(*alpha_beta(tmp_path))
    seen = addresses(mcp)
    _, agents = await addressed(mcp, seen)
    assert agents == ["agents://t/review.md", "agents://t/review@beta.md"]

    staging = mcp.plugin_set()
    staging.keep(mcp.plugins[1])
    mcp.serve(staging)
    assert (await addressed(mcp, seen))[1] == ["agents://t/review.md"]


def test_an_address_two_files_want_skips_the_later(tmp_path, caplog):
    root = write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": manifest(
                "p", commands=["./commands/", "./extra/a.md"]
            ),
            "commands/extra/a.md": command_md(),
            "extra/a.md": command_md(),
        },
    )
    with caplog.at_level(logging.WARNING):
        plugin = served((root, "kit")).plugins[0]
    assert plugin.prompts == ("kit_extra_a",)
    assert (
        "extra/a.md",
        "prompts://kit/extra/a.md is already served by commands/extra/a.md",
    ) in plugin.skipped
    base = root.resolve()
    assert (
        f"prompts://kit/extra/a.md is already served by {base / 'commands/extra/a.md'}"
        f"; {base / 'extra/a.md'} is skipped"
    ) in caplog.messages


def test_a_file_skipped_for_its_address_claims_no_name(tmp_path):
    root = write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": manifest(
                "p", commands=["./commands/", "./extra/a.md", "./more/a.md"]
            ),
            "commands/extra/a.md": command_md(),
            "extra/a.md": command_md(),
            "more/a.md": command_md(),
        },
    )
    plugin = served((root, "kit")).plugins[0]
    assert plugin.prompts == ("kit_extra_a", "kit_a")
    assert [path for path, _ in plugin.skipped] == ["extra/a.md"]


def test_an_agent_address_two_files_want_skips_the_later(tmp_path):
    root = write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": manifest(
                "p", agents=["./agents/x.md", "./x.md"]
            ),
            "agents/x.md": agent_md("one"),
            "x.md": agent_md("two"),
        },
    )
    plugin = served((root, "kit")).plugins[0]
    assert plugin.agents == ("kit_one",)
    assert ("x.md", "agents://kit/x.md is already served by agents/x.md") in (
        plugin.skipped
    )


def test_a_later_plugins_address_refuses_it(tmp_path):
    alpha = write(
        tmp_path / "alpha",
        {
            ".claude-plugin/plugin.json": manifest("alpha"),
            "commands/extra/a.md": command_md(),
        },
    )
    beta = write(
        tmp_path / "beta",
        {
            ".claude-plugin/plugin.json": manifest("beta", commands=["./extra/a.md"]),
            "extra/a.md": command_md(),
        },
    )
    mcp = served((alpha, "kit"))
    with pytest.raises(
        PluginError,
        match=r"prompts://kit/extra/a\.md is already served by plugin alpha",
    ):
        mcp.add_plugin(beta, namespace="kit")
    assert [plugin.name for plugin in mcp.plugins] == ["alpha"]


def test_a_kept_plugins_address_refuses_it(tmp_path):
    alpha = write(
        tmp_path / "alpha",
        {
            ".claude-plugin/plugin.json": manifest("alpha"),
            "commands/extra/a.md": command_md(),
        },
    )
    beta = write(
        tmp_path / "beta",
        {
            ".claude-plugin/plugin.json": manifest("beta", commands=["./extra/a.md"]),
            "extra/a.md": command_md(),
        },
    )
    mcp = QuickMCP("t")
    kept = mcp.add_plugin(beta, namespace="kit")
    staging = mcp.plugin_set()
    staging.add(alpha, namespace="kit")
    with pytest.raises(
        PluginError,
        match=r"^plugin .* is not served: "
        r"prompts://kit/extra/a\.md is already served by plugin alpha$",
    ):
        staging.keep(kept)


def test_a_later_plugins_agent_address_skips_the_agent(tmp_path, caplog):
    alpha = write(
        tmp_path / "alpha",
        {".claude-plugin/plugin.json": manifest("alpha"), "agents/x.md": agent_md("y")},
    )
    beta = write(
        tmp_path / "beta",
        {
            ".claude-plugin/plugin.json": manifest("beta"),
            "agents/x.md": agent_md("z"),
            "agents/w.md": agent_md("w"),
        },
    )
    mcp = served((alpha, "kit"))
    with caplog.at_level(logging.WARNING):
        plugin = mcp.add_plugin(beta, namespace="kit")  # not refused (PLUG-64)
    assert plugin.agents == ("kit_w",)  # kit_z never claimed its name
    assert (
        "agents/x.md",
        "agents://kit/x.md is already served by plugin alpha",
    ) in plugin.skipped
    assert (
        "plugin beta: agent file agents/x.md is skipped: "
        "agents://kit/x.md is already served by plugin alpha"
    ) in caplog.text
    assert caplog.text.count("agent file agents/x.md is skipped") == 1


def x_twice(tmp_path):
    """alpha with agents/x.md named y, and beta with agents/x.md named z, which
    alpha's address skips under kit."""
    alpha = write(
        tmp_path / "alpha",
        {".claude-plugin/plugin.json": manifest("alpha"), "agents/x.md": agent_md("y")},
    )
    beta = write(
        tmp_path / "beta",
        {".claude-plugin/plugin.json": manifest("beta"), "agents/x.md": agent_md("z")},
    )
    return alpha, beta


def test_an_agent_skipped_for_its_address_leaves_its_name_free(tmp_path):
    alpha, beta = x_twice(tmp_path)
    mcp = served((alpha, "kit"), (beta, "kit"))
    assert mcp.plugins[1].agents == ()
    gamma = write(
        tmp_path / "gamma",
        {".claude-plugin/plugin.json": manifest("gamma"), "agents/q.md": agent_md("z")},
    )
    assert mcp.add_plugin(gamma, namespace="kit").agents == ("kit_z",)


def test_an_agent_skipped_for_its_address_reenters_unchanged(tmp_path, caplog):
    alpha, beta = x_twice(tmp_path)
    with caplog.at_level(logging.WARNING):
        mcp = served((alpha, "kit"), (beta, "kit"))
        first, second = mcp.plugins
        staging = mcp.plugin_set()
        assert staging.keep(first) is first
        assert staging.keep(second) is second  # the Plugin a server holds
        mcp.serve(staging)
    said = [m for m in caplog.messages if "agent file agents/x.md is skipped" in m]
    assert said == [
        "plugin beta: agent file agents/x.md is skipped: "
        "agents://kit/x.md is already served by plugin alpha"
    ]
    assert mcp.plugins[1] is second


def test_a_kept_plugins_agent_address_skips_the_agent(tmp_path):
    alpha, beta = x_twice(tmp_path)
    mcp = QuickMCP("t")
    kept = mcp.add_plugin(beta, namespace="kit")
    staging = mcp.plugin_set()
    staging.add(alpha, namespace="kit")
    entered = staging.keep(kept)  # not refused (PLUG-64)
    assert entered.agents == ()
    assert (
        "agents/x.md",
        "agents://kit/x.md is already served by plugin alpha",
    ) in entered.skipped


def own_address_twice(tmp_path):
    """alpha and beta, beta also with agents/review@beta.md, the address its review
    takes once it is renamed review@beta."""
    first, beta = alpha_beta(tmp_path)
    write(beta[0], {"agents/review@beta.md": agent_md("other")})
    return served(first, beta)


def test_an_agents_address_another_of_its_own_takes_skips_it(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        plugin = own_address_twice(tmp_path).plugins[1]
    assert plugin.agents == ("review@beta",)
    reason = "agents://t/review@beta.md is already served by agents/review.md"
    assert ("agents/review@beta.md", reason) in plugin.skipped
    assert (
        f"plugin beta: agent file agents/review@beta.md is skipped: {reason}"
        in caplog.messages
    )


def test_an_agents_own_address_is_back_once_its_name_is(tmp_path):
    mcp = own_address_twice(tmp_path)
    staging = mcp.plugin_set()
    assert staging.keep(mcp.plugins[1]).agents == ("review", "other")


def test_a_renamed_agents_taken_address_skips_it(tmp_path):
    # alpha's own file sits where beta's renamed review would be served.
    alpha = write(
        tmp_path / "first",
        {
            ".claude-plugin/plugin.json": manifest("alpha"),
            "agents/review.md": agent_md("review"),
            "agents/review@beta.md": agent_md("other"),
        },
    )
    mcp = served((alpha, None), alpha_beta(tmp_path)[1])
    beta = mcp.plugins[1]
    assert beta.agents == ()
    assert (
        "agents/review.md",
        "agents://t/review@beta.md is already served by plugin alpha",
    ) in beta.skipped


@pytest.mark.parametrize(
    ("path", "said", "address"),
    [
        ("commands/_index.md", "prompt", "prompts://kit/_index.md"),
        ("commands/_index.json/x.md", "prompt", "prompts://kit/_index.json/x.md"),
        ("agents/_index.md", "agent file", "agents://kit/_index.md"),
    ],
)
def test_a_file_at_an_index_address_is_skipped(tmp_path, caplog, path, said, address):
    body = agent_md("i") if said == "agent file" else command_md()
    root = write(tmp_path / "p", {path: body, "commands/ok.md": command_md()})
    with caplog.at_level(logging.WARNING):
        plugin = served((root, "kit")).plugins[0]
    reason = f"its address {address} is its folder's index"
    assert [each for each in plugin.skipped if each[0] == path] == [(path, reason)]
    assert plugin.prompts == ("kit_ok",)
    assert plugin.agents == ()
    assert f"{said} {root.resolve() / path} is skipped: {reason}" in caplog.messages
