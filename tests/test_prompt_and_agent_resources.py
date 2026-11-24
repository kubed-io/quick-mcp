"""Prompts and agents as resources (PLUG-66): each file's address, and its claim."""

from __future__ import annotations

import json
import logging

import pytest
from fastmcp import Client, FastMCP
from fastmcp.prompts import Prompt
from gateway_server import kit, over_http
from mcp_types import INVALID_PARAMS
from plugin_server import (
    DEMO,
    ENTRY,
    agent_md,
    command_md,
    listed,
    served,
    skill,
    write,
)
from skill_server import DEMO as SKILL_DEMO
from skill_server import ROOT as SKILL_ROOT
from skill_server import ExtensionParams, read_directory, read_json, request

from kubed.mcp import QuickMCP
from kubed.mcp.agents import EXTENSION_ID as AGENTS_ID
from kubed.mcp.agents import META_KEY as AGENT_KEY
from kubed.mcp.plugins import PluginError, View
from kubed.mcp.prompts import META_KEY as PROMPT_KEY
from kubed.mcp.protocol import MCPError
from kubed.mcp.skills import (
    DIRECTORY_READ,
    NameTheIndex,
    SkillsExtension,
    load_skill,
    serve,
)

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


# ---- the files, their folders and indexes (PLUG-66 to PLUG-68, PLUG-71) -------------

FLOW = {  # selenium-flow's shape, served with no namespace on a server named so
    "skills/driving-a-browser/SKILL.md": skill("driving-a-browser"),
    "commands/build_flow.md": (
        "---\ndescription: Build a flow.\nargument-hint: <site>\n"
        "arguments:\n- name: site\n  required: true\n---\nBuild a flow for {{ site }}.\n"
    ),
    "commands/repair_flow.md": (
        "---\ndescription: Repair a flow.\ndisable-model-invocation: true\n"
        "arguments:\n- name: flow\n  required: true\n---\nRepair {{ flow }}.\n"
    ),
    "agents/explorer.md": agent_md("explorer", "Explores a site."),
}
FLOW_FILES = {
    "commands/build_flow.md": "prompts://selenium-flow/build_flow.md",
    "commands/repair_flow.md": "prompts://selenium-flow/repair_flow.md",
    "agents/explorer.md": "agents://selenium-flow/explorer.md",
}


def flow(tmp_path) -> QuickMCP:
    return served((write(tmp_path / "flow", FLOW), None), name="selenium-flow")


async def rows_of(client, uri: str) -> list[dict]:
    """The children ``resources/directory/read`` gives folder ``uri``."""
    return (await read_directory(client, uri=uri))["resources"]


async def read_text(client, uri: str) -> str:
    contents = await client.read_resource(uri)
    assert len(contents) == 1
    return contents[0].text


async def test_each_prompt_and_agent_reads_at_its_address(tmp_path):
    async with Client(flow(tmp_path).server) as client:
        for file, uri in FLOW_FILES.items():
            (content,) = await client.read_resource(uri)
            assert content.text == FLOW[file]
            assert content.mime_type == "text/markdown"


async def test_they_are_never_listed(tmp_path):
    async with Client(flow(tmp_path).server) as client:
        resources = [str(r.uri) for r in await client.list_resources()]
        templates = [t.uriTemplate for t in await client.list_resource_templates()]
    assert resources == ["skill://selenium-flow/driving-a-browser/SKILL.md"]
    assert templates == []


async def test_the_namespace_is_the_top_folder(tmp_path):
    async with Client(flow(tmp_path).server) as client:
        rows = await rows_of(client, "prompts://selenium-flow")
        agents = await rows_of(client, "agents://selenium-flow")
        index_md = await read_text(client, "prompts://selenium-flow/_index.md")
        index_json = await read_json(client, "prompts://selenium-flow/_index.json")
    assert [r["uri"] for r in rows] == [
        "prompts://selenium-flow/build_flow.md",
        "prompts://selenium-flow/repair_flow.md",
    ]
    assert rows[0]["name"] == "build_flow"
    assert rows[0]["description"] == "Build a flow."
    assert rows[0]["mimeType"] == "text/markdown"
    assert [(r["uri"], r["name"], r["description"]) for r in agents] == [
        ("agents://selenium-flow/explorer.md", "explorer", "Explores a site.")
    ]
    assert index_md == (
        "prompts://selenium-flow/build_flow.md: Build a flow.\n"
        "prompts://selenium-flow/repair_flow.md: Repair a flow.\n"
    )
    assert index_json == rows


def kit_with(tmp_path, name: str, files: dict[str, str]):
    """A plugin ``name`` holding ``files`` and a skill: its first plugin brings both
    the Skills extension and the agents extension, which binds
    resources/directory/read."""
    files = {".claude-plugin/plugin.json": manifest(name), **files}
    return write(tmp_path / name, {f"skills/{name}/SKILL.md": skill(name), **files})


async def test_a_nested_file_makes_its_folders(tmp_path):
    root = kit_with(tmp_path, "p", {"commands/db/migrate.md": command_md("Migrate.")})
    async with Client(served((root, "kit")).server) as client:
        top = await rows_of(client, "prompts://kit")
        db = await rows_of(client, "prompts://kit/db")
        db_md = await read_text(client, "prompts://kit/db/_index.md")
    assert top == [
        {
            "uri": "prompts://kit/db",
            "mimeType": "inode/directory",
            "name": "kit/db",
            "description": "A folder of prompts in kit",
        }
    ]
    assert [(r["uri"], r["name"]) for r in db] == [
        ("prompts://kit/db/migrate.md", "kit_db_migrate")
    ]
    assert db_md == "prompts://kit/db/migrate.md: Migrate.\n"


async def test_a_namespace_above_is_a_folder_only_with_a_plugin_under_it(tmp_path):
    tools = kit_with(tmp_path, "tools", {"commands/a.md": command_md()})
    top = kit_with(tmp_path, "top", {"commands/b.md": command_md()})
    async with Client(served((tools, "kit/tools")).server) as client:
        with pytest.raises(MCPError) as caught:
            await read_directory(client, uri="prompts://kit")
        under = await rows_of(client, "prompts://kit/tools")
    assert (caught.value.error.code, caught.value.error.message) == (
        INVALID_PARAMS,
        "Directory not found",
    )
    assert [r["uri"] for r in under] == ["prompts://kit/tools/a.md"]

    async with Client(served((tools, "kit/tools"), (top, "kit")).server) as client:
        rows = await rows_of(client, "prompts://kit")
    assert [(r["uri"], r["name"]) for r in rows] == [
        ("prompts://kit/b.md", "kit_b"),
        ("prompts://kit/tools", "kit/tools"),
    ]
    assert rows[1] == {
        "uri": "prompts://kit/tools",
        "mimeType": "inode/directory",
        "name": "kit/tools",
        "description": "A folder of prompts in kit/tools",
    }


async def test_an_agent_folder_is_a_folder_of_agents(tmp_path):
    root = kit_with(tmp_path, "p", {"agents/review/security.md": agent_md("security")})
    async with Client(served((root, "kit")).server) as client:
        top = await rows_of(client, "agents://kit")
        review = await rows_of(client, "agents://kit/review")
    assert top == [
        {
            "uri": "agents://kit/review",
            "mimeType": "inode/directory",
            "name": "kit/review",
            "description": "A folder of agents in kit",
        }
    ]
    assert [(r["uri"], r["name"]) for r in review] == [
        ("agents://kit/review/security.md", "kit_review_security")
    ]


async def test_an_address_is_percent_quoted(tmp_path):
    root = kit_with(tmp_path, "p", {"commands/caf\u00e9/x.md": command_md("X.")})
    async with Client(served((root, "kit")).server) as client:
        text = await read_text(client, "prompts://kit/caf%C3%A9/x.md")
        top = await rows_of(client, "prompts://kit")
        folder = await rows_of(client, "prompts://kit/caf%C3%A9")
    assert text == command_md("X.")
    assert [(r["uri"], r["name"]) for r in top] == [
        ("prompts://kit/caf%C3%A9", "kit/caf\u00e9")
    ]
    assert [r["uri"] for r in folder] == ["prompts://kit/caf%C3%A9/x.md"]


@pytest.mark.parametrize(
    ("namespace", "files"),
    [
        ("kit/x.md", {"commands/y.md": command_md("Y.")}),
        ("kit", {"commands/x.md/y.md": command_md("Y.")}),
    ],
    ids=["namespace", "folder"],
)
async def test_a_file_and_a_folder_at_one_uri_keep_the_files_row(
    tmp_path, namespace, files
):
    # No spec row covers this yet: the file's row is kept, and nothing is refused.
    first = kit_with(tmp_path, "a", {"commands/x.md": command_md("The file.")})
    second = kit_with(tmp_path, "b", files)
    async with Client(served((first, "kit"), (second, namespace)).server) as client:
        top = await rows_of(client, "prompts://kit")
        text = await read_text(client, "prompts://kit/x.md")
    at = [r for r in top if r["uri"] == "prompts://kit/x.md"]
    assert len(at) == 1
    assert (at[0]["mimeType"], at[0]["description"]) == ("text/markdown", "The file.")
    assert text == command_md("The file.")


async def test_a_folder_read_names_its_index(tmp_path):
    async with Client(flow(tmp_path).server) as client:
        with pytest.raises(MCPError) as caught:
            await client.read_resource("prompts://selenium-flow")
    assert caught.value.error.message == (
        "Resource not found: 'prompts://selenium-flow'. prompts://selenium-flow is a "
        "folder: read prompts://selenium-flow/_index.md to list it."
    )


async def test_a_plugin_the_view_leaves_out_has_no_files(tmp_path):
    first = kit_with(tmp_path, "kit", {"commands/x.md": command_md()})
    second = kit_with(tmp_path, "lab", {"commands/x.md": command_md()})
    mcp = served((first, "kit"), (second, "lab"))
    mcp.plugin_view(lambda: View("kit", lambda plugin: plugin.namespace == "kit"))
    async with Client(mcp.server) as client:
        assert await read_text(client, "prompts://kit/x.md") == command_md()
        with pytest.raises(MCPError) as file:
            await client.read_resource("prompts://lab/x.md")
        with pytest.raises(MCPError) as index:
            await client.read_resource("prompts://lab/_index.md")
        with pytest.raises(MCPError) as folder:
            await read_directory(client, uri="prompts://lab")
    assert file.value.error.message == "Resource not found: 'prompts://lab/x.md'"
    assert index.value.error.message.startswith("Resource not found")
    assert folder.value.error.message == "Directory not found"


async def test_a_rewriters_text_is_what_reads(tmp_path):
    mcp = flow(tmp_path)
    seen: list[tuple[str, str]] = []

    def make(site, text):
        seen.append((site.kind, site.uri))
        return text.replace("Build", "Make")

    mcp.add_rewriter(make)
    async with Client(mcp.server) as client:
        text = await read_text(client, "prompts://selenium-flow/build_flow.md")
        index_md = await read_text(client, "prompts://selenium-flow/_index.md")
        rows = await rows_of(client, "prompts://selenium-flow")
        index_json = await read_json(client, "prompts://selenium-flow/_index.json")
    assert text.startswith("---\ndescription: Make a flow.")
    assert ("index", "prompts://selenium-flow/_index.json") in seen
    assert ("index", "agents://selenium-flow/_index.md") in seen
    assert "Make a flow." in index_md
    assert rows[0]["description"] == "Make a flow."
    assert index_json == rows


async def test_an_index_site_is_its_folder_in_the_first_plugin(tmp_path):
    root = kit_with(tmp_path, "p", {"commands/db/migrate.md": command_md()})
    extra = write(
        tmp_path / "q",
        {
            ".claude-plugin/plugin.json": manifest("q", commands=["./extra/x.md"]),
            "extra/x.md": command_md(),
        },
    )
    mcp = served((root, "kit"), (extra, "lab"))
    sites: dict[str, tuple[str, str]] = {}

    def record(site, text):
        if site.kind == "index":
            sites[site.uri] = (site.plugin.name, site.path)
        return text

    mcp.add_rewriter(record)
    await listed(mcp)
    assert {uri: at for uri, at in sites.items() if uri.startswith("prompts://")} == {
        "prompts://kit/_index.md": ("p", "commands"),
        "prompts://kit/_index.json": ("p", "commands"),
        "prompts://kit/db/_index.md": ("p", "commands/db"),
        "prompts://kit/db/_index.json": ("p", "commands/db"),
        "prompts://lab/_index.md": ("q", ""),
        "prompts://lab/_index.json": ("q", ""),
        "prompts://lab/extra/_index.md": ("q", "extra"),
        "prompts://lab/extra/_index.json": ("q", "extra"),
    }


async def test_a_renamed_agents_index_site_is_where_its_file_was_read(tmp_path):
    alpha = write(
        tmp_path / "alpha",
        {
            ".claude-plugin/plugin.json": manifest("alpha"),
            "agents/sub_x.md": agent_md(None),
        },
    )
    beta = write(
        tmp_path / "beta",
        {
            ".claude-plugin/plugin.json": manifest("beta"),
            "agents/sub/x.md": agent_md(None),
        },
    )
    mcp = served((alpha, "kit"), (beta, "kit/tools"))
    assert mcp.plugins[1].agents == ("kit_sub_x@beta",)
    sites: dict[str, tuple[str, str]] = {}

    def record(site, text):
        if site.kind == "index":
            sites[site.uri] = (site.plugin.name, site.path)
        return text

    mcp.add_rewriter(record)
    await listed(mcp)
    assert sites["agents://kit/tools/_index.md"] == ("beta", "agents")
    assert sites["agents://kit/_index.md"] == ("alpha", "agents")


async def test_a_failed_prompt_has_no_file(tmp_path):
    mcp = flow(tmp_path)

    def fail(site, text):
        if site.uri == "prompts://selenium-flow/build_flow.md":
            raise ValueError("no")
        return text

    mcp.add_rewriter(fail)
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as caught:
            await client.read_resource("prompts://selenium-flow/build_flow.md")
        index = await read_json(client, "prompts://selenium-flow/_index.json")
    assert caught.value.error.message.startswith("Resource not found")
    assert [r["uri"] for r in index] == ["prompts://selenium-flow/repair_flow.md"]


async def test_a_failed_index_is_missing_but_its_folder_is_not(tmp_path):
    mcp = flow(tmp_path)

    def fail(site, text):
        if site.uri == "prompts://selenium-flow/_index.md":
            raise ValueError("no")
        return text

    mcp.add_rewriter(fail)
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as caught:
            await client.read_resource("prompts://selenium-flow/_index.md")
        rows = await rows_of(client, "prompts://selenium-flow")
        index = await read_json(client, "prompts://selenium-flow/_index.json")
    assert caught.value.error.message.startswith("Resource not found")
    assert index == rows
    assert len(rows) == 2


CRLF = b"---\r\ndescription: C.\r\n---\r\nDo it.\r\n"


def crlf(tmp_path, kind: str = "commands") -> QuickMCP:
    """A server over one CRLF file, ``<kind>/c.md`` under ``kit``."""
    root = tmp_path / "p"
    (root / kind).mkdir(parents=True)
    (root / kind / "c.md").write_bytes(CRLF)
    return served((root, "kit"))


async def test_a_file_reads_as_its_bytes(tmp_path):  # Review Focus 3
    async with Client(crlf(tmp_path).server) as client:
        assert await read_text(client, "prompts://kit/c.md") == CRLF.decode("utf-8")


async def test_a_rewritten_crlf_file_reads_as_the_chains_text(tmp_path):
    # The chain is shown the text as read_text gives it, so what it returns is LF.
    mcp = crlf(tmp_path)
    mcp.add_rewriter(lambda site, text: text.replace("Do it", "Done"))
    async with Client(mcp.server) as client:
        text = await read_text(client, "prompts://kit/c.md")
    assert text == "---\ndescription: C.\n---\nDone.\n"


async def test_a_rewriter_that_changes_nothing_serves_the_bytes(tmp_path):
    mcp = crlf(tmp_path, "agents")
    mcp.add_rewriter(lambda site, text: text)
    async with Client(mcp.server) as client:
        assert await read_text(client, "agents://kit/c.md") == CRLF.decode("utf-8")


async def test_a_renamed_agent_reads_at_its_served_name(tmp_path):
    mcp = served(*alpha_beta(tmp_path))
    async with Client(mcp.server) as client:
        renamed = await read_text(client, "agents://t/review@beta.md")
        plain = await read_text(client, "agents://t/review.md")
    assert renamed == plain == agent_md("review")


async def test_the_files_read_behind_a_prefix(tmp_path):  # Review Focus 4
    async with over_http(kit(tmp_path), header="kb+") as client:
        ask = await request(client, "resources/read", uri="kb+prompts://lib/ask.md")
        index = await request(
            client, "resources/read", uri="kb+agents://lib/_index.json"
        )
        rows = await read_directory(client, uri="kb+prompts://lib")
    text = ask["contents"][0]["text"]
    assert "kb+prompts://lib/ask.md" in text
    assert "kb+skill://lib/guide/SKILL.md" in text
    assert [r["uri"] for r in json.loads(index["contents"][0]["text"])] == [
        "kb+agents://lib/scout.md"
    ]
    assert [r["uri"] for r in rows["resources"]] == ["prompts://lib/ask.md"]


async def test_the_extensions_are_unchanged():  # PLUG-71
    mcp = served((DEMO, None))
    assert await listed(mcp) == (["greet"], ["helper"], [ENTRY])
    async with Client(mcp.server) as client:
        greeting = await client.get_prompt("greet", {"who": "x"})
        detail = await request(client, "agents/get", name="helper")
    assert greeting.messages[0].content.text.startswith("Greet x warmly. ")
    assert sorted(detail) == [
        "_meta",
        "agent",
        "cacheScope",
        "instructions",
        "modelRoutingHint",
        "resultType",
        "tools",
        "ttlMs",
    ]


# Rows that register (PLUG-69, PLUG-70)


SERVER_INFO = "io.modelcontextprotocol/serverInfo"


def by_name(rows: list[dict]) -> dict[str, dict]:
    """Each prompt row's ``io.kubed/prompt``, by the name it registers."""
    return {r["_meta"][PROMPT_KEY]["name"]: r["_meta"][PROMPT_KEY] for r in rows}


async def test_the_root_index_alone_registers_both_commands(tmp_path):
    async with Client(flow(tmp_path).server) as client:
        rows = await read_json(client, "prompts://selenium-flow/_index.json")
        listed = {
            p.name: [
                a.model_dump(mode="json", by_alias=True, exclude_none=True)
                for a in p.arguments or []
            ]
            for p in await client.list_prompts()
        }
    assert [set(r["_meta"]) for r in rows] == [{PROMPT_KEY}, {PROMPT_KEY}]
    found = by_name(rows)
    assert set(found) == {"build_flow", "repair_flow"}
    build, repair = found["build_flow"], found["repair_flow"]
    assert build["arguments"] == listed["build_flow"]
    assert build["arguments"] == [{"name": "site", "required": True}]
    assert build["argument-hint"] == "<site>"
    assert "disable-model-invocation" not in build
    assert repair["arguments"] == listed["repair_flow"]
    assert repair["disable-model-invocation"] is True
    assert "user-invocable" not in repair and "argument-hint" not in repair


async def test_a_prompt_files_row_maps_its_address_to_its_name(tmp_path):
    root = kit_with(tmp_path, "p", {"commands/x.prompt.md": command_md("X.")})
    async with Client(served((root, "kit")).server) as client:
        rows = await read_json(client, "prompts://kit/_index.json")
    (row,) = rows
    assert row["uri"] == "prompts://kit/x.prompt.md"
    assert row["_meta"] == {PROMPT_KEY: {"name": "kit_x", "arguments": []}}


async def test_a_nested_prompts_row_carries_its_meta(tmp_path):
    root = kit_with(tmp_path, "p", {"commands/db/migrate.md": command_md("Migrate.")})
    async with Client(served((root, "kit")).server) as client:
        index = await read_json(client, "prompts://kit/db/_index.json")
        rows = await rows_of(client, "prompts://kit/db")
    assert index == rows
    assert [r["_meta"][PROMPT_KEY]["name"] for r in rows] == ["kit_db_migrate"]


def read_by(tmp_path, text: str) -> QuickMCP:
    """A server under ``kit`` whose reader makes a plain prompt of every command, its
    one command ``commands/x.md`` holding ``text``."""
    mcp = QuickMCP("t")
    mcp.command_reader(
        lambda text, path, name, plugin: Prompt.from_function(lambda: "x", name=name)
    )
    mcp.add_plugin(write(tmp_path / "p", {"commands/x.md": text}), namespace="kit")
    return mcp


@pytest.mark.parametrize(
    "text",
    [
        "Just a body.\n",
        "---\n- a list\n---\nBody.\n",
        "---\nargument-hint\n---\nBody.\n",
        "---\n: [\n---\nBody.\n",
    ],
    ids=["no-header", "a-list", "a-string", "bad-yaml"],
)
async def test_a_readers_prompt_without_a_header_registers_too(tmp_path, text):
    # Review Focus 5
    async with Client(read_by(tmp_path, text).server) as client:
        (row,) = await read_json(client, "prompts://kit/_index.json")
    assert row["_meta"] == {PROMPT_KEY: {"name": "kit_x", "arguments": []}}


async def test_a_readers_prompt_has_its_files_invocation_keys(tmp_path):
    text = (
        "---\nuser-invocable: false\nargument-hint: 2024-01-01\n"
        "disable-model-invocation: [a, 1]\nother: kept off\n---\nBody.\n"
    )
    async with Client(read_by(tmp_path, text).server) as client:
        (row,) = await read_json(client, "prompts://kit/_index.json")
    # A YAML date does not survive JSON, so argument-hint is left out.
    assert row["_meta"] == {
        PROMPT_KEY: {
            "name": "kit_x",
            "arguments": [],
            "disable-model-invocation": ["a", 1],
            "user-invocable": False,
        }
    }


async def test_an_agents_row_is_what_agents_get_gives(tmp_path):
    mcp = QuickMCP("selenium-flow")
    mcp.agent_meta(
        lambda name, plugin: {"io.kubed.mcp-kb/agent": {"library": plugin.namespace}}
    )
    mcp.add_plugin(write(tmp_path / "flow", FLOW))
    async with Client(mcp.server) as client:
        rows = await read_json(client, "agents://selenium-flow/_index.json")
        listed = await rows_of(client, "agents://selenium-flow")
        got = await request(client, "agents/get", name="explorer")
    (row,) = rows
    assert listed == rows
    # The SDK stamps every result with its server's info: that is no agent's key.
    meta = {k: v for k, v in got["_meta"].items() if k != SERVER_INFO}
    assert meta["io.kubed.mcp-kb/agent"] == {"library": "selenium-flow"}
    assert row["_meta"] == {**meta, AGENT_KEY: {"name": "explorer", **meta[AGENT_KEY]}}
    assert list(row["_meta"][AGENT_KEY]) == ["name", "frontmatter", "skills"]


async def test_a_renamed_agents_row_has_its_served_name(tmp_path):
    async with Client(served(*alpha_beta(tmp_path)).server) as client:
        rows = await read_json(client, "agents://t/_index.json")
    assert [(r["uri"], r["_meta"][AGENT_KEY]["name"]) for r in rows] == [
        ("agents://t/review.md", "review"),
        ("agents://t/review@beta.md", "review@beta"),
    ]


async def test_a_rewritten_files_row_registers_its_rewritten_text(tmp_path):
    mcp = flow(tmp_path)
    mcp.add_rewriter(
        lambda site, text: text.replace("argument-hint: <site>", "user-invocable: no")
    )
    async with Client(mcp.server) as client:
        rows = await read_json(client, "prompts://selenium-flow/_index.json")
    build = by_name(rows)["build_flow"]
    assert build["user-invocable"] is False
    assert "argument-hint" not in build


async def test_the_rows_text_is_prefixed_and_their_fields_are_not(tmp_path):
    # Review Focus 4
    async with over_http(kit(tmp_path), header="kb+") as client:
        index = await request(
            client, "resources/read", uri="kb+agents://lib/_index.json"
        )
        rows = await read_directory(client, uri="kb+agents://lib")
    text = index["contents"][0]["text"]
    (indexed,) = json.loads(text)
    assert indexed["_meta"][AGENT_KEY]["skills"] == {
        "guide": "kb+skill://lib/guide/SKILL.md"
    }
    (row,) = rows["resources"]
    assert row["uri"] == "agents://lib/scout.md"
    assert row["_meta"][AGENT_KEY]["name"] == "lib_scout"
    assert row["_meta"][AGENT_KEY]["skills"] == {"guide": "skill://lib/guide/SKILL.md"}


async def test_a_prompts_row_behind_a_prefix_is_unprefixed(tmp_path):
    async with over_http(kit(tmp_path), header="kb+") as client:
        prompts = await read_directory(client, uri="kb+prompts://lib")
    (ask,) = prompts["resources"]
    assert ask["uri"] == "prompts://lib/ask.md"
    assert ask["_meta"] == {
        PROMPT_KEY: {
            "name": "lib_ask",
            "arguments": [{"name": "topic", "required": True}],
        }
    }


@pytest.mark.parametrize("rewritten", [False, True], ids=["as-read", "rewritten"])
async def test_a_files_read_carries_its_rows_meta(tmp_path, rewritten):
    # The read and the row are one resource, so their _meta is one.
    mcp = flow(tmp_path)
    if rewritten:
        mcp.add_rewriter(lambda site, text: text.replace("Build", "Make"))
    async with Client(mcp.server) as client:
        for scheme in ("prompts", "agents"):
            index = await read_json(client, f"{scheme}://selenium-flow/_index.json")
            for row in index:
                read = await request(client, "resources/read", uri=row["uri"])
                (content,) = read["contents"]
                assert content["_meta"] == row["_meta"]


# ---- the folder methods without skills (PLUG-68) -----------------------------------


def prompts_only(tmp_path):
    """A plugin with one command in a folder, and no skill."""
    return write(tmp_path / "p", {"commands/db/migrate.md": command_md("Migrate.")})


def holders(server) -> list[str]:
    """The extension identifiers FastMCP bound resources/directory/read for.

    FastMCP has no public record of which extension bound a method, so this reads its
    ``_extension_methods``.
    """
    return [
        i for i, bound in server._extension_methods.items() if DIRECTORY_READ in bound
    ]


async def test_folders_answer_with_no_skill(tmp_path):
    mcp = served((prompts_only(tmp_path), None))
    async with Client(mcp.server, mode="legacy") as client:
        extensions = client.session.server_capabilities.extensions or {}
        top = await rows_of(client, "prompts://t")
        with pytest.raises(MCPError) as caught:
            await client.read_resource("prompts://t/db")
        index_md = await read_text(client, "prompts://t/db/_index.md")
    assert [(r["uri"], r["mimeType"]) for r in top] == [
        ("prompts://t/db", "inode/directory")
    ]
    assert caught.value.error.message == (
        "Resource not found: 'prompts://t/db'. prompts://t/db is a folder: "
        "read prompts://t/db/_index.md to list it."
    )
    assert index_md == "prompts://t/db/migrate.md: Migrate.\n"
    # Answered, never declared: no Skills extension without a skill (PLUG-20).
    assert SkillsExtension.identifier not in extensions
    assert extensions[AGENTS_ID] == {"listChanged": False}
    assert holders(mcp.server) == [AGENTS_ID]


# Whichever extension registers first binds resources/directory/read.
FIRST = {"prompts-first": AGENTS_ID, "skills-first": SkillsExtension.identifier}


@pytest.mark.parametrize("order", FIRST)
async def test_a_skills_plugin_after_a_prompts_one(tmp_path, order):
    # Review Focus 1, and its reverse: both browsable, each bound and added once.
    plugins = [(prompts_only(tmp_path), None), (DEMO, "kit")]
    mcp = served(*(plugins if order == "prompts-first" else reversed(plugins)))
    async with over_http(mcp, mode="legacy") as client:
        extensions = client.session.server_capabilities.extensions or {}
        prompts = await rows_of(client, "prompts://t")
        guide = await rows_of(client, "skill://kit/demo-guide")
    assert [r["uri"] for r in prompts] == ["prompts://t/db"]
    assert "skill://kit/demo-guide/SKILL.md" in [r["uri"] for r in guide]
    assert extensions[SkillsExtension.identifier] == {"directoryRead": True}
    assert extensions[AGENTS_ID] == {"listChanged": False}
    assert sum(isinstance(m, NameTheIndex) for m in mcp.server.middleware) == 1
    assert holders(mcp.server) == [FIRST[order]]


@pytest.mark.parametrize("order", FIRST)
async def test_a_servers_own_skills_beside_its_plugins(tmp_path, order):
    # skills.serve on the server's own catalogue, before or after its first plugin.
    mcp = QuickMCP("t")
    if order == "skills-first":
        serve(mcp.server, [load_skill(SKILL_DEMO)])
    mcp.add_plugin(prompts_only(tmp_path))
    if order == "prompts-first":
        serve(mcp.server, [load_skill(SKILL_DEMO)])
    async with Client(mcp.server, mode="legacy") as client:
        extensions = client.session.server_capabilities.extensions or {}
        own = await rows_of(client, SKILL_ROOT)
        prompts = await rows_of(client, "prompts://t")
    assert f"{SKILL_ROOT}/SKILL.md" in [r["uri"] for r in own]
    assert [r["uri"] for r in prompts] == ["prompts://t/db"]
    assert extensions[SkillsExtension.identifier] == {"directoryRead": True}
    assert sum(isinstance(m, NameTheIndex) for m in mcp.server.middleware) == 1
    assert holders(mcp.server) == [FIRST[order]]


async def test_a_skills_extension_registered_again_keeps_the_method():
    # FastMCP's same-identifier path: an explicit registration replaces the bundled
    # one, and the method it held stays bound.
    mcp = served((DEMO, None))
    mcp.server.add_extension(SkillsExtension())
    async with Client(mcp.server) as client:
        guide = await rows_of(client, "skill://t/demo-guide")
        prompts = await rows_of(client, "prompts://t")
    assert "skill://t/demo-guide/SKILL.md" in [r["uri"] for r in guide]
    assert [r["uri"] for r in prompts] == ["prompts://t/greet.md"]
    assert holders(mcp.server) == [SkillsExtension.identifier]


async def test_a_parent_mounting_a_plugin_server_binds_the_method():
    # The parent registers a clone of the Skills extension, which binds the method
    # there. "Directory not found" for the mounted server's folders is a known
    # limitation (no children pass-through across a mount), not a contract.
    parent = FastMCP("parent")
    parent.mount(served((DEMO, None)).server)
    async with Client(parent) as client:
        with pytest.raises(MCPError) as caught:
            await read_directory(client, uri="skill://t/demo-guide")
    assert (caught.value.error.code, caught.value.error.message) == (
        INVALID_PARAMS,
        "Directory not found",
    )
    assert holders(parent) == [SkillsExtension.identifier]


async def test_a_skills_server_built_with_serve_still_binds_directory_read():
    server = FastMCP("s")
    serve(server, [load_skill(SKILL_DEMO)])
    async with Client(server) as client:
        rows = await rows_of(client, SKILL_ROOT)
        extensions = client.session.server_capabilities.extensions or {}
    assert f"{SKILL_ROOT}/SKILL.md" in [r["uri"] for r in rows]
    assert extensions[SkillsExtension.identifier] == {"directoryRead": True}
    assert holders(server) == [SkillsExtension.identifier]


async def test_a_handler_added_by_hand_is_refused_by_fastmcp():
    # A handler outside any extension is not another extension's binding, so the
    # Skills extension binds the method and FastMCP refuses it, as it would anyway.
    # FastMCP has no public way to add a raw handler: this is its low-level server's.
    server = FastMCP("s")

    async def handler(ctx, params):
        return {"resources": []}

    server._mcp_server.add_request_handler(DIRECTORY_READ, ExtensionParams, handler)
    with pytest.raises(ValueError, match="'resources/directory/read' is already"):
        serve(server, [load_skill(SKILL_DEMO)])
    assert holders(server) == []
