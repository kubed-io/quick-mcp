"""Serving plugins: loaders, the manifest, namespaces, and each kind over the wire.

Every serving test goes through a real FastMCP ``Client`` against ``plugins/demo``, a
small Claude-layout plugin, or a copy of it changed for the case.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastmcp import Client
from mcp_types import INVALID_PARAMS, METHOD_NOT_FOUND
from plugin_server import (
    DEMO,
    ENTRY,
    Folders,
    agent_md,
    command_md,
    copy,
    listed,
    served,
    skill,
    write,
)
from skill_server import request

from kubed.mcp import QuickMCP
from kubed.mcp.agents import EXTENSION_ID, META_KEY
from kubed.mcp.plugins import (
    AgentFile,
    LocalLoader,
    Plugin,
    PluginError,
    read_plugin,
)
from kubed.mcp.protocol import MCPError
from kubed.mcp.skills import EntryOnlySkillProvider

pytestmark = pytest.mark.unit


async def skill_uris(mcp: QuickMCP) -> list[str]:
    async with Client(mcp.server) as client:
        listed = await client.list_resources()
    return sorted(str(r.uri) for r in listed if str(r.uri).startswith("skill://"))


# ---- the manifest ------------------------------------------------------------------


def test_the_manifest_gives_the_metadata_and_the_conventions_the_paths():
    plugin = read_plugin(DEMO)
    assert (plugin.name, plugin.description, plugin.version) == (
        "demo",
        "A small plugin for the plugin tests.",
        "1.2.3",
    )
    assert plugin.components.skills == ("skills",)
    assert plugin.components.commands == ("commands",)
    assert plugin.components.agents == ("agents",)
    assert plugin.components.notes == ()


def test_a_plugin_with_no_manifest_is_named_by_its_folder(tmp_path):
    root = write(tmp_path / "bare", {"skills/x/SKILL.md": skill("x")})
    plugin = read_plugin(root)
    assert (plugin.name, plugin.description, plugin.version) == ("bare", None, None)
    assert plugin.components.skills == ("skills",)
    assert plugin.components.commands == ()


@pytest.mark.parametrize(
    ("manifest", "reason"),
    [
        ("{not json", "is not readable JSON"),
        ("[]", "must be a JSON object"),
        ('{"skills": 3}', "skills must be a string or a list of strings"),
    ],
)
def test_a_broken_manifest_refuses_the_plugin(tmp_path, manifest, reason):
    root = copy(tmp_path, files={".claude-plugin/plugin.json": manifest})
    with pytest.raises(PluginError, match=reason) as refused:
        QuickMCP("t").add_plugin(root)
    assert str(root) in str(refused.value)


async def test_declared_skills_add_the_rest_replace_and_an_escape_is_dropped(
    tmp_path, caplog
):
    root = copy(
        tmp_path,
        files={
            ".claude-plugin/plugin.json": json.dumps(
                {
                    "name": "demo",
                    "skills": ["./extra", "../outside", "/etc"],
                    "commands": "./cmds/",
                    "agents": ["./team/lead.md", "./../../lead.md"],
                }
            ),
            "extra/more/SKILL.md": skill("more"),
            "cmds/hello.md": "---\ndescription: Hi.\n---\nHi.\n",
            "team/lead.md": "---\nname: lead\ndescription: Leads.\n---\nLead.\n",
        },
    )
    write(tmp_path, {"outside/escaped/SKILL.md": skill("escaped")})
    with caplog.at_level(logging.WARNING):
        mcp = served((root, None))
    plugin = mcp.plugins[0]
    assert plugin.components.skills == ("skills", "extra")
    assert plugin.components.commands == ("cmds",)
    assert plugin.components.agents == ("team/lead.md",)
    assert plugin.skills == (ENTRY, "skill://t/more/SKILL.md")
    assert plugin.prompts == ("hello",)
    assert plugin.agents == ("lead",)
    warned = caplog.text
    assert "skills path '../outside' leaves the plugin root" in warned
    assert "skills path '/etc' leaves the plugin root" in warned
    assert "agents path './../../lead.md' leaves the plugin root" in warned
    assert await skill_uris(mcp) == [ENTRY, "skill://t/more/SKILL.md"]


def test_a_manifest_symlinked_out_of_the_root_is_not_read(tmp_path):
    root = write(tmp_path / "linked", {"skills/x/SKILL.md": skill("x")})
    outside = write(tmp_path, {"foreign.json": '{"name": "foreign"}'}) / "foreign.json"
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "plugin.json").symlink_to(outside)
    assert read_plugin(root).name == "linked"


# ---- the marketplace entry ---------------------------------------------------------
# Merged as Claude Code merges an entry with plugin.json (its marketplace reference,
# "How an entry combines with plugin.json" and "Strict mode").

IGNORED_AGENTS = 'Default agents/ folder is ignored because the manifest sets "agents"'


async def test_an_entry_beside_plugin_json_adds_its_components(tmp_path):
    root = copy(
        tmp_path,
        files={
            "extra/more/SKILL.md": skill("more"),
            "team/lead.md": agent_md("lead"),
        },
    )
    entry = {
        "name": "kit",
        "source": "./",
        "description": "From the entry.",
        "version": "9.9.9",
        "skills": ["./extra"],
        "agents": "./team/lead.md",
    }
    mcp = QuickMCP("t")
    plugin = mcp.add_plugin(root, manifest=entry)
    # The entry's name and description are shown; plugin.json's version wins.
    assert (plugin.name, plugin.description, plugin.version) == (
        "kit",
        "From the entry.",
        "1.2.3",
    )
    assert plugin.components.skills == ("skills", "extra")
    assert plugin.components.agents == ("team/lead.md",)
    assert plugin.components.notes == (IGNORED_AGENTS,)
    assert await listed(mcp) == (
        ["greet"],
        ["lead"],
        [ENTRY, "skill://t/more/SKILL.md"],
    )


def test_an_entrys_paths_follow_plugin_json_s(tmp_path):
    root = copy(
        tmp_path,
        files={
            ".claude-plugin/plugin.json": json.dumps(
                {"name": "demo", "commands": ["./commands/greet.md"]}
            ),
            "more/wave.md": command_md(),
        },
    )
    plugin = read_plugin(root, manifest={"commands": ["./more/wave.md"]})
    assert plugin.components.commands == ("commands/greet.md", "more/wave.md")


async def test_with_no_plugin_json_the_entry_is_the_manifest(tmp_path):
    root = write(
        tmp_path / "bare",
        {
            "skills/x/SKILL.md": skill("x"),
            "agents/helper.md": agent_md("helper"),
            "cmds/only.md": command_md(),
            "commands/hidden.md": command_md(),
        },
    )
    # Whatever strict says.
    entry = {"name": "e", "strict": False, "version": "2.0.0", "commands": "./cmds/"}
    mcp = QuickMCP("t")
    plugin = mcp.add_plugin(root, manifest=entry)
    assert (plugin.name, plugin.description, plugin.version) == ("e", None, "2.0.0")
    assert plugin.components.commands == ("cmds",)
    assert plugin.components.notes == (
        'Default commands/ folder is ignored because the manifest sets "commands"',
    )
    assert await listed(mcp) == (["only"], ["helper"], ["skill://t/x/SKILL.md"])


@pytest.mark.parametrize("field", ["skills", "commands", "agents", "hooks"])
def test_strict_false_naming_a_component_beside_plugin_json_refuses(tmp_path, field):
    root = copy(tmp_path)
    mcp = QuickMCP("t")
    entry = {"name": "demo", "strict": False, field: "./x"}
    with pytest.raises(PluginError, match="conflicting manifests") as refused:
        mcp.add_plugin(root, manifest=entry)
    assert str(root) in str(refused.value)
    assert mcp.plugins == ()


def test_strict_false_naming_no_component_is_plugin_json_alone(tmp_path):
    root = copy(tmp_path)
    entry = {"name": "demo", "strict": False, "description": "Listed."}
    plugin = read_plugin(root, manifest=entry)
    assert plugin.components == read_plugin(root).components
    assert plugin.description == "Listed."


def test_an_entrys_paths_are_published_content(tmp_path):
    root = copy(tmp_path, files={"extra/more/SKILL.md": skill("more")})
    entry = {"skills": ["../outside", "extra", "/etc"], "agents": ["./notes.txt"]}
    plugin = read_plugin(root, manifest=entry)
    assert plugin.components.skills == ("skills", "extra")
    assert plugin.components.agents == ()
    assert plugin.components.notes == (
        "skills path '../outside' leaves the plugin root",
        "skills path 'extra' does not start with ./",
        "skills path '/etc' leaves the plugin root",
        "agents entry 'notes.txt' is not a .md file",
        IGNORED_AGENTS,
    )


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (["./"], "the marketplace entry must be a JSON object"),
        ({"agents": 3}, "agents must be a string or a list of strings"),
    ],
)
def test_a_broken_entry_refuses_the_plugin(tmp_path, entry, reason):
    with pytest.raises(PluginError, match=reason):
        QuickMCP("t").add_plugin(copy(tmp_path), manifest=entry)


# ---- loaders -----------------------------------------------------------------------


def test_a_file_url_is_the_local_loader(tmp_path):
    root = copy(tmp_path)
    plugin = QuickMCP("t").add_plugin(root.as_uri())
    assert plugin.root == root
    assert plugin.address == root.as_uri()


def test_a_missing_folder_is_refused(tmp_path):
    with pytest.raises(PluginError, match="is not a folder"):
        QuickMCP("t").add_plugin(tmp_path / "nope")


def test_an_unknown_scheme_names_add_loader():
    with pytest.raises(PluginError, match=r"git\+https://.*mcp\.add_loader\(\)"):
        QuickMCP("t").add_plugin("git+https://example.com/plugin.git")


async def test_a_custom_loader_is_registered_and_used(tmp_path):
    copy(tmp_path)
    loader = Folders(tmp_path)
    mcp = QuickMCP("t")
    mcp.add_loader(loader)
    plugin = mcp.add_plugin("fixture://demo")
    assert loader.asked == ["fixture://demo"]
    assert plugin.root == tmp_path / "demo"
    assert await skill_uris(mcp) == [ENTRY]


class Recording(LocalLoader):
    def __init__(self):
        self.asked = []

    def materialise(self, address):
        self.asked.append(address)
        return super().materialise(address)


def test_a_servers_file_loader_replaces_the_local_one():
    mcp, loader = QuickMCP("t"), Recording()
    mcp.add_loader(loader)
    mcp.add_plugin(DEMO)
    assert loader.asked == [str(DEMO)]


def test_a_second_loader_for_a_scheme_is_refused(tmp_path):
    mcp = QuickMCP("t")
    mcp.add_loader(Recording())
    with pytest.raises(ValueError, match=r"a loader for file:// is already registered"):
        mcp.add_loader(LocalLoader())
    mcp.add_loader(Folders(tmp_path))
    with pytest.raises(ValueError, match=r"fixture:// is already registered"):
        mcp.add_loader(Folders(tmp_path))


def test_a_refused_loader_replaces_nothing(tmp_path):
    mcp = QuickMCP("t")
    mcp.add_loader(Folders(tmp_path))
    both = Recording()
    both.schemes = ("file", "fixture")
    with pytest.raises(ValueError):
        mcp.add_loader(both)
    mcp.add_plugin(DEMO)
    assert both.asked == []


# ---- namespaces --------------------------------------------------------------------


async def test_no_namespace_is_the_servers_name():
    mcp = served((DEMO, None), name="selenium-flow")
    assert mcp.plugins[0].namespace == "selenium-flow"
    entry = "skill://selenium-flow/demo-guide/SKILL.md"
    assert mcp.plugins[0].skills == (entry,)
    assert await skill_uris(mcp) == [entry]
    async with Client(mcp.server) as client:
        guide = await client.read_resource(
            "skill://selenium-flow/demo-guide/references/GUIDE.md"
        )
        prompts = [p.name for p in await client.list_prompts()]
        detail = await request(client, "agents/get", name="helper")
    assert guide[0].text == (DEMO / "skills/demo-guide/references/GUIDE.md").read_text()
    # The server is already a scope for prompts and agents: no prefix.
    assert prompts == ["greet"]
    assert detail["_meta"][META_KEY]["skills"] == {"demo-guide": entry}


@pytest.mark.parametrize("namespace", ["", " ", "/"])
def test_an_empty_namespace_is_refused(namespace):
    with pytest.raises(ValueError, match="empty"):
        QuickMCP("t").add_plugin(DEMO, namespace=namespace)


async def test_a_skill_is_its_name_under_the_namespace(tmp_path):
    root = write(tmp_path / "flat", {"skills/lint/SKILL.md": skill("lint")})
    assert await skill_uris(served((root, None))) == ["skill://t/lint/SKILL.md"]
    assert await skill_uris(served((root, "lib"))) == ["skill://lib/lint/SKILL.md"]


async def test_a_namespaced_plugin_serves_under_it():
    mcp = served((DEMO, "kit"))
    async with Client(mcp.server) as client:
        listed = await client.list_resources()
        prompts = await client.list_prompts()
        agents = await request(client, "agents/list")
        detail = await request(client, "agents/get", name="kit_helper")
        guide = await client.read_resource("skill://kit/demo-guide/references/GUIDE.md")
    assert [str(r.uri) for r in listed if str(r.uri).startswith("skill://")] == [
        "skill://kit/demo-guide/SKILL.md"
    ]
    assert [p.name for p in prompts] == ["kit_greet"]
    assert [a["name"] for a in agents["agents"]] == ["kit_helper"]
    assert detail["_meta"][META_KEY]["skills"] == {
        "demo-guide": "skill://kit/demo-guide/SKILL.md"
    }
    assert guide[0].text.startswith("# The guide")


async def test_two_plugins_merge_under_one_namespace(tmp_path):
    other = write(
        tmp_path / "other",
        {
            "skills/other-guide/SKILL.md": skill("other-guide"),
            "commands/wave.md": "---\ndescription: Wave.\n---\nWave.\n",
        },
    )
    mcp = served((DEMO, "kit"), (other, "kit"))
    assert [p.name for p in mcp.plugins] == ["demo", "other"]
    assert await skill_uris(mcp) == [
        "skill://kit/demo-guide/SKILL.md",
        "skill://kit/other-guide/SKILL.md",
    ]
    async with Client(mcp.server) as client:
        prompts = sorted(p.name for p in await client.list_prompts())
        listing = await request(client, "skills/list")
    assert prompts == ["kit_greet", "kit_wave"]
    assert sorted(s["uri"] for s in listing["skills"]) == [
        "skill://kit/demo-guide/SKILL.md",
        "skill://kit/other-guide/SKILL.md",
    ]


async def test_a_clash_is_refused_and_serves_nothing(tmp_path):
    clash = write(
        tmp_path / "clash",
        {
            "skills/demo-guide/SKILL.md": skill("demo-guide"),
            "commands/fresh.md": "---\ndescription: Fresh.\n---\nFresh.\n",
        },
    )
    mcp = served((DEMO, "kit"))
    with pytest.raises(PluginError, match=r"skill://kit/demo-guide/SKILL\.md .*demo"):
        mcp.add_plugin(clash, namespace="kit")
    assert [p.name for p in mcp.plugins] == ["demo"]
    async with Client(mcp.server) as client:
        prompts = [p.name for p in await client.list_prompts()]
    assert prompts == ["kit_greet"]


def test_a_prompt_clash_is_refused(tmp_path):
    clash = write(
        tmp_path / "clash",
        {"commands/greet.md": "---\ndescription: Again.\n---\nAgain.\n"},
    )
    mcp = served((DEMO, None))
    with pytest.raises(PluginError, match="prompt greet is already served"):
        mcp.add_plugin(clash)


async def test_another_namespace_does_not_clash():
    mcp = served((DEMO, None), (DEMO, "kit"))
    assert await skill_uris(mcp) == ["skill://kit/demo-guide/SKILL.md", ENTRY]


@pytest.mark.parametrize("namespace", ["..", "a/../b", "a b"])
def test_a_namespace_must_be_a_path_of_names(namespace):
    with pytest.raises(ValueError, match="namespace"):
        QuickMCP("t").add_plugin(DEMO, namespace=namespace)


# ---- skills ------------------------------------------------------------------------


async def test_an_invalid_skill_is_not_served_and_the_rest_is(tmp_path, caplog):
    root = copy(tmp_path, files={"skills/broken/SKILL.md": skill("not-broken")})
    with caplog.at_level(logging.ERROR):
        mcp = served((root, None))
    assert await skill_uris(mcp) == [ENTRY]
    assert "not a valid Agent Skill" in caplog.text
    assert mcp.plugins[0].prompts == ("greet",)


async def test_a_plugin_with_no_skills_has_no_skills_extension(tmp_path):
    root = write(
        tmp_path / "plain", {"commands/a.md": "---\ndescription: A.\n---\nA.\n"}
    )
    async with Client(served((root, None)).server) as client:
        with pytest.raises(MCPError) as listed:
            await request(client, "skills/list")
    assert listed.value.error.code == METHOD_NOT_FOUND


# ---- commands as prompts -----------------------------------------------------------


async def test_a_command_is_listed_with_its_arguments():
    async with Client(served((DEMO, None)).server) as client:
        prompts = await client.list_prompts()
    assert [p.name for p in prompts] == ["greet"]
    assert prompts[0].description == "Greet someone by name."
    assert [(a.name, a.required) for a in prompts[0].arguments] == [
        ("who", True),
        ("how", False),
    ]


async def test_a_command_renders_its_placeholders_and_defaults():
    async with Client(served((DEMO, None)).server) as client:
        result = await client.get_prompt("greet", {"who": "Dr K"})
    assert result.messages[0].content.text == (
        'Greet Dr K warmly. A flow\'s ${param} and {"json": 1} stay as written.\n'
    )


async def test_a_required_argument_is_required():
    async with Client(served((DEMO, None)).server) as client:
        with pytest.raises(MCPError, match="Missing required arguments: who"):
            await client.get_prompt("greet", {})


async def test_the_invocation_frontmatter_is_kept():
    prompt = await served((DEMO, None)).server.get_prompt("greet")
    assert prompt.frontmatter["disable-model-invocation"] is True
    assert prompt.frontmatter["argument-hint"] == "<who> [how]"


async def test_an_undeclared_placeholder_refuses_the_command(tmp_path, caplog):
    root = copy(
        tmp_path,
        files={"commands/typo.md": "---\ndescription: Typo.\n---\nHi {{ whom }}.\n"},
    )
    with caplog.at_level(logging.WARNING):
        mcp = served((root, None))
    async with Client(mcp.server) as client:
        prompts = [p.name for p in await client.list_prompts()]
    assert prompts == ["greet"]
    assert "placeholders with no argument: whom" in caplog.text


# ---- agents ------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_the_agents_extension_is_declared(mode):
    async with Client(served((DEMO, None)).server, mode=mode) as client:
        extensions = client.session.server_capabilities.extensions or {}
    assert extensions[EXTENSION_ID] == {"listChanged": False}


async def test_agents_list_gives_the_cards():
    async with Client(served((DEMO, None)).server) as client:
        listing = await request(client, "agents/list")
    assert listing["resultType"] == "complete"
    assert listing["agents"] == [
        {
            "name": "helper",
            "description": "Helps with the demo plugin.",
            "capabilities": [],
        }
    ]


async def test_agents_get_gives_the_definition():
    async with Client(served((DEMO, None)).server) as client:
        detail = await request(client, "agents/get", name="helper")
    assert detail["agent"] == "helper"
    assert detail["instructions"] == "Read skill://demo-guide/SKILL.md first."
    assert detail["modelRoutingHint"] == {"role": "medium", "vision": False}
    assert detail["tools"] == []
    meta = detail["_meta"][META_KEY]
    assert meta["frontmatter"]["tools"] == ["ReadMcpResourceTool"]
    assert meta["skills"] == {"demo-guide": ENTRY}


async def test_an_unknown_agent_is_not_found():
    async with Client(served((DEMO, None)).server) as client:
        with pytest.raises(MCPError) as missing:
            await request(client, "agents/get", name="nobody")
    assert missing.value.error.code == INVALID_PARAMS
    assert missing.value.error.data == {"reason": "agent_not_found"}


async def test_a_server_with_no_plugin_has_no_agents_extension():
    async with Client(QuickMCP("t").server) as client:
        with pytest.raises(MCPError) as listed:
            await request(client, "agents/list")
    assert listed.value.error.code == METHOD_NOT_FOUND


async def test_two_plugins_agents_share_one_listing(tmp_path):
    other = write(
        tmp_path / "other",
        {"agents/scout.md": "---\nname: scout\ndescription: Scouts.\n---\nGo.\n"},
    )
    async with Client(served((DEMO, None), (other, None)).server) as client:
        listing = await request(client, "agents/list")
    assert [a["name"] for a in listing["agents"]] == ["helper", "scout"]


def test_the_registry_is_the_plugins_served():
    mcp = served((DEMO, None))
    assert mcp.plugins == (
        Plugin(
            name="demo",
            description="A small plugin for the plugin tests.",
            version="1.2.3",
            root=DEMO,
            components=read_plugin(DEMO).components,
            address=str(DEMO),
            namespace="t",
            skills=(ENTRY,),
            prompts=("greet",),
            agents=("helper",),
            agent_files=(AgentFile("helper", "agents/helper.md"),),
        ),
    )


async def test_one_provider_of_each_kind_serves_the_plugins_as_served(tmp_path):
    other = write(
        tmp_path / "other",
        {
            "skills/other-guide/SKILL.md": skill("other-guide"),
            "commands/wave.md": command_md(),
            "agents/scout.md": agent_md("scout"),
        },
    )
    mcp = served((DEMO, None), (other, None))
    providers = [
        p for p in mcp.server.providers if isinstance(p, EntryOnlySkillProvider)
    ]
    assert len(providers) == 1
    assert await mcp.server.local_provider.list_prompts() == []
    assert await listed(mcp) == (
        ["greet", "wave"],
        ["helper", "scout"],
        [ENTRY, "skill://t/other-guide/SKILL.md"],
    )
    # What is served is the served set, read on each call: a set without the second
    # plugin serves none of it.
    staging = mcp.plugin_set()
    staging.keep(mcp.plugins[0])
    mcp.serve(staging)
    assert await listed(mcp) == (["greet"], ["helper"], [ENTRY])
    assert [p.name for p in mcp.plugins] == ["demo"]


async def test_what_a_plugin_does_not_serve_is_skipped_with_the_reason(tmp_path):
    root = write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": '{"name": "p", "skills": ["./extra"]}',
            "skills/good/SKILL.md": skill("good"),
            "skills/wrong/SKILL.md": skill("other"),
            "extra/good/SKILL.md": skill("good"),
            "commands/typo.md": "---\ndescription: T.\n---\nHi {{ who }}.\n",
            "commands/a:b/c.md": command_md(),
            "agents/review/security.md": agent_md("security"),
            "agents/review_security.md": agent_md(None),
            "agents/vague.md": "---\nname: vague\n---\nGo.\n",
        },
    )
    mcp = QuickMCP("t")
    plugin = mcp.add_plugin(root)
    assert plugin.skipped == (
        ("skills/wrong", "Skill folder 'wrong' must match its SKILL.md name 'other'"),
        ("extra/good", "skill://t/good/SKILL.md is already served by skills/good"),
        ("commands/a:b/c.md", "prompt name 'a:b_c' is not a usable name"),
        ("commands/typo.md", "placeholders with no argument: who"),
        (
            "agents/review_security.md",
            "agent review_security is already served by agents/review/security.md",
        ),
        ("agents/vague.md", "its description must be a non-empty string"),
    )
    assert mcp.plugins[0].skipped == plugin.skipped
    assert await listed(mcp) == ([], ["review_security"], ["skill://t/good/SKILL.md"])
    assert served((DEMO, None)).plugins[0].skipped == ()


# ---- layout and names, as Claude Code reads them ---------------------------------
# The fixtures are shaped like Claude Code's own --plugin-dir probes (material
# delivery F63): agents and commands at any depth, each folder a `_` segment where
# Claude Code writes `:`; skills one level below skills/.


PROBE = {
    "agents/flat.md": agent_md("flat"),
    "agents/review/security.md": agent_md("security"),
    "agents/explorer/AGENT.md": agent_md("explorer"),
    "agents/bare/AGENT.md": agent_md(None),
    "agents/noname.md": agent_md(None),
    "agents/x@y/r.md": agent_md("r"),
    "commands/flat.md": command_md(),
    "commands/db/migrate.md": command_md(),
    "commands/a/b/deep.md": command_md(),
    "commands/a:b/c.md": command_md(),
    "skills/top/SKILL.md": skill("top"),
    "skills/group/nested/SKILL.md": skill("nested"),
    "skills/wrong/SKILL.md": skill("other"),
}


async def test_the_probe_plugin_is_named_as_claude_code_names_it(tmp_path, caplog):
    root = write(tmp_path / "probe", PROBE)
    with caplog.at_level(logging.WARNING):
        mcp = served((root, None))
    assert await listed(mcp) == (
        ["a_b_deep", "db_migrate", "flat"],
        ["bare_AGENT", "explorer_explorer", "flat", "noname", "review_security"],
        ["skill://t/top/SKILL.md"],
    )
    plugin = mcp.plugins[0]
    # Path order: a/b/deep.md before db/migrate.md before flat.md.
    assert plugin.prompts == ("a_b_deep", "db_migrate", "flat")
    assert plugin.components.notes == (
        "skills/group/nested/SKILL.md is not served: a skill is one folder below "
        "its skills folder",
    )
    warned = caplog.text
    assert "agent name 'x@y_r' is not a usable name" in warned
    assert "prompt name 'a:b_c' is not a usable name" in warned
    assert "not a valid Agent Skill" in warned  # skills/wrong names itself "other"


async def test_under_another_namespace_the_names_take_its_prefix(tmp_path):
    root = write(tmp_path / "probe", PROBE)
    assert await listed(served((root, "kb"))) == (
        ["kb_a_b_deep", "kb_db_migrate", "kb_flat"],
        [
            "kb_bare_AGENT",
            "kb_explorer_explorer",
            "kb_flat",
            "kb_noname",
            "kb_review_security",
        ],
        ["skill://kb/top/SKILL.md"],
    )


async def test_an_unusable_name_claims_nothing(tmp_path):
    root = write(
        tmp_path / "p",
        {
            "agents/x@y/r.md": agent_md("r"),
            "agents/r.md": agent_md("r", "The usable one."),
        },
    )
    mcp = served((root, None))
    assert mcp.plugins[0].agents == ("r",)


@pytest.mark.parametrize("bad", ["a b", "a:b", "a@b"])
async def test_a_frontmatter_name_must_be_usable(tmp_path, bad, caplog):
    root = write(tmp_path / "p", {"agents/x.md": agent_md(f"'{bad}'")})
    with caplog.at_level(logging.WARNING):
        mcp = served((root, None))
    assert mcp.plugins[0].agents == ()
    assert f"agent name {bad!r} is not a usable name" in caplog.text


async def test_the_first_in_path_order_wins_a_joined_name(tmp_path, caplog):
    root = write(
        tmp_path / "p",
        {
            "agents/review/security.md": agent_md("security", "Nested."),
            "agents/review_security.md": agent_md("review_security", "Flat."),
            "commands/db/migrate.md": command_md("Nested."),
            "commands/db_migrate.md": command_md("Flat."),
        },
    )
    with caplog.at_level(logging.WARNING):
        mcp = served((root, None))
    async with Client(mcp.server) as client:
        prompts = await client.list_prompts()
        agents = (await request(client, "agents/list"))["agents"]
    assert [(p.name, p.description) for p in prompts] == [("db_migrate", "Nested.")]
    assert [(a["name"], a["description"]) for a in agents] == [
        ("review_security", "Nested.")
    ]
    base = root.resolve()
    assert (
        f"agent review_security is already served by "
        f"{base / 'agents/review/security.md'}; "
        f"{base / 'agents/review_security.md'} is skipped"
    ) in caplog.text
    assert (
        f"prompt db_migrate is already served by {base / 'commands/db/migrate.md'}; "
        f"{base / 'commands/db_migrate.md'} is skipped"
    ) in caplog.text


async def test_a_clash_across_plugins_is_still_refused(tmp_path):
    first = write(tmp_path / "one", {"commands/db/migrate.md": command_md()})
    second = write(tmp_path / "two", {"commands/db_migrate.md": command_md()})
    mcp = served((first, None))
    with pytest.raises(PluginError, match="prompt db_migrate is already served"):
        mcp.add_plugin(second)


def manifest(root: Path, data: dict) -> Path:
    return write(root, {".claude-plugin/plugin.json": json.dumps(data)})


async def test_a_manifest_skills_key_adds_to_the_default_scan(tmp_path):
    root = write(
        tmp_path / "p",
        {"skills/top/SKILL.md": skill("top"), "more/extra/SKILL.md": skill("extra")},
    )
    mcp = served((manifest(root, {"skills": "./more"}), None))
    assert mcp.plugins[0].components.skills == ("skills", "more")
    assert mcp.plugins[0].components.notes == ()
    assert await skill_uris(mcp) == [
        "skill://t/extra/SKILL.md",
        "skill://t/top/SKILL.md",
    ]


IGNORED_AGENTS = 'Default agents/ folder is ignored because the manifest sets "agents"'
IGNORED_COMMANDS = (
    'Default commands/ folder is ignored because the manifest sets "commands"'
)


@pytest.mark.parametrize(
    ("agents", "served_names", "notes"),
    [
        # A listed file replaces the folder, and takes no folder segment.
        (["./team/lead.md"], ("lead",), (IGNORED_AGENTS,)),
        # A folder is never an agents entry.
        ("./team", (), ("agents entry 'team' is not a .md file", IGNORED_AGENTS)),
        # A file inside agents/ reads the folder for that file: no note.
        (["./agents/review/security.md"], ("security",), ()),
    ],
)
async def test_a_manifest_agents_key_replaces_the_folder(
    tmp_path, agents, served_names, notes
):
    root = write(
        tmp_path / "p",
        {
            "agents/flat.md": agent_md("flat"),
            "agents/review/security.md": agent_md("security"),
            "team/lead.md": agent_md("lead"),
            "team/sub/deputy.md": agent_md("deputy"),
        },
    )
    mcp = served((manifest(root, {"agents": agents}), None))
    assert mcp.plugins[0].agents == served_names
    assert mcp.plugins[0].components.notes == notes
    assert (await listed(mcp))[1] == sorted(served_names)


@pytest.mark.parametrize(
    ("commands", "served_names", "notes"),
    [
        # A folder is read at any depth, its own subfolders the segments.
        ("./cmds", ("x_y", "z"), (IGNORED_COMMANDS,)),
        (["./cmds/z.md"], ("z",), (IGNORED_COMMANDS,)),
        # The default folder named outright reads as the default: no note.
        ("./commands/", ("db_migrate", "flat"), ()),
    ],
)
async def test_a_manifest_commands_key_replaces_the_folder(
    tmp_path, commands, served_names, notes
):
    root = write(
        tmp_path / "p",
        {
            "commands/flat.md": command_md(),
            "commands/db/migrate.md": command_md(),
            "cmds/x/y.md": command_md(),
            "cmds/z.md": command_md(),
        },
    )
    mcp = served((manifest(root, {"commands": commands}), None))
    assert mcp.plugins[0].prompts == served_names
    assert mcp.plugins[0].components.notes == notes
    assert (await listed(mcp))[0] == sorted(served_names)


async def test_a_dot_folder_under_commands_is_not_read(tmp_path):
    root = write(
        tmp_path / "p",
        {"commands/.drafts/x.md": command_md(), "commands/y.md": command_md()},
    )
    assert served((root, None)).plugins[0].prompts == ("y",)
