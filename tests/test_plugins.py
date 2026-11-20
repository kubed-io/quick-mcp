"""Serving plugins: loaders, the manifest, namespaces, and each kind over the wire.

Every serving test goes through a real FastMCP ``Client`` against ``plugins/demo``, a
small Claude-layout plugin, or a copy of it changed for the case.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
from fastmcp import Client
from mcp_types import INVALID_PARAMS, METHOD_NOT_FOUND
from skill_server import request

from kubed.mcp import QuickMCP
from kubed.mcp.agents import EXTENSION_ID, META_KEY
from kubed.mcp.plugins import LocalLoader, Plugin, PluginError, read_plugin
from kubed.mcp.protocol import MCPError

pytestmark = pytest.mark.unit

DEMO = Path(__file__).parent / "plugins" / "demo"
# A plugin added with no namespace serves under the server's name, "t" here.
ENTRY = "skill://t/demo-guide/SKILL.md"
GUIDE = "skill://t/demo-guide/references/GUIDE.md"
ROOT_ENTRY = "skill://demo-guide/SKILL.md"


def write(root: Path, files: dict[str, str]) -> Path:
    for path, body in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(body, encoding="utf-8")
    return root


def skill(name: str) -> str:
    return f"---\nname: {name}\ndescription: The {name} skill.\n---\n\n# {name}\n"


def copy(tmp_path: Path, name: str = "demo", files: dict | None = None) -> Path:
    """A copy of the demo plugin, with ``files`` written over it."""
    root = tmp_path / name
    shutil.copytree(DEMO, root)
    return write(root, files or {})


def served(*plugins: tuple[Path, str | None], name: str = "t") -> QuickMCP:
    mcp = QuickMCP(name)
    for root, namespace in plugins:
        mcp.add_plugin(root, namespace=namespace)
    return mcp


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


async def test_declared_paths_replace_the_conventions_and_an_escape_is_dropped(
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
    assert plugin.components.skills == ("extra",)
    assert plugin.components.commands == ("cmds",)
    assert plugin.components.agents == ("team/lead.md",)
    assert plugin.skills == ("skill://t/more/SKILL.md",)
    assert plugin.prompts == ("hello",)
    assert plugin.agents == ("lead",)
    warned = caplog.text
    assert "skills path '../outside' leaves the plugin root" in warned
    assert "skills path '/etc' leaves the plugin root" in warned
    assert "agents path './../../lead.md' leaves the plugin root" in warned
    assert await skill_uris(mcp) == ["skill://t/more/SKILL.md"]


def test_a_manifest_symlinked_out_of_the_root_is_not_read(tmp_path):
    root = write(tmp_path / "linked", {"skills/x/SKILL.md": skill("x")})
    outside = write(tmp_path, {"foreign.json": '{"name": "foreign"}'}) / "foreign.json"
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "plugin.json").symlink_to(outside)
    assert read_plugin(root).name == "linked"


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


class Folders:
    """A loader for ``fixture://<name>``, a folder under a base."""

    schemes = ("fixture",)

    def __init__(self, base: Path) -> None:
        self.base = base
        self.asked: list[str] = []

    def materialise(self, address: str) -> Path:
        self.asked.append(address)
        return self.base / address.removeprefix("fixture://")


async def test_a_custom_loader_is_registered_and_used(tmp_path):
    copy(tmp_path)
    loader = Folders(tmp_path)
    mcp = QuickMCP("t")
    mcp.add_loader(loader)
    plugin = mcp.add_plugin("fixture://demo")
    assert loader.asked == ["fixture://demo"]
    assert plugin.root == tmp_path / "demo"
    assert await skill_uris(mcp) == [ENTRY]


def test_a_scheme_takes_one_loader(tmp_path):
    mcp = QuickMCP("t")
    with pytest.raises(ValueError, match="already registered"):
        mcp.add_loader(LocalLoader())


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


async def test_an_empty_namespace_is_the_root():
    mcp = served((DEMO, ""))
    assert mcp.plugins[0].namespace == ""
    assert mcp.plugins[0].skills == (ROOT_ENTRY,)
    assert await skill_uris(mcp) == [ROOT_ENTRY]
    async with Client(mcp.server) as client:
        guide = await client.read_resource("skill://demo-guide/references/GUIDE.md")
        prompts = [p.name for p in await client.list_prompts()]
        detail = await request(client, "agents/get", name="helper")
    assert guide[0].text.startswith("# The guide")
    assert prompts == ["greet"]
    assert detail["_meta"][META_KEY]["skills"] == {"demo-guide": ROOT_ENTRY}


async def test_a_skill_keeps_its_path_under_skills(tmp_path):
    root = write(tmp_path / "deep", {"skills/tools/lint/SKILL.md": skill("lint")})
    assert await skill_uris(served((root, None))) == ["skill://t/tools/lint/SKILL.md"]
    assert await skill_uris(served((root, ""))) == ["skill://tools/lint/SKILL.md"]
    assert await skill_uris(served((root, "lib"))) == [
        "skill://lib/tools/lint/SKILL.md"
    ]


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


def test_the_root_and_the_servers_name_share_prompt_names():
    """Neither prefixes a prompt, so the same command in both clashes."""
    mcp = served((DEMO, None))
    with pytest.raises(PluginError, match="prompt greet is already served"):
        mcp.add_plugin(DEMO, namespace="")


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
        ),
    )
