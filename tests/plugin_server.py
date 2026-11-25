"""Helpers shared by the plugin tests: the demo plugin, copies of it, and what a client is served."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastmcp import Client
from skill_server import request

from kubed.mcp import QuickMCP

DEMO = Path(__file__).parent / "plugins" / "demo"
# A plugin added with no namespace serves under the server's name, "t" here.
ENTRY = "skill://t/demo-guide/SKILL.md"
GUIDE = "skill://t/demo-guide/references/GUIDE.md"


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


def agent_md(name: str | None, description: str = "An agent.") -> str:
    named = f"name: {name}\n" if name else ""
    return f"---\n{named}description: {description}\n---\nGo.\n"


def command_md(description: str = "A command.") -> str:
    return f"---\ndescription: {description}\n---\nDo it.\n"


async def listed(mcp: QuickMCP) -> tuple[list[str], list[str], list[str]]:
    """The prompt names, agent names and skill URIs a client is served."""
    async with Client(mcp.server) as client:
        return await listed_by(client)


async def listed_by(client: Client) -> tuple[list[str], list[str], list[str]]:
    """The prompt names, agent names and skill URIs an open client is served."""
    prompts = sorted(p.name for p in await client.list_prompts())
    agents = [a["name"] for a in (await request(client, "agents/list"))["agents"]]
    resources = await client.list_resources()
    uris = sorted(str(r.uri) for r in resources if str(r.uri).startswith("skill://"))
    return prompts, agents, uris


class Folders:
    """A loader for ``fixture://<name>``, a folder under a base."""

    schemes = ("fixture",)

    def __init__(self, base: Path) -> None:
        self.base = base
        self.asked: list[str] = []

    def materialise(self, address: str) -> Path:
        self.asked.append(address)
        return self.base / address.removeprefix("fixture://")
