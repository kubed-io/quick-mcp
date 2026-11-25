"""Helpers shared by the gateway tests: a plugin full of addresses, and a client over HTTP."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from fastmcp import Client
from fastmcp.utilities.tests import asgi_server
from mcp_types import Implementation
from plugin_server import write

from kubed.mcp import QuickMCP

KIT_TREE = {
    "skills/guide/SKILL.md": (
        "---\nname: guide\ndescription: Start at skill://lib/guide/SKILL.md.\n---\n\n"
        "# Guide\n\nRead skill://lib/guide/references/a.md, then "
        "skill://lib/guide/references/../SKILL.md. Not myskill://x, and not "
        "kb+skill://lib/guide/SKILL.md again.\n"
    ),
    "skills/guide/references/a.md": "# A\n\nBack to skill://lib/guide/SKILL.md.\n",
    "commands/ask.md": (
        "---\ndescription: Ask.\narguments:\n- name: topic\n  required: true\n---\n"
        "Ask about {{ topic }}; see skill://lib/guide/SKILL.md and prompts://lib/ask.md.\n"
    ),
    "agents/scout.md": (
        "---\nname: scout\ndescription: Scouts agents://lib/scout.md.\nskills: [guide]\n---\n"
        "Read agents://lib/scout.md and skill://lib/guide/SKILL.md.\n"
    ),
}


def kit(tmp_path: Path, header_prefix: str | None = None) -> QuickMCP:
    """A server ``t`` serving the plugin ``kit`` under the namespace ``lib``."""
    mcp = QuickMCP("t", header_prefix=header_prefix)
    mcp.add_plugin(write(tmp_path / "kit", KIT_TREE), namespace="lib")
    return mcp


@asynccontextmanager
async def over_http(
    mcp: QuickMCP,
    header: str | None = None,
    query: str | None = None,
    mode: str = "auto",
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    client_info: Implementation | None = None,
    extensions: list | None = None,
):
    """An open client over real HTTP, naming a prefix by header and/or query.

    ``headers`` and ``params`` send any others: a header as given, a parameter URL-encoded
    (``query`` is kept as sent). ``client_info`` is the client's ``clientInfo``, FastMCP's
    own when None, and ``extensions`` the client extensions it declares.

    ``mode="legacy"`` pins the handshake era, where a connection outlives a request:
    the default era makes every request its own connection.
    """
    async with asgi_server(mcp.server) as running:
        sent = {} if header is None else {"X-MCP-Resource-Prefix": header}
        transport = running.transport(headers={**sent, **(headers or {})})
        pairs = [] if query is None else [f"prefix={query}"]
        if params:
            pairs.append(urlencode(params))
        if pairs:
            transport.url = f"{running.url}?{'&'.join(pairs)}"
        async with Client(
            transport, mode=mode, client_info=client_info, extensions=extensions
        ) as client:
            yield client
