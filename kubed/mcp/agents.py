"""A plugin's agents, served through the MCP agents extension.

``io.modelcontextprotocol/agents`` is the draft Agent Discovery extension (agents-wg
proposal 0000): ``agents/list`` gives a card per agent and ``agents/get`` one agent's
definition. The server serves definitions; the host runs them. An agent is never a
resource, a prompt or a tool.

``io.kubed/agent`` in a definition's ``_meta`` is the shape every kubed server shares:
the frontmatter as written, and the ``skill://`` address of each skill it names that
the server serves.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml
from fastmcp.server.extensions import MethodBinding, ServerExtension
from mcp_types import INVALID_PARAMS, PaginatedRequestParams, RequestParams
from pydantic import Field

from mcp.shared.exceptions import MCPError

from .prompts import split_frontmatter

__all__ = ["EXTENSION_ID", "META_KEY", "Agent", "AgentsExtension", "parse_agent"]

EXTENSION_ID = "io.modelcontextprotocol/agents"
META_KEY = "io.kubed/agent"
# No notifications/agents/list_changed is ever sent.
SETTINGS = {"listChanged": False}
AGENT_NOT_FOUND = {"reason": "agent_not_found"}

_HINTS = {"opus": "large", "sonnet": "medium", "haiku": "small"}


@dataclass(frozen=True)
class Agent:
    name: str
    card: dict  # what agents/list sends
    detail: dict  # what agents/get sends


def usable_name(text: object) -> bool:
    """Whether ``text`` can be a served name: one non-empty token.

    No whitespace, no control or other invisible character, no ``/`` (a path), no
    ``@`` and no ``:`` (Claude Code's segment separator).
    """
    return (
        isinstance(text, str)
        and bool(text)
        and not any(
            c.isspace() or c in "/@:" or unicodedata.category(c).startswith("C")
            for c in text
        )
    )


def routing_hint(model: object) -> dict | None:
    """A hint for the host's model choice, from a model name, or None."""
    if not isinstance(model, str):
        return None
    for word in re.split(r"[^a-z0-9]+", model.lower()):
        if word in _HINTS:
            return {"role": _HINTS[word], "vision": False}
    return None


def _survives_json(value: object) -> bool:
    """Whether the value arrives as written after a trip through JSON.

    A YAML-only value fails to encode, and a non-string key (1 beside "1") encodes but
    comes back renamed or lost.
    """
    try:
        return json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError):
        return False


def _strings(value: object) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def parse_agent(
    text: str,
    skills: Mapping[str, str] | None = None,
    prefix: str = "",
    *,
    stem: str | None = None,
    folders: Sequence[str] = (),
) -> Agent | str:
    """The agent an agent file defines, or the sentence saying why it is skipped.

    ``skills`` maps a skill's name to the ``skill://`` URI of its ``SKILL.md``: a name
    in the frontmatter's ``skills`` is resolved through it, and left out when it is not
    there. The agent's own segment is the frontmatter's ``name``, else ``stem``, the
    file's. Its name is ``prefix``, then ``folders`` and that segment joined with
    ``_``; each segment and the whole must be a usable name.
    """
    try:
        parsed = split_frontmatter(text)
    except yaml.YAMLError:
        parsed = None
    if parsed is None:
        return "no YAML frontmatter"
    header, body = parsed
    if not isinstance(header, dict) or not header:
        return "its frontmatter is not a non-empty mapping"
    if not _survives_json(header):
        return "its frontmatter is not representable as JSON"
    own = header.get("name")
    if own is None:
        own = stem
    if own is None:
        return "it has no name"
    if not isinstance(own, str) or not own:
        return "its name must be a non-empty string"
    description = header.get("description")
    if not isinstance(description, str) or not description.strip():
        return "its description must be a non-empty string"
    segments = (*folders, own)
    name = prefix + "_".join(segments)
    if not all(usable_name(part) for part in (*segments, name)):
        return f"agent name {name!r} is not a usable name"

    served = skills or {}
    card = {
        "name": name,
        "description": " ".join(description.split()),
        "capabilities": _strings(header.get("capabilities")),
    }
    detail: dict[str, Any] = {"agent": name}
    instructions = "\n".join(body.splitlines()).strip()
    if instructions:
        detail["instructions"] = instructions
    hint = routing_hint(header.get("model"))
    if hint:
        detail["modelRoutingHint"] = hint
    detail["tools"] = []
    detail["_meta"] = {
        META_KEY: {
            "frontmatter": header,
            "skills": {
                s: served[s] for s in _strings(header.get("skills")) if s in served
            },
        }
    }
    return Agent(name=name, card=card, detail=detail)


def complete(**fields: object) -> dict:
    """An extension result in the envelope the Skills extension's results share."""
    return {"resultType": "complete", "ttlMs": 0, "cacheScope": "private", **fields}


class GetAgentParams(RequestParams):
    name: str = Field(min_length=1)


class AgentsExtension(ServerExtension):
    """``agents/list`` and ``agents/get`` over a list of agents, in name order.

    The list is read on every request, so agents added to it after the extension is
    registered are served too. A subclass serves agents from elsewhere by overriding
    ``_served()``, or ``_agents()`` when finding them takes an await.
    """

    identifier = EXTENSION_ID

    def __init__(self, agents: Sequence[Agent] | None = None) -> None:
        self.agents: list[Agent] = list(agents or ())

    def settings(self) -> dict:
        return dict(SETTINGS)

    def methods(self) -> Sequence[MethodBinding]:
        return (
            MethodBinding("agents/list", PaginatedRequestParams, self.list_agents),
            MethodBinding("agents/get", GetAgentParams, self.get_agent),
        )

    def clone(self) -> AgentsExtension:
        # The same list, so a clone serves what is added to it later too.
        twin = type(self)()
        twin.agents = self.agents
        return twin

    def _served(self) -> Sequence[Agent]:
        """The agents this request is served, read once per call."""
        return self.agents

    async def _agents(self) -> Sequence[Agent]:
        """The agents this request is served: ``_served()``, unless overridden."""
        return self._served()

    async def list_agents(self, ctx: Any, params: PaginatedRequestParams) -> dict:
        if params.cursor is not None:
            # One page, so no cursor was ever issued.
            raise MCPError(code=INVALID_PARAMS, message="Invalid agents cursor")
        agents = sorted(await self._agents(), key=lambda agent: agent.name)
        return complete(agents=[copy.deepcopy(a.card) for a in agents])

    async def get_agent(self, ctx: Any, params: GetAgentParams) -> dict:
        for agent in await self._agents():
            if agent.name == params.name:
                return complete(**copy.deepcopy(agent.detail))
        raise MCPError(
            code=INVALID_PARAMS,
            message="Agent not found",
            data=dict(AGENT_NOT_FOUND),
        )
