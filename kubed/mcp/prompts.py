"""A plugin's commands, served as MCP prompts.

A command file is YAML frontmatter and a Markdown body. Over MCP a person picks a prompt
and fills in its arguments before the model sees anything, so the frontmatter declares
them and the body names them as ``{{ name }}`` placeholders. Double braces, because a
body is free to hold ``${name}`` references and JSON, which are single braces.

Loading is strict: an undeclared placeholder is almost always a typo, and rendered blank
it asks the model for the wrong thing, so the file is refused. The frontmatter is kept
as authored on the prompt, the keys Claude Code reads for invocation
(``disable-model-invocation``, ``user-invocable``, ``argument-hint``) included; the MCP
prompt ignores them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fastmcp.exceptions import PromptError
from fastmcp.prompts import Prompt, PromptArgument
from pydantic import Field

if TYPE_CHECKING:
    from .plugins import Plugin

__all__ = [
    "INVOCATION_KEYS",
    "META_KEY",
    "FilePrompt",
    "index_meta",
    "load_prompt",
    "read_command",
]

# Where a prompt's row in its folder's index carries its registration (PLUG-69).
META_KEY = "io.kubed/prompt"
# The frontmatter keys Claude Code reads for invocation, kept on a prompt's row.
INVOCATION_KEYS = ("disable-model-invocation", "user-invocable", "argument-hint")

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def split_frontmatter(text: str) -> tuple[Any, str] | None:
    """The header as YAML parses it, and the body, or None without a header.

    The header opens only on a first line that is ``---``, trailing whitespace allowed,
    and ends at the next such line, so a ``---`` inside a value is no end. The body is
    every line after it, as written, less the blank lines that open it. A header whose
    YAML does not parse raises ``yaml.YAMLError``.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != "---":
        return None
    end = next(
        (n for n, line in enumerate(lines[1:], start=1) if line.rstrip() == "---"),
        None,
    )
    if end is None:
        return None
    header = yaml.safe_load("".join(lines[1:end]))
    return header, "".join(lines[end + 1 :]).lstrip("\r\n")


def _survives_json(value: object) -> bool:
    """Whether the value arrives as written after a trip through JSON.

    A YAML-only value fails to encode, and a non-string key (1 beside "1") encodes but
    comes back renamed or lost.
    """
    try:
        return json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError):
        return False


def index_meta(prompt: Prompt, text: str) -> dict[str, dict[str, Any]]:
    """What a prompt's row registers it by (PLUG-69): ``{META_KEY: ...}`` with its MCP
    ``name``, its ``arguments`` as ``prompts/list`` gives them, and each invocation
    key the frontmatter of ``text``, the file as served, sets as written.

    The keys are read from ``text`` and not the prompt, so a reader's ``Prompt``
    (PLUG-62) has them too. A text with no header, a header that does not parse or is
    not a mapping, gives none, and a value that does not survive a trip through JSON
    is left out.
    """
    arguments = [
        argument.model_dump(mode="json", by_alias=True, exclude_none=True)
        for argument in prompt.to_mcp_prompt().arguments or []
    ]
    try:
        split = split_frontmatter(text)
    except yaml.YAMLError:
        split = None
    header = split[0] if split is not None else None
    keys: dict[str, Any] = {}
    if isinstance(header, dict):
        keys = {
            key: header[key]
            for key in INVOCATION_KEYS
            if key in header and _survives_json(header[key])
        }
    return {META_KEY: {"name": prompt.name, "arguments": arguments, **keys}}


class FilePrompt(Prompt):
    """One command file, rendered by substituting its placeholders."""

    template: str
    defaults: dict[str, str] = Field(default_factory=dict)
    frontmatter: dict[str, Any] = Field(default_factory=dict)

    async def render(self, arguments: dict | None = None) -> str:
        # A field left blank arrives as "" from most prompt pickers, so it counts as
        # not given: a default applies, and a required argument still fails.
        given = {
            key: str(value)
            for key, value in (arguments or {}).items()
            if value not in (None, "")
        }
        missing = [
            argument.name
            for argument in self.arguments or []
            if argument.required and argument.name not in given
        ]
        if missing:
            raise PromptError(f"Missing required arguments: {', '.join(missing)}")
        values = {**self.defaults, **given}
        return PLACEHOLDER.sub(
            lambda match: values.get(match.group(1), ""), self.template
        )


def load_prompt(path: Path, name: str | None = None) -> FilePrompt:
    """The command file at ``path`` as a prompt named ``name`` (its stem by default).

    Raises ``ValueError`` on anything malformed, ``yaml.YAMLError`` on a header that
    does not parse, and ``OSError`` or ``UnicodeDecodeError`` on a file that cannot be
    read.
    """
    return parse_prompt(path.read_text(encoding="utf-8"), name or path.stem)


def read_command(text: str, path: str, name: str, plugin: Plugin) -> FilePrompt | str:
    """The default ``command_reader`` (PLUG-62): a command file's ``text`` as a prompt
    named ``name``, or the reason it is not one.

    ``path`` and ``plugin`` are a reader's other arguments, unused here: the file is
    read by its text alone, as ``parse_prompt`` reads it.
    """
    try:
        return parse_prompt(text, name)
    except (ValueError, yaml.YAMLError) as exc:
        return str(exc)


def parse_prompt(text: str, name: str) -> FilePrompt:
    """A command file's ``text`` as a prompt named ``name``.

    Raises ``ValueError`` on anything malformed and ``yaml.YAMLError`` on a header that
    does not parse.
    """
    split = split_frontmatter(text)
    if split is None:
        raise ValueError("no YAML frontmatter")
    meta, body = split
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        # Published with no metadata, it would be an unusable picker entry.
        raise ValueError(f"frontmatter must be a mapping, got {type(meta).__name__}")

    declared = meta.get("arguments") or []
    if not isinstance(declared, list):
        raise ValueError(f"arguments must be a list, got {type(declared).__name__}")

    arguments: list[PromptArgument] = []
    defaults: dict[str, str] = {}
    for raw in declared:
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ValueError(f"argument without a name: {raw!r}")
        argument = str(raw["name"])
        required = bool(raw.get("required", False))
        if required and "default" in raw:
            raise ValueError(f"required argument {argument!r} cannot have a default")
        arguments.append(
            PromptArgument(
                name=argument, description=raw.get("description"), required=required
            )
        )
        if "default" in raw:
            defaults[argument] = str(raw["default"])

    undeclared = sorted(set(PLACEHOLDER.findall(body)) - {a.name for a in arguments})
    if undeclared:
        raise ValueError(f"placeholders with no argument: {', '.join(undeclared)}")

    return FilePrompt(
        name=name,
        description=" ".join(str(meta.get("description", "")).split()) or None,
        arguments=arguments,
        template=body,
        defaults=defaults,
        frontmatter=meta,
    )
