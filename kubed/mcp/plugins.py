"""Plugins: what a server registers, read the Claude way and served by one rule set.

A server says which plugin to serve, ``mcp.add_plugin(address)``, and this layer does
the rest. A ``PluginLoader`` turns the address into a local folder, and that is all a
loader does: ``LocalLoader`` serves a folder in place, and a loader for any other
scheme is registered with ``mcp.add_loader``. The folder is read here, never by a
loader, as a Claude plugin: ``.claude-plugin/plugin.json`` for its metadata and the
component paths it declares beside or instead of the conventional ``skills/``,
``commands/`` and ``agents/``. What the manifest says is fetched content, never
trusted as a path: a component path that leaves the plugin root is dropped with a
warning naming it.

Each kind is then served by its rules: skills by ``kubed.mcp.skills``, commands as MCP
prompts (``kubed.mcp.prompts``) and agents through the agents extension
(``kubed.mcp.agents``). A server has one provider of each kind, whatever its plugins,
and each reads what its request is pinned to: the plugins served once the request's
view has run, of those its view admits (``mcp.plugin_view``), built once per build key
and reused.
A build runs the server's rewriters (``mcp.add_rewriter``) over the text it serves, from
what was read and never from the disk.
The layout and names are Claude Code's: a skill is one folder below ``skills/``, while
commands and agents are files at any depth, each folder adding a name segment, Claude
Code's ``:`` written ``_`` (``commands/db/migrate.md`` is ``db_migrate``). A plugin
added with no namespace is the server's own, its skills under ``skill://<server
name>/``; there is no root, and an empty namespace is refused. Plugins added under one
namespace merge, and a second thing at one address is refused.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

import anyio
from fastmcp import FastMCP
from fastmcp.prompts import Message, Prompt, PromptResult
from fastmcp.resources import Resource, ResourceTemplate, TextResource
from fastmcp.server.context import Context
from fastmcp.server.extensions import MethodBinding
from fastmcp.server.middleware import Middleware
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec, version_sort_key
from mcp_types import TextContent

from . import gateway, skills
from ._media import text_of
from ._rewrite import (
    Kind,
    RewriteFailed,
    Rewriter,
    Site,
    name_of,
    run,
    to_mcp_form,
)
from ._tools import TOOL_NAMES, _ResourceTools
from .agents import (
    EXTENSION_ID,
    META_KEY,
    SETTINGS,
    Agent,
    AgentsExtension,
    parse_agent,
    served_as,
)
from .agents import usable_name as _usable
from .delivery import BinaryPointer, _resolved
from .prompts import _survives_json, index_meta, read_command
from .protocol import AdvertiseLegacyExtensions

__all__ = [
    "AgentFile",
    "AgentMeta",
    "CommandReader",
    "Components",
    "LocalLoader",
    "Plugin",
    "PluginError",
    "PluginLoader",
    "PluginSet",
    "Site",
    "View",
    "read_plugin",
    "to_mcp_form",
]

log = logging.getLogger(__name__)

MANIFEST_FILES = (".claude-plugin/plugin.json", "plugin.json")
KINDS = ("skills", "commands", "agents")
# The fields beside a plugin.json that strict: false refuses, as Claude Code does.
COMPONENT_FIELDS = (*KINDS, "hooks", "outputStyles", "themes")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")


class PluginError(ValueError):
    """A plugin that is not served, and why."""


# What reads a command file (PLUG-62): ``fn(text, path, name, plugin)`` gives the
# ``Prompt`` to serve under ``name``, or the reason the command is not served.
CommandReader = Callable[[str, str, str, "Plugin"], Prompt | str]

# What adds a server's own keys to an agent's ``_meta`` (PLUG-73): ``fn(name, plugin)``
# with the agent's name in its set and the plugin as it enters the set.
AgentMeta = Callable[[str, "Plugin"], "Mapping[str, object] | None"]


def _read_command(
    reader: CommandReader, text: str, path: str, name: str, plugin: Plugin
) -> Prompt | str:
    """What ``reader`` makes of a command: the one place a reader is called.

    A raise is its reason, its message else its type's name, and a result neither a
    ``Prompt`` nor a ``str`` is a reason too. A ``Prompt`` named otherwise is served
    under ``name``, the layer's (PLUG-62).
    """
    try:
        result = reader(text, path, name, plugin)
    except Exception as exc:  # noqa: BLE001 - a server's reader fails one command, never its plugin (PLUG-62)
        return str(exc) or type(exc).__name__
    if isinstance(result, str):
        return result
    if not isinstance(result, Prompt):
        return f"command_reader returned {type(result).__name__}, not a Prompt or str"
    if result.name != name:
        return result.model_copy(update={"name": name})
    return result


# ---- loaders -----------------------------------------------------------------------


@runtime_checkable
class PluginLoader(Protocol):
    """Turns an address into a local folder holding the plugin, and nothing else.

    ``schemes`` are the address schemes it answers, such as ``("git+https",)``. A
    loader that cannot fetch raises ``PluginError``, ``OSError`` or ``ValueError``
    with the reason.
    """

    schemes: tuple[str, ...]

    def materialise(self, address: str) -> Path: ...


class LocalLoader:
    """A folder on this machine, a path or a ``file://`` URL, served where it is."""

    schemes: tuple[str, ...] = ("file",)

    def materialise(self, address: str) -> Path:
        path = Path(address)
        if _SCHEME.match(address):
            url = urlparse(address)
            if url.netloc not in ("", "localhost"):
                raise PluginError(f"{address} names another host")
            path = Path(unquote(url.path))
        if not path.is_dir():
            raise PluginError(f"{path} is not a folder")
        return path


# ---- the manifest ------------------------------------------------------------------


@dataclass(frozen=True)
class Components:
    """Where each kind is read from, as root-relative paths (``.`` is the root).

    ``skills`` are folders. ``commands`` are folders, whose ``*.md`` files are read at
    any depth, or ``.md`` files; ``agents`` likewise, a folder only by convention.
    ``notes`` name what the manifest or the layout got wrong without failing the
    plugin.
    """

    skills: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentFile:
    """One agent a plugin serves in its set (PLUG-74): ``name`` the served name,
    ``path`` its file's from the plugin root, ``/``-separated, and ``renamed`` True when
    it is served as ``<name>@<plugin>`` (PLUG-64)."""

    name: str
    path: str
    renamed: bool = False


@dataclass(frozen=True)
class Plugin:
    """A plugin: its metadata, where it was read from, and what it serves.

    ``namespace`` is the one it is served under. ``skills`` are the ``skill://`` URIs of
    the ``SKILL.md`` of each skill served, ``prompts`` the names served and ``agents``
    the names served in its set (PLUG-64). ``skipped`` is each skill folder, command
    file and agent file not served, as its path from the root and the reason, in the
    order read, then the agents its set skips. ``agent_files`` is each agent served,
    with its file, in path order.
    """

    name: str
    description: str | None
    version: str | None
    root: Path
    components: Components
    address: str = ""
    namespace: str = ""
    skills: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    agent_files: tuple[AgentFile, ...] = ()


def _inside(path: Path, base: Path) -> Path | None:
    """``path`` resolved, or None when it escapes ``base`` or cannot resolve.

    Before Python 3.13 a symlink loop makes ``resolve()`` raise ``RuntimeError``.
    """
    try:
        target = path.resolve()
    except (OSError, RuntimeError):
        return None
    return target if target.is_relative_to(base) else None


def _readable(base: Path, rel: str | Path) -> Path | None:
    """``base / rel`` resolved, when it is a regular file inside ``base``.

    Resolving before comparing refuses a ``..`` that climbs out and a symlink pointing
    out of the tree.
    """
    target = _inside(base / rel, base)
    return target if target is not None and target.is_file() else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _first(*values: object) -> str | None:
    """The first of ``values`` that is a string."""
    return next((value for value in values if isinstance(value, str)), None)


def _strings(value: object, field: str) -> tuple[str, ...] | None:
    """A field that may be written as one string or as a list of them."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise PluginError(f"{field} must be a string or a list of strings")


def _relative(text: str, field: str, notes: list[str]) -> str:
    """``text`` as a root-relative path, or ``""`` when it leaves the root.

    ``.`` and ``./`` name the root. A path that does not start ``./`` is kept, and
    noted: Claude requires the prefix.
    """
    path = text.strip()
    if path in (".", "./"):
        return "."
    relative = path.removeprefix("./").rstrip("/")
    if not relative or path.startswith("/") or ".." in PurePosixPath(relative).parts:
        notes.append(f"{field} path '{path}' leaves the plugin root")
        return ""
    if relative == path.rstrip("/"):
        notes.append(f"{field} path '{path}' does not start with ./")
    return relative


def _declared(data: Mapping, field: str, notes: list[str]) -> tuple[str, ...] | None:
    """The paths a manifest declares for one kind, or None when it declares none."""
    value = data.get(field)
    if field == "commands" and isinstance(value, dict):
        notes.append("commands object map is not served")
        return ()
    found = _strings(value, field)
    if found is None:
        return None
    kept = (_relative(item, field, notes) for item in found)
    paths = tuple(dict.fromkeys(path for path in kept if path))
    if field == "agents":
        for path in paths:
            if not path.endswith(".md"):
                notes.append(f"agents entry '{path}' is not a .md file")
        paths = tuple(path for path in paths if path.endswith(".md"))
    return paths


def _merged(
    sources: Iterable[Mapping], field: str, notes: list[str]
) -> tuple[str, ...] | None:
    """The paths each source declares for one kind, appended in order, or None when
    none declares any."""
    declared = [
        paths
        for source in sources
        if (paths := _declared(source, field, notes)) is not None
    ]
    if not declared:
        return None
    return tuple(dict.fromkeys(path for paths in declared for path in paths))


def _ignores(kind: str, declared: tuple[str, ...]) -> bool:
    """Whether paths ``declared`` for ``kind`` leave its default folder unread.

    A path that is the folder, one inside it, or the root reads it.
    """
    return not any(
        path in (".", kind) or path.startswith(f"{kind}/") for path in declared
    )


def read_plugin(
    root: str | os.PathLike[str], manifest: Mapping[str, object] | None = None
) -> Plugin:
    """The plugin in folder ``root``, from its manifest and the conventions.

    The manifest is the first of ``MANIFEST_FILES`` that is a regular file inside the
    root; a symlink out of it is not read. ``name``, ``description`` and ``version``
    come from it, the name falling back to the folder's. Declared ``skills`` add to the
    conventional ``skills/``; declared ``commands`` and ``agents`` replace their
    folder, noted when it is there. A manifest that is not a JSON object, or a
    component field that is not a string or a list of strings, is refused with
    ``PluginError``.

    ``manifest`` is the plugin's marketplace entry, merged as Claude Code merges it.
    With no ``plugin.json`` it is the manifest, whatever ``strict`` says. Beside one,
    its component paths are appended to the file's, read and checked the same way, and
    with ``strict: false`` an entry naming any component refuses the plugin. Its
    ``name`` and ``description`` win over the file's, the file's ``version`` over its.
    """
    root = Path(root)
    if not root.is_dir():
        raise PluginError(f"{root} is not a folder")
    if manifest is not None and not isinstance(manifest, Mapping):
        raise PluginError("the marketplace entry must be a JSON object")
    base = root.resolve()
    sources: list[Mapping] = []  # plugin.json, then the entry
    data: dict = {}
    for name in MANIFEST_FILES:
        path = _readable(base, name)
        if path is None:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PluginError(f"{name} is not readable JSON: {exc}") from exc
        except RecursionError:
            raise PluginError(f"{name} is nested too deeply to read") from None
        if not isinstance(data, dict):
            raise PluginError(f"{name} must be a JSON object")
        sources.append(data)
        break
    entry: dict = {}
    if manifest is not None:
        entry = dict(manifest)
        named = any(entry.get(field) is not None for field in COMPONENT_FIELDS)
        if sources and entry.get("strict") is False and named:
            raise PluginError(
                "conflicting manifests: both plugin.json and the marketplace entry "
                "specify components, and the entry is strict: false"
            )
        sources.append(entry)

    notes: list[str] = []
    found: dict[str, tuple[str, ...]] = {}
    for kind in KINDS:
        declared = _merged(sources, kind, notes)
        default = (base / kind).is_dir() and _inside(base / kind, base) is not None
        if declared is None:
            found[kind] = (kind,) if default else ()
        elif kind == "skills":
            found[kind] = tuple(dict.fromkeys(((kind,) if default else ()) + declared))
        else:
            found[kind] = declared
            if default and _ignores(kind, declared):
                notes.append(
                    f"Default {kind}/ folder is ignored because the manifest sets "
                    f'"{kind}"'
                )
    # Claude Code shows the entry's name and description; plugin.json's version wins.
    names = (_text(entry.get("name")), _text(data.get("name")))
    return Plugin(
        name=next((name for name in names if _usable(name)), root.name),
        description=_first(entry.get("description"), data.get("description")),
        version=_first(data.get("version"), entry.get("version")),
        root=root,
        components=Components(**found, notes=tuple(notes)),
    )


# ---- reading each kind -------------------------------------------------------------


def _skill_folders(
    base: Path, folder: str, notes: list[str]
) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Each skill in ``folder``, as its name and its folder, and the nested ones.

    A skill is a folder directly in ``folder`` holding a ``SKILL.md``: one level only,
    as Claude Code reads it. A declared folder that is a skill itself is its own name.
    Every ``SKILL.md`` deeper, in a folder that is no skill, is returned as nested,
    not served. A dot folder is never searched, and nothing below a skill is.
    """
    start = base / folder
    if _inside(start, base) is None or not start.is_dir():
        notes.append(f"skills entry '{folder}' is not a folder in the plugin")
        return [], []
    if (start / skills.ENTRY).is_file():
        return [(start.name, start)], []
    found: list[tuple[str, Path]] = []
    nested: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        here = Path(dirpath)
        if here == start or skills.ENTRY not in filenames:
            continue
        if here.parent == start:
            found.append((here.name, here))
        else:
            nested.append(here / skills.ENTRY)
        dirnames[:] = []
    return sorted(found), nested


def _md_files(
    base: Path, kind: str, entries: Iterable[str], notes: list[str]
) -> list[tuple[Path, tuple[str, ...], tuple[str, ...]]]:
    """The ``.md`` files each entry names, each with its folders' name segments and
    its place, the segments of its address below the namespace (PLUG-66).

    A listed file is itself, with no segment, placed at its path below ``kind``'s
    folder when the entry starts with it, else at its path below the root: the entry
    decides, never the resolved path. A folder is read at any depth, in path order,
    each folder between it and the file a segment, and each file placed at its path
    below the folder; a dot folder is not read, and a file that resolves outside the
    root is not either.
    """
    found: list[tuple[Path, tuple[str, ...], tuple[str, ...]]] = []
    for entry in entries:
        path = base / entry
        if entry.endswith(".md"):
            if _readable(base, entry) is None:
                notes.append(f"{kind} entry '{entry}' is not in the plugin")
            else:
                parts = PurePosixPath(entry).parts
                found.append((path, (), parts[1:] if parts[0] == kind else parts))
            continue
        if _inside(path, base) is None or not path.is_dir():
            notes.append(f"{kind} entry '{entry}' is not a folder in the plugin")
            continue

        def unlisted(exc: OSError, entry: str = entry) -> None:
            where = Path(exc.filename or base / entry)
            if where.is_relative_to(base):
                entry = where.relative_to(base).as_posix()
            notes.append(f"{kind} entry '{entry}' cannot be listed: {exc}")

        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(path, onerror=unlisted):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            here = Path(dirpath)
            files += (
                here / name
                for name in filenames
                if name.endswith(".md")
                and not name.startswith(".")
                and _readable(base, (here / name).relative_to(base))
            )
        for file in sorted(files, key=lambda f: f.relative_to(path).parts):
            place = file.relative_to(path).parts
            found.append((file, place[:-1], place))
    return found


def _namespace(namespace: str | None, server: str) -> str:
    """``namespace`` as served: the server's name for None, else its ``/`` segments.

    There is no root: a blank namespace is refused.
    """
    if namespace is None:
        namespace = server
    trimmed = namespace.strip("/")
    if not trimmed.strip():
        raise ValueError(
            f"namespace {namespace!r} is empty: skills are always under a namespace, "
            "None is the server's name"
        )
    parts = trimmed.split("/")
    if any(part in (".", "..") or not _usable(part) for part in parts):
        raise ValueError(f"namespace {namespace!r} is not a path of usable names")
    return trimmed


@dataclass(frozen=True)
class _File:
    """One command or agent file, as read: kept so a build can re-parse it from
    memory.

    ``path`` is root-relative, ``uri`` the ``prompts://`` or ``agents://`` address
    (PLUG-66) a rewriter is shown, ``data`` the raw bytes and ``text`` them as
    ``read_text`` gives them. ``stem`` and ``folders`` are what ``parse_agent`` names
    it by, and ``place`` the address's segments below the namespace: ``uri`` is
    always ``_file_uri`` of them. ``read_name`` is an agent's name as parsed, which
    its set may serve otherwise (PLUG-64), its address then following its served name;
    ``meta`` the keys its set adds to an agent's ``_meta`` (PLUG-73).
    """

    path: str
    uri: str
    data: bytes
    text: str
    item: Prompt | Agent
    stem: str
    folders: tuple[str, ...]
    place: tuple[str, ...]
    read_name: str = ""
    meta: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


def _decoded(data: bytes) -> str:
    """``data`` as ``read_text`` gives it: UTF-8, its newlines translated."""
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _file_uri(scheme: str, namespace: str, place: Iterable[str]) -> str:
    """The address of a command or agent file (PLUG-66): its ``place`` under the
    namespace, quoted as a skill's URI is with ``@`` kept too, so a renamed agent is
    ``<name>@<plugin>.md`` as a client copies it."""
    return f"{scheme}://{quote('/'.join((namespace, *place)), safe='/@')}"


def _at_index(place: Iterable[str], uri: str) -> str | None:
    """Why a file at ``place`` is not served, when its address has a folder index's
    name as a segment, its own name included (PLUG-66, SKL-11); else None."""
    if any(skills._is_index(segment) for segment in place):
        return f"its address {uri} is its folder's index"
    return None


def _joined(*parts: str) -> str:
    """Root-relative path segments joined, the root ``""`` left out."""
    return "/".join(part for part in parts if part)


@dataclass
class _Read:
    """A plugin's kinds, read and checked, before anything is served."""

    catalogue: dict[str, skills.Skill]
    skill_dirs: dict[str, str]
    prompts: list[_File]
    agents: list[_File]
    by_name: dict[str, str]
    prefix: str
    skipped: list[tuple[str, str]]


class _Skip:
    """Records what a plugin does not serve: its path from the root, and why."""

    def __init__(self, base: Path) -> None:
        self._base = base
        self.skipped: list[tuple[str, str]] = []

    def relative(self, path: Path) -> str:
        return path.relative_to(self._base).as_posix()

    def __call__(self, path: Path, reason: str) -> None:
        self.skipped.append((self.relative(path), reason))


@dataclass(frozen=True)
class _Served:
    """One plugin as served: the plugin, and what it serves of each kind.

    ``skill_dirs`` maps a catalogue key to the skill's root-relative folder (``""``
    for the root). ``by_name`` and ``prefix`` are what its agents were parsed with,
    ``reader`` what its commands were read with, and what a build reads them with.
    ``plugin`` and ``agents`` are as its set serves them (``_enter``); ``read`` and
    ``read_agents`` as read, before any set.
    """

    plugin: Plugin
    catalogue: Mapping[str, skills.Skill]
    skill_dirs: Mapping[str, str]
    prompts: tuple[_File, ...]
    agents: tuple[_File, ...]
    by_name: Mapping[str, str]
    prefix: str
    reader: CommandReader
    read: Plugin
    read_agents: tuple[_File, ...]


class _Skills(skills.EntryOnlySkillProvider):
    """The server's one skills provider: the skills of the request's build, in the
    order their plugins were added."""

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    async def load_skills(self) -> dict[str, skills.Skill]:
        return dict((await self._plugins.pinned()).build.catalogue)

    async def _list_resources(self) -> list[Resource]:
        # Listed only as the request lists skills (DLV-8, SKL-8); reads are not.
        if "skills" not in _resolved().listed:
            return []
        return await super()._list_resources()

    async def _list_skill_entries(self, context: Any) -> Sequence[Any]:
        # skills/list asks this, and skills/get its own _get_skill_entry, so under
        # local the plugins' skills are unlisted and still answer (DLV-9, DLV-10).
        if _resolved().via == "local":
            return []
        return await super()._list_skill_entries(context)

    async def get_tasks(self) -> list:
        return []  # see _Prompts.get_tasks

    async def _get_resource(
        self, uri: str, version: VersionSpec | None = None
    ) -> Resource | None:
        # A rewritten build holds its indexes, rendered and rewritten when it was
        # made: one it does not hold is missing, never rendered here unrewritten.
        if uri.startswith("skill://") and skills._is_index(uri):
            build = (await self._plugins.pinned()).build
            if build.rewritten:
                return build.indexes.get(uri)
        return await super()._get_resource(uri, version)


class _Prompts(Provider):
    """The server's one prompts provider: the commands of the request's build."""

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    async def _list_prompts(self) -> list[Prompt]:
        # Listed under mcp alone; prompts/get finds every one (DLV-9, DLV-10).
        if _resolved().via != "mcp":
            return []
        return list((await self._plugins.pinned()).build.prompts)

    async def _get_prompt(
        self, name: str, version: VersionSpec | None = None
    ) -> Prompt | None:
        # Every prompt of the pinned build, listed or not, matched as FastMCP's
        # Provider._get_prompt matches a listed one.
        prompts = (await self._plugins.pinned()).build.prompts
        matching = [p for p in prompts if p.name == name]
        if version:
            matching = [p for p in matching if version.matches(p.version)]
        if not matching:
            return None
        return max(matching, key=version_sort_key)

    async def get_tasks(self) -> list:
        # FastMCP lists every component when a server starts, to find the ones that
        # run as background tasks. Only a tool can, and a plugin serves none, so the
        # listing would only call the view's fn() outside any request (PLUG-53).
        return []


# The schemes of the plugins' files, reserved in a server that has a plugin (PLUG-72).
_RESERVED = ("prompts", "agents")
_FILE_SCHEMES = tuple(f"{scheme}://" for scheme in _RESERVED)
# "prompts:// and agents://", as both of PLUG-72's messages name them.
_RESERVED_TEXT = " and ".join(_FILE_SCHEMES)


def _reserved(uri: object) -> bool:
    """Whether ``uri``, a URI or a URI template, is under a reserved scheme, compared
    in lowercase and with the whitespace FastMCP drops stripped (PLUG-72). Anything
    but a string is not: FastMCP refuses it on its own terms."""
    if not isinstance(uri, str):
        return False
    scheme, sep, _ = uri.strip().partition("://")
    return bool(sep) and scheme.lower() in _RESERVED


def _own_reserved(server: FastMCP) -> str | None:
    """The URI or URI template of the server's first own resource or template under a
    reserved scheme, or None (PLUG-72). Read synchronously: ``add_plugin`` is sync,
    and is called inside running event loops."""
    # FastMCP lists a provider's components only by awaiting, and has no public sync
    # read. This is the one place quick-mcp touches LocalProvider's private store.
    components = getattr(server.local_provider, "_components", None)
    if components is None:
        log.warning(
            "FastMCP's LocalProvider has no _components: the server's own resources "
            "are not checked against %s (PLUG-72)",
            _RESERVED_TEXT,
        )
        return None
    for component in components.values():
        if isinstance(component, Resource):
            uri = str(component.uri)
        elif isinstance(component, ResourceTemplate):
            uri = component.uri_template
        else:
            continue
        if _reserved(uri):
            return uri
    return None


def _own_tool(server: FastMCP) -> str | None:
    """The name of the server's first own tool that the plugin layer serves itself
    (DLV-7), or None. Read as ``_own_reserved`` reads the server's resources."""
    components = getattr(server.local_provider, "_components", None)
    if components is None:
        log.warning(
            "FastMCP's LocalProvider has no _components: the server's own tools are "
            "not checked against %s (DLV-7)",
            " and ".join(TOOL_NAMES),
        )
        return None
    for component in components.values():
        if isinstance(component, Tool) and component.name in TOOL_NAMES:
            return component.name
    return None


def _serves_itself(name: str) -> str:
    """Why the server's own ``name`` is refused, as both of DLV-7's messages say it."""
    return (
        f"a server with a plugin serves {name} itself, over every resource, the "
        "plugins' and its own (DLV-6)"
    )


class _Files(Provider):
    """The server's one provider of prompt and agent files (PLUG-66 to PLUG-68): each
    file of the request's build at its address, its folders' children and indexes.

    A file is listed only as the request lists its kind (PLUG-67, DLV-8); an index and
    a folder never are, and all of them read by URI alone. It answers ``prompts://``
    and ``agents://`` only.
    """

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    async def _list_resources(self) -> list[Resource]:
        listed = _resolved().listed
        build = (await self._plugins.pinned()).build
        return [
            material.resource
            for uri, material in build.files.items()
            if uri.partition("://")[0] in listed
        ]

    async def get_tasks(self) -> list:
        return []  # see _Prompts.get_tasks

    async def _get_resource(
        self, uri: str, version: VersionSpec | None = None
    ) -> Resource | None:
        if not uri.startswith(_FILE_SCHEMES):
            return None
        build = (await self._plugins.pinned()).build
        material = build.files.get(uri)
        if material is not None:
            return material.resource
        if not skills._is_index(uri):
            return None
        # A rewritten build holds its indexes, as _Skills reads a skill folder's.
        if build.rewritten:
            return build.indexes.get(uri)
        children = build.folders.get(uri.rpartition("/")[0])
        return None if children is None else skills._index_of(uri, list(children))

    async def children(self, uri: str) -> list[dict] | None:
        """The children of folder ``uri`` (``skills.Folders``), or None."""
        if not uri.startswith(_FILE_SCHEMES):
            return None
        children = (await self._plugins.pinned()).build.folders.get(uri)
        return None if children is None else copy.deepcopy(list(children))


class _Agents(AgentsExtension):
    """The server's one agents extension: the agents of the request's build.

    It also binds ``resources/directory/read`` over every provider on the server
    (``skills.Folders``), so the folders answer from the first plugin, whatever its
    plugins serve (PLUG-68), unless a Skills extension registered before it binds it
    already. It declares nothing more than the agents extension does.
    """

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    def methods(self) -> Sequence[MethodBinding]:
        return skills._with_directory_read(self, super().methods())

    async def children(self, uri: str) -> list[dict] | None:
        """The children of folder ``uri`` from the server's providers, per request."""
        return await skills._served_children(self.server, uri)

    async def _agents(self) -> tuple[Agent, ...]:
        return (await self._plugins.pinned()).build.agents

    async def _listed(self) -> Sequence[Agent]:
        # Unlisted under local; agents/get still finds each (DLV-9, DLV-10).
        if _resolved().via == "local":
            return ()
        return await self._agents()

    def clone(self) -> _Agents:
        return type(self)(self._plugins)


@dataclass(frozen=True)
class _State:
    """What is served: its generation, and the plugins of the set, in order.

    Replaced whole by ``Plugins.serve`` and never changed in place.
    """

    generation: int
    current: tuple[_Served, ...]


# ---- views, builds and the pin -----------------------------------------------------


@dataclass(frozen=True)
class View:
    """What a request is served (PLUG-53): ``admits(plugin)`` says whether a plugin is
    in it, and ``key``, any hashable value, names it.

    ``admits`` runs only when a build is made (PLUG-55), and a build is reused for
    every request with its key, so the key must decide what it admits.
    """

    key: Hashable
    admits: Callable[[Plugin], bool]


def _every(plugin: Plugin) -> bool:
    return True


# The view of a server that sets none: one key, every plugin.
_EVERY = View(None, _every)

# How many builds are kept, the least recently used dropped first (PLUG-55), a
# prefixed one before any with none (GW-6).
BUILDS = 64

# A build key, made by Plugins._key alone.
_Key = tuple[Hashable, ...]


class _FileResource(TextResource):
    """A prompt's or agent's file whose row is its ``_meta`` alone, wherever it is
    listed: FastMCP's ``get_meta`` would add a ``fastmcp`` key to it (PLUG-68)."""

    def to_mcp_resource(self, **overrides: Any):
        return super().to_mcp_resource(**{"_meta": self.meta, **overrides})


@dataclass(frozen=True)
class _Material:
    """One prompt or agent file as a build serves it (PLUG-66): ``text`` what it reads
    as, ``resource`` what a read of its address gives, and ``row`` that resource's
    wire form, its row in its folder (PLUG-68)."""

    served: _Served
    file: _File
    text: str
    resource: _FileResource
    row: Mapping[str, Any]


def _material(
    served: _Served, file: _File, item: Prompt | Agent, text: str
) -> _Material:
    """``file`` as a build serves it, ``item`` its prompt or agent as built and
    ``text`` what it reads as: the one place its resource, and so its row, is made.

    The resource's ``_meta`` is what registers it (PLUG-69, PLUG-70): a prompt's
    ``io.kubed/prompt``, or everything ``agents/get`` gives as an agent's ``_meta``,
    its served name added to ``io.kubed/agent``. The row is that alone:
    ``to_mcp_resource(_meta=...)`` leaves out the ``fastmcp`` key FastMCP's
    ``get_meta`` always adds, and the tests that compare a row's whole ``_meta`` guard
    that override.
    """
    if isinstance(item, Agent):
        description = _text(item.card.get("description"))
        meta = copy.deepcopy(item.detail["_meta"])
        meta[META_KEY] = {"name": item.name, **meta[META_KEY]}
    else:
        description = item.description
        meta = index_meta(item, text)
    resource = _FileResource(
        uri=file.uri,
        name=item.name,
        description=description,
        mime_type="text/markdown",
        text=text,
        meta=meta,
    )
    row = skills._wire(resource.to_mcp_resource())
    return _Material(served, file, text, resource, row)


def _as_served(file: _File, text: str | None) -> str:
    """What ``file`` reads as: the bytes read unless a build changed its text (None)."""
    return file.data.decode("utf-8") if text is None else text


def _folders_in(material: _Material) -> tuple[str, str, list[tuple[str, int]]]:
    """The scheme and namespace of ``material``'s file, and each folder it sits in as
    its URI and its depth below the namespace, the namespace's own first (PLUG-68)."""
    file = material.file
    scheme = file.uri.partition("://")[0]
    namespace = material.served.plugin.namespace
    folders = [
        (_file_uri(scheme, namespace, file.place[:depth]), depth)
        for depth in range(len(file.place))
    ]
    return scheme, namespace, folders


def _read_place(material: _Material) -> tuple[str, ...]:
    """The place ``material``'s file was read at: its own, unless it is an agent its
    set renamed (PLUG-64), whose address is its served name's."""
    file = material.file
    if not isinstance(file.item, Agent) or file.item.name == file.read_name:
        return file.place
    read = next(f for f in material.served.read_agents if f.path == file.path)
    return read.place


def _folder_rows(materials: Iterable[_Material]) -> dict[str, tuple[dict, ...]]:
    """Each folder of ``materials``, as its children's rows sorted by URI (PLUG-68).

    A folder is every one a file sits in, from its namespace down. A namespace's
    folder is also a row in the folder above it, when that is a folder, named by the
    namespace alone. A folder two plugins share takes its row from the first.
    """
    rows: dict[str, dict] = {}  # each folder's own row, the first plugin's
    children: dict[str, dict[str, dict]] = {}  # folder -> child URI -> row
    for material in materials:
        file = material.file
        scheme, namespace, folders = _folders_in(material)
        for uri, depth in folders:
            if uri in rows:
                continue
            children[uri] = {}
            name = _joined(namespace, *file.place[:depth])
            about = f"A folder of {scheme} in {namespace}"
            rows[uri] = skills._folder(uri, name, about)
        parent = file.uri.rpartition("/")[0]
        children[parent].setdefault(file.uri, dict(material.row))
    for uri, row in rows.items():
        parent = uri.rpartition("/")[0]
        if parent in children:
            # A file and a folder at one URI keep the file's row, pending a spec ruling.
            children[parent].setdefault(uri, row)
    return {
        uri: tuple(sorted(found.values(), key=lambda row: row["uri"]))
        for uri, found in children.items()
    }


@dataclass(frozen=True)
class _Build:
    """What one build key is served: the admitted plugins and what they serve.

    ``files`` is each prompt and agent file served, by its address, and ``folders``
    each of their folders' children (PLUG-66, PLUG-68), rewritten or not.
    """

    admitted: tuple[Plugin, ...]
    catalogue: Mapping[str, skills.Skill]
    prompts: tuple[Prompt, ...]
    agents: tuple[Agent, ...]
    rewritten: bool  # the rewriter chain was not empty
    indexes: Mapping[str, TextResource | None]
    _renderers: tuple[object, ...] = ()  # kept alive with the build they rendered
    files: Mapping[str, _Material] = field(default_factory=lambda: MappingProxyType({}))
    folders: Mapping[str, tuple[dict, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class _Pin:
    """What one request reads: the state, view and key it was pinned with, and the
    build."""

    state: _State
    view: View
    key: _Key
    build: _Build


@dataclass
class _Slot:
    """One request's pin, made by its first reader and shared with the rest."""

    done: anyio.Event | None = None
    pin: _Pin | None = None
    error: Exception | None = None


# Each server's slot for the request running, opened by _PinScope.
_PINS: ContextVar[dict[Plugins, _Slot] | None] = ContextVar(
    "kubed_mcp_pins", default=None
)


class _PinScope(Middleware):
    """Opens a request's pin slot for one server's plugins, and fails the request
    when pinning failed.

    A request made inside another, a tool's ``ctx.read_resource`` say, finds the slot
    open and shares it. FastMCP logs and swallows what a provider raises, so an error
    stored in the slot is raised again here, after the request is handled, in place of
    whatever the request made of its absence (an unknown skill, say).
    """

    def __init__(self, plugins: Plugins) -> None:
        self._plugins = plugins

    async def on_request(self, context: Any, call_next: Any) -> Any:
        pins = _PINS.get() or {}
        slot = pins.get(self._plugins)
        token = None
        if slot is None:
            slot = _Slot()
            token = _PINS.set({**pins, self._plugins: slot})
        try:
            result = await call_next(context)
        except Exception:
            if slot.error is None:
                raise
            raise slot.error from None
        finally:
            if token is not None:
                _PINS.reset(token)
        if slot.error is not None:
            raise slot.error
        return result


@dataclass
class _Flight:
    """The one build being made for a key, which every reader of the key awaits."""

    done: anyio.Event
    build: _Build | None = None
    error: Exception | None = None


@dataclass
class _Memo:
    """The builds kept, least recently used first, and the ones being made."""

    builds: OrderedDict[_Key, _Build]
    flights: dict[_Key, _Flight]


def _owned(served: _Served) -> dict[str, str]:
    """What ``served`` serves, as the owners map names it: what -> its plugin. A
    file's address is keyed by its URI, as a skill's is (PLUG-66)."""
    name = served.plugin.name
    return {
        **dict.fromkeys(served.plugin.skills, name),
        **{f"prompt:{prompt}": name for prompt in served.plugin.prompts},
        **{file.uri: name for file in served.prompts},
        **{f"agent:{agent}": name for agent in served.plugin.agents},
        **{file.uri: name for file in served.agents},
    }


def _enter(
    served: _Served, owners: Mapping[str, str], meta: AgentMeta | None
) -> _Served:
    """``served`` as it enters a set whose ``owners`` are given (PLUG-64).

    Built from the plugin and agents as read, never as an earlier set served them, so
    each entry is decided afresh, in two passes. Names first: an agent keeps its read
    name when the set has none of it, else is served as ``<name>@<plugin>``, else is
    skipped, taken or not usable. Its address follows the name decided, its file's for
    its read name and ``<name>@<plugin>.md`` for a rename, and an agent whose address
    the set or an agent entered before it serves is skipped too, claiming no name
    (PLUG-66); that gives the plugin as it enters. Then ``meta``,
    once per agent named, with that name and that plugin (PLUG-73): its keys are added
    to the agent's ``_meta``, and a raise skips the agent, so it claims no name.

    A rename's or a skip's warning is logged only when the plugin as last served did
    not have it, so re-entering does not warn again, and a rename's only once the
    agent is served so; a bad result of ``meta`` is warned of on each entry. With
    nothing renamed, skipped or given keys this is the ``_Served`` read; with the
    outcome of the last entry, ``served`` itself, so a ``Plugin`` a server holds stays
    the one served.
    """
    read = served.read
    last = served.plugin
    named: list[tuple[_File, AgentFile]] = []
    clashes: dict[str, str] = {}  # a renamed agent's path -> the plain name's owner
    addresses: dict[str, str] = {}  # an entered agent's URI -> its file's path
    skipped = list(read.skipped)
    changed = False
    for file in served.read_agents:
        name = file.item.name
        owner = owners.get(f"agent:{name}")
        entry = AgentFile(name, file.path)
        if owner is not None:
            changed = True
            renamed = f"{name}@{read.name}"
            held = owners.get(f"agent:{renamed}")
            if not _usable(read.name):
                reason = f"agent name {renamed!r} is not a usable name"
                _skip_agent(read, last, skipped, file.path, reason)
                continue
            if held is not None:
                reason = f"agent {renamed} is already served by plugin {held}"
                _skip_agent(read, last, skipped, file.path, reason)
                continue
            clashes[file.path] = owner
            entry = AgentFile(renamed, file.path, renamed=True)
            place = (f"{renamed}.md",)
            uri = _file_uri("agents", read.namespace, place)
            file = replace(file, uri=uri, place=place)
        # Its address, as its name gives it: another plugin's, or a file of this one
        # entered before it, named as a clash within one plugin is (PLUG-18, PLUG-66).
        holder = owners.get(file.uri)
        if holder is not None:
            changed = True
            reason = f"{file.uri} is already served by plugin {holder}"
            _skip_agent(read, last, skipped, file.path, reason)
            continue
        first = addresses.get(file.uri)
        if first is not None:
            changed = True
            reason = f"{file.uri} is already served by {first}"
            _skip_agent(read, last, skipped, file.path, reason)
            continue
        addresses[file.uri] = file.path
        named.append((file, entry))

    entering = read
    if changed:
        entering = replace(
            read,
            agents=tuple(entry.name for _, entry in named),
            agent_files=tuple(entry for _, entry in named),
            skipped=tuple(skipped),
        )
    agents: list[_File] = []
    files: list[AgentFile] = []
    for file, entry in named:
        keys: Mapping[str, object] = file.meta
        if meta is not None:
            try:
                keys = _agent_meta(meta, entry.name, entering)
            except Exception as exc:  # noqa: BLE001 - a server's agent_meta fails one agent, never its plugin (PLUG-73)
                reason = str(exc) or type(exc).__name__
                failed = (file.path, f"agent_meta failed: {reason}")
                if failed not in last.skipped:
                    log.error(
                        "plugin %s: agent_meta failed on agent %s, so it is not "
                        "served: %s",
                        read.name,
                        entry.name,
                        reason,
                    )
                skipped.append(failed)
                changed = True
                continue
        if keys:
            changed = True
        if entry.renamed and entry.name not in last.agents:
            log.warning(
                "agent %s is already served by plugin %s, so plugin %s serves it as %s",
                file.item.name,
                clashes[file.path],
                read.name,
                entry.name,
            )
        if entry.renamed or keys:
            # A renamed file's uri and place were set with its name, above.
            file = replace(file, item=served_as(file.item, entry.name, keys), meta=keys)
        agents.append(file)
        files.append(entry)

    if not changed:
        if last is read:
            return served
        return replace(served, plugin=read, agents=served.read_agents)
    plugin = replace(
        read,
        agents=tuple(file.item.name for file in agents),
        agent_files=tuple(files),
        skipped=tuple(skipped),
    )
    entered = tuple(agents)
    if plugin == last:
        if entered == served.agents:
            return served
        # Only the agents' keys differ: the Plugin a server holds stays the one served.
        plugin = last
    return replace(served, plugin=plugin, agents=entered)


def _skip_agent(
    read: Plugin, last: Plugin, skipped: list[tuple[str, str]], path: str, reason: str
) -> None:
    """Adds the agent file at ``path`` to ``skipped``, warned of when the plugin as
    last served did not skip it so."""
    if (path, reason) not in last.skipped:
        log.warning("plugin %s: agent file %s is skipped: %s", read.name, path, reason)
    skipped.append((path, reason))


def _agent_meta(fn: AgentMeta, name: str, plugin: Plugin) -> Mapping[str, object]:
    """The keys ``fn`` adds to agent ``name``'s ``_meta``, as their JSON round trip,
    so the server cannot change them later; empty for None (PLUG-73).

    The layer's own key is dropped, and a result that is not a dict, or does not
    survive JSON, is dropped whole, each with a warning. A raise is the caller's.
    """
    result = fn(name, plugin)
    if result is None:
        return MappingProxyType({})
    about = f"plugin {plugin.name}: agent_meta for agent {name}"
    if not isinstance(result, Mapping):
        log.warning(
            "%s returned %s, not a dict; it is dropped", about, type(result).__name__
        )
        return MappingProxyType({})
    keys = dict(result)
    if META_KEY in keys:
        log.warning(
            "%s returned %s, which is the layer's; it is dropped", about, META_KEY
        )
        del keys[META_KEY]
    if not _survives_json(keys):
        log.warning("%s is not representable as JSON; it is dropped", about)
        return MappingProxyType({})
    return MappingProxyType(json.loads(json.dumps(keys, allow_nan=False)))


class _Renderer(skills.EntryOnlySkillProvider):
    """One skill of a rewritten build, alone: what its indexes are rendered from.

    A folder's children are its own skill's alone, so a renderer per skill lists each
    one in time to the skill's size rather than the build's. The skill was trimmed of
    its reserved files when it was read, so it is not trimmed (or warned of) again.
    """

    def __init__(self, key: str, skill: skills.Skill) -> None:
        super().__init__()
        self.skill = skill
        self._catalogue = {key: skill}

    async def load_skills(self) -> dict[str, skills.Skill]:
        return self._catalogue


class _Prefixed(Prompt):
    """A command that writes the prefix into what it renders (GW-7): its text is
    rewritten with the build, but an argument is given at the request, so the
    rendered text is written again. An address already prefixed is left alone.

    It wraps any ``Prompt`` a reader gives (PLUG-62), served as ``inner`` is, its own
    ``_render`` and ``convert_result`` included, and prefixes the text of each text
    message in the result. Other content passes as it is.
    """

    inner: Prompt
    prefix: str

    async def _render(self, arguments: dict[str, Any] | None = None) -> PromptResult:
        # FastMCP serves a prompt by its _render, so the inner prompt's own _render and
        # convert_result decide what is served, and only its texts are written here.
        result = await self.inner._render(arguments)
        if not result.messages:  # an InputRequiredPromptResult, say: nothing to write
            return result
        return result.model_copy(
            update={"messages": [self._message(m) for m in result.messages]}
        )

    def _message(self, message: Message) -> Message:
        content = message.content
        if not isinstance(content, TextContent):
            return message
        text = gateway.prefixed(content.text, self.prefix)
        return message.model_copy(
            update={"content": content.model_copy(update={"text": text})}
        )


class _Rewrite:
    """One build's rewriting: ``chain`` run over what each admitted plugin serves.

    Each method gives what is served in place of one item, or None when the item
    fails (PLUG-59): it is logged at error and left out of the build.
    """

    def __init__(
        self, chain: tuple[Rewriter, ...], view: Hashable, admitted: tuple[Plugin, ...]
    ) -> None:
        self._chain = chain
        self._view = view
        self._admitted = admitted
        self._skills: dict[int, Mapping[str, str]] = {}  # one per plugin per build

    def _site(self, served: _Served, kind: Kind, path: str, uri: str) -> Site:
        folders = self._skills.get(id(served))
        if folders is None:
            folders = self._skills[id(served)] = MappingProxyType(
                {served.skill_dirs[k]: skills._root(k) for k in served.catalogue}
            )
        return Site(
            plugin=served.plugin,
            namespace=served.plugin.namespace,
            kind=kind,
            path=path,
            uri=uri,
            view=self._view,
            admitted=self._admitted,
            skills=folders,
        )

    def _failed(
        self, served: _Served, by: str | None, path: str, what: str, reason: str
    ) -> None:
        """Logs an item left out. ``by`` is the rewriter that failed it, None for a
        re-parse or a rename, which every rewriter in the chain may have caused."""
        log.error(
            "plugin %s: %s failed on %s, so %s is not served for view %r: %s",
            served.plugin.name,
            by or ", ".join(name_of(rewriter) for rewriter in self._chain),
            path,
            what,
            self._view,
            reason,
        )

    def _run(self, served: _Served, site: Site, text: str, what: str) -> str | None:
        try:
            return run(self._chain, site, text)
        except RewriteFailed as exc:
            self._failed(served, exc.rewriter, site.path, what, exc.reason)
            return None

    def skill(
        self, served: _Served, key: str, skill: skills.Skill
    ) -> skills.Skill | None:
        """The skill at ``key``, its text files rewritten: the one read when none
        changes, else rebuilt, failed when FastMCP refuses it or its name changes."""
        folder = served.skill_dirs[key]
        what = skills._file_uri(key, skills.ENTRY)
        files: dict[str, str | bytes] = dict(skill.files)
        changed = False
        for file, data in skill.files.items():
            text = text_of(file, data)
            if text is None:
                continue  # served as it is (PLUG-58)
            path = _joined(folder, file)
            site = self._site(served, "skill", path, skills._file_uri(key, file))
            new = self._run(served, site, text, what)
            if new is None:
                return None
            if new != text:
                files[file] = new
                changed = True
        if not changed:
            return skill
        entry = _joined(folder, skills.ENTRY)
        try:
            built = skills.Skill.from_files(files)
        except ValueError as exc:
            self._failed(served, None, entry, what, skills._reason(exc))
            return None
        if built.name != skill.name:
            reason = f"its name is now {built.name!r}, not {skill.name!r}"
            self._failed(served, None, entry, what, reason)
            return None
        return built

    def prompt(self, served: _Served, file: _File) -> tuple[Prompt, str | None] | None:
        """A command, rewritten, and its text when that changed (else None): the one
        read when its text is unchanged, else read again under its name by the reader
        it was read with (PLUG-62)."""
        item = file.item
        what = f"prompt {item.name}"
        site = self._site(served, "command", file.path, file.uri)
        new = self._run(served, site, file.text, what)
        if new is None:
            return None
        if new == file.text:
            return item, None
        prompt = _read_command(served.reader, new, file.path, item.name, served.plugin)
        if isinstance(prompt, str):
            self._failed(served, None, file.path, what, prompt)
            return None
        return prompt, new

    def agent(self, served: _Served, file: _File) -> tuple[Agent, str | None] | None:
        """An agent, rewritten, and its text when that changed (else None): the one
        served when its text is unchanged, else re-parsed and served under the name
        its set gave it (PLUG-64), failed when it no longer parses or is named
        otherwise than it was read."""
        item = file.item
        what = f"agent {item.name}"
        site = self._site(served, "agent", file.path, file.uri)
        new = self._run(served, site, file.text, what)
        if new is None:
            return None
        if new == file.text:
            return item, None
        agent = parse_agent(
            new, served.by_name, served.prefix, stem=file.stem, folders=file.folders
        )
        if isinstance(agent, str):
            self._failed(served, None, file.path, what, agent)
            return None
        if agent.name != file.read_name:
            reason = f"its name is now {agent.name!r}, not {file.read_name!r}"
            self._failed(served, None, file.path, what, reason)
            return None
        return served_as(agent, item.name, file.meta), new

    def file_indexes(
        self,
        folders: Mapping[str, tuple[dict, ...]],
        files: Mapping[str, _Material],
    ) -> dict[str, TextResource | None]:
        """Each prompt and agent folder's ``_index.md`` and ``_index.json``, rendered
        from ``folders`` and rewritten; None for one that failed (PLUG-68).

        Its site is the first plugin of ``files`` with a file in the folder, at the
        folder's path in it: the file's folder, less one segment for each folder
        between the index's and the file where it was read. A renamed agent's address
        is its name's, at the namespace's top (PLUG-66), but its file stays where it
        was read, so the segments are counted from its place as read.
        """
        first: dict[str, tuple[_Material, int]] = {}
        for material in files.values():
            for uri, depth in _folders_in(material)[2]:
                first.setdefault(uri, (material, depth))
        found: dict[str, TextResource | None] = {}
        for folder, children in folders.items():
            material, depth = first[folder]
            parts = PurePosixPath(material.file.path).parent.parts
            # The place as read ends in a suffix of the file's folder, and depth is
            # below its length, so this drops what is there and never more.
            below = len(_read_place(material)) - 1 - depth
            path = _joined(*parts[: len(parts) - below])
            for name in (skills.INDEX_MD, skills.INDEX_JSON):
                uri = f"{folder}/{name}"
                resource = skills._index_of(uri, list(children))
                site = self._site(material.served, "index", path, uri)
                new = self._run(material.served, site, resource.text, uri)
                found[uri] = (
                    None if new is None else resource.model_copy(update={"text": new})
                )
        return found

    async def indexes(
        self, served: _Served, key: str, renderer: _Renderer
    ) -> dict[str, TextResource | None]:
        """Each folder's ``_index.md`` and ``_index.json`` of the skill at ``key``,
        both rendered from one listing of the folder by ``renderer`` and rewritten;
        None for one that failed."""
        found: dict[str, TextResource | None] = {}
        for rel in sorted(skills._folders_of(renderer.skill.files)):
            path = _joined(served.skill_dirs[key], rel)
            folder = skills._folder_uri(key, rel)
            children = await renderer.children(folder)
            if children is None:
                continue
            for name in (skills.INDEX_MD, skills.INDEX_JSON):
                uri = f"{folder}/{name}"
                resource = skills._index_of(uri, children)
                site = self._site(served, "index", path, uri)
                new = self._run(served, site, resource.text, uri)
                found[uri] = (
                    None if new is None else resource.model_copy(update={"text": new})
                )
        return found


def _within(path: str, folder: str) -> bool:
    """Whether ``path`` is ``folder`` or lies below it, by whole ``/`` segments."""
    return path == folder or path.startswith(folder + "/")


def _nested(served: _Served, earlier: Iterable[_Served]) -> str | None:
    """Why ``served`` may not join a set holding ``earlier`` (PLUG-76), else None:
    its namespace inside a skill of an earlier plugin, or a skill of its holding an
    earlier plugin's namespace. Checked in ``earlier``'s order, the first direction
    before the second for each."""
    ns = served.plugin.namespace
    for other in earlier:
        for key in other.catalogue:
            if _within(ns, key):
                return (
                    f"namespace {ns} is inside skill {skills._root(key)} "
                    f"of plugin {other.plugin.name}"
                )
        theirs = other.plugin.namespace
        for key in served.catalogue:
            if _within(theirs, key):
                return (
                    f"skill {skills._root(key)} holds namespace {theirs} "
                    f"of plugin {other.plugin.name}"
                )
    return None


class PluginSet:
    """A set of plugins built apart from what is served, then served whole (PLUG-50).

    ``add`` reads and checks a plugin as ``add_plugin`` does and ``keep`` carries one
    the server serves now, unread. A clash is checked against the plugins in this set
    alone, so a set that leaves a plugin out may add another that serves the same
    thing. A skill URI, a prompt name or a prompt's address served already refuses
    the plugin, and so does a namespace nesting with a skill (PLUG-76); an agent
    name is renamed or skipped instead, and an agent whose address is served
    skipped, decided as the plugin enters the set (PLUG-64, PLUG-66), so each
    returns the plugin as it entered. A refused plugin raises and leaves the set
    unchanged. Nothing here is served until ``mcp.serve(set)``; the set is then a
    snapshot, and adding to it afterwards changes nothing served.
    """

    def __init__(self, plugins: Plugins) -> None:
        self._plugins = plugins
        self._served: list[_Served] = []
        self._owners: dict[str, str] = {}  # what is in the set -> the plugin serving it

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        """The plugins in the set, in the order they were added to it."""
        return tuple(served.plugin for served in self._served)

    def add(
        self,
        address: str | os.PathLike[str],
        *,
        namespace: str | None = None,
        manifest: Mapping[str, object] | None = None,
    ) -> Plugin:
        """Reads and checks the plugin at ``address`` as ``add_plugin`` does (PLUG-2,
        PLUG-3) and adds it to the set. A refused plugin raises its ``PluginError``."""
        served = self._plugins._load(address, namespace, manifest, self._owners)
        try:
            entered = self._carry(served)
        except PluginError as exc:
            raise PluginError(
                f"plugin {served.plugin.address} is not served: {exc}"
            ) from None
        return entered.plugin

    def keep(self, plugin: Plugin) -> Plugin:
        """Carries a plugin the server serves now into the set, without reading it
        again, and returns it as it entered (PLUG-64). A plugin not served now is a
        ``ValueError``; one that clashes with the set is a ``PluginError``."""
        for served in self._plugins.current:
            if served.plugin is plugin:
                break
        else:
            raise ValueError(
                f"plugin {plugin.name} is not served now: keep() carries a plugin "
                "this server serves"
            )
        try:
            entered = self._carry(served)
        except PluginError as exc:
            raise PluginError(f"plugin {plugin.address} is not served: {exc}") from None
        return entered.plugin

    def _carry(self, served: _Served) -> _Served:
        """Adds ``served`` to the set as it enters it, and returns that: refused whole
        when a skill, a prompt or a prompt's address of it is owned already, or when
        its namespace and a skill of the set nest (PLUG-76), its agents named and
        addressed by the set (PLUG-64, PLUG-66)."""
        for key in _owned(served):
            if key.startswith(("agent:", "agents://")):
                continue
            owner = self._owners.get(key)
            if owner is not None:
                what = key if "://" in key else key.replace(":", " ", 1)
                raise PluginError(f"{what} is already served by plugin {owner}")
        if (reason := _nested(served, self._served)) is not None:
            raise PluginError(reason)
        entered = _enter(served, self._owners, self._plugins._agent_meta)
        self._owners.update(_owned(entered))
        self._served.append(entered)
        return entered


class Plugins:
    """A server's plugins and loaders: what ``QuickMCP`` delegates to.

    Everything a plugin serves is read and checked against the set it joins before any
    of it is served, so a refused plugin leaves no trace. What is served is a
    ``_State``, replaced whole by ``serve`` and never changed in place. The server's
    one skills provider, prompts provider and agents extension read the request's pin
    instead (``pinned``): a build of what the view admits of the state served once the
    request's view has run.
    """

    def __init__(self, server: FastMCP) -> None:
        self._server = server
        self._loaders: dict[str, PluginLoader] = {}
        # Schemes whose loader a server may replace once: the built-in file (PLUG-61).
        self._replaceable: set[str] = {"file"}
        self._skills: _Skills | None = None
        self._prompts: _Prompts | None = None
        self._files: _Files | None = None
        self._agents: _Agents | None = None
        self._tools: _ResourceTools | None = None
        self._pointer: BinaryPointer | None = None
        self._state = _State(0, ())
        self._view: Callable[[], View] | None = None
        self._rewriters: tuple[Rewriter, ...] = ()
        self._reader: CommandReader = read_command
        self._agent_meta: AgentMeta | None = None
        self._memo = _Memo(OrderedDict(), {})
        self.add_loader(LocalLoader())
        self._replaceable = {"file"}

    @property
    def current(self) -> tuple[_Served, ...]:
        return self._state.current

    @property
    def generation(self) -> int:
        return self._state.generation

    @property
    def served(self) -> tuple[Plugin, ...]:
        return tuple(served.plugin for served in self.current)

    def add_loader(self, loader: PluginLoader) -> None:
        schemes = tuple(scheme.lower() for scheme in loader.schemes)
        if not schemes:
            raise ValueError(f"{type(loader).__name__} names no scheme")
        for scheme in schemes:
            if scheme in self._loaders and scheme not in self._replaceable:
                raise ValueError(f"a loader for {scheme}:// is already registered")
        for scheme in schemes:
            self._loaders[scheme] = loader
            self._replaceable.discard(scheme)

    def _loader_for(self, address: str) -> PluginLoader:
        match = _SCHEME.match(address)
        scheme = match.group(1).lower() if match else "file"
        loader = self._loaders.get(scheme)
        if loader is None:
            raise PluginError(
                f"no loader for {scheme}:// addresses: register one with "
                "mcp.add_loader()"
            )
        return loader

    def plugin_set(self) -> PluginSet:
        return PluginSet(self)

    def serve(self, staging: PluginSet) -> None:
        """Makes ``staging`` the served set, in one assignment (PLUG-51)."""
        if staging._plugins is not self:
            raise ValueError("this plugin set belongs to another server")
        served = tuple(staging._served)
        if served:
            try:
                self._check_first()
            except PluginError as exc:
                raise PluginError(f"the plugin set is not served: {exc}") from None
        self._providers(served)
        self._state = _State(self._state.generation + 1, served)
        self._memo = _Memo(OrderedDict(), {})  # last: no build of the old set is kept

    def check_own(self, uri: object) -> None:
        """Refuses ``uri`` for the server's own resource once it has a plugin, when it
        is under a reserved scheme (PLUG-72)."""
        if self._files is not None and _reserved(uri):
            raise ValueError(
                f"{uri}: {_RESERVED_TEXT} are reserved for the plugins in a server "
                "that has one"
            )

    def check_own_tool(self, name: str | None) -> None:
        """Refuses ``name`` for the server's own tool once it has a plugin, when the
        plugin layer serves a tool by that name itself (DLV-7)."""
        if self._tools is not None and name in TOOL_NAMES:
            raise ValueError(
                f"{name}: {_serves_itself(name)}, so the server's own is refused "
                "(DLV-7)"
            )

    def _check_first(self) -> None:
        """Refuses the server's first plugin when the server has its own resource
        under a reserved scheme (PLUG-72), or its own tool by a name the plugin layer
        serves (DLV-7). Once it has a plugin, ``check_own`` and ``check_own_tool``
        keep it from getting one."""
        if self._files is not None:
            return
        uri = _own_reserved(self._server)
        if uri is not None:
            raise PluginError(
                f"the server has its own resource at {uri}, and {_RESERVED_TEXT} are "
                "the plugins'"
            )
        name = _own_tool(self._server)
        if name is not None:
            raise PluginError(
                f"the server has its own tool {name}, but {_serves_itself(name)}: "
                f"remove the server's {name} (DLV-7)"
            )

    def set_binary_pointer(self, fn: BinaryPointer) -> None:
        """Makes ``fn`` what ``read_resource`` asks where a binary's link is listed
        (DLV-6), in place of any earlier one."""
        self._pointer = fn

    def set_view(self, fn: Callable[[], View]) -> None:
        """Makes ``fn`` the view, called once per request (PLUG-53)."""
        self._view = fn
        self._memo = _Memo(OrderedDict(), {})  # last: no build of the old view is kept

    def add_rewriter(self, fn: Rewriter) -> None:
        """Adds ``fn`` to the end of the rewriters, run by every build (PLUG-56)."""
        self._rewriters = (*self._rewriters, fn)
        self._memo = _Memo(OrderedDict(), {})  # last: no build without it is kept

    def set_command_reader(self, fn: CommandReader) -> None:
        """Makes ``fn`` the reader of every plugin read from now on (PLUG-62). A
        plugin read already keeps the reader it was read with, so no build changes."""
        self._reader = fn

    def set_agent_meta(self, fn: AgentMeta) -> None:
        """Makes ``fn`` the ``agent_meta`` of every plugin entering a set from now on
        (PLUG-73), in place of any earlier one. It runs on set entry, never in a
        build, so the plugins served keep their keys, and no build changes, until
        they next enter a set."""
        self._agent_meta = fn

    def _chain(self, key: _Key) -> tuple[Rewriter, ...]:
        """The rewriters a build for ``key`` runs, in order: the one place the order
        is decided. Empty means no rewriting, and no copies.

        A prefix (GW-5) is written by one last rewriter, after the server's own, so
        none of them sees the prefix and none can undo it.
        """
        prefix = self._prefix_of(key)
        if prefix is None:
            return self._rewriters

        def write(site: Site, text: str) -> str:
            return gateway.prefixed(text, prefix)

        write.__name__ = write.__qualname__ = "gateway prefix (GW-5)"  # PLUG-59's log
        return (*self._rewriters, write)

    def _key(self, state: _State, view: View, prefix: str | None) -> _Key:
        """The build key of ``view`` over ``state`` and the request's ``prefix``: the
        one place a key is made. No prefix is None, so a server behind no gateway
        keeps the one build it always had (GW-6)."""
        return (state.generation, view.key, prefix)

    @staticmethod
    def _prefix_of(key: _Key) -> str | None:
        """The prefix part of a key ``_key`` made: the one place it is read back."""
        prefix = key[2]
        assert prefix is None or isinstance(prefix, str)
        return prefix

    async def pinned(self) -> _Pin:
        """The request's pin, made by its first reader; off a request, a fresh one.

        A failure to pin is stored in the request's slot and raised to every reader,
        and ``_PinScope`` fails the request with it.
        """
        slot = (_PINS.get() or {}).get(self)
        if slot is None:
            return await self._pin()
        while slot.pin is None and slot.error is None:
            if slot.done is None:
                done = slot.done = anyio.Event()
                try:
                    slot.pin = await self._pin()
                except Exception as exc:  # noqa: BLE001 - any failure to pin must fail the request (PLUG-53), not be swallowed by FastMCP
                    slot.error = exc
                finally:
                    if slot.pin is None and slot.error is None:
                        slot.done = None  # cancelled: the next reader pins afresh
                    done.set()
            else:
                await slot.done.wait()
        if slot.error is not None:
            raise slot.error
        return slot.pin

    async def _pin(self) -> _Pin:
        """One pin, made fresh. The view runs first, so a set it serves (PLUG-51) is
        what the request reads. The memo and the state are taken after it and before
        any await, and a build is stored only while its memo is still the server's, so
        a change made during a build is never stored where a later request finds it. A
        view may serve; nothing else."""
        view = self._view() if self._view else _EVERY
        memo = self._memo
        state = self._state
        key = self._key(state, view, gateway.prefix())
        build = await self._build_for(memo, state, view, key)
        return _Pin(state, view, key, build)

    async def _build_for(
        self, memo: _Memo, state: _State, view: View, key: _Key
    ) -> _Build:
        """The build for ``key``: kept, in flight, or made now and kept while ``memo``
        is still the server's."""
        if key in memo.builds:
            memo.builds.move_to_end(key)
            return memo.builds[key]
        flight = memo.flights.get(key)
        if flight is not None:
            await flight.done.wait()
            if flight.error is not None:
                raise flight.error
            if flight.build is None:  # its maker was cancelled: make it again
                return await self._build_for(memo, state, view, key)
            return flight.build
        flight = memo.flights[key] = _Flight(anyio.Event())
        try:
            flight.build = await self._make(state, view, key)
        except Exception as exc:
            flight.error = exc  # every reader awaiting the flight raises it too
            raise
        finally:
            del memo.flights[key]
            flight.done.set()
        if memo is self._memo:
            memo.builds[key] = flight.build
            while len(memo.builds) > BUILDS:
                # A prefixed build goes first, so a client naming prefixes never
                # pushes out a build with none (GW-6).
                drop = next(
                    (k for k in memo.builds if self._prefix_of(k) is not None), None
                )
                if drop is None:
                    memo.builds.popitem(last=False)
                else:
                    del memo.builds[drop]
        return flight.build

    async def _make(self, state: _State, view: View, key: _Key) -> _Build:
        """A build of what ``view`` admits of ``state``, from what was read: it never
        reads the disk. With no rewriter it serves the objects read; with any, the
        rewritten text, and each folder's indexes, rendered now. Either way it holds
        each prompt and agent file and their folders (PLUG-66, PLUG-68)."""
        chain = self._chain(key)
        prefix = self._prefix_of(key)
        admitted = tuple(s for s in state.current if view.admits(s.plugin))
        plugins = tuple(s.plugin for s in admitted)
        if not chain:
            read = [
                _material(s, f, f.item, _as_served(f, None))
                for s in admitted
                for f in (*s.prompts, *s.agents)
            ]
            return _Build(
                admitted=plugins,
                catalogue=MappingProxyType(
                    {k: skill for s in admitted for k, skill in s.catalogue.items()}
                ),
                prompts=tuple(f.item for s in admitted for f in s.prompts),
                agents=tuple(f.item for s in admitted for f in s.agents),
                rewritten=False,
                indexes=MappingProxyType({}),
                files=MappingProxyType({m.file.uri: m for m in read}),
                folders=MappingProxyType(_folder_rows(read)),
            )
        rewrite = _Rewrite(chain, view.key, plugins)
        catalogue: dict[str, skills.Skill] = {}
        prompts: list[Prompt] = []
        agents: list[Agent] = []
        materials: list[_Material] = []
        for served in admitted:
            for k, skill in served.catalogue.items():
                if (built := rewrite.skill(served, k, skill)) is not None:
                    catalogue[k] = built
            for file in served.prompts:
                if (rewritten := rewrite.prompt(served, file)) is None:
                    continue
                prompt, text = rewritten
                prompts.append(
                    prompt
                    if prefix is None
                    else _Prefixed(
                        **{f: getattr(prompt, f) for f in Prompt.model_fields},
                        inner=prompt,
                        prefix=prefix,
                    )
                )
                materials.append(
                    _material(served, file, prompt, _as_served(file, text))
                )
            for file in served.agents:
                if (rewritten := rewrite.agent(served, file)) is None:
                    continue
                agent, text = rewritten
                agents.append(agent)
                materials.append(_material(served, file, agent, _as_served(file, text)))
        files = {m.file.uri: m for m in materials}
        folders = _folder_rows(materials)
        # Throwaway renderers, one per skill: FastMCP keys what it loads per request
        # by a provider's id, so the build keeps them alive while a request may still
        # see them.
        renderers: list[_Renderer] = []
        indexes: dict[str, TextResource | None] = {}
        async with Context(fastmcp=self._server):
            for served in admitted:
                for k in served.catalogue:
                    if k in catalogue:
                        renderers.append(renderer := _Renderer(k, catalogue[k]))
                        indexes |= await rewrite.indexes(served, k, renderer)
        indexes |= rewrite.file_indexes(folders, files)
        return _Build(
            admitted=plugins,
            catalogue=MappingProxyType(catalogue),
            prompts=tuple(prompts),
            agents=tuple(agents),
            rewritten=True,
            indexes=MappingProxyType(indexes),
            _renderers=tuple(renderers),
            files=MappingProxyType(files),
            folders=MappingProxyType(folders),
        )

    def add(
        self,
        address: str | os.PathLike[str],
        namespace: str | None = None,
        manifest: Mapping[str, object] | None = None,
    ) -> Plugin:
        """A new set of the served plugins and the new one, served (PLUG-52)."""
        staging = self.plugin_set()
        for served in self.current:
            staging._carry(served)
        plugin = staging.add(address, namespace=namespace, manifest=manifest)
        self.serve(staging)
        return plugin

    def _load(
        self,
        address: str | os.PathLike[str],
        namespace: str | None,
        manifest: Mapping[str, object] | None,
        owners: dict[str, str],
    ) -> _Served:
        address = os.fspath(address)
        prefix = _namespace(namespace, self._server.name)
        try:
            self._check_first()
            loader = self._loader_for(address)
            try:
                root = loader.materialise(address)
            except PluginError:
                raise
            except (OSError, ValueError) as exc:
                raise PluginError(str(exc) or type(exc).__name__) from exc
            plugin = read_plugin(root, manifest)
            notes = list(plugin.components.notes)
            reader = self._reader
            read = self._read(
                replace(plugin, address=address, namespace=prefix),
                prefix,
                notes,
                owners,
                reader,
            )
        except PluginError as exc:
            raise PluginError(f"plugin {address} is not served: {exc}") from None
        for note in dict.fromkeys(notes):
            log.warning("plugin %s: %s", plugin.name, note)
        plugin = replace(
            plugin,
            address=address,
            namespace=prefix,
            components=replace(plugin.components, notes=tuple(notes)),
            skills=tuple(
                f"skill://{quote(key, safe='/')}/{skills.ENTRY}"
                for key in read.catalogue
            ),
            prompts=tuple(file.item.name for file in read.prompts),
            agents=tuple(file.item.name for file in read.agents),
            skipped=tuple(read.skipped),
            agent_files=tuple(AgentFile(f.item.name, f.path) for f in read.agents),
        )
        agents = tuple(read.agents)
        return _Served(
            plugin=plugin,
            catalogue=dict(read.catalogue),
            skill_dirs=dict(read.skill_dirs),
            prompts=tuple(read.prompts),
            agents=agents,
            by_name=dict(read.by_name),
            prefix=read.prefix,
            reader=reader,
            read=plugin,
            read_agents=agents,
        )

    def _claim(
        self,
        claims: Sequence[tuple[str, str]],
        plugin: Plugin,
        seen: dict[str, Path],
        path: Path,
        skip: _Skip,
        owners: Mapping[str, str],
        *,
        files: bool = True,
    ) -> bool:
        """Whether every ``(key, what)`` of ``claims`` is free, recording them all
        only then, so a file skipped for one never holds another. The first held
        decides: refused when served, skipped when this plugin has it already, the
        warning naming both files when ``files``."""
        for key, what in claims:
            owner = owners.get(key)
            if owner is not None:
                raise PluginError(f"{what} is already served by plugin {owner}")
            if key in seen:
                first = seen[key]
                if files:
                    log.warning(
                        "%s is already served by %s; %s is skipped", what, first, path
                    )
                else:
                    log.warning(
                        "plugin %s: %s is skipped, served already", plugin.name, what
                    )
                skip(path, f"{what} is already served by {skip.relative(first)}")
                return False
        for key, _ in claims:
            seen[key] = path
        return True

    def _read(
        self,
        plugin: Plugin,
        prefix: str,
        notes: list[str],
        owners: Mapping[str, str],
        reader: CommandReader,
    ) -> _Read:
        base = plugin.root.resolve()
        parts = plugin.components
        seen: dict[str, Path] = {}
        skip = _Skip(base)

        catalogue: dict[str, skills.Skill] = {}
        skill_dirs: dict[str, str] = {}
        by_name: dict[str, str] = {}
        folders: set[Path] = set()
        nested: list[Path] = []
        for folder in parts.skills:
            found, deeper = _skill_folders(base, folder, notes)
            nested += deeper
            for path, location in found:
                folders.add(location)
                key = f"{prefix}/{path}"
                uri = f"skill://{quote(key, safe='/')}/{skills.ENTRY}"
                if not self._claim(
                    ((uri, uri),), plugin, seen, location, skip, owners, files=False
                ):
                    continue
                skill = skills._load_skill(location)
                if isinstance(skill, str):
                    skip(location, skill)
                    continue
                catalogue[key] = skill
                folder = location.relative_to(base).as_posix()
                skill_dirs[key] = "" if folder == "." else folder
                by_name.setdefault(skill.name, uri)
        # A skills folder read as well, the root say, may hold one found as nested.
        for entry in nested:
            if entry.parent not in folders:
                notes.append(
                    f"{entry.relative_to(base).as_posix()} is not served: a skill is "
                    "one folder below its skills folder"
                )

        # The server is already a scope for prompts and agents: only an explicit
        # namespace other than its own is a prefix, and only its first segment
        # (PLUG-63).
        own = prefix == self._server.name
        named = "" if own else prefix.split("/")[0] + "_"
        prompts: list[_File] = []
        for path, folders_in, place in _md_files(
            base, "commands", parts.commands, notes
        ):
            uri = _file_uri("prompts", prefix, place)
            if reason := _at_index(place, uri):
                log.warning("prompt %s is skipped: %s", path, reason)
                skip(path, reason)
                continue
            # Copilot's .prompt.md is the file type: x.prompt.md is x (PLUG-62).
            segments = (*folders_in, path.stem.removesuffix(".prompt"))
            name = named + "_".join(segments)
            if not all(_usable(part) for part in (*segments, name)):
                reason = f"prompt name {name!r} is not a usable name"
                log.warning("prompt %s is skipped: %s", path, reason)
                skip(path, reason)
                continue
            try:
                data = path.read_bytes()
                text = _decoded(data)
            except (OSError, UnicodeDecodeError) as exc:
                log.warning("prompt %s is skipped: %s", path, exc)
                skip(path, str(exc))
                continue
            prompt = _read_command(reader, text, skip.relative(path), name, plugin)
            if isinstance(prompt, str):
                log.warning("prompt %s is skipped: %s", path, prompt)
                skip(path, prompt)
                continue
            # The name first, so a file both clash on is refused or skipped for it.
            claims = ((f"prompt:{prompt.name}", f"prompt {prompt.name}"), (uri, uri))
            if self._claim(claims, plugin, seen, path, skip, owners):
                prompts.append(
                    _File(
                        path=skip.relative(path),
                        uri=uri,
                        data=data,
                        text=text,
                        item=prompt,
                        stem=path.stem,
                        folders=tuple(folders_in),
                        place=place,
                    )
                )

        agents: list[_File] = []
        for path, segments, place in _md_files(base, "agents", parts.agents, notes):
            uri = _file_uri("agents", prefix, place)
            if reason := _at_index(place, uri):
                log.warning("agent file %s is skipped: %s", path, reason)
                skip(path, reason)
                continue
            try:
                data = path.read_bytes()
                text = _decoded(data)
            except (OSError, UnicodeDecodeError) as exc:
                log.warning("agent file %s is skipped: %s", path, exc)
                skip(path, str(exc))
                continue
            agent = parse_agent(text, by_name, named, stem=path.stem, folders=segments)
            if isinstance(agent, str):
                log.warning("agent file %s is skipped: %s", path, agent)
                skip(path, agent)
            elif self._claim(
                ((f"agent:{agent.name}", f"agent {agent.name}"), (uri, uri)),
                plugin,
                seen,
                path,
                skip,
                {},  # across plugins, the set names and addresses agents (PLUG-64)
            ):
                agents.append(
                    _File(
                        path=skip.relative(path),
                        uri=uri,
                        data=data,
                        text=text,
                        item=agent,
                        stem=path.stem,
                        folders=tuple(segments),
                        place=place,
                        read_name=agent.name,
                    )
                )
        return _Read(
            catalogue, skill_dirs, prompts, agents, by_name, named, skip.skipped
        )

    def _providers(self, served: tuple[_Served, ...]) -> None:
        # The first plugin with a skill brings the Skills extension (PLUG-20), the
        # first plugin the prompts provider, the files provider (PLUG-66), the agents
        # extension (PLUG-27), which binds resources/directory/read, NameTheIndex
        # (PLUG-68), and the resource tools (DLV-6).
        if self._skills is None and any(each.catalogue for each in served):
            self._skills = _Skills(self)
            skills.serve(self._server, self._skills)
        if not served:
            return
        if self._prompts is None:
            self._prompts = _Prompts(self)
            self._server.add_provider(self._prompts)
        if self._files is None:
            self._files = _Files(self)
            self._server.add_provider(self._files)
        if self._tools is None:
            self._tools = _ResourceTools(lambda: self._pointer)
            self._server.add_provider(self._tools)
        if self._agents is None:
            self._agents = _Agents(self)
            self._server.add_extension(self._agents)
            self._server.add_middleware(
                AdvertiseLegacyExtensions({EXTENSION_ID: SETTINGS})
            )
            skills._name_the_index(self._server)
