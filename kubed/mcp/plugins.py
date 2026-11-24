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
and each reads what its request is pinned to: the plugins served when the request
began that its view admits (``mcp.plugin_view``), built once per build key and reused.
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

import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

import anyio
import yaml
from fastmcp import FastMCP
from fastmcp.prompts import Prompt
from fastmcp.resources import Resource, TextResource
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware
from fastmcp.server.providers import Provider
from fastmcp.utilities.versions import VersionSpec

from . import skills
from ._rewrite import Kind, RewriteFailed, Rewriter, Site, name_of, run, text_of
from .agents import EXTENSION_ID, SETTINGS, Agent, AgentsExtension, parse_agent
from .agents import usable_name as _usable
from .prompts import FilePrompt, parse_prompt
from .protocol import AdvertiseLegacyExtensions

__all__ = [
    "Components",
    "LocalLoader",
    "Plugin",
    "PluginError",
    "PluginLoader",
    "PluginSet",
    "Site",
    "View",
    "read_plugin",
]

log = logging.getLogger(__name__)

MANIFEST_FILES = (".claude-plugin/plugin.json", "plugin.json")
KINDS = ("skills", "commands", "agents")
# The fields beside a plugin.json that strict: false refuses, as Claude Code does.
COMPONENT_FIELDS = (*KINDS, "hooks", "outputStyles", "themes")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")


class PluginError(ValueError):
    """A plugin that is not served, and why."""


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
class Plugin:
    """A plugin: its metadata, where it was read from, and what it serves.

    ``namespace`` is the one it is served under. ``skills`` are the ``skill://`` URIs of
    the ``SKILL.md`` of each skill served, ``prompts`` and ``agents`` the names served.
    ``skipped`` is each skill folder, command file and agent file not served, as its
    path from the root and the reason, in the order read.
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
) -> list[tuple[Path, tuple[str, ...]]]:
    """The ``.md`` files each entry names, each with its folders' name segments.

    A listed file is itself, with no segment. A folder is read at any depth, in path
    order, each folder between it and the file a segment; a dot folder is not read,
    and a file that resolves outside the root is not either.
    """
    found: list[tuple[Path, tuple[str, ...]]] = []
    for entry in entries:
        path = base / entry
        if entry.endswith(".md"):
            if _readable(base, entry) is None:
                notes.append(f"{kind} entry '{entry}' is not in the plugin")
            else:
                found.append((path, ()))
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
            found.append((file, file.relative_to(path).parent.parts))
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

    ``path`` is root-relative, ``uri`` the ``prompts://`` or ``agents://`` address a
    rewriter is shown, ``data`` the raw bytes and ``text`` them as ``read_text``
    gives them. ``stem`` and ``folders`` are what ``parse_agent`` names it by.
    """

    path: str
    uri: str
    data: bytes
    text: str
    item: FilePrompt | Agent
    stem: str
    folders: tuple[str, ...]


def _decoded(data: bytes) -> str:
    """``data`` as ``read_text`` gives it: UTF-8, its newlines translated."""
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _file_uri(scheme: str, namespace: str, folders: Iterable[str], name: str) -> str:
    """The address of a command or agent file: its path below its folder, under the
    namespace."""
    return f"{scheme}://{quote('/'.join((namespace, *folders, name)), safe='/')}"


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
    for the root). ``by_name`` and ``prefix`` are what its agents were parsed with.
    """

    plugin: Plugin
    catalogue: Mapping[str, skills.Skill]
    skill_dirs: Mapping[str, str]
    prompts: tuple[_File, ...]
    agents: tuple[_File, ...]
    by_name: Mapping[str, str]
    prefix: str


class _Skills(skills.EntryOnlySkillProvider):
    """The server's one skills provider: the skills of the request's build, in the
    order their plugins were added."""

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    async def load_skills(self) -> dict[str, skills.Skill]:
        return dict((await self._plugins.pinned()).build.catalogue)

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
        return list((await self._plugins.pinned()).build.prompts)

    async def get_tasks(self) -> list:
        # FastMCP lists every component when a server starts, to find the ones that
        # run as background tasks. Only a tool can, and a plugin serves none, so the
        # listing would only call the view's fn() outside any request (PLUG-53).
        return []


class _Agents(AgentsExtension):
    """The server's one agents extension: the agents of the request's build."""

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    async def _agents(self) -> tuple[Agent, ...]:
        return (await self._plugins.pinned()).build.agents

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

# How many builds are kept, the least recently used dropped first (PLUG-55).
BUILDS = 64

# A build key, made by Plugins._key alone.
_Key = tuple[Hashable, ...]


@dataclass(frozen=True)
class _Build:
    """What one build key is served: the admitted plugins and what they serve."""

    admitted: tuple[Plugin, ...]
    catalogue: Mapping[str, skills.Skill]
    prompts: tuple[FilePrompt, ...]
    agents: tuple[Agent, ...]
    rewritten: bool  # the rewriter chain was not empty
    indexes: Mapping[str, TextResource | None]
    _renderer: object = None  # kept alive with the build it rendered


@dataclass(frozen=True)
class _Pin:
    """What one request reads: the state, view and key it began with, and the build."""

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
    """What ``served`` serves, as the owners map names it: what -> its plugin."""
    name = served.plugin.name
    return {
        **dict.fromkeys(served.plugin.skills, name),
        **{f"prompt:{prompt}": name for prompt in served.plugin.prompts},
        **{f"agent:{agent}": name for agent in served.plugin.agents},
    }


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

    def prompt(self, served: _Served, file: _File) -> FilePrompt | None:
        """A command, rewritten: the one read when its text is unchanged, else
        re-parsed under its name."""
        item = file.item
        what = f"prompt {item.name}"
        site = self._site(served, "command", file.path, file.uri)
        new = self._run(served, site, file.text, what)
        if new is None:
            return None
        if new == file.text:
            return item
        try:
            return parse_prompt(new, item.name)
        except (ValueError, yaml.YAMLError) as exc:
            self._failed(served, None, file.path, what, str(exc))
            return None

    def agent(self, served: _Served, file: _File) -> Agent | None:
        """An agent, rewritten: the one read when its text is unchanged, else
        re-parsed, failed when it no longer parses or is named otherwise."""
        item = file.item
        what = f"agent {item.name}"
        site = self._site(served, "agent", file.path, file.uri)
        new = self._run(served, site, file.text, what)
        if new is None:
            return None
        if new == file.text:
            return item
        agent = parse_agent(
            new, served.by_name, served.prefix, stem=file.stem, folders=file.folders
        )
        if isinstance(agent, str):
            self._failed(served, None, file.path, what, agent)
            return None
        if agent.name != item.name:
            reason = f"its name is now {agent.name!r}, not {item.name!r}"
            self._failed(served, None, file.path, what, reason)
            return None
        return agent

    async def indexes(
        self,
        served: _Served,
        key: str,
        skill: skills.Skill,
        renderer: skills.EntryOnlySkillProvider,
    ) -> dict[str, TextResource | None]:
        """Each folder's ``_index.md`` and ``_index.json`` of the skill at ``key``,
        rendered by ``renderer`` and rewritten; None for one that failed."""
        found: dict[str, TextResource | None] = {}
        for rel in sorted(skills._folders_of(skill.files)):
            path = _joined(served.skill_dirs[key], rel)
            for name in (skills.INDEX_MD, skills.INDEX_JSON):
                uri = f"{skills._folder_uri(key, rel)}/{name}"
                resource = await skills._index_resource(renderer, uri)
                if not isinstance(resource, TextResource):
                    continue
                site = self._site(served, "index", path, uri)
                new = self._run(served, site, resource.text, uri)
                found[uri] = (
                    None if new is None else resource.model_copy(update={"text": new})
                )
        return found


class PluginSet:
    """A set of plugins built apart from what is served, then served whole (PLUG-50).

    ``add`` reads and checks a plugin as ``add_plugin`` does and ``keep`` carries one
    the server serves now, unread. A clash is checked against the plugins in this set
    alone, so a set that leaves a plugin out may add another that serves the same
    thing. A refused plugin raises and leaves the set unchanged. Nothing here is
    served until ``mcp.serve(set)``; the set is then a snapshot, and adding to it
    afterwards changes nothing served.
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
        self._carry(served)
        return served.plugin

    def keep(self, plugin: Plugin) -> Plugin:
        """Carries a plugin the server serves now into the set, without reading it
        again. A plugin not served now is a ``ValueError``; one that clashes with the
        set is a ``PluginError``."""
        for served in self._plugins.current:
            if served.plugin is plugin:
                break
        else:
            raise ValueError(
                f"plugin {plugin.name} is not served now: keep() carries a plugin "
                "this server serves"
            )
        try:
            self._carry(served)
        except PluginError as exc:
            raise PluginError(f"plugin {plugin.address} is not served: {exc}") from None
        return plugin

    def _carry(self, served: _Served) -> None:
        """Adds ``served`` to the set, refused whole when any of it is owned already."""
        owned = _owned(served)
        for key, _ in owned.items():
            owner = self._owners.get(key)
            if owner is not None:
                what = key if key.startswith("skill://") else key.replace(":", " ", 1)
                raise PluginError(f"{what} is already served by plugin {owner}")
        self._owners.update(owned)
        self._served.append(served)


class Plugins:
    """A server's plugins and loaders: what ``QuickMCP`` delegates to.

    Everything a plugin serves is read and checked against the set it joins before any
    of it is served, so a refused plugin leaves no trace. What is served is a
    ``_State``, replaced whole by ``serve`` and never changed in place. The server's
    one skills provider, prompts provider and agents extension read the request's pin
    instead (``pinned``): a build of what the view admits of the state the request
    began with.
    """

    def __init__(self, server: FastMCP) -> None:
        self._server = server
        self._loaders: dict[str, PluginLoader] = {}
        self._skills: _Skills | None = None
        self._prompts: _Prompts | None = None
        self._agents: _Agents | None = None
        self._state = _State(0, ())
        self._view: Callable[[], View] | None = None
        self._rewriters: tuple[Rewriter, ...] = ()
        self._memo = _Memo(OrderedDict(), {})
        self.add_loader(LocalLoader())

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
            if scheme in self._loaders:
                raise ValueError(f"a loader for {scheme}:// is already registered")
        for scheme in schemes:
            self._loaders[scheme] = loader

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
        self._providers(served)
        self._state = _State(self._state.generation + 1, served)
        self._memo = _Memo(OrderedDict(), {})  # last: no build of the old set is kept

    def set_view(self, fn: Callable[[], View]) -> None:
        """Makes ``fn`` the view, called once per request (PLUG-53)."""
        self._view = fn
        self._memo = _Memo(OrderedDict(), {})  # last: no build of the old view is kept

    def add_rewriter(self, fn: Rewriter) -> None:
        """Adds ``fn`` to the end of the rewriters, run by every build (PLUG-56)."""
        self._rewriters = (*self._rewriters, fn)
        self._memo = _Memo(OrderedDict(), {})  # last: no build without it is kept

    def _chain(self, key: _Key) -> tuple[Rewriter, ...]:
        """The rewriters a build for ``key`` runs, in order: the one place the order
        is decided. Empty means no rewriting, and no copies."""
        return self._rewriters

    def _key(self, state: _State, view: View) -> _Key:
        """The build key of ``view`` over ``state``: the one place a key is made."""
        return (state.generation, view.key)

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
        """One pin, made fresh. The memo is taken first, so a build begun before a
        change is never stored where a request after the change finds it."""
        memo = self._memo
        state = self._state
        view = self._view() if self._view else _EVERY
        key = self._key(state, view)
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
                memo.builds.popitem(last=False)
        return flight.build

    async def _make(self, state: _State, view: View, key: _Key) -> _Build:
        """A build of what ``view`` admits of ``state``, from what was read: it never
        reads the disk. With no rewriter it serves the objects read; with any, the
        rewritten text, and each skill folder's indexes, rendered now."""
        chain = self._chain(key)
        admitted = tuple(s for s in state.current if view.admits(s.plugin))
        plugins = tuple(s.plugin for s in admitted)
        if not chain:
            return _Build(
                admitted=plugins,
                catalogue=MappingProxyType(
                    {k: skill for s in admitted for k, skill in s.catalogue.items()}
                ),
                prompts=tuple(f.item for s in admitted for f in s.prompts),
                agents=tuple(f.item for s in admitted for f in s.agents),
                rewritten=False,
                indexes=MappingProxyType({}),
            )
        rewrite = _Rewrite(chain, view.key, plugins)
        catalogue: dict[str, skills.Skill] = {}
        prompts: list[FilePrompt] = []
        agents: list[Agent] = []
        for served in admitted:
            for k, skill in served.catalogue.items():
                if (built := rewrite.skill(served, k, skill)) is not None:
                    catalogue[k] = built
            for file in served.prompts:
                if (prompt := rewrite.prompt(served, file)) is not None:
                    prompts.append(prompt)
            for file in served.agents:
                if (agent := rewrite.agent(served, file)) is not None:
                    agents.append(agent)
        # A throwaway renderer over the build: FastMCP keys what it loads per request
        # by its id, so the build keeps it alive while a request may still see it.
        renderer = skills.EntryOnlySkillProvider(catalogue)
        indexes: dict[str, TextResource | None] = {}
        async with Context(fastmcp=self._server):
            for served in admitted:
                for k in served.catalogue:
                    if k in catalogue:
                        indexes |= await rewrite.indexes(
                            served, k, catalogue[k], renderer
                        )
        return _Build(
            admitted=plugins,
            catalogue=MappingProxyType(catalogue),
            prompts=tuple(prompts),
            agents=tuple(agents),
            rewritten=True,
            indexes=MappingProxyType(indexes),
            _renderer=renderer,
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
            loader = self._loader_for(address)
            try:
                root = loader.materialise(address)
            except PluginError:
                raise
            except (OSError, ValueError) as exc:
                raise PluginError(str(exc) or type(exc).__name__) from exc
            plugin = read_plugin(root, manifest)
            notes = list(plugin.components.notes)
            read = self._read(plugin, prefix, notes, owners)
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
        )
        return _Served(
            plugin=plugin,
            catalogue=dict(read.catalogue),
            skill_dirs=dict(read.skill_dirs),
            prompts=tuple(read.prompts),
            agents=tuple(read.agents),
            by_name=dict(read.by_name),
            prefix=read.prefix,
        )

    def _claim(
        self,
        key: str,
        plugin: Plugin,
        what: str,
        seen: dict[str, Path],
        path: Path,
        skip: _Skip,
        owners: dict[str, str],
        *,
        files: bool = True,
    ) -> bool:
        """Whether ``key`` is free: refused when served, skipped when this plugin
        has it already, the warning naming both files when ``files``."""
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
        seen[key] = path
        return True

    def _read(
        self, plugin: Plugin, prefix: str, notes: list[str], owners: dict[str, str]
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
                    uri, plugin, uri, seen, location, skip, owners, files=False
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
        # namespace other than its own is a prefix.
        own = prefix == self._server.name
        named = "" if own else prefix.replace("/", "_") + "_"
        prompts: list[_File] = []
        for path, folders_in in _md_files(base, "commands", parts.commands, notes):
            segments = (*folders_in, path.stem)
            name = named + "_".join(segments)
            if not all(_usable(part) for part in (*segments, name)):
                reason = f"prompt name {name!r} is not a usable name"
                log.warning("prompt %s is skipped: %s", path, reason)
                skip(path, reason)
                continue
            try:
                data = path.read_bytes()
                text = _decoded(data)
                prompt = parse_prompt(text, name)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                log.warning("prompt %s is skipped: %s", path, exc)
                skip(path, str(exc))
                continue
            if self._claim(
                f"prompt:{prompt.name}",
                plugin,
                f"prompt {prompt.name}",
                seen,
                path,
                skip,
                owners,
            ):
                prompts.append(
                    _File(
                        path=skip.relative(path),
                        uri=_file_uri("prompts", prefix, folders_in, path.name),
                        data=data,
                        text=text,
                        item=prompt,
                        stem=path.stem,
                        folders=tuple(folders_in),
                    )
                )

        agents: list[_File] = []
        for path, segments in _md_files(base, "agents", parts.agents, notes):
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
                f"agent:{agent.name}",
                plugin,
                f"agent {agent.name}",
                seen,
                path,
                skip,
                owners,
            ):
                agents.append(
                    _File(
                        path=skip.relative(path),
                        uri=_file_uri("agents", prefix, segments, path.name),
                        data=data,
                        text=text,
                        item=agent,
                        stem=path.stem,
                        folders=tuple(segments),
                    )
                )
        return _Read(
            catalogue, skill_dirs, prompts, agents, by_name, named, skip.skipped
        )

    def _providers(self, served: tuple[_Served, ...]) -> None:
        # The first plugin with a skill brings the Skills extension (PLUG-20), the
        # first plugin the prompts provider and the agents extension (PLUG-27).
        if self._skills is None and any(each.catalogue for each in served):
            self._skills = _Skills(self)
            skills.serve(self._server, self._skills)
        if not served:
            return
        if self._prompts is None:
            self._prompts = _Prompts(self)
            self._server.add_provider(self._prompts)
        if self._agents is None:
            self._agents = _Agents(self)
            self._server.add_extension(self._agents)
            self._server.add_middleware(
                AdvertiseLegacyExtensions({EXTENSION_ID: SETTINGS})
            )
