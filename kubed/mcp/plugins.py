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
and each reads the plugins served once per call. The layout and names are Claude
Code's: a skill is one folder below ``skills/``, while commands and agents are files at
any depth, each folder adding a name segment, Claude Code's ``:`` written ``_``
(``commands/db/migrate.md`` is ``db_migrate``). A plugin added with no namespace is the
server's own, its skills under ``skill://<server name>/``; there is no root, and an
empty namespace is refused. Plugins added under one namespace merge, and a second thing
at one address is refused.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

import yaml
from fastmcp import FastMCP
from fastmcp.prompts import Prompt
from fastmcp.server.providers import Provider

from . import skills
from .agents import EXTENSION_ID, SETTINGS, Agent, AgentsExtension, parse_agent
from .agents import usable_name as _usable
from .prompts import FilePrompt, load_prompt
from .protocol import AdvertiseLegacyExtensions

__all__ = [
    "Components",
    "LocalLoader",
    "Plugin",
    "PluginError",
    "PluginLoader",
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


@dataclass
class _Read:
    """A plugin's kinds, read and checked, before anything is served."""

    catalogue: dict[str, skills.Skill]
    prompts: list[FilePrompt]
    agents: list[Agent]


@dataclass(frozen=True)
class _Served:
    """One plugin as served: the plugin, and what it serves of each kind."""

    plugin: Plugin
    catalogue: Mapping[str, skills.Skill]
    prompts: tuple[FilePrompt, ...]
    agents: tuple[Agent, ...]


class _Skills(skills.EntryOnlySkillProvider):
    """The server's one skills provider: every plugin's skills, in the order added."""

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    async def load_skills(self) -> dict[str, skills.Skill]:
        return {
            key: skill
            for served in self._plugins.current
            for key, skill in served.catalogue.items()
        }


class _Prompts(Provider):
    """The server's one prompts provider: every plugin's commands."""

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    async def _list_prompts(self) -> list[Prompt]:
        return [prompt for served in self._plugins.current for prompt in served.prompts]


class _Agents(AgentsExtension):
    """The server's one agents extension: every plugin's agents."""

    def __init__(self, plugins: Plugins) -> None:
        super().__init__()
        self._plugins = plugins

    def _served(self) -> list[Agent]:
        return [agent for served in self._plugins.current for agent in served.agents]

    def clone(self) -> _Agents:
        return type(self)(self._plugins)


class Plugins:
    """A server's plugins and loaders: what ``QuickMCP`` delegates to.

    Everything a plugin serves is read and checked against what the server already
    serves before any of it is served, so a refused plugin leaves no trace. What is
    served is ``current``, a tuple replaced whole when a plugin is added and never
    changed in place; the server's one skills provider, prompts provider and agents
    extension read it.
    """

    def __init__(self, server: FastMCP) -> None:
        self._server = server
        self._loaders: dict[str, PluginLoader] = {}
        self._owners: dict[str, str] = {}  # what is served -> the plugin serving it
        self._skills: _Skills | None = None
        self._prompts: _Prompts | None = None
        self._agents: _Agents | None = None
        self.current: tuple[_Served, ...] = ()
        self.add_loader(LocalLoader())

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

    def add(
        self,
        address: str | os.PathLike[str],
        namespace: str | None = None,
        manifest: Mapping[str, object] | None = None,
    ) -> Plugin:
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
            read = self._read(plugin, prefix, notes)
        except PluginError as exc:
            raise PluginError(f"plugin {address} is not served: {exc}") from None
        for note in dict.fromkeys(notes):
            log.warning("plugin %s: %s", plugin.name, note)
        return self._serve(
            replace(
                plugin,
                address=address,
                namespace=prefix,
                components=replace(plugin.components, notes=tuple(notes)),
            ),
            read,
        )

    def _claim(
        self,
        key: str,
        plugin: Plugin,
        what: str,
        seen: dict[str, Path | None],
        path: Path | None = None,
    ) -> bool:
        """Whether ``key`` is free: refused when served, skipped when this plugin
        has it already, the warning naming both files when ``path`` is given."""
        owner = self._owners.get(key)
        if owner is not None:
            raise PluginError(f"{what} is already served by plugin {owner}")
        if key in seen:
            first = seen[key]
            if path is None or first is None:
                log.warning(
                    "plugin %s: %s is skipped, served already", plugin.name, what
                )
            else:
                log.warning(
                    "%s is already served by %s; %s is skipped", what, first, path
                )
            return False
        seen[key] = path
        return True

    def _read(self, plugin: Plugin, prefix: str, notes: list[str]) -> _Read:
        base = plugin.root.resolve()
        parts = plugin.components
        seen: dict[str, Path | None] = {}

        catalogue: dict[str, skills.Skill] = {}
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
                if not self._claim(uri, plugin, uri, seen):
                    continue
                skill = skills.load_skill(location)
                if skill is None:
                    continue
                catalogue[key] = skill
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
        prompts: list[FilePrompt] = []
        for path, segments in _md_files(base, "commands", parts.commands, notes):
            segments = (*segments, path.stem)
            name = named + "_".join(segments)
            if not all(_usable(part) for part in (*segments, name)):
                log.warning(
                    "prompt %s is skipped: prompt name %r is not a usable name",
                    path,
                    name,
                )
                continue
            try:
                prompt = load_prompt(path, name=name)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                log.warning("prompt %s is skipped: %s", path, exc)
                continue
            if self._claim(
                f"prompt:{prompt.name}", plugin, f"prompt {prompt.name}", seen, path
            ):
                prompts.append(prompt)

        agents: list[Agent] = []
        for path, segments in _md_files(base, "agents", parts.agents, notes):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                log.warning("agent file %s is skipped: %s", path, exc)
                continue
            agent = parse_agent(text, by_name, named, stem=path.stem, folders=segments)
            if isinstance(agent, str):
                log.warning("agent file %s is skipped: %s", path, agent)
            elif self._claim(
                f"agent:{agent.name}", plugin, f"agent {agent.name}", seen, path
            ):
                agents.append(agent)
        return _Read(catalogue, prompts, agents)

    def _serve(self, plugin: Plugin, read: _Read) -> Plugin:
        # The first plugin with a skill brings the Skills extension (PLUG-20), the
        # first plugin the prompts provider and the agents extension (PLUG-27).
        if read.catalogue and self._skills is None:
            self._skills = _Skills(self)
            skills.serve(self._server, self._skills)
        if self._prompts is None:
            self._prompts = _Prompts(self)
            self._server.add_provider(self._prompts)
        if self._agents is None:
            self._agents = _Agents(self)
            self._server.add_extension(self._agents)
            self._server.add_middleware(
                AdvertiseLegacyExtensions({EXTENSION_ID: SETTINGS})
            )
        plugin = replace(
            plugin,
            skills=tuple(
                f"skill://{quote(key, safe='/')}/{skills.ENTRY}"
                for key in read.catalogue
            ),
            prompts=tuple(prompt.name for prompt in read.prompts),
            agents=tuple(agent.name for agent in read.agents),
        )
        for key in plugin.skills:
            self._owners[key] = plugin.name
        for name in plugin.prompts:
            self._owners[f"prompt:{name}"] = plugin.name
        for name in plugin.agents:
            self._owners[f"agent:{name}"] = plugin.name
        served = _Served(
            plugin, dict(read.catalogue), tuple(read.prompts), tuple(read.agents)
        )
        self.current = (*self.current, served)
        return plugin
