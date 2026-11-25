"""How skills are served: the building blocks the plugin layer serves skills with.

Not an API a server calls on its own: the plugin layer (a plugin folder fetched by a
loader) will apply these rules to every skill it serves. Until it lands, ``serve`` is
the one call that composes them.

A catalogue maps a path to a ``Skill``, and the path's last segment is the skill's name:
``lib/folder/name`` is served at ``skill://lib/folder/name/SKILL.md``. An
``EntryOnlySkillProvider`` serves one, and its listing names each skill's ``SKILL.md``
and nothing else. Every other file, the skill's ``_manifest`` and the ``_index.json``
and ``_index.md`` beside each of its folders are read by exact URI, found through
``SKILL.md``.

A folder is a directory resource (SEP-2640): its URI has no trailing slash, it is never
listed, and it serves no content. ``resources/directory/read`` lists its direct
children, from ``SkillsExtension``: FastMCP's Skills extension bound for every protocol
version, with ``directoryRead``. The extension asks each provider on the server that has
an async ``children(uri)`` (``Folders``), per request.

What a provider knows of its folders, their titles and the reserved names is derived
from the catalogue it serves for the request, so a subclass that overrides
``load_skills()`` keeps all of it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import posixpath
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import mcp_types
from fastmcp import FastMCP
from fastmcp.exceptions import NotFoundError
from fastmcp.resources import Resource, TextResource
from fastmcp.resources.base import ResourceContent, ResourceResult
from fastmcp.server.context import Context
from fastmcp.server.extensions import MethodBinding, ServerExtension
from fastmcp.server.middleware import Middleware
from fastmcp.server.providers import Provider
from fastmcp.skills import Skill, SkillManifest
from fastmcp.skills import extension as _extension
from fastmcp.skills.provider import SkillCatalogProvider, SkillFileResource
from fastmcp.utilities.versions import VersionSpec
from mcp_types import INVALID_PARAMS, PaginatedRequestParams
from pydantic import Field, ValidationError

from mcp.shared.exceptions import MCPError

from . import _media
from .protocol import AdvertiseLegacyExtensions

__all__ = [
    "EntryOnlySkillProvider",
    "FolderNotFound",
    "Folders",
    "NameTheIndex",
    "Skill",
    "SkillFileResource",
    "SkillsExtension",
    "load_skill",
    "serve",
    "without_reserved",
]

log = logging.getLogger(__name__)

ENTRY = "SKILL.md"
# FastMCP answers skill://<path>/_manifest with the skill's file listing.
MANIFEST = "_manifest"
INDEX_JSON = "_index.json"
INDEX_MD = "_index.md"
INODE_DIRECTORY = "inode/directory"
DIRECTORY_READ = "resources/directory/read"
SETTINGS = {"directoryRead": True}

Catalogue = Mapping[str, Skill] | Iterable[Skill]


# ---- loading -----------------------------------------------------------------------


def _reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
    return str(exc) or type(exc).__name__


def load_skill(path: str | os.PathLike[str]) -> Skill | None:
    """The skill in folder ``path``, without its reserved files, or None.

    A folder with no ``SKILL.md`` is no skill: a warning, and None. A skill FastMCP's
    ``Skill.from_directory`` refuses is not served: an error with the reason, and None.
    """
    skill = _load_skill(Path(path))
    return None if isinstance(skill, str) else skill


def _load_skill(path: Path) -> Skill | str:
    """``load_skill``, logging the same, but the reason in place of None."""
    if not (path / ENTRY).is_file():
        log.warning("no SKILL.md at %s", path)
        return f"no {ENTRY}"
    try:
        skill = Skill.from_directory(path)
    except (OSError, ValueError) as exc:
        reason = _reason(exc)
        log.error(
            "skill at %s is not a valid Agent Skill, so it is not served: %s",
            path,
            reason,
        )
        return reason
    return without_reserved(skill)


def _is_index(path: str) -> bool:
    """Whether a file or folder at ``path`` sits where its parent's index is served."""
    return path.rpartition("/")[2] in (INDEX_JSON, INDEX_MD)


def _unserved(files: Iterable[str]) -> set[str]:
    """The files whose URI the server answers otherwise: an index, the root manifest."""
    return {path for path in files if _is_index(path) or path == MANIFEST}


def _all_folders(paths: Iterable[str]) -> set[str]:
    """Every folder the files at ``paths`` sit in, below the root."""
    return {
        "/".join(parts[:depth])
        for path in paths
        for parts in [path.split("/")[:-1]]
        for depth in range(1, len(parts) + 1)
    }


def _folders_of(paths: Iterable[str]) -> set[str]:
    """A skill's folders, the root ``""`` included.

    A folder with an index's name is none, and neither is any folder under it; nor is a
    ``_manifest`` at the root, whose URI reads as the manifest.
    """
    return {""} | {
        rel
        for rel in _all_folders(paths)
        if not any(_is_index(part) for part in rel.split("/"))
        and rel.split("/")[0] != MANIFEST
    }


def without_reserved(skill: Skill) -> Skill:
    """The skill minus any file whose URI the server answers otherwise, each named.

    That is a file with an index's name, which reads as its folder's index, and a
    ``_manifest`` at the root, which reads as the manifest. FastMCP would find such a
    file unreadable as listed and drop the whole skill from ``skills/list``. A
    ``_manifest`` below the root is an ordinary file. A folder at such a URI is named
    too, though its files stay in the skill. The same skill comes back when nothing is
    reserved.
    """
    for rel in sorted(_all_folders(skill.files)):
        if _is_index(rel):
            log.warning(
                "skill %s: folder %s is not served, "
                "its parent's index answers that URI",
                skill.name,
                rel,
            )
        elif rel == MANIFEST:
            log.warning(
                "skill %s: folder %s is not served, "
                "the skill's manifest answers that URI",
                skill.name,
                rel,
            )
    unserved = sorted(_unserved(skill.files))
    for name in unserved:
        answers = "the skill's manifest" if name == MANIFEST else "its folder's index"
        log.warning(
            "skill %s: %s is not served, %s answers that URI", skill.name, name, answers
        )
    if not unserved:
        return skill
    return Skill.from_files(
        {name: data for name, data in skill.files.items() if name not in unserved}
    )


def _heading(data: bytes) -> str | None:
    """A Markdown file's first ``# `` heading outside a code fence, if it has one.

    A file that is not UTF-8 has none, so it is a plain file of the skill rather than
    taking the skill down with it.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    fence = None
    for line in text.splitlines():
        marker = line.lstrip()[:3]
        if marker in ("```", "~~~"):
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
        elif fence is None and line.startswith("# "):
            return line[2:].strip() or None
    return None


# ---- folders -----------------------------------------------------------------------


class Folders(Protocol):
    async def children(self, uri: str) -> list[dict] | None:
        """The direct children of folder ``uri`` as wire ``Resource`` dicts, sorted by
        URI, or None when ``uri`` names no folder of this provider."""


async def _children(providers: Iterable[Provider], uri: str) -> list[dict] | None:
    """The children of folder ``uri`` from the first provider that knows it."""
    for provider in providers:
        children = getattr(provider, "children", None)
        if children is not None and (found := await children(uri)) is not None:
            return found
    return None


def _wire(resource: Resource | mcp_types.Resource) -> dict:
    """A resource as ``resources/list`` puts it on the wire."""
    if isinstance(resource, Resource):
        resource = resource.to_mcp_resource()
    return resource.model_dump(mode="json", by_alias=True, exclude_none=True)


def _folder(uri: str, name: str, description: str) -> dict:
    """A folder's directory resource."""
    return _wire(
        mcp_types.Resource(
            uri=uri, name=name, description=description, mimeType=INODE_DIRECTORY
        )
    )


def _index_md(children: list[dict]) -> str:
    """One line per child: a description's own newlines (YAML ``|``) collapse."""
    return "".join(
        f"{c['uri']}: {' '.join(c.get('description', '').split())}\n" for c in children
    )


_INDEXES = {
    INDEX_JSON: ("application/json", lambda children: json.dumps(children, indent=2)),
    INDEX_MD: ("text/markdown", _index_md),
}


async def _index_resource(folders: Folders, uri: str) -> Resource | None:
    """The index file at ``uri``, or None when it is not one beside a folder."""
    parent, _, name = uri.rpartition("/")
    if name not in _INDEXES:
        return None
    children = await folders.children(parent)
    if children is None:
        return None
    return _index_of(uri, children)


def _index_of(uri: str, children: list[dict]) -> TextResource:
    """The index file at ``uri``, one of ``_INDEXES``, listing ``children``."""
    parent, _, name = uri.rpartition("/")
    mime_type, render = _INDEXES[name]
    return TextResource(
        uri=uri,
        name=uri.split("://", 1)[-1],
        description=f"What is in {parent}",
        mime_type=mime_type,
        text=render(children),
    )


class _DirectoryReadParams(PaginatedRequestParams):
    uri: str = Field(min_length=1)


class _DirectoryRead:
    """``resources/directory/read`` over one ``Folders``: a page of direct children.

    The cursor is the decimal offset of the next child. Any offset inside the folder,
    from 1 to one less than the number of children, is accepted; anything else is
    ``-32602``.
    """

    page_size = 100

    def __init__(self, folders: Folders):
        self._folders = folders

    def binding(self) -> MethodBinding:
        return MethodBinding(DIRECTORY_READ, _DirectoryReadParams, self.read)

    async def read(self, ctx: Any, params: _DirectoryReadParams) -> dict:
        children = await self._folders.children(params.uri)
        if children is None:
            raise MCPError(code=INVALID_PARAMS, message="Directory not found")
        start = _offset(params.cursor, len(children))
        end = start + self.page_size
        result: dict = {"resources": children[start:end]}
        if end < len(children):
            result["nextCursor"] = str(end)
        return result


class _FolderExtension(Folders, Protocol):
    """An extension that binds ``resources/directory/read`` over its ``children``: the
    Skills extension and the plugin layer's agents extension."""

    identifier: str

    @property
    def server(self) -> FastMCP: ...


def _answered_elsewhere(extension: _FolderExtension, method: str) -> bool:
    """Whether another extension identifier on ``extension``'s server bound ``method``.

    FastMCP has no public record of which extension bound a method, so this reads
    ``FastMCP._extension_methods`` (FastMCP 4.2.5's identifier-to-methods map), here
    alone. Its own identifier is not another's, so a registration that replaces its own
    (FastMCP's same-identifier path) binds the method again. A handler added outside an
    extension is not counted, so FastMCP refuses the binding as it would. An unbound
    extension answers False.
    """
    try:
        server = extension.server
    except RuntimeError:
        return False
    return any(
        method in bound
        for identifier, bound in server._extension_methods.items()
        if identifier != extension.identifier
    )


def _with_directory_read(
    extension: _FolderExtension, bindings: Iterable[MethodBinding]
) -> tuple[MethodBinding, ...]:
    """``bindings``, and ``resources/directory/read`` over ``extension``'s ``children``
    unless another extension on its server bound it already: whichever registers first
    binds it, so it is bound once in every order."""
    bindings = tuple(bindings)
    if _answered_elsewhere(extension, DIRECTORY_READ):
        return bindings
    return (*bindings, _DirectoryRead(extension).binding())


def _offset(cursor: str | None, total: int) -> int:
    if cursor is None:
        return 0
    # Length first: int() refuses a string of more than 4300 digits.
    if (
        cursor.isascii()
        and cursor.isdecimal()
        and len(cursor) <= len(str(total))
        and 0 < int(cursor) < total
    ):
        return int(cursor)
    raise MCPError(code=INVALID_PARAMS, message="Invalid directory cursor")


class FolderNotFound(MCPError):
    """A read of a folder URI: not found, as a missing URI is, naming the index.

    ``hint`` is the sentence alone, and ``miss`` what a tool that reads resources says,
    as it says a plain miss.
    """

    def __init__(self, uri: str):
        self.hint = f"{uri} is a folder: read {uri}/{INDEX_MD} to list it."
        self.miss = f"no resource at {uri!r}. {self.hint}"
        super().__init__(
            code=INVALID_PARAMS,
            message=f"Resource not found: {uri!r}. {self.hint}",
            data={"uri": uri},
        )


class NameTheIndex(Middleware):
    """Turn the not-found of a folder read into one that names its ``_index.md``.

    A provider answers None for a folder URI, as for any missing one, and FastMCP
    reports a bare ``Resource not found``. The code and ``data`` stay what FastMCP
    sends; only the message gains the hint. The folder is any the server's providers
    know (``Folders``).
    """

    async def on_read_resource(self, context: Any, call_next: Any) -> Any:
        try:
            return await call_next(context)
        except NotFoundError:
            uri = str(context.message.uri)
            server = context.fastmcp_context and context.fastmcp_context.fastmcp
            if server is None or await _children(server.providers, uri) is None:
                raise
            raise FolderNotFound(uri) from None


def _name_the_index(server: FastMCP) -> None:
    """Add ``NameTheIndex`` to ``server`` unless it has one (SKL-4, PLUG-68)."""
    if not any(isinstance(m, NameTheIndex) for m in server.middleware):
        server.add_middleware(NameTheIndex())


# ---- the provider ------------------------------------------------------------------


def _authored(skill: object) -> bool:
    # FastMCP's Skill also answers isinstance() for its 4.1 wire model, MCPSkill.
    return issubclass(type(skill), Skill)


def _trimmed(skills: Catalogue | None) -> Catalogue | None:
    if isinstance(skills, Mapping):
        return {
            path: without_reserved(skill) if _authored(skill) else skill
            for path, skill in skills.items()
        }
    if skills is None:
        return None
    return [without_reserved(skill) if _authored(skill) else skill for skill in skills]


def _root(path: str) -> str:
    """The URI of the skill served at ``path``, which is its root folder's."""
    return f"skill://{quote(path, safe='/')}"


def _folder_uri(path: str, rel: str) -> str:
    """The URI of the folder at ``rel`` in the skill at ``path``, with no trailing /."""
    return f"{_root(path)}/{quote(rel, safe='/')}" if rel else _root(path)


def _file_uri(path: str, file: str) -> str:
    return f"{_root(path)}/{quote(file, safe='/')}"


def _without_unserved(manifest: SkillManifest) -> SkillManifest:
    unserved = _unserved(manifest.files)
    if not unserved:
        return manifest
    files = {p: f for p, f in manifest.files.items() if p not in unserved}
    return manifest.model_copy(update={"files": files})


class _SkillFile(SkillFileResource):
    """A skill's file other than its ``SKILL.md``, read as text when it is a text
    file (PLUG-75) and as a blob of its bytes otherwise."""

    async def read(self) -> ResourceResult:
        data = await self.provider.read_file(self)
        text = _media.text_of(self.file_path, data)
        value: str | bytes = data if text is None else text
        return ResourceResult([ResourceContent(value, mime_type=self.mime_type)])


class EntryOnlySkillProvider(SkillCatalogProvider):
    """A catalogue of skills, listed by each skill's ``SKILL.md`` alone.

    ``_get_resource`` still answers every file, the manifest and each folder's two index
    files by exact URI. A folder is every one a skill's files sit in. A file whose URI
    the server answers otherwise is never served: a catalogue given here is trimmed with
    ``without_reserved``, and one ``load_skills()`` returns is trimmed silently.

    The provider brings ``SkillsExtension``, which FastMCP registers once however many
    providers bring it.
    """

    def __init__(self, skills: Catalogue | None = None) -> None:
        super().__init__(_trimmed(skills))

    def required_extensions(self) -> Sequence[ServerExtension]:
        return (SkillsExtension(),)

    async def load_manifests(self) -> Mapping[str, SkillManifest]:
        """The catalogue's manifests, less the files whose URI is answered otherwise."""
        manifests = await super().load_manifests()
        return {path: _without_unserved(m) for path, m in manifests.items()}

    async def _folder_at(self, uri: str) -> tuple[str, SkillManifest, str] | None:
        """The skill path, its manifest and the folder's path in it, for a folder."""
        if not uri.startswith("skill://"):
            return None
        for path, manifest in (await self.load_manifests()).items():
            root = _root(path)
            if uri != root and not uri.startswith(f"{root}/"):
                continue
            # Matched as sent: a URI is a folder's only when it is spelled as one.
            for rel in _folders_of(manifest.files):
                if _folder_uri(path, rel) == uri:
                    return path, manifest, rel
            return None
        return None

    async def children(self, uri: str) -> list[dict] | None:
        found = await self._folder_at(uri)
        if found is None:
            return None
        path, manifest, rel = found
        folders = _folders_of(manifest.files)
        name = manifest.frontmatter.name
        rows = [
            _folder(
                _folder_uri(path, sub),
                f"{path}/{sub}",
                f"A folder of the {name} skill",
            )
            for sub in folders
            if sub and posixpath.dirname(sub) == rel
        ]
        for file in manifest.files:
            if posixpath.dirname(file) == rel:
                resource = await self._get_resource(_file_uri(path, file))
                if resource is not None:
                    rows.append(_wire(resource))
        return sorted(rows, key=lambda row: row["uri"])

    async def _list_resources(self) -> list[Resource]:
        return [
            r
            for r in await super()._list_resources()
            if isinstance(r, SkillFileResource) and r.file_path == ENTRY
        ]

    async def _get_resource(
        self, uri: str, version: VersionSpec | None = None
    ) -> Resource | None:
        if uri.startswith("skill://"):
            index = await _index_resource(self, uri)
            if index is not None:
                return index
        resource = await super()._get_resource(uri, version)
        if isinstance(resource, SkillFileResource) and resource.file_path != ENTRY:
            title = None
            if resource.file_path.endswith(".md"):
                title = _heading(await self.read_file(resource))
            name = posixpath.basename(resource.skill_path)
            # Still a SkillFileResource with every other field kept, `file` included:
            # FastMCP offers a skill's entry only while every file resolves to the one
            # its manifest lists.
            fields = {
                field: getattr(resource, field)
                for field in SkillFileResource.model_fields
            }
            fields["description"] = title or f"File from {name} skill"
            fields["mime_type"] = _media.media_type(resource.file_path)
            resource = _SkillFile(**fields)
        return resource


# ---- the extension -----------------------------------------------------------------


class SkillsExtension(_extension.SkillsExtension):
    """FastMCP's Skills extension, for every protocol version, with folders.

    ``skills/list`` and ``skills/get`` are FastMCP's handlers, bound with no
    ``protocol_versions`` so a handshake-era client is answered too. It declares
    ``directoryRead`` and binds ``resources/directory/read`` over every provider on the
    server that knows its folders, unless another extension there binds it already (the
    plugin layer's agents extension, registered first): either asks every provider.
    """

    def settings(self) -> dict:
        return dict(SETTINGS)

    def methods(self) -> Sequence[MethodBinding]:
        return _with_directory_read(
            self,
            (
                dataclasses.replace(binding, protocol_versions=None)
                for binding in super().methods()
            ),
        )

    def clone(self) -> SkillsExtension:
        # FastMCP's clone() builds its own class, which would drop what this one adds.
        return type(self)()

    async def children(self, uri: str) -> list[dict] | None:
        """The children of folder ``uri`` from the server's providers, per request."""
        return await _served_children(self.server, uri)


async def _served_children(server: FastMCP, uri: str) -> list[dict] | None:
    """The children of folder ``uri`` from ``server``'s providers, inside a
    ``Context``, so a catalogue loaded per request is loaded once for it (SKL-21)."""
    async with Context(fastmcp=server):
        return await _children(server.providers, uri)


def _advertises(server: FastMCP, identifier: str) -> bool:
    """Whether ``server`` already declares ``identifier`` in a legacy ``initialize``."""
    return any(
        isinstance(m, AdvertiseLegacyExtensions) and identifier in m._declared
        for m in server.middleware
    )


# ---- composing them ----------------------------------------------------------------


def serve(
    server: FastMCP, skills: Catalogue | EntryOnlySkillProvider
) -> EntryOnlySkillProvider:
    """Serve a catalogue of skills on ``server``, by the rules above.

    Each call adds one provider, ``skills`` itself when it is one; the provider brings
    ``SkillsExtension``, which FastMCP registers once. A call also adds the two
    middlewares a served skill needs: ``AdvertiseLegacyExtensions`` for the extension,
    unless one on the server declares it already, and ``NameTheIndex``, unless the
    server has one. So a later call adds neither again.
    """
    provider = (
        skills
        if isinstance(skills, EntryOnlySkillProvider)
        else EntryOnlySkillProvider(skills)
    )
    server.add_provider(provider)
    if not _advertises(server, SkillsExtension.identifier):
        server.add_middleware(
            AdvertiseLegacyExtensions({SkillsExtension.identifier: dict(SETTINGS)})
        )
    _name_the_index(server)
    return provider
