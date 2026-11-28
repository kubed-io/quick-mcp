"""Source links for profiles: the ``.pyroscope.yaml`` a server's image implies.

Pyroscope reads ``.pyroscope.yaml`` from the server's repository at the profile's
``service_git_ref``, so the file is committed. Only the built image knows the true paths
and versions its frames come from, so the command runs there and maps what is installed.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sysconfig
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import PurePosixPath

import yaml

# Distributions that live in a folder of a monorepo, keyed by normalised name.
_MONOREPO: dict[str, str] = {"fastmcp-slim": "fastmcp_slim"}

# The labels PEP 753 gives a project's source code, as `_github` normalises them.
_SOURCE_LABELS = {"source", "repository", "sourcecode", "github"}

_GITHUB = re.compile(r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")
_DEV = re.compile(r"\+g([0-9a-f]{7,40})")
_FINAL = re.compile(r"^\d+\.\d+\.\d+$")
_RC = re.compile(r"^(\d+\.\d+\.\d+)rc(\d+)$")


@dataclass(frozen=True)
class _Mapping:
    """One installed directory and the source it was built from."""

    directory: str  # absolute, ends in "/"
    source: dict[str, dict[str, str]]


class _Unmappable(Exception):
    """The image can't be mapped. The message is the stderr line, without its prefix."""


def _normalise(name: str) -> str:
    """PEP 503: lowercase, each run of ``-``, ``_`` and ``.`` as ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _ref(version: str, commit: str | None) -> str | None:
    """The git ref kubed-mcp's installed version was built from, or None."""
    dev = _DEV.search(version)
    if dev:
        return commit or dev.group(1)
    if _FINAL.match(version):
        return f"v{version}"
    rc = _RC.match(version)
    if rc:
        return f"v{rc.group(1)}-rc.{rc.group(2)}"
    return None


def _commit(dist: Distribution) -> str | None:
    """The commit pip recorded for a git install (PEP 610), or None."""
    text = dist.read_text("direct_url.json")
    if text is None:
        return None
    try:
        info = json.loads(text)
    except ValueError:
        return None
    vcs = info.get("vcs_info") if isinstance(info, dict) else None
    commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
    return commit if isinstance(commit, str) else None


def _github(dist: Distribution) -> tuple[str, str]:
    """The owner and repository the distribution's Project-URL names for its source."""
    for entry in dist.metadata.get_all("Project-URL") or []:
        label, _, url = entry.partition(", ")
        if re.sub(r"[\W_]", "", label).lower() in _SOURCE_LABELS:
            found = _GITHUB.match(url.strip())
            if found:
                return found.group(1), found.group(2)
            break
    raise _Unmappable(
        f"{dist.metadata['Name']}'s Project-URL names no GitHub repository"
    )


def _packages(dist: Distribution) -> list[tuple[str, str]]:
    """The distribution's packages as (import path, directory), by import path.

    A package is a directory holding an ``__init__.py`` while no directory above it
    does, so a ``kubed.*`` package is ``kubed/mcp_kb`` and never ``kubed/``, and a
    package's subpackages are not listed apart from it.
    """
    files = {PurePosixPath(str(f)) for f in dist.files or []}
    found = []
    for path in files:
        parent = path.parent
        if path.name != "__init__.py" or parent == PurePosixPath("."):
            continue
        if any(up / "__init__.py" in files for up in parent.parents):
            continue
        found.append((parent.as_posix(), f"{dist.locate_file(parent)}/"))
    return sorted(found)


def _distribution(name: str) -> Distribution | None:
    """The installed distribution of that name, or None."""
    try:
        return importlib.metadata.distribution(name)
    except PackageNotFoundError:
        return None


def _stdlib() -> str:
    return sysconfig.get_paths()["stdlib"]


def _python() -> str:
    return platform.python_version()


def _github_source(
    owner: str, repo: str, ref: str, path: str
) -> dict[str, dict[str, str]]:
    return {"github": {"owner": owner, "repo": repo, "ref": ref, "path": path}}


def _mappings(name: str, warn: Callable[[str], None]) -> tuple[str, list[_Mapping]]:
    """The named distribution's metadata Name and its mappings, in the spec's order."""
    dist = _distribution(name)
    if dist is None:
        raise _Unmappable(f"{name} is not installed")
    shown = dist.metadata["Name"]
    packages = _packages(dist)
    if not packages:
        raise _Unmappable(f"{shown}'s files hold no package")
    mappings = [
        _Mapping(directory, {"local": {"path": path}}) for path, directory in packages
    ]
    done = {_normalise(shown)}

    if "kubed-mcp" not in done:
        done.add("kubed-mcp")
        mcp = _distribution("kubed-mcp")
        if mcp is None:
            raise _Unmappable("kubed-mcp is not installed")
        mcp_packages = _packages(mcp)
        if not mcp_packages:
            warn(
                f"{mcp.metadata['Name']}'s files hold no package, as in an editable "
                "install: its frames stay unmapped"
            )
        else:
            owner, repo = _github(mcp)
            version = mcp.metadata["Version"]
            ref = _ref(version, _commit(mcp))
            if ref is None:
                raise _Unmappable(
                    f"{mcp.metadata['Name']} {version} has no tag or commit to map to"
                )
            mappings += [
                _Mapping(directory, _github_source(owner, repo, ref, path))
                for path, directory in mcp_packages
            ]

    if "fastmcp-slim" not in done:
        done.add("fastmcp-slim")
        slim = _distribution("fastmcp-slim")
        if slim is None:
            warn("fastmcp-slim is not installed: its frames stay unmapped")
        else:
            slim_packages = _packages(slim)
            if not slim_packages:
                warn(
                    f"{slim.metadata['Name']}'s files hold no package, as in an "
                    "editable install: its frames stay unmapped"
                )
            else:
                owner, repo = _github(slim)
                ref = f"v{slim.metadata['Version']}"
                folder = _MONOREPO["fastmcp-slim"]
                mappings += [
                    _Mapping(
                        directory, _github_source(owner, repo, ref, f"{folder}/{path}")
                    )
                    for path, directory in slim_packages
                ]

    stdlib = _stdlib().rstrip("/") + "/"
    mappings.append(
        _Mapping(stdlib, _github_source("python", "cpython", f"v{_python()}", "Lib"))
    )
    return shown, mappings


def _render(name: str, mappings: Sequence[_Mapping]) -> str:
    """The header for ``name``, then the document: the same image, the same bytes."""
    header = (
        f"# Generated by `python -m kubed.mcp.pyroscope --write {name}` "
        "inside the server's image.\n"
        "# Regenerate it there rather than edit it: CI checks it with --check.\n"
    )
    document = {
        "version": "v1",
        "source_code": {
            "mappings": [
                {
                    "path": [
                        {"prefix": m.directory},
                        {"prefix": m.directory[1:]},
                    ],
                    "language": "python",
                    "source": m.source,
                }
                for m in mappings
            ]
        },
    }
    return header + yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
