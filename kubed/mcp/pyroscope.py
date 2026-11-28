"""Source links for profiles: the ``.pyroscope.yaml`` a server's image implies.

Pyroscope reads ``.pyroscope.yaml`` from the server's repository at the profile's
``service_git_ref``, so the file is committed. Only the built image knows the true paths
and versions its frames come from, so the command runs there and maps what is installed.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import json
import platform
import re
import sys
import sysconfig
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import Distribution
from pathlib import Path, PurePosixPath

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
    """The installed distribution of that name, or None.

    ``python -m`` puts the working directory on ``sys.path``, and a ``.dist-info`` or
    ``.egg-info`` left there, as ``pip install .`` does, would stand in for the
    installed distribution. So the search leaves out the working directory.
    """
    cwd = Path.cwd().resolve()
    path = [p for p in sys.path if p and Path(p).resolve() != cwd]
    return next(iter(importlib.metadata.distributions(name=name, path=path)), None)


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


def _normalised(document: object) -> dict[str, object] | None:
    """What a ``.pyroscope.yaml`` means, apart from its order, grouping and comments.

    One entry per prefix, holding its language and its source as sorted JSON. A
    document of any other shape is None.
    """
    if not isinstance(document, dict):
        return None
    code = document.get("source_code")
    mappings = code.get("mappings") if isinstance(code, dict) else None
    if not isinstance(mappings, list):
        return None
    prefixes: dict[str, tuple[str, str]] = {}
    for mapping in mappings:
        if not isinstance(mapping, dict):
            return None
        paths = mapping.get("path")
        language = mapping.get("language")
        source = mapping.get("source")
        if (
            not isinstance(paths, list)
            or not isinstance(language, str)
            or not isinstance(source, dict)
        ):
            return None
        for entry in paths:
            if not isinstance(entry, dict) or not isinstance(entry.get("prefix"), str):
                return None
            prefixes[entry["prefix"]] = (language, json.dumps(source, sort_keys=True))
    return {"version": document.get("version"), "mappings": prefixes}


def _say(line: str) -> None:
    sys.stderr.write(f"kubed.mcp.pyroscope: {line}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Write the image's ``.pyroscope.yaml``, or check a committed copy of it."""
    parser = argparse.ArgumentParser(
        prog="python -m kubed.mcp.pyroscope",
        description="Write or check the .pyroscope.yaml that this image implies.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="print the file on stdout")
    mode.add_argument(
        "--check", action="store_true", help="compare a committed copy with the image"
    )
    parser.add_argument("distribution", help="the server's own distribution")
    parser.add_argument(
        "file", nargs="?", help="with --check: the committed file, or - for stdin"
    )
    args = parser.parse_args(argv)
    if args.write and args.file is not None:
        parser.error("--write takes no file")

    try:
        name, mappings = _mappings(args.distribution, _say)
    except _Unmappable as error:
        _say(str(error))
        return 2
    rendered = _render(name, mappings)
    if args.write:
        sys.stdout.write(rendered)
        return 0

    if args.file is None or args.file == "-":
        committed = sys.stdin.read()
    else:
        try:
            committed = Path(args.file).read_text(encoding="utf-8")
        except (OSError, ValueError) as error:
            _say(f"{args.file} can't be read: {type(error).__name__}")
            return 2
    if not committed.strip():
        _say(
            "read nothing: pass the file's path, or run docker with -i so stdin "
            "reaches the command"
        )
        return 2

    try:
        document = yaml.safe_load(committed)
    except yaml.YAMLError:
        document = None
    expected = _normalised(yaml.safe_load(rendered))
    if expected is not None and _normalised(document) == expected:
        return 0
    diff = difflib.unified_diff(
        committed.splitlines(keepends=True),
        rendered.splitlines(keepends=True),
        fromfile=".pyroscope.yaml (committed)",
        tofile=".pyroscope.yaml (this image)",
    )
    for line in diff:
        sys.stdout.write(line if line.endswith("\n") else line + "\n")
    _say(f".pyroscope.yaml is stale: regenerate it with --write {name} in this image")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
