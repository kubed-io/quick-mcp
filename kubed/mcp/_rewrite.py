"""Rewriters: what a server does to the text it serves, when a build is made.

A rewriter is ``fn(site, text) -> str``. It is shown each text file a build serves,
with the ``Site`` that says where the file is and who it is served to, and returns the
text to serve in its place. ``run`` applies a chain of them in order; one that raises
or returns anything but a ``str`` is a ``RewriteFailed`` naming it, and fails only the
item it was rewriting (PLUG-59).
"""

from __future__ import annotations

import mimetypes
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

# FastMCP registers text/markdown for .md here: a file is text when it is served so.
import fastmcp.skills.provider  # noqa: F401

if TYPE_CHECKING:
    from .plugins import Plugin

Kind = Literal["skill", "command", "agent", "index"]


@dataclass(frozen=True)
class Site:
    """Where a text file is served, and to whom (PLUG-57).

    ``path`` is the file's path in ``plugin``, root-relative; for an index, its
    folder's. ``uri`` is the address it is served at; for an index, the index's own.
    ``view`` is the view's key, ``admitted`` the plugins it admits, in the order
    served. ``skills`` maps each skill folder ``plugin`` serves, root-relative (``""``
    for the root), to the folder's ``skill://`` URI; one build shares it, read-only.
    """

    plugin: Plugin
    namespace: str
    kind: Kind
    path: str
    uri: str
    view: Hashable
    admitted: tuple[Plugin, ...]
    skills: Mapping[str, str]


Rewriter = Callable[[Site, str], str]


def text_of(path: str, data: bytes) -> str | None:
    """The text of the file at ``path``, when it is served as ``text/*`` and its bytes
    are UTF-8 (PLUG-58); else None, and no rewriter sees it."""
    mime_type = mimetypes.guess_type(path)[0]
    if mime_type is None or not mime_type.startswith("text/"):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


class RewriteFailed(ValueError):
    """A rewriter that raised or returned anything but a ``str``, and why."""

    def __init__(self, rewriter: str, reason: str) -> None:
        super().__init__(f"{rewriter}: {reason}")
        self.rewriter = rewriter
        self.reason = reason


def name_of(rewriter: Rewriter) -> str:
    """How a log names ``rewriter``: its ``__qualname__``, else its ``repr``."""
    return getattr(rewriter, "__qualname__", None) or repr(rewriter)


def run(rewriters: Sequence[Rewriter], site: Site, text: str) -> str:
    """``text`` after each of ``rewriters``, in order."""
    for rewriter in rewriters:
        try:
            result = rewriter(site, text)
        except Exception as exc:
            # Blind on purpose: a server's rewriter may raise anything, and PLUG-59
            # fails only its item. Re-raised, so ruff's BLE001 does not apply.
            reason = str(exc) or type(exc).__name__
            raise RewriteFailed(name_of(rewriter), reason) from exc
        if not isinstance(result, str):
            reason = f"returned {type(result).__name__}, not str"
            raise RewriteFailed(name_of(rewriter), reason)
        text = result
    return text
