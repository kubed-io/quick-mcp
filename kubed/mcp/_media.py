"""Media types: what a skill's file is served as, and whether it is text (PLUG-75).

A file's media type is ``TEXT_TYPES``'s for its extension, compared in lowercase, else
the one Python's ``mimetypes`` guesses, else ``application/octet-stream``. A file is
text-like when its extension is in ``TEXT_TYPES`` or its media type is a text one, and
it is a text file when it is text-like and its bytes are UTF-8: it reads as text, and
a rewriter sees it (PLUG-58). Any other file reads as a blob of its bytes.
"""

from __future__ import annotations

import mimetypes
import posixpath
from collections.abc import Mapping
from types import MappingProxyType

__all__ = ["TEXT_TYPES", "media_type", "text_like", "text_of"]

TEXT_TYPES: Mapping[str, str] = MappingProxyType(
    {
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".ini": "text/plain",
        ".cfg": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".toml": "application/toml",
        ".xml": "application/xml",
        ".sh": "application/x-sh",
        ".bash": "application/x-sh",
        ".py": "text/x-python",
        ".js": "text/javascript",
        ".ts": "text/x-typescript",
    }
)

# The application types PLUG-75 counts as text, beside every text/* one.
_TEXT_APPLICATIONS = frozenset(
    {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/javascript",
        "application/x-sh",
        "application/toml",
    }
)
_TEXT_SUFFIXES = ("+json", "+xml", "+yaml")


def _extension(path: str) -> str:
    return posixpath.splitext(path)[1].lower()


def media_type(path: str) -> str:
    """The media type the file at ``path`` is served as."""
    return (
        TEXT_TYPES.get(_extension(path))
        or mimetypes.guess_type(path)[0]
        or "application/octet-stream"
    )


def text_like(path: str) -> bool:
    """Whether the file at ``path`` reads as text when its bytes are UTF-8."""
    if _extension(path) in TEXT_TYPES:
        return True
    media = media_type(path)
    return (
        media.startswith("text/")
        or media in _TEXT_APPLICATIONS
        or media.endswith(_TEXT_SUFFIXES)
    )


def text_of(path: str, data: bytes) -> str | None:
    """The text of the file at ``path`` when it is a text file, else None."""
    if not text_like(path):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None
