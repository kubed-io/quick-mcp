"""``list_resources`` and ``read_resource``: resources for a client that cannot read
them, as tools (DLV-6).

A server with a plugin serves both, over every resource it has. With
``X-MCP-Resources: off`` they are in ``tools/list``; with ``on`` they are left out of
it, and stay callable by name.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable, Iterable
from typing import Any

from fastmcp.exceptions import DisabledError, FastMCPError, NotFoundError, ToolError
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool, ToolResult
from fastmcp.utilities.versions import VersionSpec
from mcp_types import ImageContent, TextContent, ToolAnnotations

from mcp.shared.exceptions import MCPError

from . import skills
from ._media import media_type
from .delivery import BinaryPointer, _resolved
from .gateway import prefixed
from .request import get_context

log = logging.getLogger("kubed.mcp.delivery")

TOOL_NAMES = ("list_resources", "read_resource")

# An MCP Apps address, with or without a gateway prefix before it: matched at the start.
_UI = re.compile(r"(?:[A-Za-z][A-Za-z0-9+.-]*\+)?ui://", re.IGNORECASE)
# The blobs that are image content, whatever their size.
_IMAGES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
# A blob's media type when it declares none: FastMCP gives bytes this one.
_UNDECLARED = "application/octet-stream"
NOT_TEXT = "Not returned: binary content is not text."


def _read_only(title: str) -> ToolAnnotations:
    """mcp-kb's annotations for its resource tools (TOOL-6), with the tool's title."""
    return ToolAnnotations(
        title=title,
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )


class _ResourceTools(Provider):
    """The server's one provider of ``list_resources`` and ``read_resource``.

    It lists them only when the request turned resources off, and always answers them
    by name: FastMCP finds a tool to call by ``_get_tool``, never by its listing.
    """

    def __init__(self, pointer: Callable[[], BinaryPointer | None]) -> None:
        super().__init__()
        self._pointer = pointer
        self._tools = {
            tool.name: tool
            for tool in (
                Tool.from_function(
                    self._list,
                    name="list_resources",
                    title="List resources",
                    description=(
                        "List every resource this server has, as resources/list does "
                        "for this request."
                    ),
                    annotations=_read_only("List resources"),
                ),
                Tool.from_function(
                    self._read,
                    name="read_resource",
                    title="Read a resource",
                    description="Read one resource by its URI, as resources/read does.",
                    annotations=_read_only("Read a resource"),
                ),
            )
        }

    async def _list_tools(self) -> list[Tool]:
        if _resolved().resources == "off":
            return list(self._tools.values())
        return []

    async def _get_tool(
        self, name: str, version: VersionSpec | None = None
    ) -> Tool | None:
        tool = self._tools.get(name)
        if tool is None or (version is not None and not version.matches(tool.version)):
            return None
        return tool

    async def get_tasks(self) -> list:
        return []  # neither runs as a background task: see plugins._Prompts.get_tasks

    async def _list(self) -> ToolResult:
        # The rows resources/list returns, through the server's middleware, as sent.
        listed = await get_context().list_resources()
        prefix = _resolved().prefix
        rows = []
        for resource in listed:
            row = skills._wire(resource)
            if _UI.match(row["uri"]):
                continue
            # A gateway rewrites no tool result, so the uri is written here (GW-5).
            rows.append({**row, "uri": prefixed(row["uri"], prefix)})
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({"resources": rows}, ensure_ascii=False),
                )
            ]
        )

    async def _read(self, uri: str) -> ToolResult:
        if _UI.match(uri):
            raise ToolError(
                f"{uri} is an app for a host to draw, not to read.",
                log_level=logging.INFO,
            )
        try:
            # Through the server's middleware, so the prefix, the %2F refusal and the
            # folder's hint apply as they do to resources/read.
            result = await get_context().read_resource(uri)
        except MCPError as exc:
            raise ToolError(exc.message, log_level=logging.INFO) from None
        except (NotFoundError, DisabledError):
            # What FastMCP's resources/read handler says, naming the URI as sent.
            raise ToolError(
                f"Resource not found: {uri!r}", log_level=logging.INFO
            ) from None
        except FastMCPError as exc:
            raise ToolError(str(exc), log_level=logging.INFO) from None
        return ToolResult(content=_contents(uri, result.contents, self._pointer()))


def _contents(
    uri: str, contents: Iterable[Any], pointer: BinaryPointer | None
) -> list[TextContent | ImageContent]:
    """What ``read_resource`` returns for each content read of ``uri`` (DLV-6)."""
    # The guess reads the extension past the scheme: mimetypes parses a whole URI of a
    # scheme it does not know as a URL, and guesses nothing.
    path = re.split(r"[?#]", uri.partition("://")[2] or uri, maxsplit=1)[0]
    out: list[TextContent | ImageContent] = []
    for content in contents:
        data = content.content
        if isinstance(data, str):
            out.append(TextContent(type="text", text=data))
            continue
        media = content.mime_type
        if not media or media == _UNDECLARED:
            media = media_type(path)
        if media in _IMAGES:
            out.append(
                ImageContent(
                    type="image", data=base64.b64encode(data).decode(), mime_type=media
                )
            )
            continue
        if media.startswith("text/") or media == "image/svg+xml":
            try:
                out.append(TextContent(type="text", text=data.decode("utf-8")))
                continue
            except UnicodeDecodeError:
                pass
        read = NOT_TEXT
        pointed = _pointed(pointer, uri)
        if pointed:
            read = f"{read} {pointed}"
        described = {
            "uri": uri,
            "mime_type": media,
            "bytes": len(data),
            "binary": True,
            "read": read,
        }
        out.append(
            TextContent(type="text", text=json.dumps(described, ensure_ascii=False))
        )
    return out


def _pointed(pointer: BinaryPointer | None, uri: str) -> str | None:
    """Where the server says ``uri``'s link is listed, or None (DLV-6).

    A pointer that raises, or returns anything but a string, is logged at error and
    gives none: the read still answers.
    """
    if pointer is None:
        return None
    try:
        pointed = pointer(uri)
    except Exception:
        log.exception("binary_pointer failed for %r: the read gives no pointer", uri)
        return None
    if pointed is None or isinstance(pointed, str):
        return pointed
    log.error(
        "binary_pointer returned %s for %r, not a string: the read gives no pointer",
        type(pointed).__name__,
        uri,
    )
    return None
