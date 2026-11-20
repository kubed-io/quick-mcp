"""kubed-mcp: an opinionated platform on top of FastMCP.

``QuickMCP`` builds the server, its container builds what the tools use, and
``Inject[T]`` hands it to them. The rest of this namespace is the FastMCP a tool author
touches, re-exported as the same objects, so ``isinstance`` and pydantic keep working.
Protocol plumbing is in ``kubed.mcp.protocol``, auth in ``kubed.mcp.auth`` and the
request's headers and query in ``kubed.mcp.request``.
"""

from __future__ import annotations

from importlib.metadata import version

from fastmcp.dependencies import Depends
from fastmcp.exceptions import (
    FastMCPError,
    NotFoundError,
    PromptError,
    ResourceError,
    ToolError,
    ValidationError,
)
from fastmcp.prompts import Prompt, PromptArgument
from fastmcp.resources import (
    Resource,
    ResourceContent,
    ResourceResult,
    ResourceTemplate,
    TextResource,
)
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp_types import TextContent

from .app import QuickMCP, Serving
from .inject import Inject, lookup

__version__ = version("kubed-mcp")

__all__ = [
    "Context",
    "Depends",
    "FastMCPError",
    "Image",
    "Inject",
    "Middleware",
    "NotFoundError",
    "Prompt",
    "PromptArgument",
    "PromptError",
    "QuickMCP",
    "Resource",
    "ResourceContent",
    "ResourceError",
    "ResourceResult",
    "ResourceTemplate",
    "Serving",
    "TextContent",
    "TextResource",
    "ToolError",
    "ToolResult",
    "ValidationError",
    "lookup",
]
