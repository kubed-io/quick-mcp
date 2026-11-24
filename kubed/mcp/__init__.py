"""kubed-mcp: an opinionated platform on top of FastMCP.

``QuickMCP`` builds the server, its container builds what the tools use, and
``Inject[T]`` hands it to them. The rest of this namespace is the FastMCP a tool author
touches, re-exported as the same objects, so ``isinstance`` and pydantic keep working.
Protocol plumbing is in ``kubed.mcp.protocol``, auth in ``kubed.mcp.auth`` and the
request's headers and query in ``kubed.mcp.request``.

Every name but ``Telemetry`` and ``TelemetrySettings`` resolves on first use, so
importing ``kubed.mcp`` loads no FastMCP until one of them is used, and a config module
can import ``TelemetrySettings`` alone.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from .telemetry import Telemetry, TelemetrySettings

if TYPE_CHECKING:
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
    "Telemetry",
    "TelemetrySettings",
    "TextContent",
    "TextResource",
    "ToolError",
    "ToolResult",
    "ValidationError",
    "lookup",
]

_LAZY: dict[str, str] = {
    "Context": "fastmcp.server.context",
    "Depends": "fastmcp.dependencies",
    "FastMCPError": "fastmcp.exceptions",
    "Image": "fastmcp.utilities.types",
    "Inject": ".inject",
    "Middleware": "fastmcp.server.middleware",
    "NotFoundError": "fastmcp.exceptions",
    "Prompt": "fastmcp.prompts",
    "PromptArgument": "fastmcp.prompts",
    "PromptError": "fastmcp.exceptions",
    "QuickMCP": ".app",
    "Resource": "fastmcp.resources",
    "ResourceContent": "fastmcp.resources",
    "ResourceError": "fastmcp.exceptions",
    "ResourceResult": "fastmcp.resources",
    "ResourceTemplate": "fastmcp.resources",
    "Serving": ".app",
    "TextContent": "mcp_types",
    "TextResource": "fastmcp.resources",
    "ToolError": "fastmcp.exceptions",
    "ToolResult": "fastmcp.tools",
    "ValidationError": "fastmcp.exceptions",
    "lookup": ".inject",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
