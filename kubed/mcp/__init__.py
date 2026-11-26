"""kubed-mcp: an opinionated platform on top of FastMCP.

``QuickMCP`` builds the server, its container builds what the tools use, and
``Inject[T]`` hands it to them. The rest of this namespace is the FastMCP a tool author
touches, re-exported as the same objects, so ``isinstance`` and pydantic keep working.
Protocol plumbing is in ``kubed.mcp.protocol``, auth in ``kubed.mcp.auth`` and the
request's headers and query in ``kubed.mcp.request``.

Every name but ``Config``, ``DeliverySettings``, ``Field``, ``NoDecode``,
``ServingSettings``, ``Source``, ``TelemetrySettings``, ``config`` and ``config_schema``
resolves on first use, so importing ``kubed.mcp`` loads no FastMCP until one of them is
used, and a config module can declare its sections and write their schema without
loading FastMCP.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from ._builtins import DeliverySettings, ServingSettings
from ._config import Config, Field, NoDecode, Source, config, config_schema
from .telemetry import TelemetrySettings

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

    from .app import QuickMCP
    from .delivery import Delivery
    from .inject import Inject, lookup

__version__ = version("kubed-mcp")

__all__ = [
    "Config",
    "Context",
    "Delivery",
    "DeliverySettings",
    "Depends",
    "FastMCPError",
    "Field",
    "Image",
    "Inject",
    "Middleware",
    "NoDecode",
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
    "ServingSettings",
    "Source",
    "TelemetrySettings",
    "TextContent",
    "TextResource",
    "ToolError",
    "ToolResult",
    "ValidationError",
    "config",
    "config_schema",
    "lookup",
]

_LAZY: dict[str, str] = {
    "Delivery": ".delivery",
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
    "TextContent": "mcp_types",
    "TextResource": "fastmcp.resources",
    "ToolError": "fastmcp.exceptions",
    "ToolResult": "fastmcp.tools",
    "ValidationError": "fastmcp.exceptions",
    "lookup": ".inject",
}


# Hidden from type checkers, so an unknown name stays an error to them.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        module = _LAZY.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value
        return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
