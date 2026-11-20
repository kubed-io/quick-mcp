"""kubed-mcp: an opinionated platform on top of FastMCP.

``QuickMCP`` builds the server, its container builds what the tools use, and
``Inject[T]`` hands it to them.
"""

from __future__ import annotations

from importlib.metadata import version

from fastmcp.dependencies import Depends
from fastmcp.server.context import Context

from .app import QuickMCP, Serving
from .inject import Inject, lookup

__version__ = version("kubed-mcp")

__all__ = [
    "Context",
    "Depends",
    "Inject",
    "QuickMCP",
    "Serving",
    "lookup",
]
