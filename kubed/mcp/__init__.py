"""The platform mcp-kb and selenium-flow share on top of FastMCP.

It will own how a server serves plugins, the MCP protocol details that come with
them, and the extension points loaded through Python entry points. Today it holds
only its own version.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("kubed-mcp")
