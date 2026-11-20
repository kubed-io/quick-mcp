"""The protocol plumbing a server's extensions, providers and middleware use.

Re-exported from FastMCP and the MCP SDK as the same objects. ``types`` is the SDK's
``mcp_types`` module, whole, for the wire types not named here. ``TasksExtension``
needs the ``tasks`` extra: ``pip install 'kubed-mcp[tasks]'``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp_types as types
from fastmcp.apps import UI_EXTENSION_ID, AppConfig, ResourceCSP
from fastmcp.prompts.base import InputRequiredPromptResult
from fastmcp.server.extensions import MethodBinding, ServerExtension
from fastmcp.server.low_level import client_supports_extension
from fastmcp.server.providers import Provider
from fastmcp.server.transforms import PromptsAsTools
from fastmcp.tools import Tool
from fastmcp.utilities.tasks import TaskConfig
from fastmcp.utilities.versions import VersionSpec
from mcp_types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PaginatedRequestParams,
    PromptReference,
    RequestParams,
    ResourceTemplateReference,
    ToolAnnotations,
)
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from mcp.shared.exceptions import MCPError

if TYPE_CHECKING:
    from fastmcp_tasks import TasksExtension

__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "MODERN_PROTOCOL_VERSIONS",
    "PARSE_ERROR",
    "UI_EXTENSION_ID",
    "AppConfig",
    "InputRequiredPromptResult",
    "MCPError",
    "MethodBinding",
    "PaginatedRequestParams",
    "PromptReference",
    "PromptsAsTools",
    "Provider",
    "RequestParams",
    "ResourceCSP",
    "ResourceTemplateReference",
    "ServerExtension",
    "TaskConfig",
    "TasksExtension",
    "Tool",
    "ToolAnnotations",
    "VersionSpec",
    "client_supports_extension",
    "types",
]


def __getattr__(name: str) -> Any:
    if name == "TasksExtension":
        try:
            from fastmcp_tasks import TasksExtension
        except ImportError as exc:
            raise ImportError(
                "TasksExtension needs the tasks extra: pip install 'kubed-mcp[tasks]'"
            ) from exc
        return TasksExtension
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
