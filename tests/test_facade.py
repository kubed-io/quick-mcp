"""The facade: every name in every ``__all__`` imports, and is FastMCP's own object."""

from __future__ import annotations

import importlib
import warnings

import pytest

FACADES = (
    "kubed.mcp",
    "kubed.mcp.protocol",
    "kubed.mcp.auth",
    "kubed.mcp.request",
    "kubed.mcp.skills",
)

# Where each re-export comes from, so a re-export is checked to be the same object.
SOURCES = {
    "kubed.mcp": {
        "Context": "fastmcp.server.context",
        "Depends": "fastmcp.dependencies",
        "FastMCPError": "fastmcp.exceptions",
        "Image": "fastmcp.utilities.types",
        "Middleware": "fastmcp.server.middleware",
        "NotFoundError": "fastmcp.exceptions",
        "Prompt": "fastmcp.prompts",
        "PromptArgument": "fastmcp.prompts",
        "PromptError": "fastmcp.exceptions",
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
    },
    "kubed.mcp.protocol": {
        "AppConfig": "fastmcp.apps",
        "INTERNAL_ERROR": "mcp_types",
        "INVALID_PARAMS": "mcp_types",
        "INVALID_REQUEST": "mcp_types",
        "METHOD_NOT_FOUND": "mcp_types",
        "MODERN_PROTOCOL_VERSIONS": "mcp_types.version",
        "PARSE_ERROR": "mcp_types",
        "InputRequiredPromptResult": "fastmcp.prompts.base",
        "MCPError": "mcp.shared.exceptions",
        "MethodBinding": "fastmcp.server.extensions",
        "PaginatedRequestParams": "mcp_types",
        "PromptReference": "mcp_types",
        "PromptsAsTools": "fastmcp.server.transforms",
        "Provider": "fastmcp.server.providers",
        "RequestParams": "mcp_types",
        "ResourceCSP": "fastmcp.apps",
        "ResourceTemplateReference": "mcp_types",
        "ServerExtension": "fastmcp.server.extensions",
        "TaskConfig": "fastmcp.utilities.tasks",
        "TasksExtension": "fastmcp_tasks",
        "Tool": "fastmcp.tools",
        "ToolAnnotations": "mcp_types",
        "UI_EXTENSION_ID": "fastmcp.apps",
        "VersionSpec": "fastmcp.utilities.versions",
        "client_supports_extension": "fastmcp.server.low_level",
    },
    "kubed.mcp.auth": {
        "AccessToken": "fastmcp.server.auth",
        "AuthProvider": "fastmcp.server.auth",
        "JWTVerifier": "fastmcp.server.auth",
        "MultiAuth": "fastmcp.server.auth",
        "TokenVerifier": "fastmcp.server.auth",
        "get_access_token": "fastmcp.server.dependencies",
    },
    "kubed.mcp.request": {
        "get_context": "fastmcp.server.dependencies",
        "get_http_request": "fastmcp.server.dependencies",
    },
    "kubed.mcp.skills": {
        "Skill": "fastmcp.skills",
        "SkillFileResource": "fastmcp.skills.provider",
    },
}

# Names this package defines rather than re-exports.
OWN = {
    "kubed.mcp": {"Delivery", "Inject", "QuickMCP", "Serving", "lookup"},
    "kubed.mcp.protocol": {"AdvertiseLegacyExtensions", "types"},
    "kubed.mcp.auth": set(),
    "kubed.mcp.request": {"Values", "request_values"},
    "kubed.mcp.skills": {
        "EntryOnlySkillProvider",
        "FolderNotFound",
        "Folders",
        "NameTheIndex",
        "SkillsExtension",
        "load_skill",
        "serve",
        "without_reserved",
    },
}


@pytest.mark.parametrize("facade", FACADES)
def test_every_name_in_all_imports_without_a_deprecation(facade: str) -> None:
    module = importlib.import_module(facade)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name in module.__all__:
            getattr(module, name)
        exec(f"from {facade} import *", {})


@pytest.mark.parametrize("facade", FACADES)
def test_every_re_export_is_the_same_object(facade: str) -> None:
    module = importlib.import_module(facade)
    assert set(module.__all__) == set(SOURCES[facade]) | OWN[facade]
    for name, source in SOURCES[facade].items():
        assert getattr(module, name) is getattr(importlib.import_module(source), name)


def test_types_is_the_sdks_wire_types() -> None:
    import mcp_types

    from kubed.mcp.protocol import types

    assert types is mcp_types
    assert types.ElicitResult is mcp_types.ElicitResult
