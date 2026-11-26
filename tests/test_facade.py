"""The facade: every name in every ``__all__`` imports, and is FastMCP's own object."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

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
    "kubed.mcp": {
        "Config",
        "Delivery",
        "Field",
        "Inject",
        "QuickMCP",
        "Serving",
        "Telemetry",
        "TelemetrySettings",
        "config",
        "config_schema",
        "lookup",
    },
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


# -- importing

# The top-level packages importing kubed.mcp never loads.
FASTMCP = ("fastmcp", "fastmcp_tasks", "mcp", "mcp_types")

# Printed last by every fresh interpreter: what it loaded, and what dir() misses of __all__.
REPORT = f"""
import json, sys
package = sys.modules["kubed.mcp"]
print(json.dumps({{
    "loaded": sorted(m for m in sys.modules if m.split(".")[0] in {FASTMCP!r} or m.startswith("kubed.mcp.")),
    "missing_from_dir": sorted(set(package.__all__) - set(dir(package))),
}}))
"""


def _run(code: str) -> list[str]:
    """``code`` then REPORT, in a new interpreter (``-I``: the installed package, no cwd)."""
    out = subprocess.run(
        [sys.executable, "-I", "-c", code + "\n" + REPORT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def _fresh(code: str) -> dict:
    return json.loads(_run(code)[-1])


@pytest.mark.parametrize(
    "code",
    [
        "import kubed.mcp",
        "from kubed.mcp import TelemetrySettings",
        "import kubed.mcp.telemetry",
        "from kubed.mcp import telemetry",
        "from kubed.mcp import TelemetrySettings\nTelemetrySettings(otel=True, tags={'team': 'a'})",
        "from kubed.mcp import Config, Field, config, config_schema",
        "from kubed.mcp import Config, Field, config, config_schema\n"
        "@config('recording')\n"
        "class Recording(Config):\n"
        "    inbox: str | None = Field(None, doc='Where', env='RECORDING_INBOX')\n"
        "assert config_schema(Recording)['properties']['recording']",
    ],
)
def test_importing_kubed_mcp_imports_no_fastmcp(code: str) -> None:
    assert _fresh(code) == {
        "loaded": ["kubed.mcp._config", "kubed.mcp.telemetry"],
        "missing_from_dir": [],
    }


def test_the_first_fastmcp_name_used_imports_it() -> None:
    lines = _run(
        """
import json, sys, kubed.mcp
before = "fastmcp" in sys.modules
q = kubed.mcp.QuickMCP
import kubed.mcp.app
print(json.dumps({"before": before, "after": "fastmcp" in sys.modules,
    "same": q is kubed.mcp.app.QuickMCP, "cached": "QuickMCP" in vars(kubed.mcp)}))
"""
    )
    seen = json.loads(lines[-2])
    assert seen == {"before": False, "after": True, "same": True, "cached": True}


def test_an_unknown_name_is_an_attribute_error_as_in_any_module() -> None:
    import kubed.mcp

    name = "Nope"  # a variable, so ruff's B009 doesn't ask for plain attribute access
    with pytest.raises(
        AttributeError, match=r"^module 'kubed\.mcp' has no attribute 'Nope'$"
    ):
        getattr(kubed.mcp, name)
    assert not hasattr(kubed.mcp, name)
    with pytest.raises(
        ImportError, match=r"^cannot import name 'Nope' from 'kubed\.mcp'"
    ):
        exec("from kubed.mcp import Nope", {})


def test_a_submodule_still_imports_from_the_package() -> None:
    from kubed.mcp import protocol

    assert protocol is importlib.import_module("kubed.mcp.protocol")


@pytest.mark.parametrize("module", ["kubed.mcp.telemetry", "kubed.mcp._config"])
def test_the_config_modules_import_only_pydantic_and_the_standard_library(
    module: str,
) -> None:
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

    tops: set[str] = set()
    relative: list[str] = []

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                tops.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative.append(node.module or "")
                else:
                    tops.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.If):
                guard = node.test
                is_type_checking = (
                    isinstance(guard, ast.Name) and guard.id == "TYPE_CHECKING"
                ) or (
                    isinstance(guard, ast.Attribute) and guard.attr == "TYPE_CHECKING"
                )
                if not is_type_checking:
                    walk(node.body)
                walk(node.orelse)

    walk(tree.body)
    assert tops <= set(sys.stdlib_module_names) | {"__future__", "pydantic"}
    assert relative == []
