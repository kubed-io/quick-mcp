"""The installed distribution: the import path, its version, and the namespace it lives in."""

from __future__ import annotations

import importlib
from importlib.metadata import version


def test_imports() -> None:
    module = importlib.import_module("kubed.mcp")
    assert module.__name__ == "kubed.mcp"


def test_version_is_the_installed_distributions() -> None:
    import kubed.mcp

    assert kubed.mcp.__version__ == version("kubed-mcp")


def test_kubed_is_a_namespace_package() -> None:
    # A kubed/__init__.py would make `kubed` a regular package, and whichever of
    # kubed-mcp, kubed-mcp-kb or kubed-selenium-flow installed second would shadow the rest.
    import kubed

    assert getattr(kubed, "__file__", None) is None
    assert not hasattr(kubed, "__version__")
