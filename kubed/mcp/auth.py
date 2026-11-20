"""Auth for a server: the providers, the verifiers and the caller's token.

Re-exported from FastMCP as the same objects.
"""

from __future__ import annotations

from fastmcp.server.auth import (
    AccessToken,
    AuthProvider,
    JWTVerifier,
    MultiAuth,
    TokenVerifier,
)
from fastmcp.server.dependencies import get_access_token

__all__ = [
    "AccessToken",
    "AuthProvider",
    "JWTVerifier",
    "MultiAuth",
    "TokenVerifier",
    "get_access_token",
]
