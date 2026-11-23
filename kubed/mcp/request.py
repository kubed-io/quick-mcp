"""What the current request says about itself: its query, its headers, its Context.

``request_values()`` is the one reader of a request's query parameters and headers. A
server reads its own settings out of what it returns.
"""

from __future__ import annotations

from collections.abc import Callable

from fastmcp.server.dependencies import get_context, get_http_headers, get_http_request
from starlette.requests import Request

Values = dict[str, list[str]]

__all__ = ["Values", "get_context", "get_http_request", "request_values"]


def request_values(
    request: Request | None = None,
    *,
    snapshot: Callable[[Values], Values] | None = None,
) -> tuple[Values, Values] | None:
    """(query parameters, headers) of ``request``, the current one, or None off HTTP.

    Given a Starlette ``request``, such as a custom route's, it is that request that is
    read, by the same rules, and ``snapshot`` does not apply to it: it is for the
    ambient background-task case only.

    Every value is a list, because a repeated name is how a client says "any of
    these", and header names are lowercased. Off a live request, a background task's
    worker still has the headers snapshotted when it was submitted: they come back
    with no query parameters, passed through ``snapshot`` first when one is given.
    With neither, as on stdio, it is None.
    """
    if request is None:
        try:
            request = get_http_request()
        except RuntimeError:  # no live HTTP request
            request = None
    if request is not None:
        params: Values = {}
        for key, value in request.query_params.multi_items():
            params.setdefault(key, []).append(value)
        headers: Values = {}
        for key, value in request.headers.items():
            headers.setdefault(key.lower(), []).append(value)
        return params, headers
    carried = get_http_headers(include_all=True)
    snapshotted = {key.lower(): [value] for key, value in carried.items()}
    if not snapshotted:
        return None
    return {}, snapshot(snapshotted) if snapshot is not None else snapshotted
