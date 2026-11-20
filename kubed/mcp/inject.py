"""``Inject[T]`` and ``lookup(T)``: a tool's parameter, built by the container.

``Inject[T]`` is ``Annotated[T, Depends(lookup(T))]``. FastMCP's own dependency
injection does the per-call work: it hides the parameter from the tool's schema,
refuses a client that sends it, and resolves it for the call. Called directly, the
function is a plain function, and an argument passed for the parameter is used as
given.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

from fastmcp.dependencies import Depends
from fastmcp.server.dependencies import get_context

from .container import _CALL, SERVING, Container, name_of

T = TypeVar("T")


def _serving() -> Container:
    """The container of the QuickMCP serving this call."""
    call = _CALL.get()
    if call is not None:
        return call.container
    try:
        server = get_context().fastmcp
    except RuntimeError:
        raise RuntimeError("lookup() resolves only in a call to a QuickMCP") from None
    container = SERVING.get(server)
    if container is None:
        raise RuntimeError(f"{server.name!r} is a FastMCP server, not a QuickMCP")
    return container


@functools.cache
def lookup(key: type[T]) -> Callable[[], Awaitable[T]]:
    """One resolver per key, so FastMCP's per-call cache knows two uses for one."""

    async def resolve() -> Any:
        return await _serving().resolve(key)

    resolve.__name__ = resolve.__qualname__ = f"lookup[{name_of(key)}]"
    return resolve


if TYPE_CHECKING:
    # A type checker reads Inject[Grid] as Grid.
    Inject = Annotated[T, "kubed.mcp.Inject"]
else:

    class Inject:
        """``Inject[T]``: a parameter of type ``T``, built by the container."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            raise TypeError("Inject is used as Inject[T], in an annotation")

        def __class_getitem__(cls, key: Any) -> Any:
            return Annotated[key, Depends(lookup(key))]
