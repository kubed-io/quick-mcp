"""Resource reads in the call metrics, labelled ``mcp.resource.template``: imported only
when ``otel`` is on.

FastMCP's ``UsageMiddleware`` records tool calls and prompt renders; ``_CallMetrics``
records ``resources/read`` too. The label crosses two middlewares. The metrics are
outermost, to record what the client received, but only a middleware inside the pin and
the prefix sees the plain URI, under the request's pinned build. So ``_ReadObserver``
sits there and writes the label into a holder the metrics leave in ``_MATCHED``, and the
metrics' histogram reads it back as the read's point is recorded.

The label is the template the URI matches, else the URI. The lookup asks for templates
only: a successful read that matches none has found a static resource, so a static
lookup would decide nothing, and for a skill's file it re-reads the file. A static
resource whose URI also matches a template is labelled by that template.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import extract_version_spec
from fastmcp.server.middleware import Middleware
from fastmcp.server.middleware.usage import UsageMiddleware
from fastmcp.utilities.versions import VersionSpec

from .gateway import _TakeThePrefix

log = logging.getLogger(__name__)

_TEMPLATE = "mcp.resource.template"


@dataclass
class _Matched:
    """One read's label: the observer writes it, the read's point reads it."""

    label: str | None = None


_MATCHED: ContextVar[_Matched | None] = ContextVar("_MATCHED", default=None)


class _Labelled:
    """A histogram whose resource-read points take their label from ``_MATCHED``.

    A point without ``mcp.resource.template`` goes on as it came. UsageMiddleware drops
    the name on a routing error, so such a read never carries a label.
    """

    def __init__(self, histogram: Any) -> None:
        self._histogram = histogram

    def record(
        self, amount: float, attributes: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        if isinstance(attributes, Mapping) and _TEMPLATE in attributes:
            attributes = dict(attributes)
            matched = _MATCHED.get()
            if matched is not None and matched.label is not None:
                attributes[_TEMPLATE] = matched.label
            else:
                del attributes[_TEMPLATE]
        self._histogram.record(amount, attributes, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # A copy is built without __init__, and asks for names before _histogram is set.
        histogram = self.__dict__.get("_histogram")
        if histogram is None:
            raise AttributeError(name)
        return getattr(histogram, name)


def _takes_the_reads(record: Any) -> bool:
    """Whether ``record`` is callable as ``on_read_resource`` calls it."""
    if not callable(record):
        return False
    try:
        inspect.signature(record).bind(
            None, None, None, method="", name_attribute="", name=""
        )
    except (TypeError, ValueError):
        return False
    return True


class _CallMetrics(UsageMiddleware):
    """FastMCP's call metrics, with ``resources/read`` recorded too.

    It uses two private names of FastMCP's UsageMiddleware, ``_record`` and
    ``_duration``. Without them, or with a ``_record`` that no longer takes the keywords
    the reads pass, it logs one warning and reads pass straight through: tools and
    prompts are still recorded.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        duration = getattr(self, "_duration", None)
        self._reads = _takes_the_reads(
            getattr(UsageMiddleware, "_record", None)
        ) and callable(getattr(duration, "record", None))
        if self._reads:
            self._duration = _Labelled(duration)
        else:
            log.warning(
                "Resource reads are not in the call metrics: "
                "FastMCP's UsageMiddleware has changed"
            )

    async def on_read_resource(self, context: Any, call_next: Any) -> Any:
        if not self._reads:
            return await call_next(context)
        token = _MATCHED.set(_Matched())
        try:
            return await self._record(
                context,
                call_next,
                method="resources/read",
                name_attribute=_TEMPLATE,
                name="",
            )
        finally:
            _MATCHED.reset(token)

    def observe(self, server: FastMCP) -> None:
        """Put the read observer right after the prefix, or last without one.

        Never raises (TEL-6).
        """
        if not self._reads:
            return
        try:
            middleware = server.middleware
            after = next(
                (
                    i + 1
                    for i, found in enumerate(middleware)
                    if isinstance(found, _TakeThePrefix)
                ),
                len(middleware),
            )
            middleware.insert(after, _ReadObserver(server))
        except Exception as error:  # noqa: BLE001  # TEL-6: logged by type only, never its text
            log.error(
                "Resource reads are not labelled in the call metrics: %s",
                type(error).__name__,
            )


class _ReadObserver(Middleware):
    """Labels a read the call metrics are recording, from the plain URI under the pin.

    The label is the ``uri_template`` of the template the URI matches. A read that
    succeeds without one is labelled by its URI, and one that fails without one is not
    labelled. A cancellation goes through unlabelled.
    """

    def __init__(self, server: FastMCP) -> None:
        self._server = server

    async def on_read_resource(self, context: Any, call_next: Any) -> Any:
        matched = _MATCHED.get()
        if matched is None:
            return await call_next(context)
        uri = str(context.message.uri)
        try:
            result = await call_next(context)
        except Exception:
            matched.label = await self._template(context, uri)
            raise
        matched.label = await self._template(context, uri) or uri
        return result

    async def _template(self, context: Any, uri: str) -> str | None:
        """The ``uri_template`` of the template ``uri`` matches, at its version."""
        try:
            # As FastMCP's resources/read handler reads it from the request's _meta.
            version = extract_version_spec(context.message.meta)
            template = await self._server.get_resource_template(
                uri, version=VersionSpec(eq=version) if version else None
            )
        except Exception:  # noqa: BLE001  # TEL-6: a lookup that fails gives no template
            return None
        return None if template is None else template.uri_template
