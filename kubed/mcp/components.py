"""``Component``: a server part that takes config, registers, and runs as it serves.

A component names the config section it takes, registers with the server at the end of
construction, starts when serving starts and stops when it ends. A server passes its
components to ``QuickMCP(components=...)`` or ``load_config(components=...)``.
Serving, telemetry and delivery are components too, built into every server and
started before the server's own.

This module imports no FastMCP, so a component can be declared, and its section's
schema written, without loading it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from .app import QuickMCP

log = logging.getLogger(__name__)


class Component:
    """A part of a server: its config section, its registration, its start and stop.

    ``section`` is a section class carrying its own ``@config(key)``, composed into the
    server's config as ``sections=`` are, or None for no section. The ``section`` each
    method is given is the loaded instance of the class that won its key, a server's
    subclass included, and None when the component names none.

    Every method does nothing until overridden:

    - ``register(mcp, section)`` runs once, at the end of ``QuickMCP.__init__``, in
      the order the components were given, after the built-in ones;
    - ``start(section)`` runs each time serving starts, in that order, before the
      server's own ``lifespan``. One that raises stops the components already started,
      in reverse, and fails the boot;
    - ``stop()`` runs when serving ends, in reverse, after the server's ``lifespan``.
      Every stop runs: one that raises an ``Exception`` is logged at error by its
      type, never raised. Anything else, a cancellation, is raised once every stop
      has run.

    Entries may overlap: FastMCP marks its lifespan entered only once the server's has
    been, so two entries racing at start-up each run every ``start`` before either
    runs a ``stop``. A start may run again before the earlier entry's stop, so a
    component keeps no single-slot state it can't share.
    """

    section: ClassVar[type[BaseModel] | None] = None

    def register(self, mcp: QuickMCP, section: BaseModel | None) -> None:
        """Registers with ``mcp``: its tools, middleware, routes or auth."""

    async def start(self, section: BaseModel | None) -> None:
        """Starts what the component runs while the server serves."""

    async def stop(self) -> None:
        """Stops what ``start`` began."""


def _components(components: Iterable[Component]) -> tuple[Component, ...]:
    """``components=`` as a tuple, each one checked to be a ``Component`` instance."""
    if isinstance(components, (Component, type)) or not isinstance(
        components, Iterable
    ):
        raise TypeError(
            "components= takes Component instances in a sequence, got "
            + _what(components)
        )
    checked = tuple(components)
    for each in checked:
        if not isinstance(each, Component):
            raise TypeError(f"components= takes Component instances, got {_what(each)}")
    return checked


def _what(value: object) -> str:
    if isinstance(value, type):
        return f"the class {value.__qualname__}"
    return f"an instance of {type(value).__qualname__}"


async def _start_all(
    parts: Sequence[tuple[Component, BaseModel | None]],
) -> list[Component]:
    """Starts each component with its section, in order, and returns those started.

    A start that raises stops the ones already started, in reverse, and the error is
    raised again. The one that raised is not stopped: it did not start.
    """
    started: list[Component] = []
    try:
        for component, section in parts:
            await component.start(section)
            started.append(component)
    except BaseException:
        await _stop_all(started)
        raise
    return started


async def _stop_all(started: Sequence[Component]) -> None:
    """Stops ``started`` in reverse. Every stop runs.

    A stop that raises an ``Exception`` is logged at error by the component's class and
    the error's type, never its text, and isn't raised. Anything else, a cancellation,
    is raised once every stop has run.
    """
    escaped: BaseException | None = None
    for component in reversed(started):
        try:
            await component.stop()
        except Exception as error:  # noqa: BLE001  # logged by type only, never its text
            log.error(
                "%s did not stop cleanly: %s",
                type(component).__qualname__,
                type(error).__name__,
            )
        except BaseException as error:  # noqa: BLE001  # raised once every stop has run
            if escaped is None:
                escaped = error
    if escaped is not None:
        raise escaped
