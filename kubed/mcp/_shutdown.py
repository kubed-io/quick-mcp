"""Open ``subscriptions/listen`` streams end cleanly when an HTTP server stops.

A listen stream lasts until its client disconnects. When uvicorn stops, it closes its
listeners and waits ``timeout_graceful_shutdown`` for the open requests to finish, and
a listen never does: uvicorn cancels it at the timeout and logs an ERROR, and the SDK's
handler logs a traceback for it. ``run()`` puts two pieces in its way, on HTTP only.

``_stop_hook()`` wraps ``uvicorn.Server.handle_exit``, the method uvicorn's SIGINT and
SIGTERM handlers call, so the stop is known the moment the signal lands, before uvicorn
starts its wait. It patches uvicorn's class for the life of the hook, checked against
uvicorn 0.54.0, whose ``capture_signals()`` binds ``self.handle_exit`` inside
``serve()``.

``_EndListens`` is a pure ASGI middleware. Once a listen's body is read, it gives the
SDK an ``http.disconnect`` when the stop comes, which ends the stream's handler as a
client leaving would. Then it closes the response with an empty final body, so the
client reads the end of the stream. Every other request is passed straight to the app
and keeps the graceful timeout to finish; one still running past it is cancelled, and
uvicorn logs it, as before.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

import uvicorn
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger(__name__)

_LISTEN = b"subscriptions/listen"
_DISCONNECT: Message = {"type": "http.disconnect"}
_END: Message = {"type": "http.response.body", "body": b"", "more_body": False}


class _Stop:
    """Whether the server has begun to stop, and an event that is set when it has.

    The event belongs to the loop of the first request that asks for it, and starts
    set when the stop came first. The stop is recorded from a signal handler, so the
    event is set through ``call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        self.stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def event(self) -> asyncio.Event:
        if self._event is None:
            self._loop = asyncio.get_running_loop()
            self._event = asyncio.Event()
            if self.stopping:
                self._event.set()
        return self._event

    def record(self) -> None:
        self.stopping = True
        event, loop = self._event, self._loop
        if event is not None and loop is not None:
            loop.call_soon_threadsafe(event.set)


@contextmanager
def _stop_hook() -> Iterator[_Stop]:
    """``uvicorn.Server.handle_exit`` records the stop first, while the hook is open.

    Yields the ``_Stop`` it records to. The original method is put back on exit,
    however the block ends. sse-starlette replaces the method with its own when it is
    imported, which ``QuickMCP()`` does, so by ``run()`` the original is
    sse-starlette's, and it calls uvicorn's.
    """
    stop = _Stop()
    original = uvicorn.Server.handle_exit

    @functools.wraps(original)
    def handle_exit(self: uvicorn.Server, sig: int, frame: FrameType | None) -> None:
        stop.record()
        original(self, sig, frame)

    uvicorn.Server.handle_exit = handle_exit  # type: ignore[method-assign]
    try:
        yield stop
    finally:
        uvicorn.Server.handle_exit = original  # type: ignore[method-assign]


def _is_listen(scope: Scope) -> bool:
    """A POST naming ``subscriptions/listen`` in its ``Mcp-Method`` header.

    The SDK refuses a request whose header and body name different methods, so the
    header alone says what the request is.
    """
    if scope["type"] != "http" or scope.get("method") != "POST":
        return False
    return any(
        name == b"mcp-method" and value == _LISTEN
        for name, value in scope.get("headers", ())
    )


class _EndListens:
    """Ends a ``subscriptions/listen`` stream cleanly when the server begins to stop."""

    def __init__(self, app: ASGIApp, stop: _Stop) -> None:
        self.app = app
        self.stop = stop

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_listen(scope):
            await self.app(scope, receive, send)
            return
        stop = self.stop.event()
        body_read = False
        stopped = False
        started = False
        complete = False

        async def receive_until_stop() -> Message:
            nonlocal body_read, stopped
            if not body_read:
                message = await receive()
                if message["type"] == "http.request" and not message.get("more_body"):
                    body_read = True
                return message
            if stopped or stop.is_set():
                stopped = True
                return dict(_DISCONNECT)
            received = asyncio.ensure_future(receive())
            stopping = asyncio.ensure_future(stop.wait())
            try:
                await asyncio.wait(
                    (received, stopping), return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                received.cancel()
                stopping.cancel()
            if received.done() and not received.cancelled():
                return received.result()
            stopped = True
            return dict(_DISCONNECT)

        async def tracked(message: Message) -> None:
            nonlocal started, complete
            if message["type"] == "http.response.start":
                started = True
            elif message["type"] == "http.response.body" and not message.get(
                "more_body"
            ):
                complete = True
            await send(message)

        await self.app(scope, receive_until_stop, tracked)
        if not stopped or complete:
            return
        if not started:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
        await send(dict(_END))
        log.debug("listen stream ended: the server is stopping")


def _listen_ender(stop: _Stop) -> Middleware:
    """``_EndListens`` as Starlette middleware, ending listens on ``stop``."""
    return Middleware(_EndListens, stop=stop)
