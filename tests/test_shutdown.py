"""An HTTP server that stops ends its listen streams cleanly (APP-15).

The first tests run ``tests/shutdown_server.py``, a real server, in a process of its
own on a free loopback port, hold a ``subscriptions/listen`` stream open to it, and send
it SIGTERM. The rest drive the listen-ender and the stop hook in-process.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import uvicorn

SERVER = Path(__file__).with_name("shutdown_server.py")
VERSION = "2026-07-28"
META = {
    "io.modelcontextprotocol/protocolVersion": VERSION,
    "io.modelcontextprotocol/clientCapabilities": {},
}
WAIT = 15.0  # the longest any step waits before the test fails instead of hanging


# -- a real server -----------------------------------------------------------


@dataclass
class Served:
    process: subprocess.Popen[str]
    port: int
    out: queue.Queue[str] = field(default_factory=queue.Queue)

    def started(self) -> None:
        """Waits for the ``slow`` tool to say it started, past uvicorn's access log."""
        deadline = time.monotonic() + WAIT
        while self.out.get(timeout=max(deadline - time.monotonic(), 0)) != (
            "slow: started"
        ):
            pass

    def stop(self) -> tuple[float, str]:
        """SIGTERM, then the seconds until the process exited, and its stderr.

        stdout is the reader thread's alone, so only stderr is read here.
        """
        self.process.send_signal(signal.SIGTERM)
        start = time.monotonic()
        self.process.wait(timeout=WAIT)
        seconds = time.monotonic() - start
        return seconds, _stderr(self.process)


@contextmanager
def served(*args: str, **env: str) -> Iterator[Served]:
    """The server, serving, with ``env`` over the HTTP basics; killed if left running."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, str(SERVER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "TRANSPORT": "http",
            "HOST": "127.0.0.1",
            "PORT": str(port),
            **env,
        },
    )
    server = Served(process, port)
    threading.Thread(target=_lines, args=(process, server.out), daemon=True).start()
    try:
        deadline = time.monotonic() + WAIT
        while True:
            assert process.poll() is None, _stderr(process)
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "the server never listened"
                time.sleep(0.05)
        yield server
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=WAIT)
        _stderr(process)


def _stderr(process: subprocess.Popen[str]) -> str:
    """What the exited process wrote to stderr, read once; the pipe is closed after."""
    assert process.stderr is not None
    if process.stderr.closed:
        return ""
    with process.stderr:
        return process.stderr.read()


def _lines(process: subprocess.Popen[str], out: queue.Queue[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        out.put(line.strip())


def _post(
    port: int, method: str, params: dict, **headers: str
) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {**params, "_meta": META},
    }
    conn.request(
        "POST",
        "/mcp",
        body=json.dumps(body),
        headers={
            "MCP-Protocol-Version": VERSION,
            "Mcp-Method": method,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **headers,
        },
    )
    return conn.getresponse()


class Listen:
    """A listen stream, read line by line in a thread of its own.

    It is read off the wire, chunk sizes and all: ``http.client`` reads a chunked body
    cut short as a clean end. ``ended`` is ``"eof"`` when the body ended with the last
    chunk, ``"cut"`` when the connection closed before it, or the error the read met.
    """

    def __init__(self, port: int) -> None:
        self.lines: list[bytes] = []
        self.ended: str | None = None
        self.acknowledged = threading.Event()
        self.pinged = threading.Event()
        self._thread = threading.Thread(target=self._read, args=(port,), daemon=True)
        self._thread.start()
        assert self.acknowledged.wait(WAIT), self.lines

    def _read(self, port: int) -> None:
        try:
            response = _post(
                port,
                "subscriptions/listen",
                {"notifications": {"toolsListChanged": True}},
            )
            assert response.status == 200
            assert response.getheader("content-type", "").startswith(
                "text/event-stream"
            )
            assert response.getheader("transfer-encoding") == "chunked"
            while line := response.fp.readline():
                self.lines.append(line)
                if b"notifications/subscriptions/acknowledged" in line:
                    self.acknowledged.set()
                if line.startswith(b": ping"):
                    self.pinged.set()
            self.ended = "eof" if self.lines[-2:] == [b"0\r\n", b"\r\n"] else "cut"
        except Exception as error:  # noqa: BLE001  # the test reads it
            self.ended = repr(error)

    def end(self) -> str | None:
        self._thread.join(WAIT)
        return self.ended


class Call:
    """A ``slow`` tool call, in a thread of its own."""

    def __init__(self, port: int, seconds: float) -> None:
        self.result: dict | None = None
        self.error: str | None = None
        self._thread = threading.Thread(
            target=self._call, args=(port, seconds), daemon=True
        )
        self._thread.start()

    def _call(self, port: int, seconds: float) -> None:
        try:
            params = {"name": "slow", "arguments": {"seconds": seconds}}
            response = _post(port, "tools/call", params, **{"Mcp-Name": "slow"})
            self.result = json.loads(response.read())
        except Exception as error:  # noqa: BLE001  # the test reads it
            self.error = repr(error)

    def end(self) -> dict | None:
        self._thread.join(WAIT)
        assert self.error is None, self.error
        return self.result


def assert_stopped_cleanly(
    server: Served, seconds: float, err: str, within: float = 1.5
) -> None:
    """Stopped in ``within`` seconds of the signal, logging nothing above INFO (no WARNING).

    A listen left open would hold the stop for the whole graceful timeout, 15 s.
    """
    # uvicorn raises the signal it caught again, once it has shut down
    assert server.process.returncode == -signal.SIGTERM, err
    assert "ERROR" not in err and "Traceback" not in err, err
    # a record starts with its level and a colon, in both the app's and uvicorn's format
    assert not re.search(r"^WARNING:", err, re.MULTILINE), err
    assert "Application shutdown complete" in err, err
    assert seconds < within, (seconds, err)


@pytest.mark.integration
def test_a_listen_ends_cleanly_when_the_server_stops() -> None:
    with served() as server:
        listen = Listen(server.port)
        seconds, err = server.stop()
    assert_stopped_cleanly(server, seconds, err)
    assert listen.end() == "eof"


@pytest.mark.integration
def test_a_listen_that_has_pinged_ends_cleanly_when_the_server_stops() -> None:
    with served("ping") as server:
        listen = Listen(server.port)
        assert listen.pinged.wait(WAIT), listen.lines
        seconds, err = server.stop()
    assert_stopped_cleanly(server, seconds, err)
    assert listen.end() == "eof"


@pytest.mark.integration
def test_a_call_in_flight_still_finishes_beside_an_ended_listen() -> None:
    with served() as server:
        listen = Listen(server.port)
        call = Call(server.port, 1.0)
        server.started()
        seconds, err = server.stop()
    assert_stopped_cleanly(server, seconds, err, within=2.5)
    assert listen.end() == "eof"
    result = call.end()
    assert result is not None and result["result"]["content"][0]["text"] == "slept 1.0"


@pytest.mark.integration
def test_a_call_past_the_graceful_timeout_is_still_cancelled() -> None:
    with served(GRACEFUL_TIMEOUT="1") as server:
        listen = Listen(server.port)
        Call(server.port, 5.0)
        server.started()
        seconds, err = server.stop()
    # the listen had ended: the one task cancelled is the call's
    assert "Cancel 1 running task(s), timeout graceful shutdown exceeded" in err, err
    assert 1 <= seconds < 3, (seconds, err)
    assert listen.end() == "eof"


# -- the listen-ender, in-process --------------------------------------------


def _scope(method: str = "subscriptions/listen") -> dict:
    return {
        "type": "http",
        "method": "POST",
        "headers": [
            (b"mcp-method", method.encode()),
            (b"content-type", b"application/json"),
        ],
    }


class Wire:
    """The server's side of one request: the body, then what the client does."""

    def __init__(self, *messages: dict) -> None:
        self.incoming: asyncio.Queue[dict] = asyncio.Queue()
        for message in messages or ({"type": "http.request", "body": b"{}"},):
            self.incoming.put_nowait(message)
        self.sent: list[dict] = []
        self.receives = 0

    async def receive(self) -> dict:
        """The next message; a client that sends nothing more fails the test in time."""
        self.receives += 1
        return await asyncio.wait_for(self.incoming.get(), WAIT)

    async def send(self, message: dict) -> None:
        self.sent.append(message)


def _ender(app, stop):
    from kubed.mcp._shutdown import _EndListens

    return _EndListens(app, stop)


def _stop():
    from kubed.mcp._shutdown import _Stop

    return _Stop()


START = {
    "type": "http.response.start",
    "status": 200,
    "headers": [(b"content-type", b"text/event-stream")],
}
END = {"type": "http.response.body", "body": b"", "more_body": False}


async def test_a_listen_stopped_before_it_answered_gets_an_empty_event_stream() -> None:
    stop = _stop()
    seen: list[dict] = []

    async def app(scope, receive, send) -> None:
        seen.append(await receive())
        stop.record()
        seen.append(await receive())

    wire = Wire()
    await _ender(app, stop)(_scope(), wire.receive, wire.send)
    assert seen == [
        {"type": "http.request", "body": b"{}"},
        {"type": "http.disconnect"},
    ]
    assert wire.sent == [START, END]


async def test_a_listen_stopped_mid_stream_gets_only_the_final_body(caplog) -> None:
    stop = _stop()

    async def app(scope, receive, send) -> None:
        await receive()
        await send(START)
        await send(
            {"type": "http.response.body", "body": b": ping\r\n\r\n", "more_body": True}
        )
        stop.record()
        assert await receive() == {"type": "http.disconnect"}
        assert await receive() == {"type": "http.disconnect"}

    wire = Wire()
    with caplog.at_level("DEBUG", logger="kubed.mcp._shutdown"):
        await _ender(app, stop)(_scope(), wire.receive, wire.send)
    assert wire.sent[2:] == [END] and len(wire.sent) == 3
    assert [(r.levelname, r.name) for r in caplog.records] == [
        ("DEBUG", "kubed.mcp._shutdown")
    ]


async def test_the_body_is_read_as_sent_even_after_the_stop() -> None:
    stop = _stop()
    stop.record()
    seen: list[dict] = []

    async def app(scope, receive, send) -> None:
        seen.extend([await receive(), await receive(), await receive()])

    first = {"type": "http.request", "body": b'{"a":', "more_body": True}
    rest = {"type": "http.request", "body": b"1}"}
    wire = Wire(first, rest)
    await _ender(app, stop)(_scope(), wire.receive, wire.send)
    assert seen == [first, rest, {"type": "http.disconnect"}]
    assert wire.receives == 2
    assert wire.sent == [START, END]


async def test_a_client_that_leaves_ends_its_listen_as_before() -> None:
    stop = _stop()

    async def app(scope, receive, send) -> None:
        await receive()
        assert await receive() == {"type": "http.disconnect"}

    wire = Wire({"type": "http.request", "body": b"{}"}, {"type": "http.disconnect"})
    await _ender(app, stop)(_scope(), wire.receive, wire.send)
    assert wire.sent == []


async def test_a_listen_that_answered_gets_nothing_more() -> None:
    stop = _stop()

    async def app(scope, receive, send) -> None:
        await receive()
        stop.record()
        assert await receive() == {"type": "http.disconnect"}
        await send(START)
        await send({"type": "http.response.body", "body": b"x"})

    wire = Wire()
    await _ender(app, stop)(_scope(), wire.receive, wire.send)
    assert wire.sent == [START, {"type": "http.response.body", "body": b"x"}]


@pytest.mark.parametrize(
    "scope",
    [
        _scope("tools/call"),
        {**_scope(), "method": "GET"},
        {"type": "websocket", "headers": _scope()["headers"]},
        {"type": "lifespan"},
    ],
    ids=["another-method", "get", "websocket", "lifespan"],
)
async def test_anything_but_a_listen_goes_straight_to_the_app(scope) -> None:
    stop = _stop()
    stop.record()
    wire = Wire()
    got: list[tuple] = []

    async def app(scope, receive, send) -> None:
        got.append((receive, send))

    await _ender(app, stop)(scope, wire.receive, wire.send)
    assert got == [(wire.receive, wire.send)]


async def test_a_cancelled_receive_leaves_nothing_waiting() -> None:
    stop = _stop()
    waiting: list[asyncio.Task] = []

    async def app(scope, receive, send) -> None:
        await receive()
        pending = asyncio.ensure_future(receive())
        await asyncio.sleep(0.05)
        waiting.extend(
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        )
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    wire = Wire()
    await _ender(app, stop)(_scope(), wire.receive, wire.send)
    assert waiting, "no task was running beside the listen to wait for"
    await asyncio.wait(waiting, timeout=1)
    assert all(t.done() for t in waiting)
    assert wire.sent == []


# -- the stop hook -------------------------------------------------------------


async def test_the_hook_wakes_a_listen_on_the_signal_uvicorn_handles(
    monkeypatch,
) -> None:
    """uvicorn's own method is a stand-in: sse-starlette's, if it is installed over
    uvicorn's, would mark every SSE stream in the process as ending."""
    from kubed.mcp._shutdown import _stop_hook

    called: list[tuple] = []

    def handle_exit(self, sig, frame) -> None:
        called.append((self, sig, frame))

    monkeypatch.setattr(uvicorn.Server, "handle_exit", handle_exit)
    server = object.__new__(uvicorn.Server)
    with _stop_hook() as stop:
        event = stop.event()
        assert not event.is_set()
        uvicorn.Server.handle_exit(server, signal.SIGTERM, None)
        assert stop.stopping and called == [(server, signal.SIGTERM, None)]
        await asyncio.wait_for(event.wait(), WAIT)
    assert uvicorn.Server.handle_exit is handle_exit


def test_a_stop_before_the_first_listen_starts_its_event_set() -> None:
    stop = _stop()
    stop.record()

    async def first() -> bool:
        return stop.event().is_set()

    assert asyncio.run(first()) is True
