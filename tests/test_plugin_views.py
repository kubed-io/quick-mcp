"""Views: each request served the plugins its view admits, pinned once, built once
per key (PLUG-53 to PLUG-55)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from fastmcp import Client, Context
from fastmcp.utilities.tests import asgi_server
from mcp_types import INTERNAL_ERROR
from plugin_server import (
    DEMO,
    ENTRY,
    agent_md,
    command_md,
    listed_by,
    served,
    skill,
    write,
)
from skill_server import request

from kubed.mcp import QuickMCP
from kubed.mcp.plugins import _PINS, View, _Slot
from kubed.mcp.protocol import MCPError
from kubed.mcp.request import request_values

pytestmark = pytest.mark.unit


def scoped() -> View:
    found = request_values()
    scope = (found[1].get("x-scope") or [None])[0] if found else None
    return View(scope, lambda plugin: scope is None or plugin.namespace == scope)


def plugin(tmp_path: Path, name: str) -> Path:
    """A plugin ``name``: a skill ``<name>-guide`` with a reference, a command and an
    agent."""
    return write(
        tmp_path / name,
        {
            f"skills/{name}-guide/SKILL.md": skill(f"{name}-guide"),
            f"skills/{name}-guide/references/NOTE.md": "# A note\n",
            "commands/wave.md": command_md(),
            "agents/scout.md": agent_md("scout"),
        },
    )


def scopes(tmp_path: Path) -> QuickMCP:
    """Plugins ``one`` and ``two``, each under its own namespace, seen by scope."""
    mcp = served((plugin(tmp_path, "one"), "one"), (plugin(tmp_path, "two"), "two"))
    mcp.plugin_view(scoped)
    return mcp


def alone(tmp_path: Path) -> QuickMCP:
    """The same server with plugin ``one`` only, and no view."""
    return served((plugin(tmp_path, "one"), "one"))


@asynccontextmanager
async def over_http(mcp: QuickMCP, scope: str):
    async with asgi_server(mcp.server) as running:
        transport = running.transport(headers={"x-scope": scope})
        async with Client(transport) as client:
            yield client


def counting(mcp: QuickMCP) -> list:
    """Each call of the view, admitting every plugin under one key."""
    calls: list = []

    def view() -> View:
        calls.append(1)
        return View(None, lambda plugin: True)

    mcp.plugin_view(view)
    return calls


def slow_make(monkeypatch, mcp: QuickMCP) -> list:
    """The key of each build made, each taking a moment."""
    made: list = []
    plain = type(mcp._plugins)._make

    async def slow(self, state, view, key):
        made.append(key)
        await anyio.sleep(0.01)
        return await plain(self, state, view, key)

    monkeypatch.setattr(type(mcp._plugins), "_make", slow)
    return made


async def test_each_scope_is_served_its_plugins(tmp_path):
    mcp = scopes(tmp_path)
    async with over_http(mcp, "one") as client:
        assert await listed_by(client) == (
            ["one_wave"],
            ["one_scout"],
            ["skill://one/one-guide/SKILL.md"],
        )
    async with over_http(mcp, "two") as client:
        assert await listed_by(client) == (
            ["two_wave"],
            ["two_scout"],
            ["skill://two/two-guide/SKILL.md"],
        )


OUTSIDE = [
    ("resources/read", {"uri": "skill://two/two-guide/SKILL.md"}),
    ("resources/read", {"uri": "skill://two/two-guide/_index.md"}),
    ("resources/read", {"uri": "skill://two/two-guide"}),  # a folder read
    ("resources/directory/read", {"uri": "skill://two/two-guide"}),
    ("skills/get", {"uri": "skill://two/two-guide/SKILL.md"}),
    ("prompts/get", {"name": "two_wave"}),
    ("agents/get", {"name": "two_scout"}),
]


@pytest.mark.parametrize(("method", "params"), OUTSIDE)
async def test_outside_the_view_fails_as_a_missing_one_does(tmp_path, method, params):
    failures = []
    for mcp in (scopes(tmp_path / "a"), alone(tmp_path / "b")):
        async with over_http(mcp, "one") as client:
            with pytest.raises(MCPError) as caught:
                await request(client, method, **params)
        error = caught.value.error
        failures.append((error.code, error.message, error.data))
    assert failures[0] == failures[1]


async def test_the_view_runs_once_per_request():
    mcp = served((DEMO, None))
    calls = counting(mcp)

    # Skills, an index and a prompt, in one request.
    @mcp.tool
    async def read_all(ctx: Context) -> str:
        await ctx.read_resource(ENTRY)
        await ctx.read_resource("skill://t/demo-guide/_index.md")
        await ctx.fastmcp.render_prompt("greet", {"who": "x"})
        return "done"

    async with Client(mcp.server) as client:
        await client.call_tool("read_all", {})
        await request(client, "agents/list")
    assert len(calls) == 2


async def test_a_tools_nested_reads_keep_its_calls_set():  # Review Focus 1
    mcp = served((DEMO, None))
    calls = counting(mcp)

    @mcp.tool
    async def swap_and_render(ctx: Context) -> str:
        await ctx.read_resource(ENTRY)
        mcp.serve(mcp.plugin_set())  # today: the render below is "Unknown prompt"
        await ctx.fastmcp.render_prompt("greet", {"who": "x"})
        return "read and rendered"

    async with Client(mcp.server) as client:
        result = await client.call_tool("swap_and_render", {})
    assert result.data == "read and rendered"
    assert len(calls) == 1


async def test_concurrent_reads_in_one_request_pin_once(monkeypatch):
    mcp = served((DEMO, None))
    calls, made = counting(mcp), slow_make(monkeypatch, mcp)

    @mcp.tool
    async def both(ctx: Context) -> str:
        await asyncio.gather(
            ctx.read_resource(ENTRY), ctx.fastmcp.render_prompt("greet", {"who": "x"})
        )
        return "done"

    async with Client(mcp.server) as client:
        await client.call_tool("both", {})
    assert len(calls) == 1 and len(made) == 1


async def test_concurrent_requests_share_one_build(monkeypatch):
    mcp = served((DEMO, None))
    made = slow_make(monkeypatch, mcp)
    async with Client(mcp.server) as client:
        await asyncio.gather(
            client.list_prompts(),
            request(client, "agents/list"),
            request(client, "skills/list"),
        )
    assert len(made) == 1


def first_hangs(monkeypatch, mcp: QuickMCP, name: str) -> anyio.Event:
    """``Plugins.<name>`` hanging on its first call, until cancelled, and plain after:
    the event is set once the first call has begun."""
    entered, plain = anyio.Event(), getattr(type(mcp._plugins), name)

    async def hanging(self, *args):
        if not entered.is_set():
            entered.set()
            await anyio.sleep_forever()
        return await plain(self, *args)

    monkeypatch.setattr(type(mcp._plugins), name, hanging)
    return entered


async def until_awaited(event: anyio.Event) -> None:
    while event.statistics().tasks_waiting == 0:
        await anyio.sleep(0)


async def test_a_reader_left_by_a_cancelled_pinner_pins_itself(monkeypatch):
    mcp = served((DEMO, None))
    plugins, entered, pins = mcp._plugins, first_hangs(monkeypatch, mcp, "_pin"), []
    slot = _Slot()
    token = _PINS.set({plugins: slot})  # one request's slot, as _PinScope opens it
    try:
        with anyio.fail_after(2):
            async with anyio.create_task_group() as tg:
                pinner = anyio.CancelScope()

                async def pin_first() -> None:
                    with pinner:
                        await plugins.pinned()

                async def read() -> None:
                    pins.append(await plugins.pinned())

                tg.start_soon(pin_first)
                await entered.wait()
                tg.start_soon(read)
                await until_awaited(slot.done)
                pinner.cancel()
    finally:
        _PINS.reset(token)
    assert pins == [slot.pin] and pins[0].build.prompts


async def test_a_request_left_by_a_cancelled_build_builds_itself(monkeypatch):
    mcp = served((DEMO, None))
    plugins, entered, builds = mcp._plugins, first_hangs(monkeypatch, mcp, "_make"), []
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tg:
            maker = anyio.CancelScope()

            async def make_first() -> None:
                with maker:
                    await plugins.pinned()

            async def share() -> None:
                builds.append((await plugins.pinned()).build)

            tg.start_soon(make_first)
            await entered.wait()
            tg.start_soon(share)
            (flight,) = plugins._memo.flights.values()
            await until_awaited(flight.done)
            maker.cancel()
    assert builds[0] is not None and builds[0].prompts
    assert list(plugins._memo.builds.values()) == builds  # and it is kept


async def test_builds_are_kept_per_key_least_recently_used_first():
    key, built = {"now": 0}, []
    mcp = served((DEMO, None))
    mcp.plugin_view(
        lambda: View(key["now"], lambda plugin: built.append(key["now"]) or True)
    )

    async def ask(k):
        key["now"] = k
        async with Client(mcp.server) as client:
            await client.list_prompts()

    for k in range(65):
        await ask(k)  # 0 is dropped when 64 is built
    await ask(64)
    await ask(1)  # kept; 1 is now the most recent
    await ask(0)  # rebuilt, which drops 2, not 1
    await ask(1)
    await ask(2)
    assert built == [*range(65), 0, 2]


async def test_a_new_generation_never_reuses_a_build():
    built = []
    mcp = served((DEMO, None))
    mcp.plugin_view(lambda: View("k", lambda plugin: built.append(1) or True))
    async with Client(mcp.server) as client:
        await client.list_prompts()
    staging = mcp.plugin_set()
    staging.keep(mcp.plugins[0])
    mcp.serve(staging)
    async with Client(mcp.server) as client:
        await client.list_prompts()
    assert len(built) == 2


async def test_a_new_view_is_not_served_an_old_build():  # Review Focus 2
    mcp = served((DEMO, None))
    async with Client(mcp.server) as client:
        assert [p.name for p in await client.list_prompts()] == ["greet"]
    mcp.plugin_view(lambda: View(None, lambda plugin: False))  # same key, None
    async with Client(mcp.server) as client:
        assert await client.list_prompts() == []


async def test_a_swap_during_a_request_does_not_mix_sets():
    mcp = served((DEMO, None))
    swapped = []

    def swap() -> View:
        if not swapped:
            swapped.append(1)
            mcp.serve(mcp.plugin_set())  # mid-request
        return View(None, lambda plugin: True)

    mcp.plugin_view(swap)
    async with Client(mcp.server) as client:
        index = await client.read_resource("skill://t/demo-guide/_index.md")
        assert "skill://t/demo-guide/SKILL.md" in index[0].text
        with pytest.raises(MCPError):
            await request(client, "resources/read", uri=ENTRY)


async def broken_make(self, state, view, key):
    raise RuntimeError("a bug in _make")


FAILING_VIEWS = {
    "raises": lambda: 1 / 0,
    "unhashable": lambda: View([], lambda plugin: True),
}
READS = [
    ("skills/list", {}),
    ("skills/get", {"uri": ENTRY}),
    ("prompts/list", {}),
    ("agents/list", {}),
    ("resources/read", {"uri": ENTRY}),
]


@pytest.mark.parametrize("failure", [*FAILING_VIEWS, "make"])
@pytest.mark.parametrize(("method", "params"), READS)
async def test_a_failure_to_pin_fails_the_request(
    monkeypatch, failure, method, params
):  # Review Focus 5
    mcp = served((DEMO, None))
    if failure == "make":
        monkeypatch.setattr(type(mcp._plugins), "_make", broken_make)
    else:
        mcp.plugin_view(FAILING_VIEWS[failure])
    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as caught:
            await request(client, method, **params)
    # The pin's own error, never what its absence would read as, an unknown skill.
    assert caught.value.error.code == INTERNAL_ERROR
    with pytest.raises(
        (ZeroDivisionError, TypeError, RuntimeError)
    ):  # off a request it just raises
        await mcp._plugins.pinned()


async def test_the_build_key_is_one_extendable_value(monkeypatch):  # #28's seam (GW-6)
    extra, built = {"now": "a"}, []
    mcp = served((DEMO, None))
    mcp.plugin_view(
        lambda: View("k", lambda plugin: built.append(extra["now"]) or True)
    )
    plain = type(mcp._plugins)._key
    monkeypatch.setattr(
        type(mcp._plugins),
        "_key",
        lambda self, state, view: (*plain(self, state, view), extra["now"]),
    )
    for now in ("a", "b", "a"):
        extra["now"] = now
        async with Client(mcp.server) as client:
            await client.list_prompts()
    assert built == ["a", "b"]  # one view key, two build keys, each built once
    assert plain(mcp._plugins, mcp._plugins._state, View("k", bool)) == (
        mcp.generation,
        "k",
    )
