"""QuickMCP, its container and Inject[T], through a real FastMCP Client."""

from __future__ import annotations

import argparse
import itertools
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fastmcp import Client
from fastmcp.server.transforms import Namespace
from fastmcp.utilities.tests import asgi_server
from starlette.requests import Request
from starlette.responses import JSONResponse

import kubed.mcp.container
from kubed.mcp import Context, Inject, QuickMCP, lookup

BUILT: dict[str, int] = {}
EVENTS: list[str] = []
WORKSPACE = {"name": "alpha"}
_ids = itertools.count(1)


@pytest.fixture(autouse=True)
def _clean() -> None:
    BUILT.clear()
    EVENTS.clear()
    WORKSPACE["name"] = "alpha"


def _count(name: str) -> int:
    BUILT[name] = BUILT.get(name, 0) + 1
    return next(_ids)


class Grid:
    def __init__(self) -> None:
        self.id = _count("Grid")
        self.url = "http://grid.example.com"


class Caller:
    def __init__(self, grid: Grid) -> None:
        self.id = _count("Caller")
        self.grid = grid


class Actions:
    """A transient built as a nested dependency of another transient."""

    def __init__(self, caller: Caller) -> None:
        self.caller = caller


class Workspace(str):
    """The session key: which workspace a call is about."""


def workspace() -> Workspace:
    return Workspace(WORKSPACE["name"])


class Origins:
    def __init__(self, name: Workspace, grid: Grid) -> None:
        self.id = _count("Origins")
        self.name = name
        self.grid = grid


@dataclass
class Settings:
    transport: str = "http"
    host: str = "0.0.0.0"
    port: int = 8000


def server(**options) -> QuickMCP:
    mcp = QuickMCP("test", **options)
    mcp.singleton(Grid)
    mcp.transient(Caller)
    mcp.transient(Actions)
    mcp.transient(Workspace, workspace)
    mcp.session(Origins, by=Workspace, ttl=60)

    @mcp.tool
    async def navigate(
        url: str,
        grid: Inject[Grid],
        caller: Inject[Caller],
        actions: Inject[Actions],
        ctx: Context,
    ) -> dict:
        """Open a URL."""
        return {
            "url": url,
            "grid": grid.id,
            "caller": caller.id,
            "nested_is_same": actions.caller is caller,
            "ctx": type(ctx).__name__,
        }

    @mcp.tool
    def origins(origins: Inject[Origins]) -> dict:
        """The calling workspace's origins."""
        return {"id": origins.id, "name": origins.name}

    return mcp


# -- Inject[T] -----------------------------------------------------------------


async def test_an_injected_parameter_is_not_in_the_schema() -> None:
    async with Client(server().server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    schema = tools["navigate"].input_schema
    assert sorted(schema["properties"]) == ["url"]
    assert schema["required"] == ["url"]
    assert tools["origins"].input_schema.get("properties", {}) == {}


async def test_a_client_sending_an_injected_name_is_refused() -> None:
    async with Client(server().server) as client:
        result = await client.call_tool(
            "navigate",
            {"url": "https://example.com", "grid": "x"},
            raise_on_error=False,
        )
    assert result.is_error
    assert "Unexpected keyword argument" in result.content[0].text
    assert "Grid" not in BUILT


async def test_a_direct_call_with_explicit_arguments_needs_no_container() -> None:
    mcp = QuickMCP("direct")

    @mcp.tool
    def where(url: str, grid: Inject[Grid]) -> str:
        return f"{url} via {grid.url}"

    fake = Grid.__new__(Grid)
    fake.url = "http://fake.example.com"
    assert where("https://example.com", fake) == (
        "https://example.com via http://fake.example.com"
    )
    assert "Grid" not in BUILT


async def test_inject_works_in_resource_templates_and_prompts() -> None:
    mcp = server()

    @mcp.resource("grid://{name}/status")
    def status(name: str, grid: Inject[Grid]) -> str:
        return f"{name}: {grid.url}"

    @mcp.prompt
    def ask(who: str, grid: Inject[Grid]) -> str:
        return f"{who}, use {grid.url}"

    async with Client(mcp.server) as client:
        templates = await client.list_resource_templates()
        read = await client.read_resource("grid://main/status")
        prompts = {prompt.name: prompt for prompt in await client.list_prompts()}
        prompt = await client.get_prompt("ask", {"who": "Dr K"})
    assert templates[0].uri_template == "grid://{name}/status"
    assert read[0].text == "main: http://grid.example.com"
    assert [argument.name for argument in prompts["ask"].arguments] == ["who"]
    assert prompt.messages[0].content.text == "Dr K, use http://grid.example.com"


def test_lookup_is_one_resolver_per_key() -> None:
    assert lookup(Grid) is lookup(Grid)
    assert lookup(Grid) is not lookup(Caller)


async def test_an_unregistered_key_is_refused_at_its_first_call() -> None:
    mcp = QuickMCP("bare")

    @mcp.tool
    async def needs(grid: Inject[Grid]) -> str:
        return grid.url

    @mcp.tool
    async def looks_up() -> str:
        return (await lookup(Grid)()).url

    async with Client(mcp.server) as client:
        injected = await client.call_tool("needs", {}, raise_on_error=False)
        looked = await client.call_tool("looks_up", {}, raise_on_error=False)
    assert injected.is_error
    assert looked.is_error
    assert "Grid is not registered" in looked.content[0].text


# -- lifetimes -------------------------------------------------------------------


async def test_a_singleton_is_built_once() -> None:
    async with Client(server().server) as client:
        first = (await client.call_tool("navigate", {"url": "a"})).data
        second = (await client.call_tool("navigate", {"url": "b"})).data
    assert first["grid"] == second["grid"]
    assert BUILT["Grid"] == 1


async def test_a_transient_is_built_once_per_call_and_shared_when_nested() -> None:
    async with Client(server().server) as client:
        first = (await client.call_tool("navigate", {"url": "a"})).data
        second = (await client.call_tool("navigate", {"url": "b"})).data
    assert first["caller"] != second["caller"]
    assert first["nested_is_same"] and second["nested_is_same"]
    assert BUILT["Caller"] == 2
    assert first["ctx"] == "Context"


async def test_a_session_is_built_once_per_value_of_its_key() -> None:
    async with Client(server().server) as client:
        first = (await client.call_tool("origins", {})).data
        again = (await client.call_tool("origins", {})).data
        WORKSPACE["name"] = "beta"
        other = (await client.call_tool("origins", {})).data
    assert first == again
    assert first["name"] == "alpha"
    assert other["name"] == "beta"
    assert other["id"] != first["id"]
    assert BUILT["Origins"] == 2


async def test_a_session_is_rebuilt_after_end_session() -> None:
    mcp = server()
    async with Client(mcp.server) as client:
        first = (await client.call_tool("origins", {})).data
        await mcp.end_session("alpha")
        second = (await client.call_tool("origins", {})).data
    assert second["id"] != first["id"]
    assert BUILT["Origins"] == 2


async def test_a_session_is_rebuilt_after_its_ttl(monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(kubed.mcp.container, "monotonic", lambda: now[0])
    async with Client(server().server) as client:
        first = (await client.call_tool("origins", {})).data
        now[0] += 59
        kept = (await client.call_tool("origins", {})).data
        now[0] += 61
        rebuilt = (await client.call_tool("origins", {})).data
    assert kept == first
    assert rebuilt["id"] != first["id"]
    assert BUILT["Origins"] == 2


# -- the order rule ------------------------------------------------------------


class Pinned:
    """A singleton that asks for a transient: refused."""

    def __init__(self, caller: Caller) -> None:
        self.caller = caller


class Sticky:
    """A session that asks for a transient other than its own key: refused."""

    def __init__(self, caller: Caller) -> None:
        self.caller = caller


async def test_a_longer_lifetime_never_depends_on_a_shorter_one() -> None:
    mcp = server()
    mcp.singleton(Pinned)
    mcp.session(Sticky, by=Workspace, ttl=60)

    @mcp.tool
    async def pinned() -> str:
        return type(await lookup(Pinned)()).__name__

    @mcp.tool
    async def sticky() -> str:
        return type(await lookup(Sticky)()).__name__

    @mcp.tool
    def injected(pinned: Inject[Pinned]) -> str:
        return "never"

    async with Client(mcp.server) as client:
        singleton = await client.call_tool("pinned", {}, raise_on_error=False)
        session = await client.call_tool("sticky", {}, raise_on_error=False)
        through_inject = await client.call_tool("injected", {}, raise_on_error=False)
        allowed = (await client.call_tool("origins", {})).data
    assert singleton.is_error and session.is_error and through_inject.is_error
    assert "Pinned is a singleton and cannot depend on Caller, a transient" in (
        singleton.content[0].text
    )
    assert "Sticky is a session and cannot depend on Caller, a transient" in (
        session.content[0].text
    )
    # A session may take its own key, which never changes for the session's life.
    assert allowed["name"] == "alpha"


# -- async factories and teardown ---------------------------------------------


class Bus:
    pass


class Monitor:
    def __init__(self, bus: Bus) -> None:
        self.bus = bus


class Scrubber:
    pass


class Page:
    pass


async def make_bus() -> Bus:
    EVENTS.append("bus built")
    return Bus()


@asynccontextmanager
async def make_monitor(bus: Bus):
    EVENTS.append("monitor started")
    yield Monitor(bus)
    EVENTS.append("monitor stopped")


@asynccontextmanager
async def make_scrubber(name: Workspace):
    EVENTS.append(f"scrubber {name} started")
    yield Scrubber()
    EVENTS.append(f"scrubber {name} stopped")


@asynccontextmanager
async def make_page(monitor: Monitor):
    EVENTS.append("page opened")
    yield Page()
    EVENTS.append("page closed")


async def test_async_factories_and_context_managers_tear_down_in_reverse() -> None:
    mcp = server()
    mcp.singleton(Bus, make_bus)
    mcp.singleton(Monitor, make_monitor)
    mcp.session(Scrubber, make_scrubber, by=Workspace, ttl=60)
    mcp.transient(Page, make_page)

    @mcp.tool
    async def watch(
        monitor: Inject[Monitor], scrubber: Inject[Scrubber], page: Inject[Page]
    ) -> bool:
        EVENTS.append("tool ran")
        return isinstance(monitor.bus, Bus)

    async with Client(mcp.server) as client:
        assert (await client.call_tool("watch", {})).data is True
        EVENTS.append("call ended")
        await mcp.end_session("alpha")
        EVENTS.append("session ended")
        WORKSPACE["name"] = "beta"
        await client.call_tool("watch", {})
        EVENTS.append("client closing")
    assert EVENTS == [
        "bus built",
        "monitor started",
        "scrubber alpha started",
        "page opened",
        "tool ran",
        "page closed",  # a transient ends with its call
        "call ended",
        "scrubber alpha stopped",  # a session ends when it is ended
        "session ended",
        "scrubber beta started",
        "page opened",
        "tool ran",
        "page closed",
        "client closing",
        "scrubber beta stopped",  # at shutdown: open sessions, then singletons
        "monitor stopped",
    ]


async def test_a_lifespan_passed_in_runs_inside_the_containers() -> None:
    @asynccontextmanager
    async def lifespan(server):
        EVENTS.append("lifespan up")
        yield {"answer": 42}
        EVENTS.append("lifespan down")

    mcp = server(lifespan=lifespan)
    mcp.singleton(Monitor, make_monitor)
    mcp.singleton(Bus, make_bus)

    @mcp.tool
    async def answer(ctx: Context, monitor: Inject[Monitor]) -> int:
        return ctx.lifespan_context["answer"]

    async with Client(mcp.server) as client:
        assert (await client.call_tool("answer", {})).data == 42
    assert EVENTS[0] == "lifespan up"
    assert EVENTS[-2:] == ["lifespan down", "monitor stopped"]


async def test_singletons_are_built_again_after_a_restart() -> None:
    mcp = server()
    for _ in range(2):
        async with Client(mcp.server) as client:
            await client.call_tool("navigate", {"url": "a"})
    assert BUILT["Grid"] == 2


class Booted:
    """Built before the server starts, by the caller, who owns its lifetime."""

    def __init__(self) -> None:
        self.id = _count("Booted")

    async def __aenter__(self) -> Booted:
        EVENTS.append("booted entered")
        return self

    async def __aexit__(self, *exc) -> None:
        EVENTS.append("booted exited")


class Reader:
    def __init__(self, booted: Booted) -> None:
        self.booted = booted


async def test_a_singleton_instance_is_injected_and_never_built_or_torn_down() -> None:
    booted = Booted()
    mcp = QuickMCP("instance")
    mcp.singleton(Booted, instance=booted)
    mcp.transient(Reader)

    @mcp.tool
    def ids(booted: Inject[Booted], reader: Inject[Reader]) -> list[int]:
        return [booted.id, reader.booted.id]

    for _ in range(2):  # a restart keeps it: it was never the container's
        async with Client(mcp.server) as client:
            assert (await client.call_tool("ids", {})).data == [booted.id] * 2
    assert BUILT["Booted"] == 1
    assert EVENTS == []


def test_a_singleton_takes_a_factory_or_an_instance_not_both() -> None:
    with pytest.raises(TypeError, match="a factory or an instance, not both"):
        QuickMCP("both").singleton(Grid, Grid, instance=Grid())


async def test_registering_a_key_again_replaces_its_instance_or_recipe() -> None:
    mcp = server()
    fake = Grid.__new__(Grid)
    fake.id = -1
    mcp.singleton(Grid, instance=fake)
    async with Client(mcp.server) as client:
        assert (await client.call_tool("navigate", {"url": "a"})).data["grid"] == -1
    mcp.singleton(Grid)
    async with Client(mcp.server) as client:
        assert (await client.call_tool("navigate", {"url": "a"})).data["grid"] > 0
    assert BUILT["Grid"] == 1


# -- override and config -------------------------------------------------------


async def test_override_replaces_what_a_key_resolves_to() -> None:
    mcp = server()
    fake = Grid.__new__(Grid)
    fake.id = -1
    mcp.override(Grid, fake)
    async with Client(mcp.server) as client:
        result = (await client.call_tool("navigate", {"url": "a"})).data
    assert result["grid"] == -1
    assert "Grid" not in BUILT


async def test_the_config_is_injectable_as_its_own_type() -> None:
    settings = Settings(port=9000)
    mcp = QuickMCP("configured", config=settings)

    @mcp.tool
    def port(settings: Inject[Settings]) -> int:
        return settings.port

    async with Client(mcp.server) as client:
        assert (await client.call_tool("port", {})).data == 9000


async def test_a_factory_parameter_with_no_registered_type_needs_a_default() -> None:
    def grid(url, retries: int = 3) -> Grid:
        return Grid()

    mcp = QuickMCP("strict")
    mcp.singleton(Grid, grid)

    @mcp.tool
    async def where() -> str:
        return (await lookup(Grid)()).url

    async with Client(mcp.server) as client:
        result = await client.call_tool("where", {}, raise_on_error=False)
    assert result.is_error
    assert "needs 'url', which has no registered type and no default" in (
        result.content[0].text
    )


# -- FastMCP's own ---------------------------------------------------------------


async def test_a_transform_added_applies_to_every_tool() -> None:
    mcp = server()
    mcp.add_transform(Namespace("api"))
    async with Client(mcp.server) as client:
        names = {tool.name for tool in await client.list_tools()}
        result = (await client.call_tool("api_navigate", {"url": "a"})).data
    assert {"api_navigate", "api_origins"} <= names
    assert "navigate" not in names
    assert result["grid"] > 0


async def test_a_custom_route_answers_over_http() -> None:
    mcp = server()

    @mcp.custom_route("/health", methods=["GET"], name="health")
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    assert health.__name__ == "health"  # the decorator hands back the handler
    async with asgi_server(mcp.server) as running, running.http_client() as http:
        answer = await http.get(str(running.app.url_path_for("health")))
    assert answer.status_code == 200
    assert answer.json() == {"status": "ok"}


# -- run() ---------------------------------------------------------------------


@pytest.fixture
def ran(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        "fastmcp.FastMCP.run", lambda self, **kwargs: calls.append(kwargs)
    )
    return calls


def test_run_reads_serving(ran) -> None:
    QuickMCP("http").run(Settings(host="127.0.0.1", port=9001))
    assert ran == [
        {
            "transport": "http",
            "host": "127.0.0.1",
            "port": 9001,
            "show_banner": False,
        }
    ]


def test_run_reads_an_argparse_namespace(ran) -> None:
    args = argparse.Namespace(transport="stdio", host="0.0.0.0", port=8000, extra=1)
    QuickMCP("stdio").run(args, path="/mcp")
    assert ran == [{"transport": "stdio", "show_banner": False}]


def test_run_falls_back_to_the_config_and_passes_http_options(ran) -> None:
    QuickMCP("configured", config=Settings()).run(path="/mcp")
    assert ran[0]["port"] == 8000
    assert ran[0]["path"] == "/mcp"


def test_run_needs_a_serving(ran) -> None:
    with pytest.raises(TypeError, match="run\\(\\) needs a Serving"):
        QuickMCP("bare").run()
    assert ran == []


def test_run_hands_fastmcps_logging_to_the_server(ran) -> None:
    fastmcp_logger = logging.getLogger("fastmcp")
    fastmcp_logger.addHandler(logging.NullHandler())
    fastmcp_logger.propagate = False
    QuickMCP("logged").run(Settings())
    assert fastmcp_logger.handlers == []
    assert fastmcp_logger.propagate is True


def test_serving_over_http_leaves_fastmcps_logging_with_the_server(
    monkeypatch,
) -> None:
    """FastMCP's own run, not a stand-in: given a log level it would put its handlers
    back on its logger for the rest of the process. The config has one; run() never
    passes it."""
    seen: list[tuple[list[logging.Handler], bool]] = []

    async def serve(self, sockets=None) -> None:
        logger = logging.getLogger("fastmcp")
        seen.append((list(logger.handlers), logger.propagate))

    monkeypatch.setattr("uvicorn.Server.serve", serve)
    fastmcp_logger = logging.getLogger("fastmcp")
    QuickMCP("served").run(
        argparse.Namespace(
            transport="http", host="127.0.0.1", port=0, log_level="DEBUG"
        ),
        uvicorn_config={"log_config": None},
    )
    assert seen == [([], True)]
    assert fastmcp_logger.handlers == []
    assert fastmcp_logger.propagate is True
