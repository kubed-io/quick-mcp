"""``load_config()``: a server's config, loaded before the server is built."""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager

import pytest
from fastmcp import Client
from pydantic import BaseModel

import kubed.mcp._config
from kubed.mcp import (
    Component,
    Config,
    Field,
    LoadedConfig,
    QuickMCP,
    ServingSettings,
    Source,
    config,
    load_config,
)

pytestmark = pytest.mark.unit


@config()
class Root(Config):
    name: str = Field("kb", env="KB_NAME")
    title: str = "t"


@config("store")
class Store(Config):
    url: str = Field("mem://", env="STORE_URL", arg="--store-url")


class Keeper(Component):
    section = Store

    def __init__(self) -> None:
        self.registered: list = []

    def register(self, mcp: QuickMCP, section: BaseModel | None) -> None:
        self.registered.append((mcp, section))


def to_one(c: Config) -> BaseModel:
    return ServingSettings(port=1)


def every_source(f):
    return {
        "sections": [Root],
        "components": [Keeper()],
        "argv": ["--store-url", "s://"],
        "environ": {"CONFIG_FILE": str(f), "KB_NAME": "env"},
        "hooks": {"serving": to_one},
    }


# -- the loader


def test_load_config_gives_the_config_and_its_sources() -> None:
    loaded = load_config(
        sections=[Root],
        components=[Keeper()],
        environ={"KB_NAME": "env"},
        argv=["--store-url", "s://"],
    )
    assert (loaded.config.name, loaded.config.store.url) == ("env", "s://")
    assert loaded.sources["name"] == Source("env", "KB_NAME")
    assert loaded.sources["store.url"] == Source("arg", "--store-url")
    assert loaded.sources["serving.port"] == Source("default")


def test_load_config_takes_hooks_and_values_in_code() -> None:
    hooked = load_config(hooks={"serving": to_one}, environ={})
    assert hooked.sources["serving.port"] == Source("hook")
    assert load_config(sections=[Root], config={"name": "c"}).sources["name"] == Source(
        "code"
    )


def test_load_config_logs_nothing(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        load_config(environ={"PORT": "9000"})
    assert not [r for r in caplog.records if r.name.startswith("kubed.mcp")]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"components": [Keeper], "environ": {}},
        {"components": Keeper(), "environ": {}},
        {"config": {}, "argv": []},
        {"environ": {"PORT": "x"}},
        {"hooks": {"nope": to_one}, "environ": {}},
    ],
)
def test_load_config_refuses_what_quick_mcp_refuses(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)) as by_server:
        QuickMCP("t", **kwargs)
    with pytest.raises(type(by_server.value)) as by_loader:
        load_config(**kwargs)
    assert str(by_loader.value) == str(by_server.value)


def test_a_loaded_config_is_read_only() -> None:
    loaded = load_config(sections=[Root], environ={})
    with pytest.raises(AttributeError):
        loaded.config = None
    with pytest.raises(AttributeError):
        loaded.extra = 1
    with pytest.raises(TypeError):
        loaded.sources["name"] = Source("code")


# -- a server from a load


def lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "kubed.mcp.config"]


async def test_a_load_passed_in_is_the_load_quick_mcp_does(tmp_path, caplog) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("title: from-file\n")
    with caplog.at_level(logging.INFO, logger="kubed.mcp.config"):
        inside = QuickMCP("t", **every_source(f))
        outside = QuickMCP("t", loaded=load_config(**every_source(f)))
        assert lines(caplog) == []  # nothing at load or construction
        for mcp in (inside, outside):
            async with Client(mcp.server):
                pass
    assert outside.config.model_dump() == inside.config.model_dump()
    assert dict(outside.config_sources) == dict(inside.config_sources)
    paths = ("title", "name", "store.url", "serving.port", "delivery.via")
    assert [outside.config_sources[p] for p in paths] == [
        Source("file", str(f)),
        Source("env", "KB_NAME"),
        Source("arg", "--store-url"),
        Source("hook"),
        Source("default"),
    ]
    first, second = lines(caplog)
    assert first == second


def test_a_server_built_from_a_load_reads_no_source_again(
    tmp_path, monkeypatch
) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("title: from-file\n")
    environ = {"CONFIG_FILE": str(f), "STORE_URL": "env://"}
    keeper = Keeper()
    loaded = load_config(sections=[Root], components=[keeper], environ=environ)
    f.unlink()
    environ["STORE_URL"] = "changed://"

    def boom(*args, **kwargs):
        raise AssertionError("read again")

    for name in ("_load", "_compose", "_read_file", "_read_env", "_read_argv"):
        monkeypatch.setattr(kubed.mcp._config, name, boom)
    mcp = QuickMCP("t", loaded=loaded)
    assert mcp.config is loaded.config and mcp.config_sources is loaded.sources
    assert (mcp.config.title, mcp.config.store.url) == ("from-file", "env://")
    assert keeper.registered == [(mcp, loaded.config.store)]


@pytest.mark.parametrize(
    "given, names",
    [
        ({"sections": []}, "sections="),
        ({"components": []}, "components="),
        ({"hooks": {}}, "hooks="),
        ({"config": {}}, "config="),
        ({"argv": []}, "argv="),
        ({"environ": {}}, "environ="),
        (
            {"environ": {}, "sections": [Root], "argv": ["--port", "1"]},
            "sections=, argv=, environ=",
        ),
    ],
)
def test_loaded_takes_no_other_config_argument(given, names) -> None:
    loaded = load_config(environ={})
    message = (
        rf"^loaded= is already loaded: pass {re.escape(names)} "
        r"to load_config\(\), not with loaded=$"
    )
    with pytest.raises(TypeError, match=message):
        QuickMCP("t", loaded=loaded, **given)


@pytest.mark.parametrize(
    "bad, what",
    [
        (object(), "an instance of object"),
        (LoadedConfig, "the class LoadedConfig"),
        ({"port": 1}, "an instance of dict"),
    ],
)
def test_loaded_takes_only_what_load_config_returns(bad, what) -> None:
    message = rf"^loaded= takes what load_config\(\) returns, got {what}$"
    with pytest.raises(TypeError, match=message):
        QuickMCP("t", loaded=bad)
    # and the private _Loaded inside one is refused too:
    with pytest.raises(TypeError, match=r"got an instance of _Loaded"):
        QuickMCP("t", loaded=load_config(environ={})._loaded)


async def test_fastmcp_options_go_through_beside_loaded() -> None:
    entered: list[str] = []

    @asynccontextmanager
    async def lifespan(server):
        entered.append("lifespan")
        yield {}

    loaded = load_config(environ={})
    mcp = QuickMCP(
        "t", loaded=loaded, instructions="hi", lifespan=lifespan, header_prefix="Kb"
    )
    assert mcp.server.instructions == "hi"
    async with Client(mcp.server):
        pass
    assert entered == ["lifespan"]


def test_a_load_builds_one_server() -> None:
    keeper = Keeper()
    loaded = load_config(components=[keeper], environ={})
    refused = (
        r"^loaded= is already loaded: pass environ= to load_config\(\), "
        r"not with loaded=$"
    )
    # refused: doesn't claim the load
    with pytest.raises(TypeError, match=refused):
        QuickMCP("refused", loaded=loaded, environ={})
    a = QuickMCP("a", loaded=loaded)
    message = (
        r"^loaded= builds one server, and this one already built a server: "
        r"call load_config\(\) again for another$"
    )
    with pytest.raises(TypeError, match=message):
        QuickMCP("b", loaded=loaded)
    assert [m for m, _ in keeper.registered] == [a]


def test_a_refused_header_prefix_leaves_the_load_unbuilt() -> None:
    loaded = load_config(environ={})
    with pytest.raises(ValueError, match="header-name token"):
        QuickMCP("t", loaded=loaded, header_prefix="bad prefix")
    assert QuickMCP("t", loaded=loaded).config is loaded.config
