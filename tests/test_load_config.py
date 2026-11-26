"""``load_config()``: a server's config, loaded before the server is built."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from kubed.mcp import (
    Component,
    Config,
    Field,
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
