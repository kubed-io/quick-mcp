"""The ``Max`` input: a cap on listed skills, resolved by levels (DLV-17)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fastmcp import Client
from gateway_server import over_http
from test_delivery_inputs import warnings

from kubed.mcp import DeliverySettings, QuickMCP
from kubed.mcp.delivery import _Partial, _resolved

pytestmark = pytest.mark.unit

MODES = ["legacy", "auto"]
META = "io.kubed/delivery"


def result(client):
    """The handshake result the client holds, whichever handshake it made."""
    return client.initialize_result or client.session.discover_result


def server(delivery=None, header_prefix=None) -> QuickMCP:
    """A server whose ``cap`` tool reports the request's resolved cap."""
    mcp = QuickMCP(
        "t",
        config={} if delivery is None else {"delivery": delivery},
        header_prefix=header_prefix,
    )

    @mcp.tool
    def cap() -> dict:
        return {"max": _resolved().max}

    return mcp


async def cap(client) -> int | None:
    return (await client.call_tool("cap", {})).data["max"]


@pytest.mark.parametrize(
    ("options", "sent", "expected"),
    [
        ({}, {}, 100),
        ({}, {"headers": {"X-MCP-Max": "150"}}, 150),
        ({"header_prefix": "Kb"}, {"headers": {"Kb-Max": "5", "X-MCP-Max": "150"}}, 5),
        ({}, {"params": {"max": "150"}}, 150),
        ({"delivery": DeliverySettings(max=150)}, {}, 150),
        ({"delivery": DeliverySettings(max="150")}, {"params": {"max": "5"}}, 5),
        ({}, {"headers": {"X-MCP-Max": " ALL "}}, None),
        ({}, {"headers": {"X-MCP-Max": "007"}}, 7),
    ],
)
async def test_the_cap_resolves_at_every_level(options, sent, expected):
    async with over_http(server(**options), **sent) as client:
        assert await cap(client) == expected


@pytest.mark.parametrize("answer", [_Partial(max="5"), SimpleNamespace(max=5)])
async def test_delivery_default_sets_the_cap(answer):
    mcp = server()
    mcp.delivery_default(lambda client: answer)
    async with over_http(mcp) as client:
        assert await cap(client) == 5


@pytest.mark.parametrize("raw", ["0", "-1", "+5", "1.5", "ten", "all of them"])
async def test_an_untaken_cap_falls_through(raw, caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(
            server(), headers={"X-MCP-Max": raw}, params={"max": "7"}
        ) as client:
            assert await cap(client) == 7
    (record,) = warnings(caplog)
    assert record.getMessage() == (
        f"header X-MCP-Max {raw!r} is ignored, and the next source answers: "
        "it is a whole number of 1 or more, or all for no cap"
    )


async def test_a_non_ascii_digit_cap_falls_through(caplog):
    # A header cannot carry it, so it comes by the query: fullwidth one and two.
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(server(), params={"max": "\uff11\uff12"}) as client:
            assert await cap(client) == 100
    (record,) = warnings(caplog)
    assert record.getMessage().startswith(
        "query parameter max '\uff11\uff12' is ignored"
    )


async def test_an_untaken_config_cap_warns_once_at_build(caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        mcp = server(DeliverySettings(max="0"))
        async with Client(mcp.server) as client:
            assert await cap(client) == 100
    (record,) = warnings(caplog)
    assert record.getMessage().startswith("config delivery.max '0' is ignored")


async def test_a_bool_cap_falls_through(caplog):  # Review Focus 3
    mcp = server()
    mcp.delivery_default(lambda client: SimpleNamespace(max=True))
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        async with over_http(mcp) as client:
            assert await cap(client) == 100
    (record,) = warnings(caplog)
    assert record.getMessage().startswith("delivery_default max 'True' is ignored")


async def test_the_config_cap_loads_from_yaml_and_env(tmp_path):  # Review Focus 3
    (tmp_path / "c.yaml").write_text("delivery:\n  max: 150\n")
    from_yaml = QuickMCP("t", environ={"CONFIG_FILE": str(tmp_path / "c.yaml")})
    from_env = QuickMCP("t", environ={"DELIVERY_MAX": "150"})
    assert from_yaml.config.delivery.max == 150
    assert from_env.config.delivery.max == "150"
    for mcp in (from_yaml, from_env):
        assert mcp._resolver.resolve(None).max == 150


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("sent", "echoed"),
    [({}, 100), ({"X-MCP-Max": "all"}, "all"), ({"X-MCP-Max": "150"}, 150)],
)
async def test_the_echo_carries_the_cap(mode, sent, echoed):  # DLV-12
    async with over_http(QuickMCP("t"), mode=mode, headers=sent) as client:
        assert result(client).meta[META] == {
            "resources": "on",
            "listed": ["skills"],
            "via": "mcp",
            "prefix": None,
            "max": echoed,
        }


@pytest.mark.parametrize(("sent", "shown"), [(True, "True"), (1.0, "1.0")])
async def test_a_bool_or_float_config_cap_is_ignored(sent, shown, caplog):
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        mcp = server(DeliverySettings(max=sent))
        async with Client(mcp.server) as client:
            assert await cap(client) == 100
    (record,) = warnings(caplog)
    assert record.getMessage().startswith(f"config delivery.max {shown!r} is ignored")


async def test_a_yaml_bool_config_cap_is_ignored(tmp_path, caplog):
    (tmp_path / "c.yaml").write_text("delivery:\n  max: true\n")
    with caplog.at_level(logging.WARNING, logger="kubed.mcp.delivery"):
        mcp = QuickMCP("t", environ={"CONFIG_FILE": str(tmp_path / "c.yaml")})
        assert mcp._resolver.resolve(None).max == 100
    (record,) = warnings(caplog)
    assert record.getMessage().startswith("config delivery.max 'True' is ignored")
