"""The ``Max`` input: a cap on listed skills, resolved by levels (DLV-17)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client
from fastmcp.skills.provider import SkillManifestProvider
from gateway_server import over_http
from plugin_server import skill, write
from skill_server import read_directory, request
from test_delivery_inputs import warnings

from kubed.mcp import DeliverySettings, QuickMCP
from kubed.mcp.delivery import _Partial, _resolved
from kubed.mcp.skills import Skill, serve

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


def plugin(tmp_path: Path, name: str, count: int) -> Path:
    """A plugin folder ``name`` holding ``count`` skills, ``<name>-000`` on."""
    return write(
        tmp_path / name,
        {
            f"skills/{name}-{i:03}/SKILL.md": skill(f"{name}-{i:03}")
            for i in range(count)
        },
    )


def capped(
    tmp_path: Path, *plugins: tuple[str, int], own: int = 0, **options
) -> QuickMCP:
    """A server ``t``: ``own`` skills of its own, then each plugin under its name."""
    mcp = QuickMCP("t", **options)
    if own:
        serve(
            mcp.server,
            [
                Skill.from_files(
                    {
                        "SKILL.md": (
                            f"---\nname: own-{i:03}\ndescription: Its own.\n---\n"
                        )
                    }
                )
                for i in range(own)
            ],
        )
    for name, count in plugins:
        mcp.add_plugin(plugin(tmp_path, name, count), namespace=name)
    return mcp


async def page_sizes(client, method: str, key: str) -> list[int]:
    """The length of each page's ``key`` list, following ``nextCursor``. For
    ``resources/list`` it counts only the rows holding ``skill://``."""
    sizes, cursor = [], None
    while True:
        page = await request(client, method, **({"cursor": cursor} if cursor else {}))
        counted = [
            row
            for row in page[key]
            if method != "resources/list" or "skill://" in str(row["uri"])
        ]
        sizes.append(len(counted))
        cursor = page.get("nextCursor")
        if not cursor:
            return sizes


async def listing(client) -> list[str]:
    """Every ``skills/list`` entry URI, following ``nextCursor``."""
    found, cursor = [], None
    while True:
        page = await request(
            client, "skills/list", **({"cursor": cursor} if cursor else {})
        )
        found += [entry["uri"] for entry in page["skills"]]
        cursor = page.get("nextCursor")
        if not cursor:
            return found


def uris(name: str, numbers: Iterable[int]) -> list[str]:
    return [f"skill://{name}/{name}-{i:03}/SKILL.md" for i in numbers]


def own(numbers: Iterable[int]) -> list[str]:
    return [f"skill://own-{i:03}/SKILL.md" for i in numbers]


def visibility_checks(monkeypatch) -> list[str]:
    """Wrap FastMCP's ``_is_served``, recording each skill path it is called with."""
    seen: list[str] = []
    original = SkillManifestProvider._is_served

    async def spy(self, path, manifest):
        seen.append(path)
        return await original(self, path, manifest)

    monkeypatch.setattr(SkillManifestProvider, "_is_served", spy)
    return seen


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


@pytest.mark.parametrize("mode", MODES)
async def test_the_default_lists_100_of_101(tmp_path, mode):
    async with Client(capped(tmp_path, ("a", 101)).server, mode=mode) as client:
        assert await listing(client) == uris("a", range(100))


@pytest.mark.parametrize(
    ("options", "sent", "count"),
    [
        ({}, {"headers": {"X-MCP-Max": "150"}}, 150),
        ({}, {"params": {"max": "5"}}, 5),
        ({"header_prefix": "Kb"}, {"headers": {"Kb-Max": "5"}}, 5),
        ({"config": {"delivery": {"max": 150}}}, {}, 150),
        ({}, {"headers": {"X-MCP-Max": "all"}}, 160),
    ],
)
async def test_the_cap_is_set_per_request(tmp_path, options, sent, count):
    async with over_http(capped(tmp_path, ("a", 160), **options), **sent) as client:
        assert await listing(client) == uris("a", range(count))


async def test_delivery_default_caps_the_listing(tmp_path):
    mcp = capped(tmp_path, ("a", 160))
    mcp.delivery_default(lambda client: _Partial(max="5"))
    async with over_http(mcp) as client:
        assert await listing(client) == uris("a", range(5))


@pytest.mark.parametrize("mode", MODES)
async def test_the_order_holds_on_every_call_and_page(tmp_path, mode):  # Review Focus 1
    mcp = capped(tmp_path, ("a", 60), ("b", 60), list_page_size=7)
    async with Client(mcp.server, mode=mode) as client:
        seen = [await listing(client) for _ in range(3)]
        pages = await page_sizes(client, "skills/list", "skills")
    assert seen == [uris("a", range(60)) + uris("b", range(40))] * 3
    assert pages == [7] * 14 + [2]


async def test_the_order_plugins_are_added_in_chooses_who_survives(tmp_path):
    mcp = capped(tmp_path, ("b", 60), ("a", 60))
    async with Client(mcp.server) as client:
        assert await listing(client) == uris("a", range(40)) + uris("b", range(60))


@pytest.mark.parametrize(("mine", "kept_theirs"), [(3, 97), (100, 0), (105, 0)])
async def test_the_servers_own_skills_are_never_cut(tmp_path, mine, kept_theirs):
    async with Client(capped(tmp_path, ("a", 101), own=mine).server) as client:
        assert await listing(client) == uris("a", range(kept_theirs)) + own(range(mine))


@pytest.mark.parametrize("via", ["mcp", "extension"])
async def test_extension_mode_is_cut_alike(tmp_path, via):
    async with over_http(
        capped(tmp_path, ("a", 101)), headers={"X-MCP-Via": via}
    ) as client:
        assert await listing(client) == uris("a", range(100))


async def test_the_cut_adds_no_visibility_check(tmp_path, monkeypatch):  # #86
    checked = visibility_checks(monkeypatch)
    for sent in ({"X-MCP-Max": "all"}, {}):
        checked.clear()
        async with over_http(
            capped(tmp_path / str(len(sent)), ("a", 101)), headers=sent
        ) as client:
            await listing(client)
        assert len(checked) == 101  # each plugin skill once, capped or not


async def test_a_cut_skill_is_still_served(tmp_path):  # DLV-10
    root = "skill://a/a-100"
    async with Client(capped(tmp_path, ("a", 101)).server) as client:
        assert f"{root}/SKILL.md" not in await listing(client)
        got = await request(client, "skills/get", uri=f"{root}/SKILL.md")
        assert got["skill"]["uri"] == f"{root}/SKILL.md"
        for tail in ("SKILL.md", "_manifest", "_index.md", "_index.json"):
            assert await client.read_resource(f"{root}/{tail}")
        assert (await read_directory(client, uri=root))["resources"]
