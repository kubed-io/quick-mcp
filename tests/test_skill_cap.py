"""The ``Max`` input: a cap on listed skills, resolved by levels (DLV-17)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client
from fastmcp.skills.provider import SkillManifestProvider
from gateway_server import over_http
from plugin_server import agent_md, command_md, skill, write
from skill_server import read_directory, request
from test_delivery_inputs import warnings
from test_handshake import errors

from kubed.mcp import DeliverySettings, QuickMCP, plugins
from kubed.mcp.delivery import Resolved, SkillCount, _Partial, _resolved
from kubed.mcp.plugins import Plugins, View
from kubed.mcp.protocol import MCPError
from kubed.mcp.skills import EntryOnlySkillProvider, Skill, serve

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


async def rows(client) -> list[str]:
    """Every ``resources/list`` row URI holding ``skill://``, in the order sent,
    following ``nextCursor``."""
    found, cursor = [], None
    while True:
        page = await request(
            client, "resources/list", **({"cursor": cursor} if cursor else {})
        )
        found += [
            str(r["uri"]) for r in page["resources"] if "skill://" in str(r["uri"])
        ]
        cursor = page.get("nextCursor")
        if not cursor:
            return found


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


@pytest.mark.parametrize("mode", MODES)
async def test_resources_list_cuts_at_the_same_places(tmp_path, mode):  # Review Focus 1
    mcp = capped(tmp_path, ("a", 60), ("b", 60), own=3, list_page_size=7)
    async with Client(mcp.server, mode=mode) as client:
        listed = await listing(client)
        shown = await rows(client)
        pages = await page_sizes(client, "resources/list", "resources")
    assert listed == uris("a", range(60)) + uris("b", range(37)) + own(range(3))
    assert sorted(shown) == sorted(listed)
    assert pages == [7] * 14 + [2]


async def test_a_prefixed_listing_is_cut_alike(tmp_path):  # Review Focus 2
    async with over_http(capped(tmp_path, ("a", 101)), header="kb+") as client:
        prefix = result(client).meta[META]["prefix"]
        listed = await listing(client)
        shown = await rows(client)
    assert prefix == "kb+"  # the cut ran on the prefixed build (GW-6)
    assert listed == uris("a", range(100))
    assert all(u.startswith("skill://") for u in shown)  # a row's uri never is (GW-8)
    assert sorted(shown) == listed


async def test_prompt_and_agent_rows_are_never_cut(tmp_path):
    root = plugin(tmp_path, "a", 101)
    write(root, {f"commands/c{i}.md": command_md() for i in range(3)})
    write(root, {f"agents/g{i}.md": agent_md(f"g{i}") for i in range(3)})
    mcp = QuickMCP("t")
    mcp.add_plugin(root, namespace="a")
    headers = {"X-MCP-Max": "5", "X-MCP-Listed": "skills,prompts,agents"}
    async with over_http(mcp, headers=headers) as client:
        everything = [str(r.uri) for r in await client.list_resources()]
        prompts = await client.list_prompts()
        agents = (await request(client, "agents/list"))["agents"]
    assert len([u for u in everything if u.startswith("skill://")]) == 5
    assert len([u for u in everything if u.startswith("prompts://")]) == 3
    assert len([u for u in everything if u.startswith("agents://")]) == 3
    assert (len(prompts), len(agents)) == (3, 3)


async def test_list_resources_is_cut(tmp_path):  # DLV-6
    async with over_http(
        capped(tmp_path, ("a", 101)), headers={"X-MCP-Resources": "off"}
    ) as client:
        found = json.loads(
            (await client.call_tool("list_resources", {})).content[0].text
        )
    assert (
        len([r for r in found["resources"] if r["uri"].startswith("skill://")]) == 100
    )


async def test_local_listing_skills_keeps_the_same_room(tmp_path):  # Dr K, call 3
    headers = {"X-MCP-Via": "local", "X-MCP-Listed": "skills"}
    async with over_http(
        capped(tmp_path, ("a", 101), own=3), headers=headers
    ) as client:
        assert await listing(client) == own(range(3))
        assert sorted(await rows(client)) == sorted(
            uris("a", range(97)) + own(range(3))
        )


async def test_no_cap_lists_every_row(tmp_path):
    async with over_http(
        capped(tmp_path, ("a", 101)), headers={"X-MCP-Max": "all"}
    ) as client:
        assert sorted(await rows(client)) == uris("a", range(101))


@pytest.mark.parametrize("mode", MODES)
async def test_resources_list_runs_no_listing_pass(
    tmp_path, monkeypatch, mode
):  # Review Focus 5
    checked = visibility_checks(monkeypatch)
    async with Client(capped(tmp_path, ("a", 101), own=3).server, mode=mode) as client:
        assert len(await rows(client)) == 100
    assert checked == []


@pytest.mark.parametrize(
    ("own_", "plugin_", "cap_", "via", "count"),
    [
        (0, 101, 100, "mcp", (100, 1)),
        (3, 101, 100, "mcp", (100, 4)),
        (5, 101, 3, "extension", (5, 101)),
        (0, 101, None, "mcp", (101, 0)),
        (3, 101, 100, "local", (3, 0)),
        (2, 0, 100, "mcp", (2, 0)),
    ],
)
def test_count(own_, plugin_, cap_, via, count):
    assert plugins._count(own_, plugin_, cap_, via) == SkillCount(*count)


def seen_by(mcp: QuickMCP) -> list[Resolved]:
    got: list[Resolved] = []
    mcp.instructions(
        lambda d: got.append(d) or ("cut" if d.skills and d.skills.cut else "whole")
    )
    return got


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("mine", "headers", "max_", "count"),
    [
        (0, {}, 100, (100, 1)),
        (3, {}, 100, (100, 4)),
        (0, {"X-MCP-Max": "all"}, None, (101, 0)),
        (3, {"X-MCP-Via": "local"}, 100, (3, 0)),
        (5, {"X-MCP-Max": "3"}, 3, (5, 101)),
    ],
)
async def test_instructions_see_the_cap_and_the_count(
    tmp_path, mode, mine, headers, max_, count
):
    mcp = capped(tmp_path, ("a", 101), own=mine)
    got = seen_by(mcp)
    async with over_http(mcp, mode=mode, headers=headers) as client:
        text = result(client).instructions
        listed = await listing(client)
    (delivery,) = got
    assert delivery.max == max_
    assert delivery.skills == SkillCount(*count)
    assert delivery.skills.listed == len(listed)
    assert text == ("cut" if count[1] else "whole")


@pytest.mark.parametrize("mode", MODES)
async def test_the_count_follows_skills_list_not_resources_list(tmp_path, mode):
    mcp = capped(tmp_path, ("a", 101))
    got = seen_by(mcp)
    async with over_http(mcp, mode=mode, headers={"X-MCP-Listed": "none"}) as client:
        assert await rows(client) == []
        assert len(await listing(client)) == 100
    assert got[0].skills == SkillCount(100, 1)


@pytest.mark.parametrize("mode", MODES)
async def test_the_count_reuses_the_cached_build(
    tmp_path, monkeypatch, mode
):  # Review Focus 5, #86
    mcp = capped(tmp_path, ("a", 101))
    seen_by(mcp)
    made = []
    plain = Plugins._make
    monkeypatch.setattr(
        Plugins, "_make", lambda self, *a: made.append(1) or plain(self, *a)
    )
    checked = visibility_checks(monkeypatch)
    async with over_http(mcp, mode=mode) as client:
        assert checked == []  # the handshake ran no listing
        await rows(client)
        assert checked == []
        await listing(client)
    assert len(made) == 1


async def test_skills_is_none_off_the_handshake(tmp_path):
    mcp = capped(tmp_path, ("a", 101))
    seen_by(mcp)

    @mcp.tool
    def count() -> dict:
        return {"skills": _resolved().skills}

    async with over_http(mcp) as client:
        assert (await client.call_tool("count", {})).data == {"skills": None}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("mine", "count"), [(0, None), (3, SkillCount(3, 0))])
async def test_a_server_without_plugin_skills(mode, mine, count):
    mcp = QuickMCP("t")
    if mine:
        serve(
            mcp.server,
            [Skill.from_files({"SKILL.md": skill(f"own-{i:03}")}) for i in range(mine)],
        )
    got = seen_by(mcp)
    async with over_http(mcp, mode=mode):
        pass
    assert got[0].skills == count


class _Broken(EntryOnlySkillProvider):
    """An own catalogue whose load fails: FastMCP lists none of its skills."""

    async def load_skills(self) -> dict[str, Skill]:
        raise RuntimeError("catalogue down")


@pytest.mark.parametrize("mode", MODES)
async def test_a_broken_own_catalogue_hides_no_plugin_skill(tmp_path, mode, caplog):
    mcp = capped(tmp_path, ("a", 5))
    serve(mcp.server, _Broken())
    got = seen_by(mcp)
    with caplog.at_level(logging.ERROR, logger="kubed.mcp"):
        async with over_http(mcp, mode=mode) as client:
            assert await listing(client) == uris("a", range(5))
            assert sorted(await rows(client)) == uris("a", range(5))
    assert got[0].skills == SkillCount(5, 0)
    assert not [r for r in caplog.records if "skill count" in r.getMessage()]


@pytest.mark.parametrize("mode", MODES)
async def test_the_count_is_not_echoed(tmp_path, mode):
    mcp = capped(tmp_path, ("a", 101))
    seen_by(mcp)
    async with over_http(mcp, mode=mode) as client:
        assert set(result(client).meta[META]) == {
            "resources",
            "listed",
            "via",
            "prefix",
            "max",
        }


async def _never(self):
    raise AssertionError("counted with no instructions fn")


async def _boom(self):
    raise RuntimeError("boom")


async def test_no_instructions_fn_counts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(Plugins, "skill_count", _never)
    async with over_http(capped(tmp_path, ("a", 101))) as client:
        assert result(client) is not None


async def test_a_failing_count_leaves_skills_none(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Plugins, "skill_count", _boom)
    mcp = capped(tmp_path, ("a", 101))
    got = seen_by(mcp)
    with caplog.at_level(logging.ERROR, logger="kubed.mcp.delivery"):
        async with over_http(mcp):
            pass
    assert got[0].skills is None
    assert [r.getMessage() for r in errors(caplog)] == [
        "skill count failed: the instructions get no count"
    ]


def _view_raises() -> View:
    raise RuntimeError("view down")


def _build_fails() -> View:
    def admits(plugin):
        raise RuntimeError("view down")

    return View("broken", admits)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("view", "key"), [(_view_raises, None), (_build_fails, "broken")]
)
async def test_a_failing_pin_leaves_the_count_out(
    tmp_path, caplog, mode, view, key
):  # Review Focus 4
    mcp = capped(tmp_path, ("a", 3))
    mcp.plugin_view(view)
    got = seen_by(mcp)
    with caplog.at_level(logging.ERROR, logger="kubed.mcp.plugins"):
        for _ in range(2):
            async with over_http(mcp, mode=mode) as client:
                assert result(client) is not None
                with pytest.raises(MCPError) as caught:  # as today (PLUG-53)
                    await request(client, "skills/list")
                if mode == "legacy":  # auto answers "Internal server error"
                    assert caught.value.error.message == "view down"
    assert [d.skills for d in got] == [None, None]
    skipped = [r for r in caplog.records if "skill count" in r.getMessage()]
    assert [r.msg for r in skipped] == [
        "skill count skipped for view %r: %s: %s; the instructions get no count"
    ] * len(skipped)
    logged = [r.getMessage() for r in skipped]
    assert logged == [
        f"skill count skipped for view {key!r}: RuntimeError: view down; "
        "the instructions get no count"
    ]


@pytest.mark.parametrize("mode", MODES)
async def test_local_counts_without_the_view(tmp_path, caplog, mode):
    mcp = capped(tmp_path, ("a", 3), own=2)
    mcp.plugin_view(_view_raises)
    got = seen_by(mcp)
    with caplog.at_level(logging.ERROR, logger="kubed.mcp"):
        async with over_http(mcp, mode=mode, headers={"X-MCP-Via": "local"}):
            pass
    assert got[0].skills == SkillCount(2, 0)
    assert not [r for r in caplog.records if "skill count" in r.getMessage()]
