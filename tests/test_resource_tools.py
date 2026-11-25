"""``list_resources`` and ``read_resource``, for clients that cannot read resources
(DLV-6), and the names they take from the server (DLV-7)."""

from __future__ import annotations

import base64
import json
import logging

import pytest
from fastmcp import Client
from gateway_server import kit, over_http
from mcp.shared.exceptions import MCPError
from plugin_server import DEMO, ENTRY, copy

from kubed.mcp import Context, QuickMCP, ResourceContent, ResourceResult
from kubed.mcp.plugins import PluginError

pytestmark = pytest.mark.unit

TOOLS = ("list_resources", "read_resource")
OFF = {"X-MCP-Resources": "off"}
NOT_TEXT = "Not returned: binary content is not text."
LOGO_URI = "skill://t/demo-guide/logo.png"

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(16))
WEBP = b"RIFF\x16\x00\x00\x00WEBPVP8 \x80\xff\x00\x9d"
SVG = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="1"/></svg>'
LOGO = PNG + b"logo"
DATA = {
    "files://x.png": PNG,
    "files://y.webp": WEBP,
    "files://n.txt": b"hello",
    "files://i.svg": SVG.encode(),
    "files://l1.txt": "café".encode("latin-1"),
    "files://x.bin": b"\x00\x01",
    LOGO_URI: LOGO,
}
# What each blob declares: y.webp declares nothing, so FastMCP gives it
# application/octet-stream, and the tool guesses from its extension.
DECLARED = {
    "files://x.png": "image/png",
    "files://n.txt": "text/plain",
    "files://i.svg": "image/svg+xml",
    "files://l1.txt": "text/plain",
    "files://x.bin": "application/octet-stream",
}
MEDIA = {"files://x.bin": "application/octet-stream", "files://l1.txt": "text/plain"}


def _bytes_of(uri: str):
    def read() -> bytes:
        return DATA[uri]

    return read


def make(tmp_path) -> QuickMCP:
    """A server ``t`` over the demo plugin, with a PNG in its skill, beside its own."""
    root = copy(tmp_path)
    (root / "skills" / "demo-guide" / "logo.png").write_bytes(LOGO)
    mcp = QuickMCP("t")
    mcp.add_plugin(root)

    @mcp.resource("ui://app/x")
    def app() -> str:
        return "<html></html>"

    @mcp.resource("state://own")
    def own() -> str:
        return "own"

    for uri, media in DECLARED.items():
        mcp.resource(uri, name=uri.removeprefix("files://"), mime_type=media)(
            _bytes_of(uri)
        )

    @mcp.resource("files://y.webp")
    def webp() -> ResourceResult:
        return ResourceResult([ResourceContent(WEBP)])

    @mcp.resource("state://{id}")
    def state(id: str) -> str:
        return id

    return mcp


async def call(client: Client, name: str, **args):
    return await client.call_tool(name, args, raise_on_error=False)


async def rows(client: Client) -> list[dict]:
    return [
        r.model_dump(by_alias=True, exclude_none=True, mode="json")
        for r in await client.list_resources()
    ]


def listed(result) -> list[dict]:
    assert not result.is_error, result.content
    assert len(result.content) == 1
    return json.loads(result.content[0].text)["resources"]


def only(result):
    assert not result.is_error, result.content
    assert len(result.content) == 1
    return result.content[0]


# ---- listing the tools -------------------------------------------------------------


@pytest.mark.parametrize(
    "asked",
    [{"headers": OFF}, {"params": {"resources": "off"}}],
    ids=["header", "query"],
)
async def test_off_lists_the_two_read_only_tools(tmp_path, asked):
    async with over_http(make(tmp_path), **asked) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    assert set(TOOLS) <= set(tools)
    assert tools["list_resources"].title == "List resources"
    assert tools["read_resource"].title == "Read a resource"
    for name in TOOLS:
        a = tools[name].annotations
        assert (
            a.read_only_hint,
            a.destructive_hint,
            a.idempotent_hint,
            a.open_world_hint,
        ) == (True, False, True, False)
    assert list(tools["read_resource"].input_schema["properties"]) == ["uri"]


async def test_on_hides_them_and_they_stay_callable(tmp_path):
    async with over_http(make(tmp_path)) as client:
        names = {tool.name for tool in await client.list_tools()}
        rows_ = listed(await call(client, "list_resources"))
        read = only(await call(client, "read_resource", uri="state://own"))
    assert not names & set(TOOLS)
    assert "state://own" in [r["uri"] for r in rows_]
    assert read.text == "own"


async def test_a_server_with_no_plugin_has_neither(tmp_path):
    mcp = QuickMCP("t")
    async with over_http(mcp, headers=OFF) as client:
        names = {tool.name for tool in await client.list_tools()}
    assert not names & set(TOOLS)


# ---- list_resources ----------------------------------------------------------------


async def test_list_resources_is_resources_list_less_ui(tmp_path):
    headers = {**OFF, "X-MCP-Listed": "skills,prompts"}
    async with over_http(make(tmp_path), headers=headers) as client:
        rows_ = listed(await call(client, "list_resources"))
        expected = await rows(client)
    assert any(r["uri"].startswith("ui://") for r in expected)
    assert rows_ == [r for r in expected if not r["uri"].startswith("ui://")]
    uris = [r["uri"] for r in rows_]
    assert ENTRY in uris and "state://own" in uris
    assert any(uri.startswith("prompts://") for uri in uris)
    assert not any("{" in uri for uri in uris)  # no template


# ---- read_resource -----------------------------------------------------------------


@pytest.mark.parametrize("uri", [ENTRY, "state://own"])
async def test_read_resource_reads_text_as_resources_read(tmp_path, uri):
    async with Client(make(tmp_path).server) as client:
        content = only(await call(client, "read_resource", uri=uri))
        read = await client.read_resource(uri)
    assert content.type == "text"
    assert content.text == read[0].text


@pytest.mark.parametrize(
    ("uri", "media"),
    [
        ("files://x.png", "image/png"),
        ("files://y.webp", "image/webp"),
        (LOGO_URI, "image/png"),
    ],
)
async def test_images_are_image_content(tmp_path, uri, media):
    async with Client(make(tmp_path).server) as client:
        content = only(await call(client, "read_resource", uri=uri))
    assert content.type == "image"
    assert content.mime_type == media
    assert base64.b64decode(content.data) == DATA[uri]


@pytest.mark.parametrize(
    ("uri", "text"), [("files://n.txt", "hello"), ("files://i.svg", SVG)]
)
async def test_text_blobs_are_text(tmp_path, uri, text):
    async with Client(make(tmp_path).server) as client:
        content = only(await call(client, "read_resource", uri=uri))
    assert content.type == "text"
    assert content.text == text


@pytest.mark.parametrize("uri", ["files://x.bin", "files://l1.txt"])
async def test_other_binary_is_a_description(tmp_path, uri):
    async with Client(make(tmp_path).server) as client:
        content = only(await call(client, "read_resource", uri=uri))
    assert content.type == "text"
    assert list(json.loads(content.text)) == [
        "uri",
        "mime_type",
        "bytes",
        "binary",
        "read",
    ]
    assert json.loads(content.text) == {
        "uri": uri,
        "mime_type": MEDIA[uri],
        "bytes": len(DATA[uri]),
        "binary": True,
        "read": NOT_TEXT,
    }


async def test_the_pointer_follows_the_sentence(tmp_path):
    mcp = make(tmp_path)

    def first(uri: str) -> str:
        return "replaced"

    def pointer(uri: str) -> str:
        return f"Its link is listed at files://index for {uri}."

    assert mcp.binary_pointer(first) is first
    assert mcp.binary_pointer(pointer) is pointer  # a second call replaces the first
    async with Client(mcp.server) as client:
        content = only(await call(client, "read_resource", uri="files://x.bin"))
    assert json.loads(content.text)["read"] == (
        "Not returned: binary content is not text. "
        "Its link is listed at files://index for files://x.bin."
    )


@pytest.mark.parametrize(
    ("pointer", "logged"),
    [
        (lambda uri: None, False),
        (lambda uri: "", False),
        (lambda uri: 1 / 0, True),
        (lambda uri: 3, True),
    ],
    ids=["none", "empty", "raises", "not-a-string"],
)
async def test_a_pointer_that_gives_nothing_leaves_the_sentence(
    tmp_path, pointer, logged, caplog
):
    mcp = make(tmp_path)
    mcp.binary_pointer(pointer)
    with caplog.at_level(logging.ERROR, logger="kubed.mcp.delivery"):
        async with Client(mcp.server) as client:
            content = only(await call(client, "read_resource", uri="files://x.bin"))
    assert json.loads(content.text)["read"] == NOT_TEXT
    errors = [
        r
        for r in caplog.records
        if r.name == "kubed.mcp.delivery" and r.levelno == logging.ERROR
    ]
    assert bool(errors) is logged


@pytest.mark.parametrize(
    "uri",
    [
        "skill://t/nope/SKILL.md",
        "skill://t/demo-guide/references",
        "skill://t/demo-guide%2FSKILL.md",
        "files://nowhere",
    ],
)
async def test_a_miss_is_a_tool_error_with_the_reads_message(tmp_path, uri):
    async with Client(make(tmp_path).server) as client:
        with pytest.raises(MCPError) as read:
            await client.read_resource(uri)
        result = await call(client, "read_resource", uri=uri)
    assert result.is_error
    assert result.content[0].text == read.value.message


async def test_a_models_miss_is_not_logged_as_an_error(tmp_path, caplog):
    async with Client(make(tmp_path).server) as client:
        with caplog.at_level(logging.DEBUG):
            for uri in ("skill://t/nope/SKILL.md", "ui://app/x"):
                assert (await call(client, "read_resource", uri=uri)).is_error
    assert not [
        r
        for r in caplog.records
        if r.name.startswith("fastmcp") and r.levelno >= logging.ERROR
    ]


async def test_json_text_keeps_non_ascii(tmp_path):
    mcp = make(tmp_path)

    @mcp.resource("files://menu", description="Caf\u00e9 menu")
    def cafe() -> str:
        return "x"

    async with Client(mcp.server) as client:
        text = (await call(client, "list_resources")).content[0].text
    assert "Caf\u00e9 menu" in text


async def test_a_failed_read_is_a_tool_error_with_the_reads_message(tmp_path):
    mcp = make(tmp_path)

    @mcp.resource("files://boom")
    def boom() -> str:
        raise RuntimeError("the disk is gone")

    async with Client(mcp.server) as client:
        with pytest.raises(MCPError) as read:
            await client.read_resource("files://boom")
        result = await call(client, "read_resource", uri="files://boom")
    assert "the disk is gone" in read.value.message  # FastMCP's ResourceError
    assert result.is_error
    assert result.content[0].text == read.value.message


async def test_a_folder_miss_names_its_index(tmp_path):
    folder = "skill://t/demo-guide/references"
    async with Client(make(tmp_path).server) as client:
        result = await call(client, "read_resource", uri=folder)
    assert result.is_error
    assert f"read {folder}/_index.md" in result.content[0].text


@pytest.mark.parametrize("uri", ["ui://app/x", "UI://app/x", "kb+ui://app/x"])
async def test_ui_is_refused(tmp_path, uri):
    async with Client(make(tmp_path).server) as client:
        result = await call(client, "read_resource", uri=uri)
    assert result.is_error
    assert result.content[0].text == f"{uri} is an app for a host to draw, not to read."


# ---- behind a gateway prefix -------------------------------------------------------


async def test_under_a_prefix_rows_are_prefixed_and_reads_take_it(tmp_path):
    plain_uri = "skill://lib/guide/SKILL.md"
    uri = "kb+skill://lib/guide/SKILL.md"
    async with over_http(kit(tmp_path), header="kb+", headers=OFF) as client:
        rows_ = listed(await call(client, "list_resources"))
        plain = await rows(client)
        read = only(await call(client, "read_resource", uri=uri))
        direct = await client.read_resource(uri)
    # resources/list itself stays unprefixed: a gateway writes the prefix into it.
    assert plain_uri in [r["uri"] for r in plain]
    assert uri not in [r["uri"] for r in plain]
    assert uri in [r["uri"] for r in rows_]
    row = next(r for r in rows_ if r["uri"] == uri)
    plain_row = next(r for r in plain if r["uri"] == plain_uri)
    # Only the uri is written: the description is the build's, prefixed already.
    assert row == {**plain_row, "uri": uri}
    assert "kb+skill://lib/guide/SKILL.md" in row["description"]
    assert read.text == direct[0].text
    assert "kb+skill://lib/guide/references/a.md" in read.text


# ---- one request, one set (#18) ----------------------------------------------------


async def test_an_inner_read_after_serve_answers_from_the_outer_set(tmp_path):
    mcp = QuickMCP("t")
    mcp.add_plugin(copy(tmp_path))

    @mcp.tool
    async def swap(ctx: Context) -> list[str]:
        before = await ctx.read_resource(ENTRY)
        mcp.serve(mcp.plugin_set())
        after = await ctx.read_resource(ENTRY)
        return [before.contents[0].content, after.contents[0].content]

    async with Client(mcp.server) as client:
        before, after = (await client.call_tool("swap", {})).data
        with pytest.raises(MCPError):
            await client.read_resource(ENTRY)
    assert before == after


# ---- the names (DLV-7) -------------------------------------------------------------


def reason(name: str) -> str:
    return (
        f"the server has its own tool {name}, but a server with a plugin serves {name} "
        "itself, over every resource, the plugins' and its own (DLV-6): remove the "
        f"server's {name} (DLV-7)"
    )


def refused(name: str) -> str:
    return (
        f"{name}: a server with a plugin serves {name} itself, over every resource, the "
        "plugins' and its own (DLV-6), so the server's own is refused (DLV-7)"
    )


def test_an_own_tool_refuses_the_first_plugin():
    mcp = QuickMCP("t")

    @mcp.tool
    def read_resource() -> str:
        return "own"

    with pytest.raises(PluginError) as caught:
        mcp.add_plugin(DEMO)
    assert (
        str(caught.value) == f"plugin {DEMO} is not served: {reason('read_resource')}"
    )
    assert mcp.plugins == () and mcp.generation == 0

    # The other door, a set served whole, and a tool added on mcp.server itself.
    other = QuickMCP("t")
    staging = other.plugin_set()
    staging.add(DEMO)

    def list_resources() -> str:
        return "own"

    other.server.tool(list_resources)
    with pytest.raises(PluginError) as caught:
        other.serve(staging)
    assert str(caught.value) == (
        f"the plugin set is not served: {reason('list_resources')}"
    )
    assert other.plugins == () and other.generation == 0


def test_a_server_with_no_plugin_takes_the_names():
    mcp = QuickMCP("t")

    def read_resource() -> str:
        return "own"

    assert mcp.tool(read_resource) is read_resource


@pytest.mark.parametrize("form", ["bare", "called", "positional", "keyword"])
async def test_mcp_tool_refuses_the_names_once_a_plugin_is_served(form):
    mcp = QuickMCP("t")
    mcp.add_plugin(DEMO)

    def read_resource() -> str:
        return "own"

    def renamed() -> str:
        return "own"

    register = {
        "bare": lambda: mcp.tool(read_resource),
        "called": lambda: mcp.tool()(read_resource),
        "positional": lambda: mcp.tool("read_resource")(renamed),
        "keyword": lambda: mcp.tool(renamed, name="list_resources"),
    }[form]
    name = "list_resources" if form == "keyword" else "read_resource"
    with pytest.raises(ValueError) as caught:
        register()
    assert str(caught.value) == refused(name)

    def other() -> str:
        return "other"

    def called() -> str:
        return "called"

    assert mcp.tool(other) is other
    assert mcp.tool()(called) is called
    assert mcp.tool(name="kept")(renamed) is renamed
    async with Client(mcp.server) as client:
        tools = await client.call_tool(
            "read_resource", {"uri": "state://none"}, raise_on_error=False
        )
        names = {tool.name for tool in await client.list_tools()}
    assert {"other", "called", "kept"} <= names
    # The plugin layer's read_resource answers, not a server's own.
    assert tools.is_error and "Resource not found" in tools.content[0].text
