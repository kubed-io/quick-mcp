"""The addresses a client sends behind a gateway: the prefix taken off before any
provider is asked, and a miss naming the address as sent (GW-4, SKL-14, SKL-16, SKL-18).

Reads go out raw, by ``request``: FastMCP's ``Client.read_resource`` normalises a URI
before sending it, which would hide the dot segments the server is meant to resolve.
"""

from __future__ import annotations

import json
import re

import pytest
from fastmcp import Context
from fastmcp.exceptions import NotFoundError
from gateway_server import kit, over_http
from skill_server import DIRECTORY, request

from kubed.mcp.gateway import _as_sent, _plain, _TakeThePrefix
from kubed.mcp.protocol import MCPError, Middleware
from kubed.mcp.skills import FolderNotFound

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("uri", "prefix", "dots", "expected"),
    [
        ("kb+skill://lib/guide/SKILL.md", "kb+", True, "skill://lib/guide/SKILL.md"),
        ("kb+prompts://lib/ask.md", "kb+", True, "prompts://lib/ask.md"),
        ("kb+agents://lib/scout.md", "kb+", False, "agents://lib/scout.md"),
        ("xx+skill://lib/guide/SKILL.md", "kb+", True, "xx+skill://lib/guide/SKILL.md"),
        ("kb+skill://lib/guide/SKILL.md", None, True, "kb+skill://lib/guide/SKILL.md"),
        (
            "kb+skill://lib/guide/references/../SKILL.md",
            "kb+",
            True,
            "skill://lib/guide/SKILL.md",
        ),
        ("skill://lib/guide/references/..", None, True, "skill://lib/guide"),
        (
            "skill://lib/guide/references/../SKILL.md",
            None,
            False,
            "skill://lib/guide/references/../SKILL.md",
        ),
        ("skill://lib/../..", None, True, "skill://%2F"),
        ("skill://lib/guide/references%2Fa.md", None, True, "skill://%2F"),
        ("kb+skill://lib/guide%2fSKILL.md", "kb+", False, "skill://%2F"),
        (
            "kb+skill://lib/guide/./SKILL.md#top",
            "kb+",
            True,
            "skill://lib/guide/SKILL.md#top",
        ),
        (
            "kb+skill://lib/guide/SKILL.md?v=../x",
            "kb+",
            True,
            "skill://lib/guide/SKILL.md?v=../x",
        ),
        ("workspace://a%2Fb/../c", "kb+", True, "workspace://a%2Fb/../c"),
        # Not in the brief: a trailing slash names no folder, so it stays a plain miss
        # (test_skill_folders.py::test_a_trailing_slash_is_a_plain_miss).
        ("skill://lib/guide/references/", None, True, "skill://lib/guide/references/"),
        ("kb+skill://lib/guide/./", "kb+", True, "skill://lib/guide/"),
        ("skill://lib//guide/SKILL.md", None, True, "skill://lib/guide/SKILL.md"),
    ],
)
def test_the_plain_address(uri, prefix, dots, expected):  # Review Focus 3: the #/? rows
    assert _plain(uri, prefix, dots=dots) == expected


READS_BACK = [  # (method, sent under "kb+", the plain address read for comparison)
    ("resources/read", "kb+skill://lib/guide/SKILL.md", "skill://lib/guide/SKILL.md"),
    (
        "resources/read",
        "kb+skill://lib/guide/references/a.md",
        "skill://lib/guide/references/a.md",
    ),
    (
        "resources/read",
        "kb+skill://lib/guide/references/../SKILL.md",
        "skill://lib/guide/SKILL.md",
    ),
    ("resources/read", "kb+skill://lib/guide/_index.md", "skill://lib/guide/_index.md"),
    ("resources/read", "kb+skill://lib/guide/_manifest", "skill://lib/guide/_manifest"),
    ("skills/get", "kb+skill://lib/guide/SKILL.md", "skill://lib/guide/SKILL.md"),
    ("resources/directory/read", "kb+skill://lib/guide", "skill://lib/guide"),
]


@pytest.mark.parametrize(("method", "sent", "plain"), READS_BACK)
async def test_every_link_reads_back_under_the_prefix(tmp_path, method, sent, plain):
    async with over_http(kit(tmp_path), header="kb+") as client:
        got = await request(client, method, uri=sent)
        want = await request(client, method, uri=plain)
    if method == "resources/read":
        assert got["contents"]
        assert [c["uri"] for c in got["contents"]] == [sent] * len(got["contents"])
        assert [c["uri"] for c in want["contents"]] == [plain] * len(want["contents"])
        got = {**got, "contents": [{**c, "uri": plain} for c in got["contents"]]}
    assert got == want


class _Seen(Middleware):
    """Records the message each read and extension method arrives with."""

    def __init__(self) -> None:
        self.messages: list = []

    async def on_read_resource(self, context, call_next):
        self.messages.append(context.message)
        return await call_next(context)

    async def on_request(self, context, call_next):
        if context.method in ("skills/get", "resources/directory/read"):
            self.messages.append(context.message)
        return await call_next(context)


SEEN = [  # (header, method, sent, what the providers are asked)
    (
        "kb+",
        "skills/get",
        "kb+skill://lib/guide/SKILL.md",
        "skill://lib/guide/SKILL.md",
    ),
    ("kb+", "resources/directory/read", "kb+skill://lib/guide", "skill://lib/guide"),
    (
        "kb+",
        "resources/read",
        "kb+skill://lib/guide/SKILL.md",
        "skill://lib/guide/SKILL.md",
    ),
    (None, "skills/get", "skill://lib/guide/SKILL.md", "skill://lib/guide/SKILL.md"),
    (None, "resources/directory/read", "skill://lib/guide", "skill://lib/guide"),
    (
        None,
        "resources/read",
        "skill://lib/guide/SKILL.md",
        "skill://lib/guide/SKILL.md",
    ),
]


@pytest.mark.parametrize(("header", "method", "sent", "asked"), SEEN)
async def test_the_providers_are_asked_the_plain_address(
    tmp_path, header, method, sent, asked
):
    """FastMCP hands an extension method's raw params to ``on_request`` as a dict and
    a read's as a model; either way, inside the strip only the plain address is seen,
    and with nothing to take off the message is the very one that arrived."""
    mcp = kit(tmp_path)
    outside, inside = _Seen(), _Seen()
    taker = next(
        i for i, m in enumerate(mcp.server.middleware) if isinstance(m, _TakeThePrefix)
    )
    mcp.server.middleware.insert(taker, outside)
    mcp.server.add_middleware(inside)
    async with over_http(mcp, header=header) as client:
        await request(client, method, uri=sent)
    (before,), (after,) = outside.messages, inside.messages
    if method == "resources/read":
        assert (str(before.uri), str(after.uri)) == (sent, asked)
    else:
        assert type(before) is dict and type(after) is dict
        assert (before["uri"], after["uri"]) == (sent, asked)
    assert (after is before) == (sent == asked)


# An address in served text: up to whitespace or a closing bracket or quote, less the
# punctuation that ends a sentence.
_KB_ADDRESS = re.compile(r"kb\+skill://[^\s\"'<>()\[\]`]+")


def _addresses(text: str) -> set[str]:
    return {found.rstrip(".,;:") for found in _KB_ADDRESS.findall(text)}


async def test_every_address_in_the_served_text_reads(tmp_path):
    async with over_http(kit(tmp_path), header="kb+") as client:

        async def text_of(uri: str) -> str:
            result = await request(client, "resources/read", uri=uri)
            return result["contents"][0]["text"]

        index_json = await text_of("skill://lib/guide/_index.json")
        folders = {
            row["uri"] for row in json.loads(index_json) if row["mimeType"] == DIRECTORY
        }
        found: set[str] = set()
        for uri in (
            "skill://lib/guide/SKILL.md",
            "skill://lib/guide/references/a.md",
            "skill://lib/guide/_index.md",
        ):
            found |= _addresses(await text_of(uri))
        found |= _addresses(index_json)
        assert "kb+skill://lib/guide/references/a.md" in found
        assert "kb+skill://lib/guide/references/../SKILL.md" in found
        assert folders and folders <= found
        for address in sorted(found):
            # A folder serves no content (SKL-18): it is read by its listing.
            if address in folders:
                listed = await request(client, "resources/directory/read", uri=address)
                assert listed["resources"], address
            else:
                read = await request(client, "resources/read", uri=address)
                assert read["contents"], address


MISSES = [  # (method, header, sent)
    ("resources/read", "kb+", "xx+skill://lib/guide/SKILL.md"),
    ("resources/read", None, "kb+skill://lib/guide/SKILL.md"),
    ("skills/get", "kb+", "xx+skill://lib/guide/SKILL.md"),
    ("skills/get", None, "kb+skill://lib/guide/SKILL.md"),
    ("resources/directory/read", None, "kb+skill://lib/guide"),
    ("resources/read", None, "skill://lib/../.."),
    *[
        (m, h, u)
        for h in (None, "kb+")
        for (m, u) in [
            ("resources/read", f"{h or ''}skill://lib/guide/references%2Fa.md"),
            ("resources/read", f"{h or ''}skill://lib/guide/references%2fa.md"),
            ("skills/get", f"{h or ''}skill://lib%2Fguide/SKILL.md"),
            ("resources/directory/read", f"{h or ''}skill://lib/guide%2Freferences"),
        ]
    ],
]


def _named(error, missing: str, sent: str) -> tuple:
    """An error's code, message and data, with ``missing`` named ``sent`` instead."""
    data = error.data
    if isinstance(data, dict):
        data = {k: sent if v == missing else v for k, v in data.items()}
    return error.code, error.message.replace(missing, sent), data


@pytest.mark.parametrize(("method", "header", "sent"), MISSES)
async def test_what_resolves_to_nothing_fails_as_a_missing_one_does(
    tmp_path, method, header, sent
):
    missing = (
        "skill://lib/nope"
        if method == "resources/directory/read"
        else "skill://lib/nope/SKILL.md"
    )
    async with over_http(kit(tmp_path), header=header) as client:
        with pytest.raises(MCPError) as caught:
            await request(client, method, uri=sent)
        with pytest.raises(MCPError) as nope:
            await request(client, method, uri=missing)
    assert missing in nope.value.error.message or method == "resources/directory/read"
    got = caught.value.error
    assert (got.code, got.message, got.data) == _named(nope.value.error, missing, sent)


async def test_the_folder_hint_names_the_uri_as_sent(tmp_path):
    sent = "kb+skill://lib/guide/references"
    async with over_http(kit(tmp_path), header="kb+") as client:
        with pytest.raises(MCPError) as caught:
            await request(client, "resources/read", uri=sent)
    error = caught.value.error
    assert error.message == (
        f"Resource not found: {sent!r}. {sent} is a folder: read {sent}/_index.md to list it."
    )
    assert error.data == {"uri": sent}


async def test_a_tool_reading_inside_a_prefixed_request_is_told_the_address_it_sent(
    tmp_path,
):
    """A server-side read misses with FastMCP's ``NotFoundError`` and a folder with
    ``FolderNotFound``; each names the address the tool passed, not the stripped one."""
    mcp = kit(tmp_path)

    @mcp.tool
    async def read(uri: str, ctx: Context) -> str:
        try:
            await ctx.read_resource(uri)
        except FolderNotFound as folder:
            return f"folder: {folder.miss}"
        except NotFoundError as missing:
            return f"missing: {missing}"
        return "read"

    missing = "kb+skill://lib/nope/SKILL.md"
    folder = "kb+skill://lib/guide/references"
    async with over_http(mcp, header="kb+") as client:
        miss = (await client.call_tool("read", {"uri": missing})).data
        hint = (await client.call_tool("read", {"uri": folder})).data
    assert miss == f"missing: Unknown resource: {missing!r}"
    assert hint == (
        f"folder: no resource at {folder!r}. "
        f"{folder} is a folder: read {folder}/_index.md to list it."
    )


def test_any_other_error_names_the_address_as_sent():
    passed, sent = "skill://lib/nope/SKILL.md", "kb+skill://lib/nope/SKILL.md"
    renamed = _as_sent(
        MCPError(code=-32602, message=f"x {passed}", data={"uri": passed, "x": 1}),
        passed,
        sent,
    )
    assert isinstance(renamed, MCPError)
    assert (renamed.code, renamed.message, renamed.data) == (
        -32602,
        f"x {sent}",
        {"uri": sent, "x": 1},
    )
    assert _as_sent(ValueError(passed), passed, sent) is None
