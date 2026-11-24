"""The text a server serves behind a gateway: the prefix written into every plugin
address (GW-5 to GW-8, GW-10), inside the build its digests come from."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastmcp import Context
from gateway_server import kit, over_http
from skill_server import request

pytestmark = pytest.mark.unit

GUIDE = "skill://lib/guide/SKILL.md"
KB_GUIDE = "kb+skill://lib/guide/SKILL.md"


async def text_of(client, uri: str) -> str:
    return (await client.read_resource(uri))[0].text


async def test_a_prefixed_request_reads_prefixed_text(tmp_path):
    async with over_http(kit(tmp_path), header="kb+") as client:
        skill = await text_of(client, GUIDE)
        assert "kb+skill://lib/guide/references/a.md" in skill
        assert "kb+skill://lib/guide/references/../SKILL.md" in skill
        assert "myskill://x" in skill
        assert "kb+kb+" not in skill
        a = await text_of(client, "skill://lib/guide/references/a.md")
        assert "Back to kb+skill://lib/guide/SKILL.md." in a
        index = await text_of(client, "skill://lib/guide/references/_index.md")
        assert index.strip()
        for line in index.splitlines():
            assert line.startswith("kb+skill://lib/guide/references/"), line
        rows = json.loads(await text_of(client, "skill://lib/guide/_index.json"))
        assert rows
        for row in rows:
            assert row["uri"].startswith("kb+skill://lib/guide"), row
        prompt = await client.get_prompt(
            "lib_ask", {"topic": "skill://lib/guide/references/a.md"}
        )
        said = prompt.messages[0].content.text
        assert "Ask about kb+skill://lib/guide/references/a.md" in said
        assert "kb+prompts://lib/ask.md" in said
        agent = await request(client, "agents/get", name="lib_scout")
        assert "kb+agents://lib/scout.md" in json.dumps(agent["instructions"])
        assert agent["_meta"]["io.kubed/agent"]["frontmatter"]["description"] == (
            "Scouts kb+agents://lib/scout.md."
        )
        listed = (await request(client, "skills/list"))["skills"]
        assert listed[0]["frontmatter"]["description"] == (
            "Start at kb+skill://lib/guide/SKILL.md."
        )


async def test_structured_fields_stay_plain(tmp_path):  # GW-8
    async with over_http(kit(tmp_path), header="kb+") as client:
        resources = await client.list_resources()
        assert resources
        for resource in resources:
            assert "kb+" not in str(resource.uri) and "kb+" not in resource.name
        listed = await request(client, "skills/list")
        assert listed["skills"]
        for entry in listed["skills"]:
            assert "kb+" not in entry["uri"]
            for resource in entry["resources"]:
                assert "kb+" not in resource["uri"]
        children = await request(
            client, "resources/directory/read", uri="skill://lib/guide"
        )
        assert children["resources"]
        for child in children["resources"]:
            assert "kb+" not in child["uri"]
        agent = await request(client, "agents/get", name="lib_scout")
        assert agent["_meta"]["io.kubed/agent"]["skills"] == {"guide": GUIDE}


async def test_digests_agree_under_a_prefix(tmp_path):
    async with over_http(kit(tmp_path), header="kb+") as client:
        entry = (await request(client, "skills/list"))["skills"][0]
        manifest = json.loads(await text_of(client, "skill://lib/guide/_manifest"))
        by_path = {f["path"]: f for f in manifest["files"]}
        assert by_path
        for resource in entry["resources"]:
            data = (await text_of(client, resource["uri"])).encode()
            assert resource["size"] == len(data)
            assert resource["digest"] == f"sha256:{hashlib.sha256(data).hexdigest()}"
        for path, file in by_path.items():
            data = (await text_of(client, f"skill://lib/guide/{path}")).encode()
            assert file["size"] == len(data)
            assert file["hash"] == f"sha256:{hashlib.sha256(data).hexdigest()}"


@pytest.mark.parametrize("header", [None, "kb"])
async def test_with_no_prefix_the_build_is_the_objects_read(tmp_path, header):
    mcp = kit(tmp_path)

    @mcp.tool
    async def key() -> str:
        pin = await mcp._plugins.pinned()
        same = pin.build.catalogue["lib/guide"] is (
            mcp._plugins.current[0].catalogue["lib/guide"]
        )
        return json.dumps([list(pin.key), same])

    async with over_http(mcp, header=header) as client:
        result = await client.call_tool("key", {})
    assert json.loads(result.data) == [[mcp.generation, None, None], True]


async def test_prefixed_and_plain_builds_stay_apart(tmp_path):  # GW-6
    mcp = kit(tmp_path)
    plain_reads = []
    for header in (None, "kb+", None):
        async with over_http(mcp, header=header) as client:
            text = await text_of(client, GUIDE)
        if header is None:
            plain_reads.append(text)
        else:
            assert "kb+skill://lib/guide/references" in text
    assert plain_reads[0] == plain_reads[1]
    assert "kb+skill://lib/guide/references" not in plain_reads[0]
    assert set(mcp._plugins._memo.builds) == {
        (mcp.generation, None, None),
        (mcp.generation, None, "kb+"),
    }


async def test_the_writer_runs_after_the_servers_rewriters(tmp_path):
    mcp = kit(tmp_path)
    shown = []

    def append(site, text):
        shown.append(text)
        if site.path.endswith("SKILL.md"):
            return text + "Also skill://lib/guide/references/a.md.\n"
        return text

    mcp.add_rewriter(append)
    async with over_http(mcp, header="kb+") as client:
        text = await text_of(client, GUIDE)
    assert text.endswith("Also kb+skill://lib/guide/references/a.md.\n")
    assert shown
    assert not any("kb+skill://lib/guide/references" in t for t in shown)


async def test_a_prefixed_argument_is_not_prefixed_again(tmp_path):  # Review Focus 4
    async with over_http(kit(tmp_path), header="kb+") as client:
        prompt = await client.get_prompt("lib_ask", {"topic": KB_GUIDE})
    said = prompt.messages[0].content.text
    assert "Ask about " + KB_GUIDE in said
    assert "kb+kb+" not in said


async def test_a_tool_reading_inside_a_prefixed_request_gets_the_prefixed_text(
    tmp_path,
):
    mcp = kit(tmp_path)

    @mcp.tool
    async def read(ctx: Context) -> str:
        return (await ctx.read_resource(GUIDE)).contents[0].content

    async with over_http(mcp, header="kb+") as client:
        result = await client.call_tool("read", {})
    assert "kb+skill://lib/guide/references/a.md" in result.data


async def test_the_servers_own_material_is_not_rewritten(tmp_path):  # GW-10
    mcp = kit(tmp_path)
    said = "see skill://lib/guide/SKILL.md"

    @mcp.prompt
    def own() -> str:
        return said

    @mcp.resource("own://note")
    def note() -> str:
        return said

    @mcp.tool
    def tool() -> str:
        return said

    async with over_http(mcp, header="kb+") as client:
        prompt = await client.get_prompt("own", {})
        assert prompt.messages[0].content.text == said
        assert await text_of(client, "own://note") == said
        assert (await client.call_tool("tool", {})).data == said
