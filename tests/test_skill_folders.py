"""A skill's folders as SEP-2640 directory resources, and the index files beside them.

A folder is read two ways: ``resources/directory/read``, which a proxy that does not know
the method refuses, and ``_index.json`` / ``_index.md``, which plain ``resources/read``
reaches anywhere. Every test goes through a real client, and the two ways are checked
against each other rather than against the provider's own helpers.
"""

from __future__ import annotations

import logging

import pytest
from fastmcp import Client
from mcp_types import INVALID_PARAMS
from skill_server import (
    DEMO,
    DIRECTORY,
    ENTRY_URI,
    REFERENCES,
    ROOT,
    demo,
    make_skill,
    read_directory,
    read_json,
    request,
    served,
)

import kubed.mcp.skills
from kubed.mcp import Context, QuickMCP
from kubed.mcp.protocol import MCPError
from kubed.mcp.skills import (
    EntryOnlySkillProvider,
    FolderNotFound,
    Skill,
    SkillsExtension,
    load_skill,
    serve,
)

pytestmark = pytest.mark.unit

REFERENCE_URIS = [f"{REFERENCES}/GUIDE.md", f"{REFERENCES}/NOTES.md"]


# ---- the index files ------------------------------------------------------------------


async def test_index_json_reads_beside_each_folder():
    async with Client(demo().server) as client:
        root = await read_json(client, f"{ROOT}/_index.json")
        references = await read_json(client, f"{REFERENCES}/_index.json")
    assert [r["uri"] for r in root] == [ENTRY_URI, REFERENCES, f"{ROOT}/scripts"]
    assert [r["uri"] for r in references] == REFERENCE_URIS


@pytest.mark.parametrize("folder", [ROOT, REFERENCES])
async def test_index_md_is_one_line_per_child(folder):
    async with Client(demo().server) as client:
        children = await read_json(client, f"{folder}/_index.json")
        contents = await client.read_resource(f"{folder}/_index.md")
    assert contents[0].mime_type == "text/markdown"
    assert contents[0].text.splitlines() == [
        f"{c['uri']}: {c['description']}" for c in children
    ]


async def test_a_folder_child_is_a_directory_resource():
    async with Client(demo().server) as client:
        root = await read_json(client, f"{ROOT}/_index.json")
    folder = next(r for r in root if r["uri"] == REFERENCES)
    assert folder["mimeType"] == DIRECTORY
    assert folder["name"] == "demo-skill/references"
    assert folder["description"] == "A folder of the demo-skill skill"


async def test_no_folder_uri_ends_in_a_slash():
    async with Client(demo().server) as client:
        root = await read_json(client, f"{ROOT}/_index.json")
    folders = [r["uri"] for r in root if r["mimeType"] == DIRECTORY]
    assert folders
    assert not [uri for uri in folders if uri.endswith("/")]


async def test_a_markdown_child_is_described_by_its_heading():
    async with Client(demo().server) as client:
        references = await read_json(client, f"{REFERENCES}/_index.json")
        read = await client.list_resources()
    guide = next(r for r in references if r["uri"].endswith("/GUIDE.md"))
    assert guide["description"] == "Guide"
    assert guide["mimeType"] == "text/markdown"
    assert read


async def test_skill_md_is_described_as_the_listing_describes_it():
    async with Client(demo().server) as client:
        listed = next(
            r for r in await client.list_resources() if str(r.uri) == ENTRY_URI
        )
        root = await read_json(client, f"{ROOT}/_index.json")
    assert root[0]["description"] == listed.description


async def test_folders_and_indexes_are_unlisted():
    async with Client(demo().server) as client:
        rows = await client.list_resources()
        templates = await client.list_resource_templates()
    uris = [str(r.uri) for r in rows]
    assert not [r for r in rows if r.mime_type == DIRECTORY]
    assert not [u for u in uris if "/_index." in u or u in (ROOT, REFERENCES)]
    assert not [t for t in templates if t.uri_template.startswith("skill://")]


async def test_the_manifest_is_never_a_child_and_still_reads():
    async with Client(demo().server) as client:
        root = await read_json(client, f"{ROOT}/_index.json")
        manifest = await read_json(client, f"{ROOT}/_manifest")
    assert not [r for r in root if "_manifest" in r["uri"] or "_index" in r["uri"]]
    assert "references/GUIDE.md" in [f["path"] for f in manifest["files"]]


async def test_an_index_of_no_folder_is_not_found():
    async with Client(demo().server) as client:
        with pytest.raises(MCPError) as caught:
            await client.read_resource(f"{ROOT}/nope/_index.json")
    assert caught.value.error.code == INVALID_PARAMS
    assert caught.value.error.message.startswith("Resource not found")


@pytest.mark.parametrize("name", ["_index.md", "_index.json"])
async def test_a_file_with_an_index_name_never_breaks_its_folder(
    tmp_path, caplog, name
):
    """The index wins: the file is no child, reads as the index, and is named at load."""
    with caplog.at_level(logging.WARNING):
        mcp = served(tmp_path, {f"sub/{name}": "a file", "sub/a.txt": "a"})
    assert f"sub/{name}" in caplog.text
    async with Client(mcp.server) as client:
        listed = await read_directory(client, uri="skill://tmp-skill/sub")
        index = await read_json(client, "skill://tmp-skill/sub/_index.json")
        markdown = (await client.read_resource("skill://tmp-skill/sub/_index.md"))[
            0
        ].text
        entry = (await request(client, "skills/list"))["skills"][0]
    assert [r["uri"] for r in listed["resources"]] == ["skill://tmp-skill/sub/a.txt"]
    assert listed["resources"] == index
    assert markdown == "skill://tmp-skill/sub/a.txt: File from tmp-skill skill\n"
    assert f"skill://tmp-skill/sub/{name}" not in [r["uri"] for r in entry["resources"]]


@pytest.mark.parametrize("name", ["_index.md", "_index.json"])
async def test_a_file_with_an_index_name_is_not_in_the_manifest(tmp_path, name):
    """read_skill verifies every file the manifest names, and this URI reads as the
    index, so the manifest must not name it."""
    mcp = served(tmp_path, {f"sub/{name}": "a file", "sub/a.txt": "a"})
    async with Client(mcp.server) as client:
        manifest = await read_json(client, "skill://tmp-skill/_manifest")
        read = await client.read_skill("skill://tmp-skill/SKILL.md")
    paths = [f["path"] for f in manifest["files"]]
    assert f"sub/{name}" not in paths
    assert "sub/a.txt" in paths
    assert set(read.files) == set(paths)


async def test_a_provider_built_from_a_skill_leaves_out_index_named_files(caplog):
    """A skill built in code, not read from a folder, is still offered."""
    skill = Skill.from_files(
        {
            "SKILL.md": "---\nname: tmp-skill\ndescription: A skill for tests.\n---\n",
            "references/A.md": "# A\n",
            "references/_index.md": "a file",
        }
    )
    mcp = QuickMCP("t")
    with caplog.at_level(logging.WARNING):
        serve(mcp.server, [skill])
    assert "references/_index.md" in caplog.text
    async with Client(mcp.server) as client:
        skills = (await request(client, "skills/list"))["skills"]
    assert [entry["uri"] for entry in skills] == ["skill://tmp-skill/SKILL.md"]
    assert sorted(r["uri"] for r in skills[0]["resources"]) == [
        "skill://tmp-skill/SKILL.md",
        "skill://tmp-skill/references/A.md",
    ]


async def test_a_root_manifest_file_is_not_served(tmp_path, caplog):
    """FastMCP answers that URI with the manifest; below the root it is a file."""
    with caplog.at_level(logging.WARNING):
        mcp = served(tmp_path, {"_manifest": "a file", "references/_manifest": "b"})
    assert "_manifest is not served" in caplog.text
    async with Client(mcp.server) as client:
        skills = (await request(client, "skills/list"))["skills"]
        manifest = await read_json(client, "skill://tmp-skill/_manifest")
        root = await read_directory(client, uri="skill://tmp-skill")
        references = await read_directory(client, uri="skill://tmp-skill/references")
    assert [entry["uri"] for entry in skills] == ["skill://tmp-skill/SKILL.md"]
    named = [r["uri"] for r in skills[0]["resources"]]
    assert "skill://tmp-skill/_manifest" not in named
    assert "skill://tmp-skill/references/_manifest" in named
    assert manifest["skill"] == "tmp-skill"
    assert [f["path"] for f in manifest["files"]] == [
        "SKILL.md",
        "references/_manifest",
    ]
    assert [r["uri"] for r in root["resources"]] == [
        "skill://tmp-skill/SKILL.md",
        "skill://tmp-skill/references",
    ]
    assert [r["uri"] for r in references["resources"]] == [
        "skill://tmp-skill/references/_manifest"
    ]


async def test_a_root_manifest_folder_is_no_folder(tmp_path, caplog):
    """Its URI reads as the manifest; its files still read and stay in the skill. Below
    the root, ``_manifest/`` is an ordinary folder."""
    with caplog.at_level(logging.WARNING):
        mcp = served(
            tmp_path, {"_manifest/x.md": "# X\n", "references/_manifest/y.md": "# Y\n"}
        )
    assert "folder _manifest is not served" in caplog.text
    async with Client(mcp.server) as client:
        root = await read_json(client, "skill://tmp-skill/_index.json")
        manifest = await read_json(client, "skill://tmp-skill/_manifest")
        inner = (await client.read_resource("skill://tmp-skill/_manifest/x.md"))[0]
        with pytest.raises(MCPError) as caught:
            await read_directory(client, uri="skill://tmp-skill/_manifest")
        nested = await read_directory(
            client, uri="skill://tmp-skill/references/_manifest"
        )
        skills = (await request(client, "skills/list"))["skills"]
    assert [r["uri"] for r in root] == [
        "skill://tmp-skill/SKILL.md",
        "skill://tmp-skill/references",
    ]
    assert manifest["skill"] == "tmp-skill"
    assert "_manifest/x.md" in [f["path"] for f in manifest["files"]]
    assert inner.text == "# X\n"
    assert caught.value.error.code == INVALID_PARAMS
    assert [r["uri"] for r in nested["resources"]] == [
        "skill://tmp-skill/references/_manifest/y.md"
    ]
    assert "skill://tmp-skill/_manifest/x.md" in [
        r["uri"] for r in skills[0]["resources"]
    ]


@pytest.mark.parametrize("name", ["_index.md", "_index.json"])
async def test_a_folder_with_an_index_name_never_breaks_its_parent(
    tmp_path, caplog, name
):
    with caplog.at_level(logging.WARNING):
        mcp = served(
            tmp_path,
            {
                f"sub/{name}/inner/x.txt": "x",
                f"sub/{name}/y.txt": "y",
                "sub/a.txt": "a",
            },
        )
    assert f"sub/{name}" in caplog.text
    async with Client(mcp.server) as client:
        listed = await read_directory(client, uri="skill://tmp-skill/sub")
        index = (await client.read_resource(f"skill://tmp-skill/sub/{name}"))[0]
        with pytest.raises(MCPError) as caught:
            await read_directory(client, uri=f"skill://tmp-skill/sub/{name}")
        inner = (await client.read_resource(f"skill://tmp-skill/sub/{name}/y.txt"))[0]
        manifest = await read_json(client, "skill://tmp-skill/_manifest")
        entry = (await request(client, "skills/list"))["skills"][0]
    assert [r["uri"] for r in listed["resources"]] == ["skill://tmp-skill/sub/a.txt"]
    assert f"sub/{name}/y.txt" in [f["path"] for f in manifest["files"]]
    assert f"skill://tmp-skill/sub/{name}/y.txt" in [
        r["uri"] for r in entry["resources"]
    ]
    assert index.mime_type in ("application/json", "text/markdown")
    assert "skill://tmp-skill/sub/a.txt" in index.text
    assert caught.value.error.code == INVALID_PARAMS
    assert inner.text == "y"


async def test_a_multiline_description_is_one_index_line(tmp_path):
    front = "name: tmp-skill\ndescription: |\n  Line one.\n  Line two."
    async with Client(served(tmp_path, {}, front=front).server) as client:
        children = await read_json(client, "skill://tmp-skill/_index.json")
        markdown = (await client.read_resource("skill://tmp-skill/_index.md"))[0].text
    assert "\n" in children[0]["description"]
    assert markdown == "skill://tmp-skill/SKILL.md: Line one. Line two.\n"


async def test_a_file_child_reads_with_its_own_type(tmp_path):
    async with Client(served(tmp_path, {"references/A.md": "# A\n"}).server) as client:
        listed = await read_directory(client, uri="skill://tmp-skill/references")
        read = (await client.read_resource("skill://tmp-skill/references/A.md"))[0]
    assert [r["uri"] for r in listed["resources"]] == [
        "skill://tmp-skill/references/A.md"
    ]
    assert read.mime_type == "text/markdown"


async def test_a_file_that_is_not_markdown_is_a_file_from_the_skill():
    async with Client(demo().server) as client:
        scripts = await read_json(client, f"{ROOT}/scripts/_index.json")
        markdown = (await client.read_resource(f"{ROOT}/scripts/_index.md"))[0]
    assert scripts[0]["description"] == "File from demo-skill skill"
    assert markdown.text == f"{ROOT}/scripts/run.sh: File from demo-skill skill\n"


async def test_a_markdown_file_with_no_heading_is_a_file_from_the_skill():
    async with Client(demo().server) as client:
        references = await read_json(client, f"{REFERENCES}/_index.json")
    notes = next(r for r in references if r["uri"].endswith("/NOTES.md"))
    assert notes["description"] == "File from demo-skill skill"


async def test_a_heading_inside_a_code_fence_is_not_the_title(tmp_path):
    body = "```bash\n# install it\n```\n\n~~~\n# nor this\n~~~\n\n# Real Title\n"
    async with Client(served(tmp_path, {"references/A.md": body}).server) as client:
        references = await read_json(client, "skill://tmp-skill/references/_index.json")
    assert references[0]["description"] == "Real Title"


async def test_a_markdown_file_that_is_not_utf8_never_loses_the_skill(tmp_path):
    mcp = served(tmp_path, {"references/BAD.md": b"# Caf\xe9\n"})
    async with Client(mcp.server) as client:
        assert (
            SkillsExtension.identifier in client.session.server_capabilities.extensions
        )
        references = await read_json(client, "skill://tmp-skill/references/_index.json")
        entry = (await request(client, "skills/list"))["skills"][0]
        read = await client.read_skill("skill://tmp-skill/SKILL.md")
    assert references[0]["description"] == "File from tmp-skill skill"
    assert "skill://tmp-skill/references/BAD.md" in [
        r["uri"] for r in entry["resources"]
    ]
    assert read.files["references/BAD.md"] == b"# Caf\xe9\n"


async def test_an_empty_directory_is_no_folder(tmp_path):
    """Folders come from the skill's files, so a directory with none is none."""
    mcp = served(tmp_path, {"a/b/c.txt": "c"}, folders=("assets", "a/empty"))
    async with Client(mcp.server) as client:
        root = await read_json(client, "skill://tmp-skill/_index.json")
        a = await read_json(client, "skill://tmp-skill/a/_index.json")
        misses = []
        for uri in ("skill://tmp-skill/assets", "skill://tmp-skill/a/empty"):
            with pytest.raises(MCPError) as index:
                await client.read_resource(f"{uri}/_index.json")
            with pytest.raises(MCPError) as method:
                await read_directory(client, uri=uri)
            with pytest.raises(MCPError) as read:
                await client.read_resource(uri)
            misses.append((index.value.error, method.value.error, read.value.error))
    assert [r["uri"] for r in root] == [
        "skill://tmp-skill/SKILL.md",
        "skill://tmp-skill/a",
    ]
    assert [r["uri"] for r in a] == ["skill://tmp-skill/a/b"]
    for index, method, read in misses:
        assert index.message.startswith("Resource not found")
        assert (method.code, method.message) == (INVALID_PARAMS, "Directory not found")
        assert "_index" not in read.message


async def test_a_nested_folder_is_listed_once_by_its_parent(tmp_path):
    mcp = served(tmp_path, {"a/b/c.txt": "c", "a/d.md": "# Dee\n"})
    async with Client(mcp.server) as client:
        root = await read_json(client, "skill://tmp-skill/_index.json")
        a = await read_json(client, "skill://tmp-skill/a/_index.json")
        b = await read_json(client, "skill://tmp-skill/a/b/_index.json")
    assert [r["uri"] for r in root] == [
        "skill://tmp-skill/SKILL.md",
        "skill://tmp-skill/a",
    ]
    assert [(r["uri"], r["mimeType"]) for r in a] == [
        ("skill://tmp-skill/a/b", DIRECTORY),
        ("skill://tmp-skill/a/d.md", "text/markdown"),
    ]
    assert a[1]["description"] == "Dee"
    assert [r["uri"] for r in b] == ["skill://tmp-skill/a/b/c.txt"]


# ---- resources/directory/read ---------------------------------------------------------


async def test_the_root_lists_its_direct_children():
    async with Client(demo().server) as client:
        result = await read_directory(client, uri=ROOT)
    assert [r["uri"] for r in result["resources"]] == [
        ENTRY_URI,
        REFERENCES,
        f"{ROOT}/scripts",
    ]
    assert result["resources"][1]["mimeType"] == DIRECTORY
    assert "nextCursor" not in result


async def test_descending_lists_the_folder_alone():
    async with Client(demo().server) as client:
        result = await read_directory(client, uri=REFERENCES)
    assert [r["uri"] for r in result["resources"]] == REFERENCE_URIS


@pytest.mark.parametrize("mode", ["auto", "legacy"])
@pytest.mark.parametrize("folder", [ROOT, REFERENCES])
async def test_the_method_returns_what_the_index_holds(folder, mode):
    async with Client(demo().server, mode=mode) as client:
        result = await read_directory(client, uri=folder)
        index = await read_json(client, f"{folder}/_index.json")
    assert result["resources"] == index


async def test_a_file_child_is_its_resources_list_row():
    async with Client(demo().server) as client:
        listed = next(
            r
            for r in (await client.session.list_resources()).resources
            if str(r.uri) == ENTRY_URI
        )
        result = await read_directory(client, uri=ROOT)
    row = listed.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert result["resources"][0] == row


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"uri": f"{ROOT}/nope"}, "Directory not found"),
        ({"uri": ENTRY_URI}, "Directory not found"),
        ({"uri": f"{REFERENCES}/"}, "Directory not found"),
        ({"uri": f"{ROOT}/"}, "Directory not found"),
        ({"uri": "workspace://current"}, "Directory not found"),
        ({"uri": ROOT, "cursor": "abc"}, "Invalid directory cursor"),
        ({"uri": ROOT, "cursor": "0"}, "Invalid directory cursor"),
        ({"uri": ROOT, "cursor": "99"}, "Invalid directory cursor"),
        ({"uri": ROOT, "cursor": "1" * 5000}, "Invalid directory cursor"),
        ({"uri": ""}, ""),
        ({}, ""),
    ],
    ids=[
        "missing",
        "file",
        "trailing-slash",
        "root-slash",
        "other-scheme",
        "cursor-text",
        "cursor-zero",
        "cursor-past-end",
        "cursor-5000-digits",
        "empty-uri",
        "no-uri",
    ],
)
async def test_refusals_are_invalid_params(params, message):
    async with Client(demo().server) as client:
        with pytest.raises(MCPError) as caught:
            await read_directory(client, **params)
    assert caught.value.error.code == INVALID_PARAMS
    assert message in str(caught.value)


async def test_pages_join_to_the_whole_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(kubed.mcp.skills._DirectoryRead, "page_size", 2)
    mcp = served(tmp_path, {f"references/R{n}.md": f"# R{n}\n" for n in range(5)})
    pages = []
    async with Client(mcp.server) as client:
        result = await read_directory(client, uri="skill://tmp-skill/references")
        pages.append(result)
        while "nextCursor" in result:
            result = await read_directory(
                client, uri="skill://tmp-skill/references", cursor=result["nextCursor"]
            )
            pages.append(result)
        index = await read_json(client, "skill://tmp-skill/references/_index.json")
    assert [len(page["resources"]) for page in pages] == [2, 2, 1]
    assert [r for page in pages for r in page["resources"]] == index
    assert len(index) == 5


async def test_the_method_matches_the_uri_as_sent_while_a_read_normalises():
    dotted = f"{ROOT}/x/../references"
    async with Client(demo().server) as client:
        with pytest.raises(MCPError) as caught:
            await read_directory(client, uri=dotted)
        index = await read_json(client, f"{dotted}/_index.json")
    assert caught.value.error.code == INVALID_PARAMS
    assert [r["uri"] for r in index] == REFERENCE_URIS


# ---- a folder serves no content -------------------------------------------------------


@pytest.mark.parametrize("folder", [ROOT, REFERENCES])
async def test_reading_a_folder_names_its_index(folder):
    async with Client(demo().server) as client:
        with pytest.raises(MCPError) as caught:
            await client.read_resource(folder)
    error = caught.value.error
    assert error.code == INVALID_PARAMS
    assert error.message.startswith(f"Resource not found: {folder!r}. ")
    assert f"{folder}/_index.md" in error.message
    assert error.data == {"uri": folder}


async def test_a_trailing_slash_is_a_plain_miss():
    async with Client(demo().server) as client:
        with pytest.raises(MCPError) as caught:
            await client.read_resource(f"{REFERENCES}/")
    assert caught.value.error.code == INVALID_PARAMS
    assert "_index" not in caught.value.error.message


async def test_a_server_side_read_raises_the_folder_miss():
    """What a tool that reads resources for a client catches, to say ``miss``."""
    mcp = demo()

    @mcp.tool
    async def read(uri: str, ctx: Context) -> str:
        try:
            await ctx.fastmcp.read_resource(uri)
        except FolderNotFound as folder:
            return folder.miss
        return "read"

    async with Client(mcp.server) as client:
        said = (await client.call_tool("read", {"uri": REFERENCES})).data
        file = (await client.call_tool("read", {"uri": REFERENCE_URIS[0]})).data
    assert said == (
        f"no resource at {REFERENCES!r}. {REFERENCES} is a folder: "
        f"read {REFERENCES}/_index.md to list it."
    )
    assert file == "read"


# ---- catalogues -----------------------------------------------------------------------


NESTED = "skill://lib/folder/demo-skill"


async def test_a_nested_skill_has_its_own_folders():
    mcp = QuickMCP("t")
    serve(mcp.server, {"lib/folder/demo-skill": load_skill(DEMO)})
    async with Client(mcp.server) as client:
        root = await read_directory(client, uri=NESTED)
        references = await read_json(client, f"{NESTED}/references/_index.json")
        with pytest.raises(MCPError) as library:
            await read_directory(client, uri="skill://lib")
        with pytest.raises(MCPError) as folder:
            await client.read_resource(f"{NESTED}/references")
    assert [r["uri"] for r in root["resources"]] == [
        f"{NESTED}/SKILL.md",
        f"{NESTED}/references",
        f"{NESTED}/scripts",
    ]
    assert root["resources"][1]["name"] == "lib/folder/demo-skill/references"
    assert [(r["uri"], r["description"]) for r in references] == [
        (f"{NESTED}/references/GUIDE.md", "Guide"),
        (f"{NESTED}/references/NOTES.md", "File from demo-skill skill"),
    ]
    assert library.value.error.message == "Directory not found"
    assert f"{NESTED}/references/_index.md" in folder.value.error.message


async def test_each_provider_answers_its_own_folders():
    other = Skill.from_files(
        {
            "SKILL.md": "---\nname: other\ndescription: Another skill.\n---\n",
            "x/y.md": "# Y\n",
        }
    )
    mcp = QuickMCP("t")
    serve(mcp.server, [load_skill(DEMO)])
    serve(mcp.server, [other])
    async with Client(mcp.server) as client:
        demo_root = await read_directory(client, uri=ROOT)
        other_x = await read_directory(client, uri="skill://other/x")
        with pytest.raises(MCPError) as folder:
            await client.read_resource("skill://other/x")
    assert demo_root["resources"][0]["uri"] == ENTRY_URI
    assert [(r["uri"], r["description"]) for r in other_x["resources"]] == [
        ("skill://other/x/y.md", "Y")
    ]
    assert "skill://other/x/_index.md" in folder.value.error.message


class Switching(EntryOnlySkillProvider):
    """A catalogue built per request from what ``current`` holds when it comes."""

    def __init__(self, current: dict[str, Skill]):
        super().__init__()
        self.current = current
        self.loads = 0

    async def load_skills(self):
        self.loads += 1
        return dict(self.current)


def skill(files: dict[str, str]) -> Skill:
    return Skill.from_files(
        {
            "SKILL.md": "---\nname: tmp-skill\ndescription: A skill for tests.\n---\n",
            **files,
        }
    )


async def test_folders_and_indexes_follow_a_catalogue_loaded_per_request(caplog):
    """``load_skills()`` overridden: nothing the provider knows was fixed when it was built."""
    current = {"lib/tmp-skill": skill({"a/x.md": "# Ex\n"})}
    provider = Switching(current)
    mcp = QuickMCP("t")
    serve(mcp.server, provider)
    base = "skill://lib/tmp-skill"
    async with Client(mcp.server) as client:
        before = await read_json(client, f"{base}/_index.json")
        a = (await client.read_resource(f"{base}/a/_index.md"))[0].text

        current["lib/tmp-skill"] = skill(
            {"b/y.md": "# Why\n", "b/_index.md": "reserved", "b/c/z.txt": "z"}
        )
        with caplog.at_level(logging.WARNING):
            after = await read_directory(client, uri=base)
            b = await read_json(client, f"{base}/b/_index.json")
            entry = (await request(client, "skills/list"))["skills"][0]
            read = await client.read_skill(f"{base}/SKILL.md")
            with pytest.raises(MCPError) as gone:
                await client.read_resource(f"{base}/a/_index.json")
            with pytest.raises(MCPError) as folder:
                await client.read_resource(f"{base}/b")
    assert [r["uri"] for r in before] == [f"{base}/SKILL.md", f"{base}/a"]
    assert a == f"{base}/a/x.md: Ex\n"
    assert [r["uri"] for r in after["resources"]] == [f"{base}/SKILL.md", f"{base}/b"]
    assert [(r["uri"], r["description"]) for r in b] == [
        (f"{base}/b/c", "A folder of the tmp-skill skill"),
        (f"{base}/b/y.md", "Why"),
    ]
    assert f"{base}/b/_index.md" not in [r["uri"] for r in entry["resources"]]
    assert set(read.files) == {"SKILL.md", "b/y.md", "b/c/z.txt"}
    assert gone.value.error.message.startswith("Resource not found")
    assert f"{base}/b/_index.md" in folder.value.error.message
    # The catalogue loaded per request is trimmed without a warning every request.
    assert not caplog.records


@pytest.mark.parametrize(
    "call",
    ["directory", "index", "file", "folder", "list"],
)
async def test_a_catalogue_is_loaded_once_per_request(call):
    provider = Switching(
        {"tmp-skill": skill({"a/x.md": "# Ex\n", "a/y.md": "# Why\n"})}
    )
    mcp = QuickMCP("t")
    serve(mcp.server, provider)
    async with Client(mcp.server) as client:
        provider.loads = 0
        if call == "directory":
            await read_directory(client, uri="skill://tmp-skill/a")
        elif call == "index":
            await client.read_resource("skill://tmp-skill/a/_index.md")
        elif call == "file":
            await client.read_resource("skill://tmp-skill/a/x.md")
        elif call == "folder":
            with pytest.raises(MCPError):
                await client.read_resource("skill://tmp-skill/a")
        else:
            await request(client, "skills/list")
    assert provider.loads == 1


def test_make_skill_is_a_valid_skill(tmp_path):
    """The throwaway skill the tests serve is one FastMCP accepts."""
    assert (
        Skill.from_directory(make_skill(tmp_path, {"a.md": "# A\n"})).name
        == "tmp-skill"
    )
