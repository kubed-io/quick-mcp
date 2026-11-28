"""A cold build keeps the event loop answering: it yields between items, builds are made
one at a time in the order asked for, and each one made or waited on is a span (#99,
PLUG-55, TEL-24)."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from fastmcp import Client
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from plugin_server import agent_md, command_md, served, skill, write

from kubed.mcp import QuickMCP
from kubed.mcp import plugins as plugins_module
from kubed.mcp.plugins import View

pytestmark = pytest.mark.unit

# A skill with nested folders, each kind of file an index describes, a hidden path and
# the reserved names: what a folder's index is rendered from.
KIT = {
    "skills/kit/SKILL.md": skill("kit"),
    "skills/kit/references/HEADED.md": "# A heading\n\nBody.\n",
    "skills/kit/references/plain.md": "No heading here.\n",
    "skills/kit/references/with space.md": "# Spaced out\n",
    "skills/kit/references/deep/fenced.md": "```\n# not a heading\n```\nBody.\n",
    "skills/kit/references/deep/run.sh": "#!/bin/sh\necho hi\n",
    "skills/kit/notes/.secret.md": "# Hidden\n",
    "skills/kit/.hidden/x.md": "# Hidden too\n",
    "skills/kit/x/_index.md": "# Reserved\n",
    "skills/kit/x/_manifest": "an ordinary file below the root\n",
    "skills/kit/x/keep.md": "# Kept\n",
    "skills/kit/_manifest": "reserved at the root\n",
    "skills/kit/_index.md/inner.md": "# In a folder that is no folder\n",
}
BINARY = {"skills/kit/assets/logo.png": bytes(range(256))}
KIT_FOLDERS = [
    "skill://t/kit",
    "skill://t/kit/references",
    "skill://t/kit/references/deep",
    "skill://t/kit/assets",
    "skill://t/kit/x",
]


def kit(tmp_path: Path) -> Path:
    root = write(tmp_path / "kit", KIT)
    for path, data in BINARY.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    return root


async def test_a_rewritten_build_renders_the_indexes_the_provider_does(tmp_path):
    root = kit(tmp_path)
    plain = served((root, None))
    rewritten = served((root, None))
    rewritten.add_rewriter(lambda site, text: text)  # the build renders the indexes
    reads: dict[str, list] = {}
    for mcp in (plain, rewritten):
        async with Client(mcp.server) as client:
            for folder in KIT_FOLDERS:
                for name in ("_index.md", "_index.json"):
                    uri = f"{folder}/{name}"
                    (read,) = await client.read_resource(uri)
                    reads.setdefault(uri, []).append(
                        (str(read.uri), read.mime_type, read.text)
                    )
    assert (await rewritten._plugins.pinned()).build.rewritten
    assert all(first == second for first, second in reads.values())
    root_md = reads["skill://t/kit/_index.md"][0][2]
    references = reads["skill://t/kit/references/_index.json"][0][2]
    # Every kind of row is there: a folder, a heading, none, a fenced one, a binary.
    assert "skill://t/kit/references: A folder of the kit skill" in root_md
    assert "skill://t/kit/_manifest" not in root_md
    assert '"description": "A heading"' in references
    assert '"description": "File from kit skill"' in references
    assert "Spaced out" in references
    deep = reads["skill://t/kit/references/deep/_index.md"][0][2]
    assert "fenced.md: File from kit skill" in deep
    assert "image/png" in reads["skill://t/kit/assets/_index.json"][0][2]
    x = reads["skill://t/kit/x/_index.md"][0][2]
    assert "skill://t/kit/x/_manifest" in x and "x/_index.md:" not in x
    # A file in a folder that is no folder is listed nowhere.
    assert not any("inner.md" in each[0][2] for each in reads.values())


def many(tmp_path: Path, *, rewriter: bool = True) -> tuple[QuickMCP, list]:
    """Five skills of a few files each, two commands and two agents, and, unless
    ``rewriter`` is False, a rewriter that logs each call as (view key, URI)."""
    tree = {}
    for i in range(5):
        tree |= {
            f"skills/s{i}/SKILL.md": skill(f"s{i}"),
            f"skills/s{i}/references/a.md": "# A\n",
            f"skills/s{i}/references/b.md": "# B\n",
        }
    tree |= {f"commands/c{i}.md": command_md() for i in range(2)}
    tree |= {f"agents/a{i}.md": agent_md(f"a{i}") for i in range(2)}
    mcp = served((write(tmp_path / "many", tree), None))
    calls: list = []

    def log(site, text):
        calls.append((site.view, site.uri))
        return text

    if rewriter:
        mcp.add_rewriter(log)
    return mcp, calls


def every(key: str) -> View:
    return View(key, lambda plugin: True)


def _is_index(entry: object) -> bool:
    return isinstance(entry, tuple) and entry[1].rsplit("/", 1)[-1] in _INDEXES


_INDEXES = ("_index.md", "_index.json")


async def test_a_build_yields_to_the_loop_between_items(tmp_path):
    mcp, log = many(tmp_path)
    done = anyio.Event()

    async def build() -> None:
        await mcp._plugins._pin_view(every("a"))
        done.set()

    async def tick() -> None:
        while not done.is_set():
            log.append("tick")
            await anyio.sleep(0)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(build)
            tg.start_soon(tick)

    def between(kind) -> bool:
        """Whether a tick sits between two of the rewriter's calls of ``kind``."""
        at = [i for i, entry in enumerate(log) if entry != "tick" and kind(entry)]
        return any(at[0] < i < at[-1] for i, entry in enumerate(log) if entry == "tick")

    # Both passes yield: rewriting the skills, commands and agents, and the indexes.
    assert between(lambda entry: not _is_index(entry)), log
    assert between(_is_index), log


async def until_building(mcp: QuickMCP, key: str) -> None:
    """Until a build for view ``key`` is in flight."""
    while not any(k[1] == key for k in mcp._plugins._memo.flights):
        await anyio.sleep(0)


async def test_builds_are_made_one_at_a_time_in_the_order_asked(tmp_path):
    mcp, log = many(tmp_path)
    builds: dict[str, list] = {"a": [], "b": []}

    async def pin(key: str) -> None:
        builds[key].append((await mcp._plugins._pin_view(every(key))).build)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(pin, "a")
            await until_building(mcp, "a")
            tg.start_soon(pin, "b")
            tg.start_soon(pin, "a")  # a second reader of the key being built
    keys = [key for key, _ in log]
    assert keys == sorted(keys) and set(keys) == {"a", "b"}, keys
    first, second = builds["a"]
    assert second is first
    assert builds["b"][0] is not first


async def test_an_unrewritten_build_is_not_queued_behind_a_rewritten_one(tmp_path):
    mcp, _ = many(tmp_path, rewriter=False)  # only a prefix rewrites
    plugins = mcp._plugins
    memo, state = plugins._memo, plugins._state
    prefixed = (state.generation, "a", "kb+")
    plain = (state.generation, "b", None)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(plugins._build_for, memo, state, every("a"), prefixed)
            while not plugins._building.locked():  # paused at a checkpoint, holding it
                await anyio.sleep(0)
            build = await plugins._build_for(memo, state, every("b"), plain)
            # Made while the rewritten build still holds the lock, not after it.
            assert prefixed in memo.flights and plugins._building.locked()
    assert not build.rewritten and memo.builds[prefixed].rewritten


async def test_a_build_cancelled_midway_or_queued_frees_the_next_and_leaves_no_flight(
    tmp_path, spans
):
    mcp, log = many(tmp_path)
    plugins, made = mcp._plugins, []

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            scopes = {"a": anyio.CancelScope(), "c": anyio.CancelScope()}

            async def cancelled(key: str) -> None:
                with scopes[key]:
                    await plugins._pin_view(every(key))
                    made.append(key)  # never: it is cancelled before it is done

            async def other() -> None:
                await plugins._pin_view(every("b"))
                made.append("b")

            tg.start_soon(cancelled, "a")
            await until_building(mcp, "a")
            tg.start_soon(cancelled, "c")  # queued on the lock, then cancelled there
            tg.start_soon(other)  # queued behind it
            while not log or plugins._building.statistics().tasks_waiting < 2:
                await anyio.sleep(0)  # "a" is under way at a checkpoint; "c", "b" wait
            scopes["c"].cancel()
            scopes["a"].cancel()
    assert made == ["b"]  # the lock was released: the build waiting on it was made
    assert not plugins._memo.flights and not plugins._building.locked()
    assert [k[1] for k in plugins._memo.builds] == ["b"]
    outcomes = [each["kubed.build.outcome"] for each in finished(spans)]
    assert sorted(outcomes) == ["cancelled", "cancelled", "made"]
    # A cancelled build is no success: it has no counts.
    assert [
        "kubed.build.skills" in each
        for each in finished(spans)
        if each["kubed.build.outcome"] == "cancelled"
    ] == [False, False]
    for key in ("a", "c"):  # each built afresh, whole
        again = (await plugins._pin_view(every(key))).build
        assert again.rewritten and len(again.catalogue) == 5
    assert [k[1] for k in plugins._memo.builds] == ["b", "a", "c"]
    assert not plugins._building.locked()


@pytest.fixture
def spans(monkeypatch) -> InMemorySpanExporter:
    """The spans the plugin layer finishes, from an SDK tracer patched in."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(plugins_module, "_TRACER", provider.get_tracer("test"))
    return exporter


def finished(spans: InMemorySpanExporter) -> list[dict]:
    return [
        dict(span.attributes or {})
        for span in spans.get_finished_spans()
        if span.name == "kubed.mcp.build"
    ]


async def test_a_build_made_or_waited_on_is_a_span_and_a_kept_one_none(tmp_path, spans):
    mcp, _ = many(tmp_path)
    plugins = mcp._plugins
    memo, state, view = plugins._memo, plugins._state, every("a")
    key = (state.generation, "a", "kb+")  # a prefixed key, as a gateway's request has
    builds = []

    async def build() -> None:
        builds.append(await plugins._build_for(memo, state, view, key))

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(build)
            await until_building(mcp, "a")
            tg.start_soon(build)  # a second reader, waiting on the first's flight
    made, waited = finished(spans)
    generation = state.generation
    assert made == {
        "kubed.build.generation": generation,
        "kubed.build.prefixed": True,
        "kubed.build.outcome": "made",
        "kubed.build.skills": 5,
        "kubed.build.prompts": 2,
        "kubed.build.agents": 2,
    }
    assert waited == {
        "kubed.build.generation": generation,
        "kubed.build.prefixed": True,
        "kubed.build.outcome": "waited",
    }
    assert builds[0] is builds[1]
    assert await plugins._build_for(memo, state, view, key) is builds[0]  # kept
    assert len(finished(spans)) == 2  # a kept build is no span
    await plugins._pin_view(every("b"))  # a request with no prefix
    assert finished(spans)[2]["kubed.build.prefixed"] is False
