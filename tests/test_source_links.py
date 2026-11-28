from __future__ import annotations

import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import sysconfig
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

import kubed.mcp.pyroscope as source_map
from kubed.mcp.pyroscope import (
    _commit,
    _github,
    _Mapping,
    _mappings,
    _normalise,
    _packages,
    _ref,
    _render,
    _Unmappable,
    main,
)

SITE = PurePosixPath("/opt/venv/lib/python3.14/site-packages")
FULL = "311c15298c0c036bce9e99d306c166689c4525a9"


class Present(PurePosixPath):
    """A path that exists: Python 3.14's ``Distribution.files`` skips what isn't there."""

    def exists(self) -> bool:
        return True


class Fake(importlib.metadata.Distribution):
    """A distribution read from strings, installed under ``site``."""

    def __init__(
        self,
        name: str,
        version: str,
        files: Iterable[str] = (),
        urls: Iterable[str] = (),
        direct_url: dict[str, Any] | None = None,
        site: PurePosixPath = SITE,
    ) -> None:
        self._name = name
        self._version = version
        self._files = list(files)
        self._urls = list(urls)
        self._direct_url = direct_url
        self._site = site

    def read_text(self, filename: str) -> str | None:
        if filename == "METADATA":
            lines = ["Metadata-Version: 2.1", f"Name: {self._name}"]
            lines.append(f"Version: {self._version}")
            lines.extend(f"Project-URL: {url}" for url in self._urls)
            return "\n".join(lines) + "\n"
        if filename == "RECORD":
            return "".join(f"{path},,\n" for path in self._files)
        if filename == "direct_url.json" and self._direct_url is not None:
            return json.dumps(self._direct_url)
        return None

    def locate_file(self, path: Any) -> PurePosixPath:
        return Present(self._site, path)


def kb(**kw: Any) -> Fake:
    return Fake(
        "kubed-mcp-kb",
        "1.4.0",
        [
            "kubed/mcp_kb/__init__.py",
            "kubed/mcp_kb/app.py",
            "kubed/mcp_kb/tools/__init__.py",
        ],
        ["Repository, https://github.com/kubed-io/mcp-kb"],
        **kw,
    )


def mcp(version: str = "0.6.1", **kw: Any) -> Fake:
    return Fake(
        "kubed-mcp",
        version,
        ["kubed/mcp/__init__.py", "kubed/mcp/telemetry.py"],
        [
            "Homepage, https://github.com/kubed-io",
            "Repository, https://github.com/kubed-io/quick-mcp",
        ],
        **kw,
    )


def slim(**kw: Any) -> Fake:
    return Fake(
        "fastmcp-slim",
        "4.2.6",
        ["fastmcp/__init__.py", "fastmcp/server/__init__.py"],
        ["Repository, https://github.com/PrefectHQ/fastmcp"],
        **kw,
    )


@pytest.fixture
def image(monkeypatch: pytest.MonkeyPatch):
    """Install fakes as the image's distributions: image(*fakes, stdlib=..., python=...)."""

    def install(
        *fakes: Fake,
        stdlib: str = "/usr/local/lib/python3.14",
        python: str = "3.14.10",
    ) -> None:
        found = {_normalise(f.metadata["Name"]): f for f in fakes}
        monkeypatch.setattr(
            source_map, "_distribution", lambda name: found.get(_normalise(name))
        )
        monkeypatch.setattr(source_map, "_stdlib", lambda: stdlib)
        monkeypatch.setattr(source_map, "_python", lambda: python)

    return install


@pytest.mark.parametrize(
    ("version", "commit", "ref"),
    [
        ("0.6.1", None, "v0.6.1"),
        ("0.8.0rc2", None, "v0.8.0-rc.2"),
        ("0.7.1.dev3+g311c152", None, "311c152"),
        ("0.7.1.dev3+g311c152.d20261128", None, "311c152"),
        ("0.8.0rc2.dev1+g311c152", None, "311c152"),
        ("0.7.1.dev3+g311c152", FULL, FULL),
        ("0.6.1", FULL, "v0.6.1"),
        ("0.7.1.dev3", None, None),
        ("0.7.0.post1", None, None),
    ],
)
def test_kubed_mcps_version_maps_to_its_ref(
    version: str, commit: str | None, ref: str | None
) -> None:
    assert _ref(version, commit) == ref


def test_a_git_install_maps_to_its_full_commit(image) -> None:
    git = {
        "url": "https://github.com/kubed-io/quick-mcp",
        "vcs_info": {"vcs": "git", "commit_id": FULL},
    }
    editable = {"url": "file:///src/quick-mcp", "dir_info": {"editable": True}}
    assert _commit(mcp("0.7.1.dev3+g311c152", direct_url=git)) == FULL
    assert _commit(mcp(direct_url=editable)) is None
    assert _commit(mcp()) is None
    image(kb(), mcp("0.7.1.dev3+g311c152", direct_url=git), slim())
    _, mappings = _mappings("kubed-mcp-kb", print)
    assert mappings[1].source["github"]["ref"] == FULL


@pytest.mark.parametrize(
    ("urls", "repository"),
    [
        (
            ["Repository, https://github.com/PrefectHQ/fastmcp"],
            ("PrefectHQ", "fastmcp"),
        ),
        (
            [
                "Homepage, https://github.com/kubed-io",
                "Repository, https://github.com/kubed-io/quick-mcp",
            ],
            ("kubed-io", "quick-mcp"),
        ),
        (["Source Code, https://github.com/example/server.git"], ("example", "server")),
        (["source, https://github.com/example/server/"], ("example", "server")),
    ],
)
def test_the_repository_comes_from_project_url(
    urls: list[str], repository: tuple[str, str]
) -> None:
    assert _github(Fake("x", "1.0", urls=urls)) == repository


@pytest.mark.parametrize(
    "urls",
    [
        [],
        ["Homepage, https://gofastmcp.com"],
        ["Repository, https://gitlab.com/example/server"],
        ["Repository, https://github.com/example"],
    ],
)
def test_no_github_repository_is_unmappable(urls: list[str]) -> None:
    with pytest.raises(
        _Unmappable, match=r"^x's Project-URL names no GitHub repository$"
    ):
        _github(Fake("x", "1.0", urls=urls))


def test_a_namespace_package_is_its_subpackage() -> None:
    assert _packages(kb()) == [("kubed/mcp_kb", f"{SITE}/kubed/mcp_kb/")]


def test_a_package_inside_a_package_is_not_listed() -> None:
    nested = slim()
    nested._files += [
        "fastmcp/contrib/mcp_mixin/__init__.py",
        "fastmcp/contrib/mcp_mixin/mixin.py",
    ]
    assert _packages(nested) == [("fastmcp", f"{SITE}/fastmcp/")]


def test_the_mappings_of_mcp_kbs_image(image) -> None:
    image(kb(), mcp(), slim())
    assert _mappings("kubed-mcp-kb", print) == (
        "kubed-mcp-kb",
        [
            _Mapping(f"{SITE}/kubed/mcp_kb/", {"local": {"path": "kubed/mcp_kb"}}),
            _Mapping(
                f"{SITE}/kubed/mcp/",
                {
                    "github": {
                        "owner": "kubed-io",
                        "repo": "quick-mcp",
                        "ref": "v0.6.1",
                        "path": "kubed/mcp",
                    }
                },
            ),
            _Mapping(
                f"{SITE}/fastmcp/",
                {
                    "github": {
                        "owner": "PrefectHQ",
                        "repo": "fastmcp",
                        "ref": "v4.2.6",
                        "path": "fastmcp_slim/fastmcp",
                    }
                },
            ),
            _Mapping(
                "/usr/local/lib/python3.14/",
                {
                    "github": {
                        "owner": "python",
                        "repo": "cpython",
                        "ref": "v3.14.10",
                        "path": "Lib",
                    }
                },
            ),
        ],
    )


def test_each_prefix_is_listed_with_and_without_its_slash(image) -> None:
    image(kb(), mcp(), slim())
    document = yaml.safe_load(_render(*_mappings("kubed-mcp-kb", print)))
    assert document["version"] == "v1"
    first = document["source_code"]["mappings"][0]
    assert first == {
        "path": [
            {"prefix": f"{SITE}/kubed/mcp_kb/"},
            {"prefix": f"{str(SITE)[1:]}/kubed/mcp_kb/"},
        ],
        "language": "python",
        "source": {"local": {"path": "kubed/mcp_kb"}},
    }


def test_the_same_image_writes_the_same_bytes(image) -> None:
    image(kb(), mcp(), slim())
    text = _render(*_mappings("kubed-mcp-kb", print))
    assert text == _render(*_mappings("kubed-mcp-kb", print))
    assert text.splitlines()[:2] == [
        "# Generated by `python -m kubed.mcp.pyroscope --write kubed-mcp-kb` inside the server's image.",
        "# Regenerate it there rather than edit it: CI checks it with --check.",
    ]


def test_an_image_without_a_venv_keeps_every_prefix(image) -> None:
    system = PurePosixPath("/usr/local/lib/python3.14/site-packages")
    image(kb(site=system), mcp(site=system), slim(site=system))
    _, mappings = _mappings("kubed-mcp-kb", print)
    assert [m.directory for m in mappings] == [
        f"{system}/kubed/mcp_kb/",
        f"{system}/kubed/mcp/",
        f"{system}/fastmcp/",
        "/usr/local/lib/python3.14/",
    ]


def test_a_distribution_is_mapped_once(image) -> None:
    image(mcp(), slim())
    _, mappings = _mappings("kubed-mcp", print)
    assert mappings[0].source == {"local": {"path": "kubed/mcp"}}
    assert len(mappings) == 3


def test_what_it_cannot_map(image) -> None:
    image(mcp(), slim())
    with pytest.raises(_Unmappable, match=r"^nope is not installed$"):
        _mappings("nope", print)
    image(Fake("kubed-mcp-kb", "1.4.0", ["kubed_mcp_kb.py"]), mcp(), slim())
    with pytest.raises(_Unmappable, match=r"^kubed-mcp-kb's files hold no package$"):
        _mappings("kubed-mcp-kb", print)
    image(kb(), mcp("0.7.1.dev3"), slim())
    with pytest.raises(
        _Unmappable,
        match=r"^kubed-mcp 0\.7\.1\.dev3 has no tag or commit to map to$",
    ):
        _mappings("kubed-mcp-kb", print)


def test_what_it_leaves_out(image) -> None:
    lines: list[str] = []
    editable = Fake(
        "kubed-mcp",
        "0.7.1.dev3+g311c152",
        ["__editable__.kubed_mcp-0.7.1.pth"],
        ["Repository, https://github.com/kubed-io/quick-mcp"],
    )
    image(kb(), editable)
    _, mappings = _mappings("kubed-mcp-kb", lines.append)
    assert [m.directory for m in mappings] == [
        f"{SITE}/kubed/mcp_kb/",
        "/usr/local/lib/python3.14/",
    ]
    assert lines == [
        "kubed-mcp's files hold no package, as in an editable install: its frames stay unmapped",
        "fastmcp-slim is not installed: its frames stay unmapped",
    ]


FIXTURE = Path(__file__).with_name("data") / "mcp-kb.pyroscope.yaml"
PREFIX = "kubed.mcp.pyroscope: "


def written(image, capsys, *fakes, **env) -> str:
    image(*fakes, **env)
    assert main(["--write", "kubed-mcp-kb"]) == 0
    return capsys.readouterr().out


def test_write_prints_the_file(image, capsys) -> None:
    image(kb(), mcp(), slim())
    assert main(["--write", "kubed-mcp-kb"]) == 0
    out, err = capsys.readouterr()
    assert out == _render(*_mappings("kubed-mcp-kb", print))
    assert err == ""


@pytest.mark.parametrize("spelling", ["kubed_mcp_kb", "Kubed-MCP-KB"])
def test_the_name_is_read_as_pip_reads_it(image, capsys, spelling) -> None:
    text = written(image, capsys, kb(), mcp(), slim())
    assert main(["--write", spelling]) == 0
    assert capsys.readouterr().out == text


def test_it_reproduces_mcp_kbs_committed_file(image, capsys) -> None:
    image(kb(), mcp("0.6.1"), slim())  # the versions mcp-kb's file names
    assert main(["--check", "kubed-mcp-kb", str(FIXTURE)]) == 0
    assert capsys.readouterr() == ("", "")


def test_check_ignores_order_grouping_and_comments(image, capsys, monkeypatch) -> None:
    document = yaml.safe_load(written(image, capsys, kb(), mcp(), slim()))
    split = [
        dict(m, path=[p])
        for m in reversed(document["source_code"]["mappings"])
        for p in m["path"]
    ]
    document["source_code"]["mappings"] = split
    text = "# by hand\n" + yaml.safe_dump(document)
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    assert main(["--check", "kubed-mcp-kb"]) == 0


@pytest.mark.parametrize("given", [[], ["-"]])
def test_check_reads_stdin(image, capsys, monkeypatch, given) -> None:
    text = written(image, capsys, kb(), mcp(), slim())
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    assert main(["--check", "kubed-mcp-kb", *given]) == 0


def test_a_moved_ref_is_stale(image, capsys, monkeypatch) -> None:
    committed = written(image, capsys, kb(), mcp("0.6.1"), slim())
    image(kb(), mcp("0.7.0"), slim())
    monkeypatch.setattr(sys, "stdin", io.StringIO(committed))
    assert main(["--check", "kubed-mcp-kb"]) == 1
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[:2] == [
        "--- .pyroscope.yaml (committed)",
        "+++ .pyroscope.yaml (this image)",
    ]
    assert any(x.startswith("-") and "v0.6.1" in x for x in lines[2:])
    assert any(x.startswith("+") and "v0.7.0" in x for x in lines[2:])
    assert err == (
        PREFIX + ".pyroscope.yaml is stale: regenerate it with "
        "--write kubed-mcp-kb in this image\n"
    )


@pytest.mark.parametrize(
    "committed", ["version: v1\n", "not: [a, map\n", "- just\n- a list\n"]
)
def test_a_file_that_is_not_a_source_map_is_stale(
    image, capsys, monkeypatch, committed
) -> None:
    image(kb(), mcp(), slim())
    monkeypatch.setattr(sys, "stdin", io.StringIO(committed))
    assert main(["--check", "kubed-mcp-kb"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("--- .pyroscope.yaml (committed)\n")
    assert out.endswith("\n")


@pytest.mark.parametrize("empty", ["", "\n  \n"])
def test_empty_input_says_how_to_pass_it(image, capsys, monkeypatch, empty) -> None:
    image(kb(), mcp(), slim())
    monkeypatch.setattr(sys, "stdin", io.StringIO(empty))
    assert main(["--check", "kubed-mcp-kb"]) == 2
    assert capsys.readouterr() == (
        "",
        PREFIX + "read nothing: pass the file's path, "
        "or run docker with -i so stdin reaches the command\n",
    )


def test_an_unreadable_file_exits_2(image, capsys, tmp_path) -> None:
    image(kb(), mcp(), slim())
    missing = str(tmp_path / "nope.yaml")
    assert main(["--check", "kubed-mcp-kb", missing]) == 2
    assert capsys.readouterr() == (
        "",
        f"{PREFIX}{missing} can't be read: FileNotFoundError\n",
    )


def test_what_it_cannot_map_exits_2_naming_it(image, capsys) -> None:
    image(mcp(), slim())
    assert main(["--write", "nope"]) == 2
    assert capsys.readouterr() == ("", PREFIX + "nope is not installed\n")


def test_kubed_mcp_not_installed_exits_2(image, capsys) -> None:
    image(kb(), slim())
    assert main(["--write", "kubed-mcp-kb"]) == 2
    assert capsys.readouterr() == ("", PREFIX + "kubed-mcp is not installed\n")


def test_kubed_mcp_without_a_github_repository_exits_2(image, capsys) -> None:
    image(
        kb(),
        Fake(
            "kubed-mcp",
            "0.6.1",
            ["kubed/mcp/__init__.py"],
            ["Homepage, https://example.com"],
        ),
        slim(),
    )
    assert main(["--write", "kubed-mcp-kb"]) == 2
    assert capsys.readouterr() == (
        "",
        PREFIX + "kubed-mcp's Project-URL names no GitHub repository\n",
    )


def test_what_it_leaves_out_is_said_on_stderr(image, capsys) -> None:
    image(kb(), mcp())
    assert main(["--write", "kubed-mcp-kb"]) == 0
    assert capsys.readouterr().err == (
        PREFIX + "fastmcp-slim is not installed: its frames stay unmapped\n"
    )


def test_write_takes_no_file(image) -> None:
    image(kb(), mcp(), slim())
    with pytest.raises(SystemExit) as raised:
        main(["--write", "kubed-mcp-kb", ".pyroscope.yaml"])
    assert raised.value.code == 2


def test_the_real_entry_point_in_this_environment(tmp_path: Path) -> None:
    run = [sys.executable, "-m", "kubed.mcp.pyroscope"]
    wrote = subprocess.run(
        [*run, "--write", "fastmcp-slim"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert wrote.returncode == 0, wrote.stderr
    mappings = yaml.safe_load(wrote.stdout)["source_code"]["mappings"]
    assert mappings[0]["source"] == {"local": {"path": "fastmcp"}}
    assert mappings[-1]["source"] == {
        "github": {
            "owner": "python",
            "repo": "cpython",
            "ref": f"v{platform.python_version()}",
            "path": "Lib",
        }
    }
    stdlib = sysconfig.get_paths()["stdlib"].rstrip("/") + "/"
    assert mappings[-1]["path"][0]["prefix"] == stdlib
    checked = subprocess.run(
        [*run, "--check", "fastmcp-slim"],
        input=wrote.stdout,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

    # A source checkout in the working directory does not stand in for the installed one.
    info = tmp_path / "fastmcp_slim-9.9.9.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fastmcp-slim\nVersion: 9.9.9\n"
        "Project-URL: Repository, https://github.com/PrefectHQ/fastmcp\n"
    )
    (info / "RECORD").write_text("fastmcp/__init__.py,,\n")
    (tmp_path / "fastmcp").mkdir()
    (tmp_path / "fastmcp" / "__init__.py").write_text("")
    shadowed = subprocess.run(
        [*run, "--write", "fastmcp-slim"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert shadowed.returncode == 0, shadowed.stderr
    assert shadowed.stdout == wrote.stdout
    assert str(tmp_path) not in shadowed.stdout
