# AGENTS.md

Agent context for `kubed-mcp`. Read this before working in this repo.

A Python library, published on PyPI as `kubed-mcp` and imported as `kubed.mcp`: an opinionated
platform on top of FastMCP that mcp-kb and selenium-flow share. It owns how a server serves plugins
(skills, agents, commands), the MCP protocol details that come with it, and the extension points
loaded through Python entry points. Today it is a skeleton and holds no logic.

## Read first

- **[The wiki](https://github.com/kubed-io/quick-mcp/wiki)** is the manual, checked out as a
  submodule at `docs/`. `docs/` is the wiki's own repository: [CONTRIBUTING.md](CONTRIBUTING.md)
  says which branch to commit on, and you never push the wiki's `master`.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** is the loop: setup, what CI runs, how the changelog is
  written.
- **[README.md](README.md)** advertises. It links out and explains nothing.

This repo ships **a Python distribution and nothing else**: no image, no Kubernetes manifests.

## Layout

| Path | What it is |
| --- | --- |
| `kubed/mcp/` | the package; `kubed/` is a namespace and `mcp/` is the library |
| `tests/` | pytest, no network |
| `docs/` | the wiki, as a submodule |
| `.github/` | workflows, dependabot, zizmor, and the review instructions |

## Things that cost someone an afternoon

- **`kubed/` is a PEP 420 namespace and has no `__init__.py`.** That is what lets `kubed.mcp` install
  beside `kubed.mcp_kb` and `kubed.selenium_flow`. Adding one breaks the pair silently, in whichever
  environment resolves second.
- **An entry-point group is an interface.** A member is a class implementing it, found by name. Never
  change a group's contract without versioning it, because the implementations live in other
  packages.
- **A glob ending in `**` is written `dir/**/*`.** Before Python 3.13 a trailing `**` matches
  directories only, so the pattern serves nothing on the oldest interpreter and everything on the
  newest, with no error either way.

## House rules that apply here

- **The maintainer is Dr K.** This repository is public: no first name, surname, personal email or
  homelab hostname in a file, a commit message or a PR. `CODEOWNERS` is the one place a handle lives.
- **The changelog is the release notes, not a work log.** `publish.yml` hands `[Unreleased]` to
  `duplocloud/version-bump`, which puts it straight into the GitHub Release. One short line per
  entry, saying what someone can now do. Internal work earns no line and takes the `no changelog`
  label. Only ever edit `[Unreleased]`, and never write a version heading: versions come from git
  tags. Full rules in [CONTRIBUTING.md](CONTRIBUTING.md#the-changelog).
- **A public name is a contract.** Servers import this package, so renaming, retyping or removing one
  is a breaking change. Adding an optional argument is safe.
- **No backwards compatibility for its own sake.** There are two consumers, both ours: change both,
  and purge the old shape rather than carry it.
- `.github/zizmor.yml` records *why* each relaxed rule is relaxed. An ignore without a reason does
  not belong there.
- No `Makefile`. Everything is `pyproject.toml` plus scripts.
- Verify an action's current latest major with `gh api repos/<owner>/<repo>/releases/latest` before
  any `uses:`, and use the `ubuntu-26.04` runner, never `ubuntu-latest`.
