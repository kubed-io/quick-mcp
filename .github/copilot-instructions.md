# Copilot code review — kubed-mcp

## Purpose & scope

You are reviewing pull requests for a **Python library**: `kubed-mcp`, imported as `kubed.mcp`, an
opinionated platform on top of FastMCP that mcp-kb and selenium-flow share. It owns how a server
serves plugins, the MCP protocol details that come with it, and extension points loaded through
Python entry points. It ships as a Python distribution on PyPI, with no image and no deployment
manifest. Its servers take FastMCP through it, so a re-exported name is a public contract.

**Read these repo files first — they are the source of truth, and you should back your comments
with them:**

- **`AGENTS.md`** — the non-negotiables and the reasoning behind them.
- **`CONTRIBUTING.md`** — the layout and what CI will say about a change.

When a convention is undocumented, the sibling repos `kubed-io/mcp-kb` and `kubed-io/selenium-flow`
set the house style for CI, release flow and layout.

## The maintainer is **Dr K**. Never write a real name.

This repository is public. The maintainer is **Dr K** in prose, commit messages, PR descriptions and
comments.

**No first name, no surname, no personal email, no personal domain, no personal GitHub handle**
outside `CODEOWNERS`. Not in a code comment, a test fixture or a transcript quoting a live instance.
A real hostname pasted in as evidence is the same leak. **Flag any real name or homelab hostname in a
diff as a blocker.**

## The layout

| Path | Holds |
|---|---|
| `kubed/mcp/` | the library |
| `tests/` | pytest, with no network |
| `docs/` | the GitHub wiki, as a submodule |

`kubed/` is a **PEP 420 namespace package and has no `__init__.py`**, which is what lets `kubed.mcp`
install beside `kubed.mcp_kb` and `kubed.selenium_flow`. A PR that adds one is a finding.

## Review priorities (highest first)

1. **Public API** — servers import this package, so a renamed, retyped or removed public name is a
   breaking change. It needs a **BREAKING:** changelog line and the matching change in both servers;
   carrying the old shape for compatibility is not required.
2. **Entry-point groups** — a group is an interface and its members live in other packages. A change
   to a group's contract without versioning is a finding. A new group needs a real use case: an
   implementation that must live in another project, or a real branch in how something is done.
3. **Correctness** — error paths and edge cases; re-derive test expectations from the documented
   behaviour, not from the current implementation.
4. **Dead code & simplification** — unused code and imports, redundant abstractions.
5. **Tests** — a `kubed/mcp/` change should carry a test in `tests/`.

## Signal over volume

Fewer, higher-value comments beat exhaustive nitpicking. Noise trains the team to ignore you.

- **Worth a comment:** correctness, the public API, and the entry-point contracts — always.
- **Usually skip:** a wording tweak, a slightly stale comment, a missing type on an internal helper.
- **Verify a bug is real before flagging it.** Trace the guards in the same function and confirm the
  failure path is reachable.
- **Assume the library is correct before calling something unsafe.** If FastMCP plausibly handles the
  concern, assume it does unless the code shows otherwise.

## Python conventions specific to this repo

- **Annotate the public API.** `param: str | None = None`, never `param: str = None`; no bare `dict`
  or untyped `**kwargs` on a public name.
- **A glob ending in `**` is written `dir/**/*`.** Before Python 3.13 a trailing `**` matches
  directories only.
- Prefer `pathlib` over `os.path`, f-strings over `%`/`.format`, and
  `from __future__ import annotations` at the top of new modules.

## Project non-negotiables — do not approve changes that break these

- **Versions come from git tags.** Never ask for a hand-written version bump; the release flow owns
  versions.
- **No secret in a log, an error message or a path.**

## CI conventions

Workflow changes are governed by
[github-workflows.instructions.md](./instructions/github-workflows.instructions.md), YAML style by
[yaml.instructions.md](./instructions/yaml.instructions.md), and Python by
[python.instructions.md](./instructions/python.instructions.md).

- **Required checks must never be path-filtered.** A path-filtered required check never reports on a
  PR that misses the filter, and the PR can then never merge.
- **Every quality gate has a config file with reasons.** `.github/zizmor.yml` records *why* each rule
  is relaxed. A PR that adds an ignore without a reason is a finding.
- **A draft runs no checks.**

## Changelog entries are release notes — review them as such

`CHANGELOG.md`'s `[Unreleased]` section is handed verbatim to the release. It is worth a comment when
it is wrong, and the failure is almost always one of these:

- **A paragraph instead of a line.** Propose the one-line version; the reasoning belongs in
  `AGENTS.md` or the PR. Only **BREAKING:** may stretch.
- **An entry for internal work.** CI, refactors, dependency bumps, tests and docs earn no entry; the
  `no changelog` label is the intended answer.
- **An edit to a released section.** Everything below `[Unreleased]` is immutable.
- **A hand-written version heading or version bump.**

## The agent workflows

The agent workflows run Dr K's issue → spec → plan → code → review flow on the self-hosted `claude`
runner. Each calls the workflow of the same name in kubed-io/actions, where the jobs live. A PR from
an `issue-<N>` branch with the `agent` label is the agents' work.

- **They start only on `repository_dispatch`**, which n8n sends. Don't add `issues`, `issue_comment`
  or `pull_request_target` triggers.
- **`kubed-io/actions/*@main`** is first-party and has no tags; see `.github/zizmor.yml`.
- **`.issue/` is the agents' scratch** and is gitignored. Any `.issue/` file in a diff is a finding.

## Review style

- Be specific and actionable: cite the file and line, and name the fix.
- Explain the "why" in one line.
- Stay within the diff and its blast radius.

## What not to flag (settled)

- **Actions are pinned to release tags, not commit hashes.** A recorded decision; see `unpinned-uses`
  in `.github/zizmor.yml`.
- **A terse changelog entry is correct, not lazy.** One line is the house style.
