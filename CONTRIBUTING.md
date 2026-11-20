# Contributing

## Getting set up

```bash
git clone --recurse-submodules git@github.com:kubed-io/quick-mcp.git
```

`docs/` is the GitHub wiki, checked out as a submodule; an existing clone gets it with
`git submodule update --init`. The build does not need it, so a clone without it works.

`kubed/` is a **PEP 420 namespace package** and has no `__init__.py`. That is what lets `kubed.mcp`
install beside `kubed.mcp_kb` and `kubed.selenium_flow`.

## Where things live

| Path | Holds |
|---|---|
| `kubed/mcp/` | the library |
| `tests/` | pytest, with no network |
| `docs/` | the wiki submodule: user pages, and the spec and plans |

## Signed commits

Every commit is signed. A pull request with an unverified commit does not merge.

## Pull requests

Work on a branch and open a pull request against `main`. A pull request is assigned to its author
and needs a new `[Unreleased]` changelog entry, or the `no changelog` label. Every change needs the
owner's review. A draft runs no checks.

## The wiki

What the library does is the spec in `docs/`, one file per area with numbered requirements, each
true of the code or marked *(proposed)*. When behaviour changes, the spec changes in the same pull
request. Designs and plans are dated files beside it, each with a History header.

A change in `docs/` is a commit in the wiki repository: commit there on the wiki branch named after
your branch, push it, then commit the `docs` pointer here. Never push the wiki's `master`.

## Changing a public name

Servers import this package, so renaming, retyping or removing a public name is a breaking change.
Its two consumers are both ours: the change ships with a **BREAKING:** changelog line and the matching
change in mcp-kb and selenium-flow, and the old shape is removed rather than carried. Adding an optional
argument is safe.

## The changelog

`CHANGELOG.md` is not a work log. **It is the release notes**, verbatim: `publish.yml` hands the
`[Unreleased]` section to `duplocloud/version-bump`, which stamps a version heading on it and puts
it straight into the GitHub Release. Write for the stranger who reads it:

- **One short line per entry**, saying what someone can now do. Not a paragraph, not the reasoning,
  not what it replaced.
- **Lead with the capability and stop when the sentence is answered.** Bold the lead only on entries
  someone would choose the release for.
- **Internal work usually earns no line at all** — CI, refactors, dependency bumps, tests, docs.
  The pull request takes the **`no changelog`** label instead.
- **Only `Added` / `Changed` / `Fixed` / `Removed`**, in that order, and only the ones you have.
  Only a **BREAKING:** entry may run long.

Two rules about *where* you write:

- **Only ever edit `[Unreleased]`.** Every section below it carries a version number and is
  immutable.
- **Never add a version heading or bump a version.** Versions come from git tags, and the release
  flow owns them.

## Before you push

`AGENTS.md` carries the rules that are easy to break by accident. Read it before changing a public
name or an entry-point group.
