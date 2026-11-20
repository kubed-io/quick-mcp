# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
  These ARE the release notes. One SHORT line per entry, written for a user —
  never a paragraph. Say what someone can now do, not how it was built and not
  why. Only **BREAKING:** may stretch. Internal work — CI, refactors, tests,
  types, docs — usually earns no line at all, and never more than a terse one.
  Deeper detail lives in AGENTS.md or the PR, not here.

  A version may open with a short preamble above its first heading, framing the
  release as a whole. That is prose, and the "never a paragraph" rule does not
  reach it — it is about the entries.

  ONLY EVER EDIT THE [Unreleased] SECTION. Every section below it carries a
  version number and is IMMUTABLE — those notes shipped with a release and must
  never be reworded, reordered, or removed. Add new work under [Unreleased].
  publish.yml (duplocloud/version-bump) rolls [Unreleased] into a dated version
  section at release time. See CONTRIBUTING.md / AGENTS.md.
-->

## [Unreleased]

### Added

- **`QuickMCP`:** build a FastMCP server in a factory, and serve it from its config with `run()`.
- **`Inject[T]`:** a tool, resource or prompt takes an object the container builds, kept out of its schema.
- Singleton, session and transient lifetimes, with async factories, ordered teardown and `override` for tests.
- Import FastMCP's names from `kubed.mcp`, `kubed.mcp.protocol` and `kubed.mcp.auth`, the same objects under one pin.
- `request_values()` reads a request's query parameters and headers, every value a list.
- `kubed-mcp[tasks]` installs FastMCP's background tasks.

### Changed

- Needs FastMCP 4.2 or later.

## [0.0.2] - 2026-11-20

## [0.0.1] - 2026-11-20

### Added

- Repository started: a skeleton with no code yet.
