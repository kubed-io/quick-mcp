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

- `add_plugin(…, manifest=entry)` and `read_plugin(root, manifest=…)` merge a marketplace entry with `plugin.json` as Claude Code does.
- `Plugin.skipped` names each skill, command and agent file a plugin does not serve, and why.
- `mcp.plugin_set()` and `mcp.serve(set)` swap a whole set of plugins at once, counted by `mcp.generation`.
- `mcp.plugin_view(fn)` serves each request only the plugins its view admits.
- `mcp.add_rewriter(fn)` rewrites the text a server serves, with digests that match.
- `to_mcp_form` rewrites `${CLAUDE_PLUGIN_ROOT}`, `${CLAUDE_SKILL_DIR}` and `<plugin>:<skill>` as `skill://` addresses.
- A server's own `file` loader replaces the built-in one.
- A server behind a multiplexing gateway takes its `X-MCP-Resource-Prefix` (or `?prefix=`) off the addresses clients send and writes it into the text it serves.
- `kubed.mcp.gateway.prefix()` and `prefixed(text, prefix)` prefix a server's own text the same way.
- `mcp.command_reader(fn)` reads a server's own command dialects as prompts.
- Same-named agents from two plugins are both served, the later as `name@plugin`.
- `Plugin.agent_files` gives each agent's file and whether it was renamed.
- `mcp.agent_meta(fn)` adds a server's own `_meta` to `agents/get`.
- A Copilot prompt file, `x.prompt.md`, is served as the prompt `x`.

### Changed

- A `skill://`, `prompts://` or `agents://` address holding an escaped `/` (`%2F`) now fails as a missing one does.
- `resources/read` resolves `.` and `..` segments in a `skill://`, `prompts://` or `agents://` address.
- Under a namespace, prompt and agent names are prefixed with its first segment only.

## [0.1.2] - 2026-11-23

### Added

- `mcp.websocket_route(path, endpoint, name=)` serves a WebSocket beside the custom routes.
- `await mcp.get_tool(name)` is the tool as registered, not as `list_tools` shows a caller.
- `request_values(request)` reads a given Starlette request, such as a custom route's.
- `kubed.mcp.skills.SkillFileResource`, the resource a skill's file is served as, for a provider that tells entries from files.

## [0.1.1] - 2026-11-23

### Changed

- A plugin's agents and commands are read at any depth, each folder a `_` segment of the name, as Claude Code reads them: `commands/db/migrate.md` is `db_migrate`; a skill is one folder below `skills/`.

### Fixed

- A manifest's `skills` key adds to the default `skills/` scan instead of replacing it.

## [0.1.0] - 2026-11-23

### Added

- **`QuickMCP`:** build a FastMCP server in a factory, and serve it from its config with `run()`.
- **`Inject[T]`:** a tool, resource or prompt takes an object the container builds, kept out of its schema.
- Singleton, session and transient lifetimes, with async factories, ordered teardown and `override` for tests.
- `mcp.singleton(T, instance=obj)` hands the container an object built before the server starts.
- `mcp.custom_route` and `mcp.add_transform`, FastMCP's own, straight on `QuickMCP`.
- Import FastMCP's names from `kubed.mcp`, `kubed.mcp.protocol` and `kubed.mcp.auth`, the same objects under one pin.
- `request_values()` reads a request's query parameters and headers, every value a list.
- `kubed-mcp[tasks]` installs FastMCP's background tasks.
- `kubed.mcp.skills` serves a catalogue of Agent Skills strictly, each listed by its `SKILL.md` alone, nested paths included.
- A served skill's folders read as directory resources, with `_index.json` and `_index.md` beside each.
- The Skills extension answers handshake-era clients too, and `AdvertiseLegacyExtensions` declares an extension in a legacy `initialize`.
- **`mcp.add_plugin(address, namespace=None)`:** serve a Claude-format plugin's skills, commands and agents in one call.
- Commands are served as MCP prompts, with declared arguments and strict `{{ name }}` placeholders.
- Agents are served through the agents extension, `agents/list` and `agents/get`.
- `mcp.add_loader()` registers a `PluginLoader` for another address scheme; local folders and `file://` work out of the box.
- Plugins added under one namespace merge, and a clash is refused.

### Changed

- Needs FastMCP 4.2 or later.

## [0.0.2] - 2026-11-20

## [0.0.1] - 2026-11-20

### Added

- Repository started: a skeleton with no code yet.
