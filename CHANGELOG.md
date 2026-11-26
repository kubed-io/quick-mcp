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

## [0.4.1] - 2026-11-26

### Added

- `kubed.mcp.gateway.unprefixed(uri, prefix)` takes a gateway prefix off an address.

### Changed

- `skills/get` and `resources/directory/read` resolve `.` and `..` segments as `resources/read` does, on a direct connection as well as behind a gateway.

## [0.4.0] - 2026-11-26

### Added

- **Config sections:** declare pydantic sections with `@config("key")`, register them with `QuickMCP(sections=...)`, inject each with `Inject[Section]`, and write the whole config's JSON Schema with `config_schema()`.
- **Config from a file, env vars and flags:** `QuickMCP(argv=sys.argv[1:])` loads one YAML file (`CONFIG_FILE` or `--config-file`) and the env var and flag each `Field(env=..., arg=...)` names; a test passes `environ=`.
- Type a config secret `SecretStr`, `SecretBytes` or `Secret[T]` and read it with `get_secret_value()`, and mark a field `NoDecode` to get its env var or flag as raw text.
- `mcp.config_sources` says which source set each config item, and quick-mcp never shows a secret in its own errors, logs or dumps.
- **Components:** subclass `Component` and pass `QuickMCP(components=[...])` to give part of a server its own config section, its objects in the container, and a start and stop around serving.
- Every server has the built-in config sections `serving`, `telemetry` and `delivery`, set by `TRANSPORT`, `HOST`, `PORT`, `LOG_LEVEL`, `TELEMETRY_*` and `DELIVERY_*` or their `--` flags; subclass `ServingSettings`, `TelemetrySettings` or `DeliverySettings` to change one.
- `QuickMCP(hooks={"serving": fn})` builds a built-in's section from the server's own config, in place of its env vars, flags and file keys, and `config_schema(hooked=["serving"])` writes that server's schema.
- Set `mcp.auth` after construction, until the server starts serving.
- At start, telemetry logs one line naming the signals that started and where they go, or `telemetry: off`.
- With `otel` on, a log line written inside a span ends with `trace_id=… span_id=…`, so Loki can link it to its trace.
- An unreachable Pyroscope is logged once, and so is its recovery: the server checks it on the push interval.
- Call metrics include `resources/read`, labelled by `mcp.resource.template`.

### Changed

- **BREAKING:** `QuickMCP(config=...)` now takes the config's values, a mapping or section instances, and the config's classes go in `sections=`. The loaded config is `mcp.config`, injectable as `Config` or by section, and no longer as `type(config)`.
- **BREAKING:** `run()` takes no serving settings, `run(port=...)` is a `TypeError`, and `Serving` is gone: set `serving` in the config (`TRANSPORT`, `--port`, a `serving:` file key) or through `hooks={"serving": ...}`; `run()` applies `serving.log_level` to the root logger.
- **BREAKING:** `Telemetry` is gone: a server reads telemetry's settings with `Inject[TelemetrySettings]`.
- **BREAKING:** `TelemetrySettings.profiles` is a `SecretStr`: read it with `get_secret_value()`.
- **BREAKING:** the config's boot line is logged when the server starts, not when `QuickMCP(...)` is built.
- **BREAKING:** `serving`, `telemetry` and `delivery` are the built-ins' keys and their env vars and flags are taken: a server's own section there must subclass the built-in's class, and its own `PORT`, `HOST`, `TRANSPORT`, `LOG_LEVEL`, `TELEMETRY_*` or `DELIVERY_*` clashes unless it hooks that built-in.
- **BREAKING:** a config's `delivery.listed` is a comma string, `skills,prompts`, not a list.
- **BREAKING:** a section subclass that redefines a field with an `env` or `arg` must restate them in its `Field(...)`, or the config is refused with a `TypeError`.

## [0.3.0] - 2026-11-25

### Added

- **`kubed-mcp[telemetry]` and `TelemetrySettings`:** set `telemetry.profiles` to push CPU and memory profiles to Pyroscope.
- **`telemetry.otel`:** export traces through the standard `OTEL_*` variables, linked to profiles and to log lines.
- **Call metrics with `telemetry.otel`:** each tool call and prompt render, per name, pushed over OTLP or scraped with `OTEL_METRICS_EXPORTER=prometheus`.
- A server's config module can import `kubed.mcp.TelemetrySettings` without loading FastMCP.

## [0.2.0] - 2026-11-25

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
- Each prompt and agent a plugin serves reads as markdown at `prompts://` and `agents://`, with folder indexes.
- `QuickMCP(…, header_prefix="Kb")` reads every `X-MCP-*` header as `Kb-*` too, the server's own name winning.
- `X-MCP-Listed` (or `?listed=`) chooses which of skills, prompts and agents `resources/list` shows, and a config's `delivery` sets the default.
- `X-MCP-Resources: off` (or `?resources=off`) lists read-only `list_resources` and `read_resource` tools for clients that cannot read resources.
- `mcp.binary_pointer(fn)` tells `read_resource` where a binary's link is listed.
- `X-MCP-Via: extension` or `local` (or `?via=`) stops listing what a client's hook or local install already has, while every read still answers.
- `mcp.delivery_default(fn)` sets a client's delivery defaults from its `clientInfo`.
- VS Code clients get `list_resources` and `read_resource` with no header sent.
- `mcp.instructions(fn)` writes a server's instructions per request, from what the client sent.
- `initialize` and `server/discover` results report what the server resolved, in `_meta["io.kubed/delivery"]`.

### Changed

- A `skill://`, `prompts://` or `agents://` address holding an escaped `/` (`%2F`) now fails as a missing one does.
- `resources/read` resolves `.` and `..` segments in a `skill://`, `prompts://` or `agents://` address.
- Under a namespace, prompt and agent names are prefixed with its first segment only.
- A skill's JSON, YAML, TOML and script files read as text, and rewriters and the gateway prefix reach them.
- A plugin is refused when its namespace would sit inside another plugin's skill, or one of its skills would hold another plugin's namespace.
- `prompts://` and `agents://` belong to the plugins: a server's own resource there, or a second plugin's prompt at one address, is refused.
- A plugin is refused beside a server's own `list_resources` or `read_resource` tool, and the other way round: the plugin layer serves both.

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
