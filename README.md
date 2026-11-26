# 🧩 kubed-mcp

**An opinionated platform on top of FastMCP.** How an MCP server serves plugins — skills, agents, commands — and the protocol details that come with it, written once and shared. 🧰

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![FastMCP](https://img.shields.io/badge/FastMCP-4-8a2be2)](https://gofastmcp.com/)

> **Pre-release.** `QuickMCP`, its container, the facade over FastMCP and serving a Claude-format plugin are in.

---

## The whole idea, in one breath

Our MCP servers each serve plugins, and each used to settle the same protocol details by hand. `kubed-mcp` is where they are settled once. A server says *what* it serves, in names and config, and the library does *how*.

```
a server's plugins  ──▶  kubed-mcp  ──▶  FastMCP  ──▶  hosts
```

---

## 🧱 What it will be

- **One way of serving plugins.** Skills, agents and commands, the same way every time, whatever put them there.
- **The protocol, handled.** Modes, directory reads and the [Skills extension](https://modelcontextprotocol.io/extensions/skills/overview), where FastMCP has no support of its own yet.
- **Extension points.** Python entry points, found by name, so an implementation can live in another package.

It is built for [mcp-kb](https://github.com/kubed-io/mcp-kb) and [selenium-flow](https://github.com/kubed-io/selenium-flow), and open to anyone serving plugins over MCP.

---

## 📦 Install

```bash
pip install kubed-mcp
```

Import it as `kubed.mcp`:

```python
import sys

from kubed.mcp import Inject, QuickMCP

mcp = QuickMCP("grid", argv=sys.argv[1:])
mcp.singleton(Grid)
mcp.add_plugin("plugin/")  # its skills, commands and agents

@mcp.tool
async def navigate(url: str, grid: Inject[Grid]) -> dict: ...

mcp.run()  # TRANSPORT, HOST, PORT and LOG_LEVEL, or --transport, --host, --port and --log-level
```

---

## 🤝 Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) is the loop; [AGENTS.md](AGENTS.md) is for the agents working here. MIT licensed.
