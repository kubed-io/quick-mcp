# 🧩 kubed-mcp

**An opinionated platform on top of FastMCP.** How an MCP server serves plugins — skills, agents, commands — and the protocol details that come with it, written once and shared. 🧰

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![FastMCP](https://img.shields.io/badge/FastMCP-4-8a2be2)](https://gofastmcp.com/)

> **Pre-release.** This is a skeleton: the repository, its devops and a name on PyPI. There is no code to use yet.

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

Import it as `kubed.mcp`. Today that gets you a package with nothing in it.

---

## 🤝 Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) is the loop; [AGENTS.md](AGENTS.md) is for the agents working here. MIT licensed.
