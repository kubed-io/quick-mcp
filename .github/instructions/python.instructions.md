---
description: 'Python conventions for this repo'
applyTo: 'kubed/**/*.py,scripts/**/*.py,tests/**/*.py'
---
# Python conventions

Applies to every `.py` file here. Cross-cutting review priorities are in
[copilot-instructions.md](../copilot-instructions.md); this file is the
language-specific rules. These are hard conventions — flag violations.

## The supported range is whatever `pyproject.toml` declares

`requires-python` is the floor, and the test matrix sweeps the whole range. A newer syntax or
stdlib feature is a **build break on the oldest supported interpreter**, not a style question.
Raising the floor is its own PR, deliberately, never a side effect.

- A glob ending in `**` matches directories only before 3.13, so in code always write
  `dir/**/*`.
- `X | Y` in an annotation is fine **because every module starts with
  `from __future__ import annotations`**. Used at runtime — `isinstance(x, A | B)`,
  a `TypeAlias` value, a pydantic field evaluated eagerly — it needs care.

## This is a library: the public API is a contract

Servers install this package and import it, so a name that is public is a name somebody depends on.

- Annotate every public parameter and return. `param: str | None = None`, never
  `param: str = None`; a bare `dict` or `**kwargs` is a finding.
- Prefer a `Literal[...]` over `str` when the set of values is closed.
- Renaming, retyping or removing a public name is a breaking change and needs the changelog's
  **BREAKING:** line. Adding an optional argument is safe.
- Entry-point groups are interfaces. Never change a group's contract without versioning it.
- Docstrings on public names are what an integrator reads. Write them for that reader.

## Style

- `from __future__ import annotations` at the top of every new module.
- `pathlib` over `os.path`; f-strings over `%` or `.format`.
- Module docstrings explain *why the module exists*, not a restatement of the names inside it.
- ruff is the linter and the gate, run as `ruff check .` over the whole checkout, tests included.

## Errors

- Never swallow an exception to return a falsy value. A silent empty result is
  indistinguishable from a real one.
- Never put a token, a URL with credentials, or a full request body into an exception message or a
  log line.

## Tests

- `tests/` runs with no network. A test that reaches out is an integration test and is marked
  `@pytest.mark.integration`.
- Test through the real entry point, not only the helper.
- `asyncio_mode = "auto"`, so an `async def test_` needs no decorator.
