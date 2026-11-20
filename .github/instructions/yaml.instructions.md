---
description: 'YAML formatting conventions for this repo'
applyTo: '**/*.yaml,**/*.yml'
---
# YAML conventions

Formatting rules for every YAML file in this repo (CI workflows, dependabot, the zizmor config, …).
Cross-cutting review rules are in `.github/copilot-instructions.md`.

## Formatting
- 2-space indentation, spaces never tabs; consistent throughout the file.
- **Sequences align with their parent key** (block style, not indented two extra
  spaces):
  ```yaml
  fruits:
  - apple
  - banana
  - cherry
  ```
- Lowercase keys; separate words with `_` or `-`.
- Don't quote strings unless they contain special characters or start with a number
  (e.g. a value like `"true"`, `"3.0"`, or one with a `:` needs quotes).
- A `description` over 100 characters is a `|` block wrapped at 100 columns, never one long line.
- Use `#` comments to explain non-obvious sections.

## Review focus
- Flag tabs, inconsistent indentation, and sequences indented under their key.
- Flag needless quoting and non-lowercase keys.
