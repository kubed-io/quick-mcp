"""What a gateway says about itself, and how its prefix is written into text.

A gateway in front of this server rewrites the structured fields of what it relays
(names, URIs), never the text of a skill or a prompt. So a skill that links to
``skill://lib/guide/SKILL.md`` would send the model to an address the gateway does
not serve. A client names the gateway's prefix, and ``prefixed`` writes it into
those addresses.
"""

from __future__ import annotations

import logging
import re

from kubed.mcp.request import _first_time, request_values

log = logging.getLogger(__name__)

HEADER = "x-mcp-resource-prefix"
QUERY = "prefix"
SCHEMES = ("skill", "prompts", "agents")

# A letter, then letters, digits, +, - or ., ending in +.
_VALID = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*\+")
# The start of an address of a served scheme, not the tail of a longer word or scheme.
_ADDRESS = re.compile(r"(?<![A-Za-z0-9+.-])(?=(?:skill|prompts|agents)://)")


def prefixed(text: str, prefix: str | None) -> str:
    """``text`` with ``prefix`` written before each address of a served scheme.

    With no prefix it is ``text`` itself. An address already carrying a prefix is left
    alone, so the result is idempotent. A prefix that is not valid is a ``ValueError``.
    """
    if prefix is None:
        return text
    if not _VALID.fullmatch(prefix):
        raise ValueError(
            f"{prefix[:64]!r} is not a valid prefix: a prefix is a letter, then "
            "letters, digits, +, - or ., ending in +"
        )
    return _ADDRESS.sub(lambda _: prefix, text)


def prefix() -> str | None:
    """The current request's prefix, or None.

    The header wins over the query parameter; an empty one is no answer. A prefix that
    is not valid is None, with one warning per connection.
    """
    found = request_values()
    if found is None:
        return None
    params, headers = found
    value = ""
    if headers.get(HEADER):
        value = headers[HEADER][0].strip()
    if not value and params.get(QUERY):
        value = params[QUERY][0].replace(" ", "+")
    if not value:
        return None
    if _VALID.fullmatch(value):
        return value
    if _first_time("prefix"):
        log.warning(
            "X-MCP-Resource-Prefix %r is not a valid prefix, so this connection is "
            "served none: a prefix is a letter, then letters, digits, +, - or ., "
            "ending in +",
            value[:64],
        )
    return None
