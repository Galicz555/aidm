"""Minimal YAML frontmatter reader/writer used by vault tooling.

We don't pull PyYAML — the surface is small (top-level scalars + inline
lists + block lists), and any malformed file should degrade to "no
frontmatter" rather than crash the orchestrator. Identical semantics to
the legacy parser that lived in ``session_resume.py``; that module now
delegates here.
"""

from __future__ import annotations

import re
from typing import Any

_FRONTMATTER_BLOCK_RE = re.compile(r"\A---\n(.*?)\n---\s*", re.DOTALL)
_FRONTMATTER_STRIP_RE = re.compile(r"\A---\n.*?\n---\s*", re.DOTALL)


def parse(text: str) -> dict[str, Any]:
    """Return the parsed frontmatter dict, or ``{}`` if absent/malformed.

    Supports:

    * ``key: value`` — string scalar
    * ``key: [a, b, c]`` — inline list of strings
    * ``key:`` followed by ``- a`` / ``- b`` indented lines — block list
    """

    match = _FRONTMATTER_BLOCK_RE.match(text)
    if not match:
        return {}
    body = match.group(1)
    out: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            current_list_key = None
            continue
        if current_list_key is not None and line.lstrip().startswith("- "):
            value = line.lstrip()[2:].strip()
            existing = out.get(current_list_key)
            if isinstance(existing, list):
                existing.append(_unquote(value))
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if not key:
            continue
        if not rest:
            out[key] = []
            current_list_key = key
            continue
        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            items = [_unquote(piece.strip()) for piece in inner.split(",") if piece.strip()]
            out[key] = items
            continue
        out[key] = _coerce_scalar(_unquote(rest))
    return out


def strip(text: str) -> str:
    """Return ``text`` with any leading frontmatter block removed."""

    return _FRONTMATTER_STRIP_RE.sub("", text, count=1)


def split(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter_dict, body_without_frontmatter)``."""

    return parse(text), strip(text)


def serialize(data: dict[str, Any], body: str) -> str:
    """Compose a markdown document with a YAML-ish frontmatter block.

    Field order is preserved from ``data``'s insertion order. Lists are
    emitted inline (``key: [a, b]``); other values are dumped as plain
    scalars without quoting unless they contain ``:`` or start/end with
    whitespace.
    """

    lines = ["---"]
    for key, value in data.items():
        lines.append(f"{key}: {_format_scalar(value)}")
    lines.append("---")
    body = body.lstrip("\n")
    return "\n".join(lines) + ("\n\n" + body if body else "\n")


# --- internals -----------------------------------------------------------


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _coerce_scalar(value: str) -> Any:
    # Integers carry semantic weight for HEXACO scores and HP fields; we
    # coerce them so callers don't have to reparse. Everything else stays
    # as a string — quoting is the author's responsibility for ambiguous
    # values.
    if value.lstrip("-").isdigit():
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _format_scalar(value: Any) -> str:
    if isinstance(value, list):
        formatted = ", ".join(_format_inline_item(v) for v in value)
        return f"[{formatted}]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _format_inline_item(value: Any) -> str:
    text = "" if value is None else str(value)
    # Inline-list items containing the list delimiter need quoting so the
    # parser can split cleanly. Bracket characters are also load-bearing.
    if any(ch in text for ch in (",", "[", "]")):
        escaped = text.replace('"', '\\"')
        return f'"{escaped}"'
    return text
