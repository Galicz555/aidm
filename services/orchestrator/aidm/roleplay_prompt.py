"""Pure string composition: character sheet → LLM persona prompt.

The web client's "Generate prompt" button hits an endpoint that returns
the output of ``build`` for the active character. The result is a
paste-ready system prompt for another LLM (or a fresh Claude session)
to play that character — useful for solo prep, NPC simulation, or
shipping a PC to a different tool.

No LLM call is made here; this is template composition.
"""

from __future__ import annotations

from typing import Any

from aidm import frontmatter
from aidm.vault_schema import HEXACO_DIMENSIONS, HEXACO_LABELS

_HEXACO_BLOCK_HEADERS = (
    "## Psychology (HEXACO)",
    "## Pszichológia",
    "## HEXACO",
)


def build(sheet_markdown: str) -> str:  # noqa: PLR0912 — straight-line template assembly
    """Compose a roleplay/intro prompt for the character described in ``sheet_markdown``.

    Reads frontmatter for name, class/role, status, location, HP, the
    six HEXACO scores, and an optional ``lang``. Splits the body to lift
    the HEXACO descriptions out, so the assembled prompt has HEXACO
    once (under "Personality") instead of twice.
    """

    fm, body = frontmatter.split(sheet_markdown)
    name = str(fm.get("name") or "the character").strip()
    role = str(fm.get("class_role") or "").strip()
    status = str(fm.get("status") or "").strip()
    location = str(fm.get("location") or "").strip()
    lang = str(fm.get("lang") or "en").strip().lower()

    hp = _hp_line(fm)
    conditions = _list_line(fm.get("conditions"))
    inventory = _list_line(fm.get("inventory"))

    hexaco_glosses = _hexaco_glosses(body)

    lines: list[str] = []
    headline = f"You are {name}"
    if role:
        headline += f", {role}"
    lines.append(headline + ".")
    lines.append("")

    lines.append("Personality (HEXACO, 1-5):")
    for dim in HEXACO_DIMENSIONS:
        score = fm.get(f"hexaco_{dim}")
        if score is None:
            continue
        gloss = hexaco_glosses.get(dim, "")
        label = HEXACO_LABELS[dim]
        if gloss:
            lines.append(f"- {label} ({score}): {gloss}")
        else:
            lines.append(f"- {label} ({score})")
    lines.append("")

    state_bits: list[str] = []
    if status:
        state_bits.append(f"status: {status}")
    if location:
        state_bits.append(f"location: {location}")
    if hp:
        state_bits.append(hp)
    if state_bits:
        lines.append("Current state: " + "; ".join(state_bits) + ".")
    if conditions:
        lines.append(f"Conditions: {conditions}.")
    if inventory:
        lines.append(f"Inventory: {inventory}.")
    if state_bits or conditions or inventory:
        lines.append("")

    background = _strip_hexaco_block(body).strip()
    if background:
        lines.append("Background:")
        lines.append(background)
        lines.append("")

    lines.append(
        f"Speak as {name} in first person. Stay consistent with the HEXACO "
        f"profile above. Respond in {lang}."
    )
    return "\n".join(lines).rstrip() + "\n"


# --- internals -----------------------------------------------------------


def _hp_line(fm: dict[str, Any]) -> str:
    current = fm.get("hp_current")
    maximum = fm.get("hp_max")
    if current is None and maximum is None:
        return ""
    if current is not None and maximum is not None:
        return f"HP {current}/{maximum}"
    if current is not None:
        return f"HP {current}"
    return f"HP ?/{maximum}"


def _list_line(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(items)
    return str(value).strip()


def _hexaco_glosses(body: str) -> dict[str, str]:
    """Extract one-line glosses for each HEXACO dimension from the body.

    Tolerant of formatting variations: anchors on the dimension name
    (with underscore or space or dash), reads the rest of the line
    after the parenthesized score.
    """

    glosses: dict[str, str] = {}
    block = _extract_hexaco_block(body)
    if not block:
        return glosses
    for line in block.splitlines():
        stripped = line.strip().lstrip("-").lstrip("*").strip()
        if not stripped:
            continue
        for dim in HEXACO_DIMENSIONS:
            label = HEXACO_LABELS[dim]
            if label.lower() not in stripped.lower():
                continue
            # Expect: "Honesty-Humility (3): gloss text"
            colon_idx = stripped.find(":")
            if colon_idx == -1:
                break
            gloss = stripped[colon_idx + 1 :].strip()
            gloss = gloss.strip("*").strip()
            glosses[dim] = gloss
            break
    return glosses


def _extract_hexaco_block(body: str) -> str:
    """Return the body lines belonging to the HEXACO section, or ``''``."""

    lines = body.splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        if any(header in line for header in _HEXACO_BLOCK_HEADERS):
            start = idx + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def _strip_hexaco_block(body: str) -> str:
    """Return ``body`` with the HEXACO subsection removed."""

    lines = body.splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        if any(header in line for header in _HEXACO_BLOCK_HEADERS):
            start = idx
            break
    if start is None:
        return body
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    kept = lines[:start] + lines[end:]
    return "\n".join(kept).strip()
