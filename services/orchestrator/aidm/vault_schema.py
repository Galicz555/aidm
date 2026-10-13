"""Vault frontmatter schema — one source of truth for lint and auto-fill.

The vault stores Obsidian markdown files with YAML frontmatter. Each file
declares a ``type:`` (npc, character, location, ...) and the type fixes
which fields are required and which enum values are valid. Lint
(``scripts/lint_vault.py``) enforces this for humans; the orchestrator's
write tools log warnings but never reject — voice latency forbids
tool-side validation loops.

Frontmatter keys and enum values are **English** regardless of vault
language. Prose body and free-text values follow the file's ``lang:``
field.

HEXACO scores are flat keys (``hexaco_honesty_humility: 3`` etc.) rather
than a nested map, because the in-house frontmatter parser in
``session_resume.py`` only handles top-level scalars and lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEXACO_DIMENSIONS = (
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
)

HEXACO_KEYS = tuple(f"hexaco_{dim}" for dim in HEXACO_DIMENSIONS)

# Display labels for the HEXACO dimensions — used by the roleplay-prompt
# builder and the lint output. Order matches HEXACO_DIMENSIONS.
HEXACO_LABELS = {
    "honesty_humility": "Honesty-Humility",
    "emotionality": "Emotionality",
    "extraversion": "eXtraversion",
    "agreeableness": "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "openness": "Openness",
}


def hexaco_label_alternatives(dim: str) -> tuple[str, str, str]:
    """Spellings a sheet body may use for ``dim``: ``honesty-humility`` / ``honesty humility`` / raw."""
    return (dim.replace("_", "-"), dim.replace("_", " "), dim)


def hexaco_body_score(text: str, dim: str) -> str | None:
    """The parenthesised score ``text`` gives ``dim`` (``- Openness (4): …``), first match.

    Permissive about layout — the lint only needs the score next to the
    label to catch drift from the frontmatter.
    """
    labels = "|".join(re.escape(alt) for alt in hexaco_label_alternatives(dim))
    match = re.search(rf"(?:{labels})[^\n(]*\((\d+)\)", text, re.IGNORECASE)
    return match.group(1) if match else None


@dataclass(frozen=True)
class TypeSchema:
    """Per-type frontmatter contract.

    ``required`` and ``optional`` are field names that may appear at the
    top level. ``enums`` maps a field name to the set of allowed string
    values (English). ``hexaco_required`` is True for character/npc.
    """

    type_name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    enums: dict[str, frozenset[str]] = field(default_factory=dict)
    hexaco_required: bool = False


_COMMON_REQ = ("type", "lang")

# Status enums kept here so both identity-time tool writes and the
# state writer can reach them without importing each other.
_PC_STATUS_VALUES = frozenset({"active", "wounded", "dead", "missing"})
# ``deceased`` is accepted alongside ``dead`` because the persona prompts
# (dm.*.md) write ``status: deceased`` after an NPC dies.
_NPC_STATUS_VALUES = frozenset({"active", "wounded", "dead", "deceased", "missing"})
_NPC_ATTITUDE_VALUES = frozenset({"hostile", "wary", "neutral", "friendly", "loyal"})
# Story-clock segments in day order (aidm.story_events owns the
# advancement rules; the tuple carries the order, the frozenset the enum).
CLOCK_SEGMENTS = ("morning", "afternoon", "evening", "night")
CLOCK_SEGMENT_VALUES = frozenset(CLOCK_SEGMENTS)

SCHEMAS: dict[str, TypeSchema] = {
    "character": TypeSchema(
        # Identity-only after the sheet split: persona, HEXACO, fixed
        # mechanical ceiling. Per-turn volatile state lives in the
        # sibling ``<slug>.state.md`` (type ``character_state``).
        type_name="character",
        required=(*_COMMON_REQ, "name", "class_role"),
        optional=("aliases", "faction", "hp_max"),
        hexaco_required=True,
    ),
    "character_state": TypeSchema(
        # Sibling state file written every turn by SceneSheetWriter.
        # No persona body, no HEXACO — those live on the identity file.
        type_name="character_state",
        required=("type", "status", "location", "last_updated"),
        optional=("hp_current", "conditions", "inventory", "physical_state"),
        enums={"status": _PC_STATUS_VALUES},
    ),
    "npc": TypeSchema(
        type_name="npc",
        required=(*_COMMON_REQ, "name"),
        optional=("aliases", "faction", "voice_id", "speech_register"),
        hexaco_required=True,
    ),
    "npc_state": TypeSchema(
        type_name="npc_state",
        required=("type", "status", "location", "last_updated", "in_scene"),
        optional=(
            "mood",
            "scene_goal",
            "social_mask",
            "attitude",
            "physical_state",
            "inventory",
        ),
        enums={
            "status": _NPC_STATUS_VALUES,
            "attitude": _NPC_ATTITUDE_VALUES,
        },
    ),
    "location": TypeSchema(
        type_name="location",
        required=(*_COMMON_REQ, "name", "region"),
        optional=("aliases", "controlled_by", "notable_features"),
    ),
    "faction": TypeSchema(
        type_name="faction",
        required=(*_COMMON_REQ, "name", "status"),
        optional=("leader", "goal", "allies", "enemies"),
        enums={"status": frozenset({"active", "dormant", "dissolved"})},
    ),
    "item": TypeSchema(
        type_name="item",
        required=(*_COMMON_REQ, "name", "kind"),
        optional=("owner", "properties"),
    ),
    "arc": TypeSchema(
        type_name="arc",
        required=(*_COMMON_REQ, "name", "status"),
        # ``trial_turns`` / ``outcome``: code-owned state of a First
        # Nightmare arc (aidm.first_nightmare).
        optional=("driving_question", "related", "trial_turns", "outcome"),
        enums={"status": frozenset({"active", "resolved", "dormant"})},
    ),
    "lore": TypeSchema(
        type_name="lore",
        required=(*_COMMON_REQ, "title"),
        optional=("topic", "era"),
    ),
    "rules": TypeSchema(
        type_name="rules",
        required=(*_COMMON_REQ, "title"),
        # ``plan_sources``: vault files a code-driven feature feeds its
        # structured call (rules/spell-voice.md, aidm.first_nightmare).
        optional=("system", "plan_sources"),
    ),
    "campaign": TypeSchema(
        type_name="campaign",
        required=(*_COMMON_REQ, "title", "status"),
        # ``features``: opt-in campaign mechanics, e.g. ``[first-nightmare]``
        # (aidm.first_nightmare).
        optional=("tone", "player_guide", "features"),
        enums={"status": frozenset({"active", "paused", "concluded"})},
    ),
    "run": TypeSchema(
        type_name="run",
        required=("type", "title", "status", "created"),
        optional=("parent_campaign", "summary", "lang"),
        enums={"status": frozenset({"active", "completed", "archived"})},
    ),
    "session_log": TypeSchema(
        type_name="session_log",
        required=("type", "date"),
        optional=("campaign", "characters", "locations", "status"),
        enums={"status": frozenset({"draft", "final"})},
    ),
    "clock": TypeSchema(
        # Run-scoped ``clock.md`` — the deterministic story clock owned
        # by aidm.story_events. Presence of this file opts a run into
        # the event engine. Hidden from the LLM tool surface.
        type_name="clock",
        required=("type", "day", "segment", "turn", "last_updated"),
        # Legacy ``transition_*`` keys from the retired narration hold may
        # linger in old files; unknown keys are ignored, dropped on write.
        optional=("last_advance_turn", "last_hint_turn", "turns_per_segment"),
        enums={"segment": CLOCK_SEGMENT_VALUES},
    ),
    "event": TypeSchema(
        # Authored story event under ``runs/<slug>/events/``. Flat keys
        # only — the in-house frontmatter parser handles top-level
        # scalars and lists. Body carries ``## Directive`` prose plus an
        # optional ``## On fire`` mutation list (see aidm.story_events).
        type_name="event",
        required=(*_COMMON_REQ, "title", "status", "trigger_day", "last_updated"),
        optional=(
            "trigger_segment",
            "subject_npc",
            "trigger_npc_status",
            "trigger_npc_location",
            "trigger_npc_in_scene",
            "trigger_pc_location",
            "chain",
            "chain_step",
            "priority",
            "defer_policy",
            "if_subject_present",
            "expires_after_days",
            "fired_turn",
            "fired_day",
            "fired_segment",
            "narrated",
            "renders",
        ),
        enums={
            "status": frozenset({"pending", "fired", "cancelled", "expired"}),
            "trigger_segment": CLOCK_SEGMENT_VALUES,
            "fired_segment": CLOCK_SEGMENT_VALUES,
            "trigger_npc_status": _NPC_STATUS_VALUES,
            "trigger_npc_in_scene": frozenset({"true", "false"}),
            "defer_policy": frozenset({"hold", "expire", "next_segment"}),
            "if_subject_present": frozenset({"hold", "satisfied"}),
        },
    ),
}


def schema_for(type_name: str) -> TypeSchema | None:
    """Look up a schema by ``type:`` value. Returns None for unknown types."""

    return SCHEMAS.get(type_name)


# Tier 0/1 frontmatter keys that live on the identity sheet. Used by the
# tool-side splitter to route incoming `vault_update_*` content. Anything
# not in either tier set is silently dropped on the floor (treated as
# unknown by the schema; lint will flag it on the file it ends up in).
NPC_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "lang",
        "name",
        "aliases",
        "faction",
        "voice_id",
        "speech_register",
        *HEXACO_KEYS,
    }
)

PC_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "lang",
        "name",
        "class_role",
        "aliases",
        "faction",
        "hp_max",
        *HEXACO_KEYS,
    }
)

NPC_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "status",
        "location",
        "last_updated",
        "in_scene",
        "mood",
        "scene_goal",
        "social_mask",
        "attitude",
        "physical_state",
        "inventory",
    }
)

PC_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "status",
        "location",
        "last_updated",
        "hp_current",
        "conditions",
        "inventory",
        "physical_state",
    }
)
