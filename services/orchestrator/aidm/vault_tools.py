"""Gemini function-calling tools that expose the vault to the DM-LLM.

We deliberately keep the surface tight: list, read, grep, glob. Anything
the LLM cannot derive from these is content it doesn't need. The result
strings are formatted so the model can quote them verbatim if useful.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import OutputTransportMessageFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams

from aidm import frontmatter
from aidm.scene_sheet_writer import _STUB_GLOSS_MARKER, _stamp_in_scene_false, collect_npc_state
from aidm.scene_state import SceneRegistry
from aidm.vault import Vault, VaultPathError
from aidm.vault import slugify as _slugify
from aidm.vault_schema import (
    HEXACO_DIMENSIONS,
    NPC_IDENTITY_FIELDS,
    NPC_STATE_FIELDS,
    PC_IDENTITY_FIELDS,
    PC_STATE_FIELDS,
    hexaco_body_score,
    hexaco_label_alternatives,
)

_VAULT_MAP_HEADER: dict[str, str] = {
    "hu": (
        "Vault térkép (a kampány aktuális fájlrendszere). Használd ezt a "
        "fájlok megkereséséhez ahelyett, hogy `vault_list_dir`-rel "
        "tapogatóznál. A vault gyökérhez relatív útvonalakat add át a "
        "`vault_read_file`-nak (pl. `rules/firearms-and-psionics.md`, "
        "nem `firearms-and-psionics.md`)."
    ),
    "en": (
        "Vault map (the campaign's current file layout). Use this to find "
        "files instead of probing with `vault_list_dir`. Pass paths "
        "relative to the vault root to `vault_read_file` (e.g. "
        "`rules/firearms-and-psionics.md`, not `firearms-and-psionics.md`)."
    ),
}

_CANON_PRELOAD_HEADER: dict[str, str] = {
    "hu": "Kánon előtöltve (innen építkezz, ne találj ki ellentmondót)",
    "en": "Canon pre-loaded (build from this, never contradict it)",
}

_SCENE_TUNING_HEADER: dict[str, str] = {
    "hu": ("Jelenet-hangolás (élő modulátorok — minden fordulón vedd figyelembe)"),
    "en": ("Scene tuning (live modulators — honour these on every turn)"),
}

_TONE_LABEL: dict[str, str] = {
    "hu": "Kampány tónus (`tone.md`)",
    "en": "Campaign tone (`tone.md`)",
}


def build_vault_summary(vault: Vault, *, locale: str = "en", mode: str = "public") -> str:
    """Render the vault map + authoritative canon files as one block.

    Bundled into the persona ``system_instruction`` so the model has the
    map AND the campaign/run frame at every turn without re-reading.
    Removes the cold-start fabrication failure mode (DM inventing an
    opening scene that contradicts ``campaign.md``) and avoids the
    duplicate-system-message warning from the LLM provider.

    Missing files are skipped silently — the function is best-effort.
    """
    sections: list[str] = []
    map_header = _VAULT_MAP_HEADER.get(locale, _VAULT_MAP_HEADER["en"])
    try:
        tree = vault.tree()
    except Exception as exc:  # pragma: no cover
        logger.warning("build_vault_summary: tree failed: {}", exc)
        tree = ""
    if tree:
        tree = _mask_npc_roster(tree)
        sections.append(f"## {map_header}\n\n```\n{tree}\n```")
    canon_header = _CANON_PRELOAD_HEADER.get(locale, _CANON_PRELOAD_HEADER["en"])
    canon_bodies: list[str] = []
    for path in ("campaign.md", "run.md"):
        body = _safe_read(vault, path)
        if body:
            canon_bodies.append(f"### `{path}`\n\n{body.strip()}")
    if canon_bodies:
        sections.append(f"## {canon_header}\n\n" + "\n\n".join(canon_bodies))
    tuning = _build_scene_tuning(vault, locale, mode=mode)
    if tuning:
        sections.append(tuning)
    return "\n\n".join(sections)


def _build_scene_tuning(vault: Vault, locale: str, *, mode: str = "public") -> str:
    """Render the live scene-tuning block (campaign tone + run ``beat.md``).

    Tone resolution mirrors the persona prompt: private mode prefers
    ``tone.private.md`` and falls back to ``tone.md``; public mode
    reads only ``tone.md``. Both files are optional — missing means
    the DM falls back to its default persona register.
    """
    header = _SCENE_TUNING_HEADER.get(locale, _SCENE_TUNING_HEADER["en"])
    tone_label = _TONE_LABEL.get(locale, _TONE_LABEL["en"])
    parts: list[str] = []
    tone_candidates = ["tone.private.md", "tone.md"] if mode == "private" else ["tone.md"]
    tone_body = next(
        (b for f in tone_candidates if (b := _safe_read(vault, f, max_chars=2000))),
        None,
    )
    if tone_body:
        parts.append(f"### {tone_label}\n\n{tone_body.strip()}")
    # beat.md deliberately NOT here (since 2026-07-12): the summary is
    # baked into the static system_instruction once per session, so a
    # beat placed here goes stale the moment play (or the story engine)
    # rewrites it — the DM couldn't even see its own vault_set_beat
    # calls. The SceneSheetInjector re-reads beat.md fresh every turn
    # and splices it with the scene sheets instead.
    if not parts:
        return ""
    return f"## {header}\n\n" + "\n\n".join(parts)


def _mask_npc_roster(tree: str) -> str:
    """Hide NPC filenames from the system-instruction vault map.

    The map is baked once at session start, so showing every NPC by name
    leaks the roster into the model's opening — the failure mode where
    turn 1 voices an NPC that was never `npc_scene_enter`'d. Keep the
    `npcs/` directory entry visible; replace its filename listing with a
    pointer so the model must explicitly call `vault_list_dir("npcs")`
    or `vault_grep` to surface a name. Same for `characters/` — the PC
    sheet is injected separately by `SceneSheetInjector`, the LLM never
    needs the filename here.
    """
    out: list[str] = []
    for line in tree.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("npcs/", "characters/")) and not stripped.rstrip().endswith("/"):
            continue
        out.append(line)
    placeholder = "  (roster hidden — call vault_list_dir before voicing or naming any NPC)"
    inserted_npcs = False
    inserted_chars = False
    final: list[str] = []
    for line in out:
        final.append(line)
        if line.rstrip() == "npcs/" and not inserted_npcs:
            final.append(placeholder)
            inserted_npcs = True
        if line.rstrip() == "characters/" and not inserted_chars:
            final.append("  (active PC sheet is auto-injected; do not invent other PCs)")
            inserted_chars = True
    return "\n".join(final)


def _safe_read(vault: Vault, path: str, *, max_chars: int = 8000) -> str:
    try:
        body = vault.read_file(path)
    except VaultPathError:
        return ""
    if len(body) > max_chars:
        return body[:max_chars] + f"\n\n[... truncated, {len(body) - max_chars} chars]"
    return body


def has_character_sheet(vault: Vault) -> bool:
    """True if the active run has at least one character sheet.

    Drives the session-zero gate: an empty ``characters/`` means the
    pipeline boots in interviewer mode and the tool subset narrows to
    creation-relevant tools. Once a sheet exists, the next turn's tool
    selector restores the play subset.
    """
    try:
        return any(_is_identity_sheet(entry) for entry in vault.list_dir("characters"))
    except VaultPathError:
        return False


def _is_identity_sheet(entry: str) -> bool:
    """True for ``<slug>.md`` identity files, False for ``.state.md`` siblings.

    The sheet writer creates a ``<slug>.state.md`` next to every active
    sheet, so any listing-based discovery MUST exclude state files or it
    breaks the moment the writer runs once (regression 2026-07-09:
    single-PC detection counted 2 files and silently disabled PC
    injection for the rest of the run's life).
    """
    return entry.endswith(".md") and not entry.endswith(".state.md")


def discover_pc_slug(vault: Vault) -> str | None:
    """Return the active run's PC slug, or ``None`` if there is no clear single PC.

    Single-PC runs are the norm (one player, one character) — the
    ``SceneSheetInjector`` needs the slug to inject the PC sheet and
    ``SceneSheetWriter`` needs it to rewrite the PC sheet each turn.
    If the run has 0 or 2+ character sheets we leave the registry's
    ``current_pc_slug`` unset and the PC sheet simply isn't injected.
    """
    try:
        entries = [e for e in vault.list_dir("characters") if _is_identity_sheet(e)]
    except VaultPathError:
        return None
    if len(entries) != 1:
        return None
    return Path(entries[0]).stem


def discover_in_scene_npcs(vault: Vault) -> list[str]:
    """Slugs of NPCs whose state file says ``in_scene: true``.

    The sheet writer stamps ``in_scene`` on every state write and
    ``npc_scene_exit`` / the location-divergence backstop stamp it
    false, so the disk already carries scene presence across sessions
    — but until now nothing read it back. Restoring at pipeline build
    makes a fresh session (or a grace reconnect, which rebuilds the
    registry) continue the scene deterministically instead of hoping
    the DM-LLM re-calls ``npc_scene_enter`` from a resume preamble.
    """
    try:
        entries = vault.list_dir("npcs")
    except VaultPathError:
        return []
    slugs: list[str] = []
    for entry in entries:
        if not _is_identity_sheet(entry):
            continue
        slug = Path(entry).stem
        state = vault.read_state("npc", slug)
        if not state:
            continue
        fm, _ = frontmatter.split(state)
        if fm and str(fm.get("in_scene", "")).strip().lower() == "true":
            slugs.append(slug)
    return sorted(slugs)


def wire_vault_tools(
    context: LLMContext,
    llm: Any,
    vault: Vault,
    *,
    scene_registry: SceneRegistry | None = None,
    locale: str = "en",
) -> set[str]:
    """One-stop wiring: attach the schema to the context and register
    handlers on the LLM service. Tool names live in one dict
    (``_handlers_for``), so adding a tool means editing one function and
    one dict entry.

    Sets the *default* subset on the context. The runtime
    ``ToolSubsetSelector`` (see ``aidm.tool_router``) will narrow or
    widen it per turn based on the user's intent.

    ``scene_registry`` enables the per-session scene-presence tools
    (``npc_scene_enter``, ``npc_scene_exit``) that flip the
    ``in_scene`` set the ``SceneSheetInjector`` reads each turn.
    Mood/goal/mask now live on the NPC sheet itself, rewritten by
    ``SceneSheetWriter`` after each turn — no tool call needed.
    Pass ``None`` to keep the old tool surface without scene tools.

    Returns the handlers' session memory (sheet paths already inlined
    into a grep result) so a text regenerate can roll it back with the
    context that holds those results.
    """
    context.set_tools(default_tools_schema())
    inlined_sheets: set[str] = set()
    for name, handler in _handlers_for(vault, scene_registry, locale, inlined_sheets).items():
        llm.register_function(name, _markup_stripped(name, handler))
    return inlined_sheets


# Heavy tools (expensive descriptions) that are excluded from the
# default play subset. The session-log tool is only shown when the
# end-session signal arrives (HTTP /end), at which point it's the only
# tool on the menu — a forced nudge for the LLM to call it.
_END_SESSION_TOOLS: frozenset[str] = frozenset({"vault_write_session_log"})


# Tools the session-zero interviewer is allowed to call. Scene-setting,
# NPC scene-state, and session-log writes are out of scope until the
# character exists. The reads stay so the interviewer can pull
# campaign.md and the suggested-class lore to brief the player.
_SESSION_ZERO_TOOLS: frozenset[str] = frozenset(
    {
        "vault_list_dir",
        "vault_read_file",
        "vault_grep",
        "vault_glob",
        "vault_update_character",
        "vault_update_npc",
        "vault_set_character",
        "vault_append_note",
    }
)


def session_zero_tools_schema() -> ToolsSchema:
    """The tool subset shown during session-zero (no character yet).

    Drops scene-change, NPC scene-state, and session-log writes — those
    only make sense after a character exists. The interviewer keeps the
    reads, the sheet-write, and the side-panel pointer.
    """
    full = _build_tools_schema()
    return ToolsSchema(
        standard_tools=[f for f in full.standard_tools if f.name in _SESSION_ZERO_TOOLS]
    )


def default_tools_schema() -> ToolsSchema:
    """The subset shown on every play turn — drops session-ending writes.

    The model sees every tool except ``vault_write_session_log``, whose
    description is the longest in the catalog (~280 tokens) and only
    relevant at session close. The end-session HTTP endpoint flips the
    selector into ``end_session_tools_schema`` for the final turn.
    """
    full = _build_tools_schema()
    return ToolsSchema(
        standard_tools=[f for f in full.standard_tools if f.name not in _END_SESSION_TOOLS]
    )


def end_session_tools_schema() -> ToolsSchema:
    """Tool surface for the final turn after the End button is pressed.

    Shrunk to just ``vault_write_session_log`` so the LLM's only useful
    move is to write the log; the persona prompt and the injected
    end-of-session player utterance do the rest.
    """
    full = _build_tools_schema()
    return ToolsSchema(
        standard_tools=[f for f in full.standard_tools if f.name in _END_SESSION_TOOLS]
    )


def _build_tools_schema() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="vault_list_dir",
                description=(
                    "List entries inside a vault directory. Use '.' for the vault root. "
                    "Subdirectories appear with a trailing '/'. Start here if you need to "
                    "discover what's available before reading specific files."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to vault root. Default '.'.",
                    },
                },
                required=[],
            ),
            FunctionSchema(
                name="vault_read_file",
                description=(
                    "Read a single markdown file from the vault. Returns the full content "
                    "(YAML frontmatter + body). Use after vault_list_dir or vault_grep "
                    "tells you which file is relevant."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "File path relative to vault root, e.g. 'npcs/elara.md'.",
                    },
                },
                required=["path"],
            ),
            FunctionSchema(
                name="vault_grep",
                description=(
                    "Case-insensitive substring search across the CONTENT of every "
                    "markdown file in the vault. Returns up to 30 hits as path:line:text. "
                    "Use when the player mentions a name or concept and you need to find "
                    "which file describes it. NOT for filename lookup — if you already "
                    "know the path (e.g. from the vault map), call vault_read_file "
                    "directly. A hit is evidence only for what its line literally says; "
                    "do not generalise from unrelated matches."
                ),
                properties={
                    "pattern": {
                        "type": "string",
                        "description": "Substring or regex to search for. Case-insensitive.",
                    },
                },
                required=["pattern"],
            ),
            FunctionSchema(
                name="vault_glob",
                description=(
                    "Find files whose name matches a glob (e.g. 'elara*.md', '*.md'). "
                    "Useful for fuzzy filename lookup when grep would return too much."
                ),
                properties={
                    "pattern": {
                        "type": "string",
                        "description": "Filename glob pattern.",
                    },
                },
                required=["pattern"],
            ),
            FunctionSchema(
                name="vault_write_session_log",
                description=(
                    "Write a session summary to a new file under sessions/. "
                    "Filename is generated from the current timestamp; you cannot "
                    "overwrite an existing log. Use this when the player signals "
                    "the end of a session ('köszi', 'vége', 'viszlát', 'leállunk', "
                    "or English equivalents). Content should be a 200-400 word "
                    "narrative summary in the SAME LANGUAGE the player has been "
                    "speaking in THIS session (look at the user messages above — "
                    "match their language). Do NOT default to the campaign's "
                    "configured language if the player has been speaking another. "
                    "Covering: who "
                    "was involved, the key events, decisions the player made, NPCs "
                    "encountered, and any open threads to resume later. "
                    "Recommended YAML frontmatter: type: session_log, characters "
                    "(list), locations (list), status: draft. The date field is "
                    "auto-filled server-side from the filename timestamp — do not "
                    "supply it. This vault is a single campaign (its top-level "
                    "state lives in campaign.md at the vault root); the optional "
                    "campaign frontmatter field is for cross-vault provenance only."
                ),
                properties={
                    "content": {
                        "type": "string",
                        "description": "Full markdown content of the session log, including frontmatter.",
                    },
                },
                required=["content"],
            ),
            FunctionSchema(
                name="vault_append_note",
                description=(
                    "Append a discovery to vault/notes/<slug>.md. Use this whenever "
                    "the session reveals something worth remembering across future "
                    "sessions: a rumour about a place, a faction's motive that came "
                    "to light, a new connection between NPCs, an unsolved thread. "
                    "Each call appends a timestamped block — earlier notes are "
                    "preserved. Pick a stable slug (place name, NPC name, concept) "
                    "so later sessions can find the same note. Content should be "
                    "1-3 sentences in the session's language."
                ),
                properties={
                    "slug": {
                        "type": "string",
                        "description": (
                            "Topic identifier; will be sanitized. Examples: 'iron-veil', "
                            "'cult-of-the-grey', 'elara'. Reuse the same slug across "
                            "sessions to accumulate notes on the same topic."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "1-3 sentence note in the session's language.",
                    },
                },
                required=["slug", "content"],
            ),
            FunctionSchema(
                name="vault_update_character",
                description=(
                    "Create or overwrite a character sheet under vault/characters/<slug>.md. "
                    "Use when a player character or important NPC's state changes meaningfully "
                    "(joined the party, took an oath, was wounded, gained an item, died). "
                    "Content is the full markdown for the sheet; required YAML frontmatter "
                    "fields: name, class/role, hp, status, location, last_updated. Optional "
                    "but recommended: campaign (slug from vault_list_dir('campaigns')) so the "
                    "character is grouped with the right campaign in future browse views."
                ),
                properties={
                    "slug": {
                        "type": "string",
                        "description": "Character identifier; will be sanitized to lowercase-with-dashes.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full markdown content of the character sheet.",
                    },
                },
                required=["slug", "content"],
            ),
            FunctionSchema(
                name="vault_update_npc",
                description=(
                    "Create or overwrite an NPC sheet under vault/npcs/<slug>.md. "
                    "NPCs are campaign-level canon (recurring across runs). "
                    "Call this the FIRST TIME you give an NPC a name and they "
                    "will speak or matter — BEFORE rendering their first line "
                    "of dialogue. A named NPC without a sheet collapses into "
                    "narrator voice; a sheet (even minimal: HEXACO + a one-line "
                    "gloss + speech register) is what lets you voice them "
                    "distinctly across turns. Required YAML frontmatter: type: "
                    "npc, lang, name, status (active|wounded|dead|missing), "
                    "location, plus the six hexaco_* fields (1-5). "
                    "`last_seen:` is auto-stamped server-side. Recommended "
                    "optional fields: attitude (hostile|wary|neutral|friendly|"
                    "loyal), speech_register (how they talk in one line), "
                    "physical_state, aliases, faction. Reuse the same slug "
                    "across sessions; the second call overwrites. Do NOT use "
                    "for one-off walk-ons that the player will never address "
                    "by name again — only for NPCs that earn a voice."
                ),
                properties={
                    "slug": {
                        "type": "string",
                        "description": "NPC identifier; will be sanitized to lowercase-with-dashes.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full markdown content of the NPC sheet.",
                    },
                },
                required=["slug", "content"],
            ),
            FunctionSchema(
                name="vault_set_scene",
                description=(
                    "Announce a new scene to the player's UI. Call this once at the moment "
                    "you start describing a new location, encounter, or major shift in the "
                    "narrative (entering a town, opening a tomb, a flashback begins). The "
                    "side-panel updates with the name and short description; the player still "
                    "hears your spoken description as usual. Do NOT call on every turn — only "
                    "when the scene genuinely changes."
                ),
                properties={
                    "name": {
                        "type": "string",
                        "description": "Short scene name in the session language, e.g. 'Vasfátyol Fellegvár'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One- or two-sentence description shown beside the name. Optional.",
                    },
                },
                required=["name"],
            ),
            FunctionSchema(
                name="npc_scene_enter",
                description=(
                    "Mark an NPC as currently 'in scene' so their mood, scene goal "
                    "and social mask are surfaced in your prompt at the start of "
                    "every turn until they leave. Call this when an NPC appears, "
                    "joins a conversation, or otherwise becomes an active speaker. "
                    "Emit a `[state: <slug> mood=...; goal=...; mask=...]` "
                    "directive on the same turn to set the opening mood/goal. The "
                    "slug should match the NPC's vault file stem (e.g. "
                    "'nova-tessen' for npcs/nova-tessen.md)."
                ),
                properties={
                    "slug": {
                        "type": "string",
                        "description": "NPC slug matching vault/npcs/<slug>.md.",
                    },
                },
                required=["slug"],
            ),
            FunctionSchema(
                name="npc_scene_exit",
                description=(
                    "Mark an NPC as no longer in the active scene. Their state is "
                    "kept in session memory (so it carries forward into the closing "
                    "session-log snapshot) but it no longer takes up prompt tokens "
                    "in the per-turn state header. Call when the NPC leaves, falls "
                    "asleep, dies, or the scene cuts away from them."
                ),
                properties={
                    "slug": {
                        "type": "string",
                        "description": "NPC slug to retire from the active list.",
                    },
                },
                required=["slug"],
            ),
            # npc_update_state used to live here as a tool. Retired in
            # favour of the inline ``[state: <slug> mood=...; goal=...;
            # mask=...]`` directive parsed out of the assistant prose by
            # ``NPCStateInlineParser`` — zero LLM round-trip, every-turn
            # refresh affordable. Bench-regress (2026-06-27, baseline) showed
            # the tool channel hit rate at 0-10%; the inline channel
            # holds 100% emission at 30 turns.
            FunctionSchema(
                name="vault_set_beat",
                description=(
                    "Rewrite the current narrative beat (runs/<run>/beat.md). "
                    "The beat is short prose (3-8 lines) the orchestrator "
                    "re-injects into your system instruction every turn — "
                    "use it to lock in the active emotional register: "
                    "tempo (gyors üldözés / lassú felfedezés), tension "
                    "(0=nyugodt, 1=életveszély), closeness (intim / nyilvános), "
                    "humour, cruelty. Call this when the scene's *kind* "
                    "changes (combat starts, a tense negotiation cools "
                    "into camaraderie, exploration tips into horror) — "
                    "NOT on every turn. The previous beat is overwritten. "
                    "Write in the session language."
                ),
                properties={
                    "content": {
                        "type": "string",
                        "description": (
                            "Full markdown body for the beat. Bullet list or "
                            "short prose; no frontmatter required."
                        ),
                    },
                },
                required=["content"],
            ),
            FunctionSchema(
                name="vault_set_character",
                description=(
                    "Tell the player's UI which character sheet to display in the side panel. "
                    "Call this once when the player picks a character at the start of a session, "
                    "or when control shifts to a different PC. The slug must match an existing "
                    "vault/characters/<slug>.md file — use vault_glob('characters/*') first if "
                    "unsure of the exact slug."
                ),
                properties={
                    "slug": {
                        "type": "string",
                        "description": "Character slug matching vault/characters/<slug>.md.",
                    },
                },
                required=["slug"],
            ),
        ]
    )


Handler = Callable[[FunctionCallParams], Any]

# Tool-call markup that leaked into an argument value: qwen38 IQ3_S wrote
# '</content>\n<parameter name="name">…' into vault_set_beat's content
# (2026-09-27 playtest beat.md) — the XML tool format's next parameter.
# English-only by decision (docs/decisions/english-only.md); the model
# answers in the conversation's language either way.
_FAREWELL_INSTRUCTION = (
    "The session is over. Now say goodbye to the player OUT OF CHARACTER in 1-3 short "
    "sentences, in the language of the conversation. Do not continue the story: no "
    "narration, no scene, no new events."
)

_TOOL_MARKUP_RE = re.compile(r"</?(?:parameter|content|function|tool_call)\b[^>]*>")


def _strip_tool_markup(value: str) -> str:
    """``value`` cut at the first leaked tool-call tag."""
    match = _TOOL_MARKUP_RE.search(value)
    return value[: match.start()].rstrip() if match else value


def _markup_stripped(name: str, handler: Handler) -> Handler:
    """``handler`` with every string argument cut at leaked tool-call markup."""

    async def wrapped(params: FunctionCallParams) -> Any:
        cleaned = {
            key: _strip_tool_markup(value) if isinstance(value, str) else value
            for key, value in params.arguments.items()
        }
        for key, value in cleaned.items():
            if value != params.arguments[key]:
                logger.warning(
                    "{}: cut leaked tool-call markup from '{}' ({} -> {} chars)",
                    name,
                    key,
                    len(params.arguments[key]),
                    len(value),
                )
        params.arguments = cleaned
        return await handler(params)

    return wrapped


def _is_authored_identity(vault: Vault, identity_path: str) -> bool:
    """An existing identity sheet that is not a writer-seeded stub (stubs may be filled in)."""
    try:
        existing = vault.read_file(identity_path)
    except VaultPathError:
        return False
    return _STUB_GLOSS_MARKER not in existing


def _handlers_for(  # noqa: PLR0915 — many tool handlers inline is fine
    vault: Vault,
    scene_registry: SceneRegistry | None = None,
    locale: str = "en",
    inlined_sheets: set[str] | None = None,
) -> dict[str, Handler]:
    """Tool-name → handler dispatch. Same ``register_function`` API works
    across every Pipecat LLM backend (Anthropic, OpenAI/OpenRouter, local
    OpenAI-compatible servers in Phase 7).
    """
    # Session-scoped: which NPC/character sheets have already been auto-inlined
    # into a grep result. Re-inlining the same sheet every turn is the long-run
    # token leak — 1.5k chars x 20 turns = 30k chars wasted. Once is enough; the
    # earlier tool result message stays in history.
    if inlined_sheets is None:
        inlined_sheets = set()

    async def _list_dir(params: FunctionCallParams) -> None:
        path = str(params.arguments.get("path", "."))
        result = _try(lambda: "\n".join(vault.list_dir(path)) or "(empty)")
        logger.debug("vault_list_dir({}): {} chars", path, len(result))
        await params.result_callback(result)

    async def _read_file(params: FunctionCallParams) -> None:
        path = str(params.arguments.get("path", ""))
        try:
            result = vault.read_file(path)
        except VaultPathError as exc:
            result = _read_file_recovery(vault, path, str(exc))
        except Exception as exc:
            logger.exception("vault_read_file crashed: {}", exc)
            result = f"error: {type(exc).__name__}"
        logger.debug("vault_read_file({}): {} chars", path, len(result))
        await params.result_callback(result)

    async def _grep(params: FunctionCallParams) -> None:
        pattern = str(params.arguments.get("pattern", ""))
        result = _try(
            lambda: _format_grep(vault.grep(pattern), vault=vault, inlined=inlined_sheets)
        )
        logger.debug("vault_grep({}): {} chars", pattern, len(result))
        await params.result_callback(result)

    async def _glob(params: FunctionCallParams) -> None:
        pattern = str(params.arguments.get("pattern", ""))
        result = _try(lambda: "\n".join(vault.glob(pattern)) or "(no matches)")
        logger.debug("vault_glob({}): {} chars", pattern, len(result))
        await params.result_callback(result)

    async def _write_session_log(params: FunctionCallParams) -> None:
        content = str(params.arguments.get("content", ""))
        # A real session log is typically 200-400 words. Under 80 chars
        # almost certainly means the model's JSON was truncated before
        # reaching us (max_tokens hit mid-string) or the call fired with
        # no body. Log the raw args so post-mortem analysis is possible.
        if len(content) < 80:
            logger.warning(
                "vault_write_session_log: suspiciously short content ({} chars) "
                "— possible JSON truncation. raw args: {}",
                len(content),
                repr(params.arguments),
            )
        # Append a one-line scene-close roster so the next session's
        # resume preamble shows who was in scene at end. Full mood/
        # goal/mask continuity now lives on each NPC's sheet (rewritten
        # turn by turn by ``SceneSheetWriter``), so this is just a
        # pointer — the sheets carry the real continuity.
        if scene_registry is not None and scene_registry.active():
            header = "NPC closing state" if locale != "hu" else "NPC zárállapot"
            roster = ", ".join(scene_registry.active())
            content = f"{content.rstrip()}\n\n## {header}\n\n- in scene at close: {roster}\n"
        result = _try(lambda: f"session log written to {vault.write_session_log(content)}")
        logger.info("vault_write_session_log: {} bytes content", len(content))
        # The log is only on the menu for the /end turn; its result re-runs
        # the LLM, which otherwise read it as "carry on" and shipped a
        # 5872-char story continuation (2026-09-27 playtest).
        if not result.startswith("error"):
            result = f"{result}. {_FAREWELL_INSTRUCTION}"
        await params.result_callback(result)

    async def _append_note(params: FunctionCallParams) -> None:
        slug = str(params.arguments.get("slug", ""))
        content = str(params.arguments.get("content", ""))
        result = _try(lambda: f"note appended to {vault.append_note(slug, content)}")
        logger.info("vault_append_note({}): {} bytes content", slug, len(content))
        await params.result_callback(result)

    async def _update_character(params: FunctionCallParams) -> None:
        slug = str(params.arguments.get("slug", ""))
        content = str(params.arguments.get("content", ""))
        # A character sheet has at least a name, class, and one trait.
        # Under 50 chars indicates truncated or empty JSON args.
        if content and len(content) < 50:
            logger.warning(
                "vault_update_character({}): suspiciously short content ({} chars) "
                "— possible JSON truncation. raw args: {}",
                slug,
                len(content),
                repr(params.arguments),
            )
        clean_slug = _slugify(slug)
        identity_text, state_text = normalize_character_sheet(
            *_split_incoming_sheet(content, kind="pc"), locale=locale
        )
        identity_path = f"characters/{clean_slug}.md" if clean_slug else None
        wrote_identity = False
        if identity_text and identity_path:
            if _would_be_no_op(vault, identity_path, identity_text, "last_updated"):
                logger.info("vault_update_character({}): identity unchanged, skip", slug)
            else:
                _try(lambda: vault.write_character_sheet(slug, identity_text))
                wrote_identity = True
        if state_text:
            _try(lambda: vault.write_character_state(slug, state_text))
        logger.info(
            "vault_update_character({}): id={} state={} ({} bytes total)",
            slug,
            wrote_identity,
            bool(state_text),
            len(content),
        )
        # Mirror the write-happened response shape so the LLM reads it
        # as success even on a no-op (older "… unchanged" wording was
        # parsed as failure and caused 7-call retry loops).
        await params.result_callback(
            f"character sheet written to {identity_path or 'characters/<unknown>.md'}"
        )

    async def _update_npc(params: FunctionCallParams) -> None:
        slug = str(params.arguments.get("slug", ""))
        content = str(params.arguments.get("content", ""))
        if content and len(content) < 50:
            logger.warning(
                "vault_update_npc({}): suspiciously short content ({} chars) "
                "— possible JSON truncation. raw args: {}",
                slug,
                len(content),
                repr(params.arguments),
            )
        clean_slug = _slugify(slug)
        identity_text, state_text = _split_incoming_sheet(content, kind="npc")
        identity_path = f"npcs/{clean_slug}.md" if clean_slug else None
        # HEXACO drift guard runs on identity; the writer never touches it.
        # Observed failure mode: Qwen3-14B rewrote an
        # authored Mara from (agreeableness=4, openness=4) into
        # (agreeableness=4, openness=2) on first encounter, destroying
        # authored canon. Reject ≥ 2 drift; redirect to per-turn state.
        if identity_path and identity_text:
            drift = _hexaco_drift(vault, identity_path, identity_text)
            if drift:
                msg = _hexaco_guard_error(slug, drift)
                logger.info("vault_update_npc({}): rejected — HEXACO drift {}", slug, drift)
                await params.result_callback(msg)
                return
        wrote_identity = False
        canon_kept = False
        if identity_text and identity_path:
            if _would_be_no_op(vault, identity_path, identity_text, "last_updated"):
                logger.info("vault_update_npc({}): identity unchanged, skip", slug)
            elif _is_authored_identity(vault, identity_path):
                # Authored canon is shared by every run. A session-zero
                # "seed the NPCs" pass rewrote all six campaign identity sheets
                # into shorter versions, dropping ~40 lines each (2026-09-27).
                canon_kept = True
                logger.warning(
                    "vault_update_npc({}): identity is authored canon — kept; state only", slug
                )
            else:
                _try(lambda: vault.write_npc_sheet(slug, identity_text))
                wrote_identity = True
        if state_text:
            _try(lambda: vault.write_npc_state(slug, state_text))
        logger.info(
            "vault_update_npc({}): id={} state={} ({} bytes total)",
            slug,
            wrote_identity,
            bool(state_text),
            len(content),
        )
        if canon_kept:
            await params.result_callback(
                f"{identity_path} is authored canon and was NOT changed; "
                + ("per-run state updated." if state_text else "nothing written.")
                + " Keep this NPC's changes in their state (location, mood, goals)."
            )
            return
        await params.result_callback(f"npc sheet written to {identity_path or 'npcs/<unknown>.md'}")

    async def _set_scene(params: FunctionCallParams) -> None:
        name = str(params.arguments.get("name", "")).strip()
        description = str(params.arguments.get("description", "")).strip()
        if not name:
            await params.result_callback("error: name is required")
            return
        await _push_app_message(
            params,
            {"type": "scene.change", "name": name, "description": description},
        )
        # Story-clock input: a scene change is one of the deterministic
        # advancement signals (aidm.story_events consumes at end-of-turn).
        if scene_registry is not None:
            scene_registry.note_scene_change()
        logger.info("vault_set_scene: {}", name)
        await params.result_callback(f"scene set to {name}")

    async def _set_beat(params: FunctionCallParams) -> None:
        content = str(params.arguments.get("content", "")).strip()
        if not content:
            await params.result_callback("error: content is required")
            return
        result = _try(lambda: f"beat written to {vault.write_beat(content)}")
        logger.info("vault_set_beat: {} bytes", len(content))
        # Push a side-channel event so the browser can refresh its
        # beat editor without a poll.
        await _push_app_message(params, {"type": "beat.change", "content": content})
        await params.result_callback(result)

    async def _set_character(params: FunctionCallParams) -> None:
        slug = str(params.arguments.get("slug", "")).strip()
        if not slug:
            await params.result_callback("error: slug is required")
            return
        # Verify the sheet exists before announcing it — otherwise the
        # browser would fetch /api/vault/characters/<slug> and 404, and
        # the side panel would render an empty card with no signal that
        # the persona picked a non-existent slug. Go through the vault's
        # overlay resolver, NOT a raw canon path: ``characters/`` is
        # run-scoped, so the file actually lives under
        # ``runs/<run-slug>/characters/<slug>.md``. The old direct check
        # always missed and the LLM concluded the write had failed even
        # though it had just succeeded.
        try:
            vault.read_file(f"characters/{slug}.md")
        except VaultPathError:
            await params.result_callback(
                f"error: no character sheet at characters/{slug}.md "
                f"({_slug_hint(vault, 'characters', 'character')})"
            )
            return
        # Register the PC for this session too: a PC created in session zero
        # never reached the registry, so the sheet injector, the PC writer
        # patch (time hints, with_pc) and Suggest action (409) all ran
        # without a PC for the whole session (2026-09-27 playtests).
        if scene_registry is not None:
            scene_registry.set_pc(slug)
        await _push_app_message(params, {"type": "character.change", "slug": slug})
        logger.info("vault_set_character: {}", slug)
        await params.result_callback(f"character set to {slug}")

    async def _npc_scene_enter(params: FunctionCallParams) -> None:
        slug = _clean_slug(params.arguments.get("slug"))
        if not slug:
            await params.result_callback("error: slug is required")
            return
        if scene_registry is None:
            await params.result_callback("error: scene registry not available in this session")
            return
        try:
            vault.read_file(f"npcs/{slug}.md")
        except VaultPathError:
            await params.result_callback(
                f"error: no NPC sheet at npcs/{slug}.md ({_npc_slug_hint(vault)}) — "
                "call vault_update_npc first to canonicalize this NPC"
            )
            return
        if slug in scene_registry.in_scene:
            await params.result_callback(f"{slug} is already in scene")
            return
        scene_registry.enter(slug)
        logger.info("npc_scene_enter: {}", slug)
        await _push_app_message(
            params,
            {
                "type": "npc.state",
                "npcs": collect_npc_state(vault, scene_registry.active()),
            },
        )
        await params.result_callback(
            f"{slug} is now in scene — their full sheet will be injected on the "
            "next turn. Voice them from that sheet; the post-turn writer will "
            "update mood/goal/mask for you."
        )

    async def _npc_scene_exit(params: FunctionCallParams) -> None:
        slug = _clean_slug(params.arguments.get("slug"))
        if not slug:
            await params.result_callback("error: slug is required")
            return
        if scene_registry is None:
            await params.result_callback("error: scene registry not available in this session")
            return
        scene_registry.exit(slug)
        _stamp_in_scene_false(vault, slug)
        logger.info("npc_scene_exit: {}", slug)
        await _push_app_message(
            params,
            {
                "type": "npc.state",
                "npcs": collect_npc_state(vault, scene_registry.active()),
            },
        )
        await params.result_callback(f"{slug} left the scene")

    handlers: dict[str, Handler] = {
        "vault_list_dir": _list_dir,
        "vault_read_file": _read_file,
        "vault_grep": _grep,
        "vault_glob": _glob,
        "vault_write_session_log": _write_session_log,
        "vault_append_note": _append_note,
        "vault_update_character": _update_character,
        "vault_update_npc": _update_npc,
        "vault_set_scene": _set_scene,
        "vault_set_beat": _set_beat,
        "vault_set_character": _set_character,
    }
    if scene_registry is not None:
        handlers["npc_scene_enter"] = _npc_scene_enter
        handlers["npc_scene_exit"] = _npc_scene_exit
    return handlers


def _clean_slug(raw: Any) -> str:
    return str(raw or "").strip()


def _slug_hint(vault: Vault, subdir: str, label: str) -> str:
    try:
        entries = vault.list_dir(subdir)
    except VaultPathError:
        entries = []
    slugs = sorted(Path(entry).stem for entry in entries if entry.endswith(".md"))
    if slugs:
        return f"available {label} slugs: " + ", ".join(slugs)
    return f"no {label} sheets exist in this run yet"


def _npc_slug_hint(vault: Vault) -> str:
    return _slug_hint(vault, "npcs", "NPC")


# HEXACO drift threshold per axis. Two-point deltas signal a different
# *person*, not a different *mood*: e.g. agreeableness 4 → 2 is the
# difference between "warm-but-cautious" and "actively suspicious".
# One-point drift is allowed because tabletop characterization
# legitimately wobbles by a point in either direction across sessions
# as the player learns the NPC better.
_HEXACO_DRIFT_TOLERANCE: int = 2

_HEXACO_AXES: tuple[str, ...] = (
    "hexaco_honesty_humility",
    "hexaco_emotionality",
    "hexaco_extraversion",
    "hexaco_agreeableness",
    "hexaco_conscientiousness",
    "hexaco_openness",
)


def _hexaco_drift(vault: Vault, path: str, new_content: str) -> dict[str, tuple[int, int]]:
    """Return a per-axis drift map (old → new) where |Δ| ≥ tolerance.

    Empty dict means "drift within tolerance — accept the write."
    Used by ``_update_npc`` to refuse rewrites that would silently
    replace authored canon with the model's own re-characterization.
    """
    try:
        existing_body = vault.read_file(path)
    except VaultPathError:
        # No existing sheet — nothing to drift from. First-write is fine.
        return {}
    # Strip the backlink footer that read_file appends.
    marker = "\n\n# Referenced by\n"
    if marker in existing_body:
        existing_body = existing_body[: existing_body.index(marker)]
    existing_fm, _ = frontmatter.split(existing_body)
    incoming_fm, _ = frontmatter.split(new_content)
    drift: dict[str, tuple[int, int]] = {}
    for axis in _HEXACO_AXES:
        old_raw = existing_fm.get(axis)
        new_raw = incoming_fm.get(axis)
        if old_raw is None or new_raw is None:
            continue
        try:
            old_v = int(old_raw)
            new_v = int(new_raw)
        except (TypeError, ValueError):
            continue
        if abs(new_v - old_v) >= _HEXACO_DRIFT_TOLERANCE:
            drift[axis] = (old_v, new_v)
    return drift


def _hexaco_guard_error(slug: str, drift: dict[str, tuple[int, int]]) -> str:
    """Render the LLM-facing rejection message.

    The wording is shaped to redirect the model to the inline ``[state:]``
    directive — the architecturally-correct channel for scene-level mood
    shifts — rather than just saying "no". The LLM that just tried to
    rewrite HEXACO usually had a real intent (express the NPC's current
    register), so the error tells it where to put that intent.
    """
    axes_list = ", ".join(
        f"{axis.removeprefix('hexaco_')}: {old}→{new}" for axis, (old, new) in drift.items()
    )
    return (
        f"error: vault_update_npc on '{slug}' rejected — HEXACO drift "
        f"≥ {_HEXACO_DRIFT_TOLERANCE} from existing sheet ({axes_list}). "
        "HEXACO is fixed canon; do not rewrite it to express a current "
        "mood. If you want to express how the NPC is feeling or what "
        f"they want RIGHT NOW, emit `[state: {slug} mood=...; goal=...; "
        "mask=...]` at the top of your next reply — that directive is "
        "designed for exactly this. Read the existing sheet first with "
        f"`vault_read_file('npcs/{slug}.md')` if you need to remember "
        "who they are. Only call `vault_update_npc` again with the "
        "ORIGINAL HEXACO values, or with at most a 1-point shift per "
        "axis to reflect long-term character growth."
    )


def _split_incoming_sheet(content: str, *, kind: str) -> tuple[str | None, str | None]:
    """Route a full-sheet tool payload to identity + state portions.

    Frontmatter keys are bucketed by tier (identity = persona/HEXACO,
    state = mood/location/inventory/etc). The body splits on the
    ``## Recent`` heading: prose above stays with identity, Recent +
    Session-so-far go to state. If a portion ends up empty, returns
    ``None`` for that slot — the caller skips the corresponding write.
    """
    fm, body = frontmatter.split(content)
    if not fm and not body.strip():
        return None, None
    identity_set = PC_IDENTITY_FIELDS if kind == "pc" else NPC_IDENTITY_FIELDS
    state_set = PC_STATE_FIELDS if kind == "pc" else NPC_STATE_FIELDS
    identity_fm: dict[str, Any] = {}
    state_fm: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in fm.items():
        if key in identity_set:
            identity_fm[key] = value
        elif key in state_set:
            state_fm[key] = value
        elif key == "last_seen":
            # Retired field — silently dropped; state's last_updated replaces it.
            continue
        else:
            extra[key] = value
    if identity_fm:
        identity_fm.setdefault("type", "character" if kind == "pc" else "npc")
        identity_fm.update(extra)
    if state_fm:
        state_fm["type"] = "character_state" if kind == "pc" else "npc_state"

    # Body split: identity keeps prose up to ## Recent; state takes Recent + Session.
    marker = "\n## Recent"
    if marker in body:
        idx = body.index(marker)
        identity_body = body[:idx].rstrip() + "\n"
        state_body = body[idx + 1 :].rstrip() + "\n"
    else:
        identity_body = body.rstrip() + "\n" if body.strip() else ""
        state_body = ""

    identity_text = frontmatter.serialize(identity_fm, identity_body) if identity_fm else None
    state_text = (
        frontmatter.serialize(state_fm, state_body or "")
        if state_fm or state_body.strip()
        else None
    )
    if state_text and not state_fm:
        # Body-only state portion needs at least a type stamp to lint.
        state_fm = {"type": "character_state" if kind == "pc" else "npc_state"}
        state_text = frontmatter.serialize(state_fm, state_body)
    return identity_text, state_text


# The sheet's HEXACO heading at level 2 or 3, however the model worded it
# ("## Psychology (HEXACO)", "### HEXACO profile", "## Personality").
_HEXACO_HEADING_RE = re.compile(r"^(#{2,3}) [^\n]*(?:HEXACO|Psychology|Personality)", re.I | re.M)
# A score the model wrote next to the label in its own shape ("4 —", "(4):").
_LEADING_SCORE_RE = re.compile(r"^\(?\d\)?\s*[:\u2014\u2013-]?\s*")


def normalize_character_sheet(
    identity_text: str | None, state_text: str | None, *, locale: str
) -> tuple[str | None, str | None]:
    """Make a ``vault_update_character`` payload lint-valid.

    The session-zero interviewer wrote sheets without ``lang``, with
    ``status: alive`` and a HEXACO block without the ``(N)`` scores the
    lint reads (2026-09-27 playtest 1). Only what is missing or
    off-enum is touched; a valid sheet passes through byte-identical.
    """
    if identity_text:
        fm, body = frontmatter.split(identity_text)
        fixed_body = _with_hexaco_block(body, fm)
        if "lang" not in fm or fixed_body != body:
            fm.setdefault("lang", locale)
            identity_text = frontmatter.serialize(fm, fixed_body)
    if state_text:
        fm, body = frontmatter.split(state_text)
        if str(fm.get("status", "")).strip().lower() == "alive":
            fm["status"] = "active"
            state_text = frontmatter.serialize(fm, body)
    return identity_text, state_text


def _with_hexaco_block(body: str, fm: dict[str, Any]) -> str:
    """``body`` whose HEXACO bullets carry the frontmatter scores as ``(N)``.

    Only the trait bullets change: each is rewritten in place as
    ``- <Label> (N): <the model's own reading>``, so surrounding prose and
    ``###`` subsections survive. Traits the section lacks are appended to
    it; a sheet with no HEXACO section at all gets the section rendered
    like ``_consolidated_identity_body`` (scene_sheet_writer). Unchanged
    when the body already agrees with the frontmatter, or when the
    frontmatter scores are missing / not ints (the lint names that).
    """
    try:
        scores = {dim: int(fm[f"hexaco_{dim}"]) for dim in HEXACO_DIMENSIONS}
    except (KeyError, TypeError, ValueError):
        return body
    if all(hexaco_body_score(body, dim) == str(score) for dim, score in scores.items()):
        return body
    lines = body.split("\n")
    start, end = _hexaco_section_span(lines)
    missing: list[str] = []
    for dim, score in scores.items():
        label = dim.replace("_", "-").title()
        index = next((k for k in range(start, end) if _trait_bullet(dim).match(lines[k])), None)
        if index is None:
            missing.append(f"- {label} ({score})")
            continue
        indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
        reading = _trait_reading([lines[index]], dim)
        lines[index] = (
            f"{indent}- {label} ({score}): {reading}" if reading else f"{indent}- {label} ({score})"
        )
    if start == end == len(lines):  # no HEXACO section on the sheet at all
        rendered = "\n".join(["## Psychology (HEXACO)", "", *missing]) + "\n"
        return f"{body.rstrip()}\n\n{rendered}" if body.strip() else rendered
    if missing:
        insert_at = end
        while insert_at > start and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines[insert_at:insert_at] = missing
    return "\n".join(lines)


def _hexaco_section_span(lines: list[str]) -> tuple[int, int]:
    """Line range under the HEXACO heading, up to the next heading of its level or
    above; ``(len, len)`` when the sheet has none."""
    for k, line in enumerate(lines):
        match = _HEXACO_HEADING_RE.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        for end in range(k + 1, len(lines)):
            hashes = len(lines[end]) - len(lines[end].lstrip("#"))
            if 1 <= hashes <= level and lines[end][hashes : hashes + 1] == " ":
                return k + 1, end
        return k + 1, len(lines)
    return len(lines), len(lines)


def _trait_bullet(dim: str) -> re.Pattern[str]:
    labels = "|".join(re.escape(alt) for alt in hexaco_label_alternatives(dim))
    return re.compile(
        rf"^\s*[-*]\s*(?:\*\*)?(?:{labels})(?:\*\*)?\s*[:\u2014\u2013-]?\s*(.*)$", re.I
    )


def _trait_reading(lines: list[str], dim: str) -> str:
    """The prose after ``dim``'s label on its bullet, any score the model put there dropped."""
    bullet = _trait_bullet(dim)
    for line in lines:
        match = bullet.match(line)
        if match:
            return _LEADING_SCORE_RE.sub("", match.group(1).strip(), count=1).strip()
    return ""


def _would_be_no_op(vault: Vault, path: str, new_content: str, stamped_field: str) -> bool:
    """True when writing ``new_content`` to ``path`` would not change the
    file's meaningful content. Used to suppress redundant LLM-driven
    rewrites that re-send the full sheet into context without changing
    anything (the dominant context-bloat shape from the test run).

    ``stamped_field`` is the auto-stamped frontmatter key the write
    method would inject (e.g. ``last_seen`` for NPCs). It's excluded
    from the comparison because the server overwrites it every time
    and would otherwise force a write on every call.
    """
    try:
        existing = vault.read_file(path)
    except VaultPathError:
        return False
    # read_file appends a "# Referenced by" backlink footer that isn't
    # part of the file on disk — strip before comparing.
    marker = "\n\n# Referenced by\n"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    existing_fm, existing_body = frontmatter.split(existing)
    incoming_fm, incoming_body = frontmatter.split(new_content)
    existing_fm.pop(stamped_field, None)
    incoming_fm.pop(stamped_field, None)
    return existing_fm == incoming_fm and existing_body.strip() == incoming_body.strip()


async def _push_app_message(params: FunctionCallParams, payload: dict[str, Any]) -> None:
    """Send a side-channel event to the browser via the data channel.

    ``params.llm`` is the LLM service, itself a FrameProcessor. Pushing
    an ``OutputTransportMessageFrame`` downstream travels through the
    DataChannelForwarder and TTS (both pass DataFrames they do not
    recognize through unchanged) and reaches the transport, which
    ``SmallWebRTCTransport`` writes to the WebRTC data channel.
    ``LocalAudioTransport`` silently drops the frame, so the same tool
    works in both transports.
    """
    await params.llm.push_frame(
        OutputTransportMessageFrame(message=payload),
        FrameDirection.DOWNSTREAM,
    )


def _try(fn: Callable[[], str]) -> str:
    """Run a vault op and turn errors into LLM-readable strings.

    ``VaultPathError`` is the expected, LLM-recoverable failure (bad path,
    grep escaped the root, empty slug). Any other exception is unexpected —
    we log it at ERROR with traceback so real bugs stay visible, but still
    hand a generic message back to the LLM so the turn keeps moving. This
    matters more in Phase 7 when a smaller local model is likelier to pass
    weird arguments that surface obscure ``OSError`` / ``UnicodeDecodeError``
    cases.
    """
    try:
        return fn()
    except VaultPathError as exc:
        return f"error: {exc}"
    except Exception as exc:
        logger.exception("vault tool crashed: {}", exc)
        return f"error: {type(exc).__name__}"


def _read_file_recovery(vault: Vault, path: str, error: str) -> str:
    """Turn a failed ``read_file`` into the answer it implied.

    The LLM often calls ``read_file`` on a directory (and gets a bare
    "not a file" error) or on a slightly wrong path (and goes hunting
    via list_dir/glob across several turns). Both cases waste a round
    trip we can avoid: list the dir inline, or glob for near matches.
    """
    try:
        entries = vault.list_dir(path)
    except VaultPathError:
        entries = None
    if entries is not None:
        listing = "\n".join(entries) if entries else "(empty)"
        return f"error: {error}\n\n{path!r} is a directory. Contents:\n{listing}"
    # Fuzzy filename lookup: try stem-prefix then substring match.
    stem = path.rstrip("/").rsplit("/", 1)[-1] or path
    candidates: list[str] = []
    for pattern in (f"{stem}*", f"*{stem}*"):
        try:
            candidates = vault.glob(pattern)
        except VaultPathError:
            candidates = []
        if candidates:
            break
    if candidates:
        suggestion = "\n".join(candidates[:5])
        return f"error: {error}\n\nDid you mean:\n{suggestion}"
    return f"error: {error}"


def _format_grep(
    hits: list[Any],
    *,
    vault: Any = None,
    inlined: set[str] | None = None,
) -> str:
    if not hits:
        return "(no matches)"
    # Surface narrative canon (npcs / locations / factions / lore / items /
    # arcs) before rules and meta dirs so the model sees the most likely
    # answer at the top of the result block.
    ordered = sorted(hits, key=_grep_sort_key)
    body = "\n".join(f"{h.path}:{h.line_number}: {h.text}" for h in ordered)
    # NPC / character hits: the bare grep line carries only the name and
    # tempts the model to short-circuit ("got the name, narrating now").
    # Begging the model to follow up with vault_read_file works ~50% on
    # Qwen3-14B. Cheaper to inline the sheet so the next step is moot:
    # whether or not the model reads, the canon is in the context.
    sole_npc_path = _sole_character_sheet_path(ordered)
    if sole_npc_path is not None and vault is not None:
        if inlined is not None and sole_npc_path in inlined:
            body += (
                f"\n\n(sheet for {sole_npc_path} was inlined earlier in this "
                "session — see the prior tool result in your context)"
            )
        else:
            try:
                sheet = vault.read_file(sole_npc_path)
            except Exception:
                sheet = None
            if sheet:
                body += f"\n\n--- inlined sheet ({sole_npc_path}) ---\n{_compact_sheet(sheet)}"
                if inlined is not None:
                    inlined.add(sole_npc_path)
    return body


def _compact_sheet(markdown: str) -> str:
    """Lossless-for-LLM compaction of a character/NPC sheet.

    The on-disk markdown is human-friendly (Obsidian). For the LLM-only
    payload that lands in a tool result, we strip:
    - blank-line padding (one separator is enough)
    - bold ``**...**`` markers around field labels
    - ``## `` header prefixes (they become inline labels)
    - leading bullet ``- `` for HEXACO score lines

    Frontmatter (between ``---`` fences) is preserved verbatim — it's
    already compact YAML and the model relies on its structure.
    """
    lines = markdown.splitlines()
    out: list[str] = []
    in_frontmatter = False
    fm_seen = 0
    prev_blank = False
    for raw in lines:
        line = raw.rstrip()
        if line == "---":
            fm_seen += 1
            in_frontmatter = fm_seen == 1
            out.append(line)
            prev_blank = False
            continue
        if in_frontmatter:
            out.append(line)
            continue
        if not line.strip():
            if prev_blank:
                continue
            out.append("")
            prev_blank = True
            continue
        # Strip leading list bullet for HEXACO score lines.
        if line.lstrip().startswith("- "):
            line = line.lstrip()[2:]
        # Strip markdown header prefix.
        if line.startswith("## "):
            line = line[3:].rstrip(":") + ":"
        elif line.startswith("# "):
            line = line[2:]
        # Strip bold around field labels: **Speech register:** → Speech register:
        line = _BOLD_LABEL_RE.sub(r"\1", line)
        out.append(line)
        prev_blank = False
    return "\n".join(out).strip()


_BOLD_LABEL_RE = re.compile(r"\*\*([^*]+?)\*\*")


def _sole_character_sheet_path(ordered_hits: list[Any]) -> str | None:
    """Return the single npc/character file path if all NPC-class hits
    point at the same file. Returns ``None`` when there are zero or
    multiple distinct NPC/character files in the hits (the model has to
    disambiguate by reading one explicitly).
    """
    npc_paths = {
        str(h.path)
        for h in ordered_hits
        if str(getattr(h, "path", "")).startswith(("npcs/", "characters/"))
    }
    if len(npc_paths) == 1:
        return next(iter(npc_paths))
    return None


_GREP_DIR_PRIORITY: dict[str, int] = {
    "npcs": 0,
    "characters": 0,
    "locations": 1,
    "factions": 1,
    "items": 1,
    "lore": 2,
    "arcs": 2,
    "notes": 3,
    "rules": 4,
    "sessions": 5,
}


def _grep_sort_key(hit: Any) -> tuple[int, str, int]:
    path = str(getattr(hit, "path", ""))
    first = path.split("/", 1)[0]
    return (_GREP_DIR_PRIORITY.get(first, 9), path, int(getattr(hit, "line_number", 0)))
