"""Per-turn scene-sheet injection into the LLM context.

Reads the NPC sheet for every slug in ``SceneRegistry.in_scene`` and the
PC sheet for the active character, fresh from disk every turn, and
splices them into ``LLMContext.messages`` as one tagged system message
just before the latest user message. No caching — the sheet is the
source of truth and the post-turn writer may have rewritten it since the
previous read.

Position matters for llama.cpp's prefix cache: this message mutates
every turn, so everything behind it re-prefills. Bottom-of-prompt keeps
the persona + history prefix stable — only the sheets block and the new
user turn cost prefill. (It sat just after the persona block until
2026-07-09, which invalidated the whole history's KV every turn.)
Recency position also gets the volatile state more attention than
burying it above 12 turns of history.

Pipeline position: between ``ToolSubsetSelector`` and the LLM service,
exactly where the retired ``NPCStateInjector`` sat.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    OutputTransportMessageFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from aidm import first_nightmare
from aidm.scene_state import SceneRegistry
from aidm.session_summary import MAX_OPEN_THREADS
from aidm.story_events import _ANCHOR_RE as _PROPER_NOUN_RE
from aidm.turn_director import TURN_CONTRACT_TAG, DirectorState, build_notes, player_passes
from aidm.vault import Vault, VaultPathError

SCENE_SHEETS_TAG = "<!--aidm:scene-sheets-->"

_LOCALE_HEADERS = {
    "en": (
        "Scene sheets — fresh from vault. ONLY these NPCs/PCs are canon-bound; "
        "any NPC not listed here is NOT in scene — call npc_scene_enter before "
        "voicing them. React from the sheet; do not invent traits, possessions, "
        "or behavior absent from it."
    ),
    "hu": (
        "Jelenet-sheet-ek — frissen a vaultból. CSAK ezek az NPC-k/PC-k "
        "kanonikusak; bármely itt nem listázott NPC NINCS a jelenetben — hívd "
        "az npc_scene_enter-t mielőtt megszólaltatnád. A sheetből reagálj; ne "
        "találj ki olyan vonást, tárgyat vagy viselkedést, ami nincs benne."
    ),
}

_TURN_ONE_ANCHOR = {
    "en": (
        "THIS IS TURN 1. No NPC has been entered. Open from the PC sheet's "
        "`location` only — do NOT introduce any NPC, ambient voice on a "
        "radio, or distant figure. The vault map shows folders, not in-scene "
        "presence. Voicing ANY NPC requires `npc_scene_enter` first."
    ),
    "hu": (
        "EZ AZ 1. KÖR. Egy NPC sincs jelenetben. A PC sheet `location` "
        "mezőjéből indíts — NE vezess be NPC-t, rádión szóló hangot, vagy "
        "távoli alakot. A vault térkép mappákat mutat, nem jelenet-jelenlétet. "
        "Bármely NPC megszólaltatása előbb `npc_scene_enter` hívást követel."
    ),
}


class SceneSheetInjector(FrameProcessor):
    """Splice the active NPC + PC sheets into the LLM context every turn.

    No-op when no NPC is in scene and no PC is set. Re-reads from disk
    on every ``LLMContextFrame`` so a post-turn rewrite (or a human
    hand-edit) lands in the very next prompt.
    """

    def __init__(  # noqa: PLR0913 — per-turn injection has naturally many knobs
        self,
        registry: SceneRegistry,
        locale: str,
        vault: Vault,
        director: DirectorState | None = None,
        agentic_recon: bool = False,
        contract_tail: bool = True,
        minds_client: AsyncOpenAI | None = None,
        minds_model: str = "",
        minds_timeout_s: float = 60.0,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._locale = locale
        self._vault = vault
        # Turn-director channel: measured signals from the previous turn
        # rendered as a tagged system message at the prompt tail. None =
        # channel disabled (some tests construct the injector bare).
        self._director = director
        self._agentic_recon = agentic_recon
        # Contract placement: tail (after the user message) for the local
        # DM, lead (before it, next to the scene sheets) for API DMs —
        # the first all-API marathon showed DeepSeek parroting the tail
        # contract verbatim as its reply on 9 turns while obeying the
        # scene-sheets channel, the same split the 14B showed for world
        # directives (2026-07-11).
        self._contract_tail = contract_tail
        # Per-NPC minds (AIDM_NPC_MINDS_ENABLED): one small completion per
        # in-scene NPC, context = only that NPC's own sheet. None client =
        # feature off.
        self._minds_client = minds_client
        self._minds_model = minds_model
        self._minds_timeout_s = minds_timeout_s

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        # Mutate the context BEFORE forwarding it downstream — the minds
        # call awaits, so a push-first order would race the LLM service.
        sheets = self._collect_sheets()
        intents = await self._collect_intents(sheets, frame.context) if self._minds_client else []
        directives = list(self._director.event_directives) if self._director else []
        nightmare = first_nightmare.dm_block(self._vault)
        _splice(
            frame.context,
            sheets,
            self._locale,
            no_npcs_entered=not self._registry.active(),
            directives=directives,
            beat=self._read_beat(),
            intents=intents,
            # Inside the First Nightmare the waking clock is frozen; its
            # [Time] line would pull the DM back out of the dream.
            time_line=self._director.clock_line if self._director and not nightmare else "",
            open_threads=self._director.open_threads if self._director else None,
            nightmare=nightmare,
        )
        _splice_turn_contract(
            frame.context,
            self._director,
            self._locale,
            tail=self._contract_tail,
            npc_slugs=self._registry.active(),
            pc_slug=self._registry.current_pc_slug,
        )
        if self._agentic_recon:
            _set_recon_tool_choice(frame.context, sheets)
        _log_context_size(frame.context, sheets)
        await self.push_frame(frame, direction)
        await self.push_frame(
            OutputTransportMessageFrame(message=_snapshot_payload(sheets)),
            FrameDirection.DOWNSTREAM,
        )

    async def _collect_intents(
        self, sheets: list[tuple[str, str, str]], context: Any
    ) -> list[tuple[str, str]]:
        """One intent line per in-scene NPC, from only that NPC's own sheet.

        Reuses the already-read blobs (no extra disk reads); capped at 3.
        Fail-open: a call that errors, times out, or returns empty yields
        no intent for that NPC and never blocks the turn.
        """
        client = self._minds_client
        if client is None:
            return []
        npcs = [(slug, blob) for kind, slug, blob in sheets if kind == "npc"][:_MINDS_CAP]
        if not npcs:
            return []
        player, prev_dm = _last_exchange(context)

        async def _one(slug: str, blob: str) -> tuple[str, str] | None:
            prompt = _MIND_PROMPT.format(
                slug=slug, blob=blob, prev_dm=prev_dm or "(scene start)", player=player or "(none)"
            )
            try:
                resp = await client.chat.completions.create(
                    model=self._minds_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0.7,
                    timeout=self._minds_timeout_s,
                )
            except Exception as exc:  # fail-open by design
                logger.debug("npc_mind: {} skipped — {}: {}", slug, type(exc).__name__, exc)
                return None
            text = (resp.choices[0].message.content or "").strip()
            return (slug, text) if text else None

        results = await asyncio.gather(*(_one(slug, blob) for slug, blob in npcs))
        return [r for r in results if r is not None]

    def _collect_sheets(self) -> list[tuple[str, str, str]]:
        """Return ``(kind, slug, sheet_text)`` for every sheet to inject.

        ``kind`` is ``"npc"`` or ``"pc"``. Each entry is the concatenation
        of the identity file (Tier 0/1: persona + HEXACO) and the per-turn
        state file (Tier 2: mood/location/inventory + Recent + Session-
        so-far). State may be missing for an NPC that just entered scene
        for the first time; we still inject identity. Identity missing
        means no sheet at all — silently dropped (logged at debug).
        """
        out: list[tuple[str, str, str]] = []
        for slug in self._registry.active():
            blob = self._read_blob("npc", slug)
            if blob:
                out.append(("npc", slug, blob))
        pc = self._registry.current_pc_slug
        if pc:
            blob = self._read_blob("pc", pc)
            if blob:
                out.append(("pc", pc, blob))
        return out

    def _read_blob(self, kind: str, slug: str) -> str | None:
        subdir = "characters" if kind == "pc" else "npcs"
        identity = self._read_one(f"{subdir}/{slug}.md")
        if not identity:
            return None
        state = self._read_one(f"{subdir}/{slug}.state.md")
        if not state:
            return identity
        return f"{identity}\n\n{state}"

    def _read_beat(self, *, max_chars: int = 1500) -> str:
        """The run's live ``beat.md``, fresh from disk, frontmatter stripped."""
        raw = self._read_one("beat.md")
        if not raw:
            return ""
        # Strip the frontmatter block; the DM needs the prose, not the keys.
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                raw = raw[end + 4 :]
        return raw.strip()[:max_chars]

    def _read_one(self, path: str) -> str | None:
        try:
            raw = self._vault.read_file(path)
        except VaultPathError:
            logger.debug("scene_sheet_injector: no sheet at {}, skipping", path)
            return None
        marker = "\n\n# Referenced by\n"
        if marker in raw:
            raw = raw[: raw.index(marker)]
        return raw.rstrip()


# The directive channel moved here from the turn-contract tail on
# 2026-07-11: the first full 30-turn run showed the 14B ignoring 9/9
# world updates rendered at the prompt tail while demonstrably obeying
# this block ("react from the sheet"). English-only by decision
# (docs/decisions/english-only.md).
_WORLD_EVENTS_HEADER = (
    "World events — these facts have ALREADY HAPPENED in the world and "
    "the player has not been told yet. Your reply THIS TURN must bring "
    "them into the scene, visibly and concretely — do not postpone them "
    "and do not contradict them. They never replace or veto the "
    "player's declared action."
)


# Emergent commitments the summarizer tracks (session_summary.py) — the
# prose summary alone flattened them and the DM contradicted them later.
# English-only by decision (docs/decisions/english-only.md).
_OPEN_THREADS_HEADER = (
    "Open threads — unresolved commitments from earlier in this session "
    "(cause → pending consequence). Stay consistent with them and never "
    "contradict them; pay one off or advance it when the scene allows. "
    "They never replace or veto the player's declared action."
)


# English-only by decision (docs/decisions/english-only.md).
_BEAT_HEADER = (
    "Current beat — live run state, rewritten during play (this is the "
    "situation NOW; it outranks any older mood or opening framing)"
)

# Per-NPC minds (AIDM_NPC_MINDS_ENABLED). English-only by decision.
_MINDS_CAP = 3
_NPC_INTENTS_HEADER = (
    "NPC interiors — each character's private inner state, derived by that "
    "character's own mind from only what they know. This is NOT narration and "
    "NOT dialogue: never state, quote, or paraphrase it outright. Let it drive "
    "subtext — what they conceal, how they hedge, when they push or withhold. "
    "The world's outcome is still yours"
)
_MIND_PROMPT = (
    "You are {slug}, one character inside a roleplay scene. Everything you "
    "know is below — nothing else exists for you.\n\n"
    "Your sheet:\n{blob}\n\n"
    "What you just witnessed (the narrator's last description):\n{prev_dm}\n\n"
    "The player character just said/did:\n{player}\n\n"
    "Report your PRIVATE interior right now — the hidden state a narrator "
    "would render as subtext, never say aloud. Exactly three lines, terse "
    "fragments (not full sentences), no extra text:\n"
    "want: <what you're actually trying to get this turn>\n"
    "hiding: <what you will not say or are concealing; 'nothing' if truly open>\n"
    "read: <your private read on the player right now>"
)


def _splice(  # noqa: PLR0912, PLR0913 — the per-turn block has naturally many inputs
    context: Any,
    sheets: list[tuple[str, str, str]],
    locale: str,
    *,
    no_npcs_entered: bool = False,
    directives: list[str] | None = None,
    beat: str = "",
    intents: list[tuple[str, str]] | None = None,
    time_line: str = "",
    open_threads: list[str] | None = None,
    nightmare: str = "",
) -> None:
    messages = getattr(context, "messages", None)
    if not isinstance(messages, list):
        return

    # Always pop the previous turn's block: the fresh one goes before
    # the *latest* user message, so refreshing in place would leave it
    # stranded mid-history.
    existing_idx = _find_existing(messages)
    if existing_idx is not None:
        messages.pop(existing_idx)

    directives = directives or []
    intents = intents or []
    open_threads = (open_threads or [])[:MAX_OPEN_THREADS]
    if not (sheets or directives or beat or time_line or open_threads or nightmare):
        if existing_idx is not None:
            logger.debug("scene_sheet_injector: cleared (no sheets in scene)")
        return

    header = _LOCALE_HEADERS.get(locale, _LOCALE_HEADERS["en"])
    parts: list[str] = [SCENE_SHEETS_TAG]
    if time_line:
        # The in-fiction time rides here, not the turn contract: the 14B
        # ignores the contract for world info (2026-09-25 replay-002 —
        # day 7 on the clock, one continuous night in the prose).
        parts.append(f"[Time] {time_line}")
    if nightmare:
        # The trial plan rides the channel the DM obeys, like the sheets.
        parts.append(nightmare)
    if directives:
        bullets = "\n".join(f"- {d}" for d in directives)
        parts.append(f"[{_WORLD_EVENTS_HEADER}]\n{bullets}\n\n---")
    if open_threads:
        bullets = "\n".join(f"- {t}" for t in open_threads)
        parts.append(f"[{_OPEN_THREADS_HEADER}]\n{bullets}\n\n---")
    if beat:
        # Fresh from disk every turn (since 2026-07-12): the session-
        # start copy in the static system prompt went stale the moment
        # play or the story engine rewrote beat.md — the DM couldn't
        # even see its own vault_set_beat calls mid-session.
        parts.append(f"[{_BEAT_HEADER}]\n\n{beat}\n\n---")
    if sheets:
        body = "\n\n---\n\n".join(f"### {kind}: {slug}\n\n{sheet}" for kind, slug, sheet in sheets)
        parts.append(f"[{header}]\n\n{body}")
    if intents:
        blocks = []
        for slug, interior in intents:
            indented = "\n".join(f"  {line}" for line in interior.splitlines() if line.strip())
            blocks.append(f"- {slug}:\n{indented}")
        parts.append(f"[{_NPC_INTENTS_HEADER}]\n" + "\n".join(blocks))
    payload = "\n".join(parts)
    if no_npcs_entered and _is_turn_one(messages):
        anchor = _TURN_ONE_ANCHOR.get(locale, _TURN_ONE_ANCHOR["en"])
        payload += f"\n\n---\n\n[{anchor}]"
    new_message = {"role": "system", "content": payload}

    insert_at = _insert_before_last_user(messages)
    messages.insert(insert_at, new_message)
    logger.debug(
        "scene_sheet_injector: inserted at slot {} ({} sheet(s))",
        insert_at,
        len(sheets),
    )


def _is_turn_one(messages: list[Any]) -> bool:
    """No assistant reply has landed yet → this is the opening turn."""
    return not any(_role_of(m) == "assistant" and _content_text(m) for m in messages)


def _find_existing(messages: list[Any]) -> int | None:
    for idx, message in enumerate(messages):
        if _role_of(message) != "system":
            continue
        content = _content_text(message)
        if content.startswith(SCENE_SHEETS_TAG):
            return idx
    return None


def _splice_turn_contract(  # noqa: PLR0913 — contract splice has naturally many inputs
    context: Any,
    director: DirectorState | None,
    locale: str,
    *,
    tail: bool = True,
    npc_slugs: Sequence[str] = (),
    pc_slug: str | None = None,
) -> None:
    """Refresh the turn-director note; popped and re-spliced every turn.

    ``tail=True`` (local DM): appended AFTER the latest user message —
    the maximum-recency slot — because the notes are per-turn directives
    ("resolve the repeated action", "break the loop") that must outrank
    7k tokens of standing rules the 14B has proven it forgets.

    ``tail=False`` (API DMs, public mode): inserted BEFORE the latest
    user message, directly after the scene-sheets block. API models
    treat a trailing system message as content to complete — the first
    all-API marathon had 9 turns whose entire reply was the parroted
    contract — while demonstrably obeying this slot.
    """
    messages = getattr(context, "messages", None)
    if not isinstance(messages, list):
        return
    for idx, message in enumerate(messages):
        if _role_of(message) != "system":
            continue
        if _content_text(message).startswith(TURN_CONTRACT_TAG):
            messages.pop(idx)
            break
    if director is None:
        return
    note = {
        "role": "system",
        "content": build_notes(
            director,
            locale,
            tail=tail,
            npc_slugs=npc_slugs,
            pc_slug=pc_slug,
            player_passes=player_passes(_latest_user_text(messages)),
        ),
    }
    if tail:
        messages.append(note)
    else:
        messages.insert(_insert_before_last_user(messages), note)


def _latest_user_text(messages: list[Any]) -> str:
    """Text of the most recent user message, or ``""``."""
    for message in reversed(messages):
        if _role_of(message) == "user":
            return _content_text(message)
    return ""


def _set_recon_tool_choice(context: Any, sheets: list[tuple[str, str, str]]) -> None:
    """Force one tool call when the player names an entity with no sheet in scene.

    Code decides WHEN — a mid-sentence proper noun in the latest user
    message whose slug isn't one of THIS turn's actually-injected
    identity sheets — and the model decides WHAT to fetch:
    ``tool_choice="required"`` grammar-forces the first completion into
    a vault tool call, and the local provider resets the choice to auto
    after one dispatch so the post-tool narration streams normally.

    "Known" is deliberately the injected sheet *slugs*, not a substring
    search over the rendered block: NPC sheets cross-reference each
    other by name in Bonds/relationship prose (Lyra's sheet naming
    Dora, say), so a text-contains check never fires — every name in
    the vault ends up "mentioned somewhere." A name merely mentioned in
    another NPC's prose still lacks its OWN sheet and is exactly the
    confabulation risk this gate exists for.
    """
    messages = getattr(context, "messages", None)
    setter = getattr(context, "set_tool_choice", None)
    if not isinstance(messages, list) or not callable(setter):
        return
    user_text = ""
    for message in reversed(messages):
        if _role_of(message) == "user":
            user_text = _content_text(message)
            break
    known = {slug.casefold() for _kind, slug, _sheet in sheets}
    unfamiliar = sorted(
        {name for name in _PROPER_NOUN_RE.findall(user_text) if name.casefold() not in known}
    )
    if not unfamiliar:
        setter(NOT_GIVEN)
        return
    setter("required")
    logger.info("agentic recon: forcing tool call (unfamiliar: {})", unfamiliar)


def _insert_before_last_user(messages: list[Any]) -> int:
    """Slot index just before the latest user message.

    The injector runs after ``user_agg`` appended the new user turn, so
    the last user message is this turn's input. Falling back to the end
    (no user message at all) keeps the sheets adjacent to whatever the
    LLM sees last.
    """
    for idx in range(len(messages) - 1, -1, -1):
        if _role_of(messages[idx]) == "user":
            return idx
    return len(messages)


def _log_context_size(context: Any, sheets: list[tuple[str, str, str]]) -> None:
    messages = getattr(context, "messages", None)
    if not isinstance(messages, list):
        return
    total_chars = sum(len(_content_text(m)) for m in messages)
    logger.info(
        "ctx send: msgs={} chars={} sheets={} (slugs={})",
        len(messages),
        total_chars,
        len(sheets),
        [f"{kind}:{slug}" for kind, slug, _ in sheets],
    )


def _snapshot_payload(sheets: list[tuple[str, str, str]]) -> dict[str, Any]:
    return {
        "type": "scene.sheets",
        "sheets": [{"kind": kind, "slug": slug} for kind, slug, _ in sheets],
    }


def _role_of(message: Any) -> str:
    raw = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
    return str(raw or "").lower()


def _content_text(message: Any) -> str:
    raw = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _last_exchange(context: Any) -> tuple[str, str]:
    """Return ``(latest_player_text, previous_dm_text)`` from the context.

    Both are things an in-scene NPC witnessed this turn — the only extra
    material (beyond its own sheet) a per-NPC mind is allowed to see.
    """
    messages = getattr(context, "messages", None)
    if not isinstance(messages, list):
        return "", ""
    player, prev_dm = "", ""
    for message in reversed(messages):
        role = _role_of(message)
        text = _content_text(message)
        if not text:
            continue
        if not player and role == "user":
            player = text
        elif player and not prev_dm and role == "assistant":
            prev_dm = text
            break
    return player, prev_dm
