"""Post-turn JSON-patch update to every in-scene state file.

After each assistant turn the writer asks the LLM for a small structured
patch — current mood/goal/mask/status/location/inventory and a one-line
bullet describing what just happened — and applies it deterministically
to the ``<slug>.state.md`` sibling of each in-scene NPC's identity sheet
plus the active PC's. The persona body, HEXACO, and all immutable
identity fields live on the unchanged identity file and never touch
this code path.

This is the second pass after the inline ``[state: ...]`` machinery (re-
tired 2026-06-28) and the full-sheet rewrite (retired 2026-06-29). The
full-sheet rewrite cost ~600 tokens of output per sheet on Qwen3-14B
Q5 — ~20s/sheet sequential — to change four scalar fields and one
bullet. The structured patch is ~80-150 tokens; same intent, ~5x the
throughput. Identity drift is structurally impossible because the writer
prompt never includes the identity file.

Pipeline position: end-of-turn, triggered by
``LLMContextAssistantTimestampFrame``, after ``aggregator.assistant()``
has finalized the turn in ``context.messages``. Blocks the trigger
frame until all patches are applied so the next user turn's
``SceneSheetInjector`` reads up-to-date state.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from loguru import logger
from openai import AsyncOpenAI
from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTimestampFrame,
    OutputTransportMessageFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from aidm import first_nightmare, frontmatter, story_events
from aidm.advance_judge import (
    judge_falls_asleep,
    judge_repeats,
    judge_scene_ends,
    judge_trial_resolved,
)
from aidm.scene_state import NAME_STOPWORDS, SceneRegistry, npc_named, same_place
from aidm.turn_director import NEAR_DUPLICATE_THRESHOLD, DirectorState, similarity
from aidm.vault import Vault, VaultPathError
from aidm.vault import slugify as _slugify
from aidm.vault_schema import SCHEMAS

_RECENT_HEADER = "## Recent (last 5 events)"
_SESSION_HEADER = "## Session so far"
_RECENT_MAX = 5
# A session_so_far this similar to the on-disk summary is a no-op rewrite.
_SESSION_THROTTLE = 0.9
# Stub identity sheets carry this gloss marker until consolidated.
_STUB_GLOSS_MARKER = "auto-seeded the turn the DM introduced"
# Writer turns featuring a stub NPC before its identity is distilled.
_CONSOLIDATE_AFTER = 5
# NPC statuses that mean "not physically present": never auto-entered.
_GONE_STATUSES = frozenset({"dead", "deceased", "missing"})
_HEXACO_KEYS = (
    "hexaco_honesty_humility",
    "hexaco_emotionality",
    "hexaco_extraversion",
    "hexaco_agreeableness",
    "hexaco_conscientiousness",
    "hexaco_openness",
)

# Patch field whitelists per kind. Anything outside these is silently
# ignored — the LLM occasionally invents keys, and rejecting on unknown
# fields would force a noisy retry over a benign confusion.
_NPC_PATCH_KEYS: frozenset[str] = frozenset(
    {
        "mood",
        "scene_goal",
        "social_mask",
        "attitude",
        "status",
        "location",
        "physical_state",
        "inventory",
        "recent_bullet",
        "session_so_far",
    }
)
_PC_PATCH_KEYS: frozenset[str] = frozenset(
    {
        "status",
        "location",
        "hp_current",
        "conditions",
        "inventory",
        "physical_state",
        "recent_bullet",
        "session_so_far",
        # Clamped, forward-only story-clock hint. Never written to the
        # state file — extracted in rewrite_sheet and routed to
        # aidm.story_events, which owns the clock value.
        "time_advance",
        # Names physically with the PC at the end of the turn. Never
        # written to the state file — extracted in rewrite_sheet and used
        # as a presence signal (scene entry + the story-event hold).
        "with_pc",
    }
)
_TIME_ADVANCE_VALUES: frozenset[str] = frozenset({"next_segment", "next_day"})


_NPC_PATCH_PROMPT: dict[str, str] = {
    "en": (
        "You are updating the per-turn state of ONE character: {subject}. "
        "Return a single JSON object — no prose, no code fences — with ONLY "
        "the keys that should change this turn. Omit a key if it didn't move.\n\n"
        "Current state:\n```yaml\n{state_fm}\n```\n\n"
        "Recent events on this NPC (newest last):\n{recent}\n\n"
        "What happened this turn:\n"
        "Player: {user_text}\n"
        "DM: {assistant_text}\n\n"
        "Allowed keys (all optional):\n"
        "  mood           — short phrase, this turn's emotional read\n"
        "  scene_goal     — short phrase, what they're trying to do RIGHT NOW\n"
        "  social_mask    — short phrase, the face they're showing\n"
        "  attitude       — one of: hostile, wary, neutral, friendly, loyal\n"
        "  status         — one of: active, wounded, dead, deceased, missing\n"
        "  location       — short phrase, where they are now\n"
        "  physical_state — short phrase, wounds/exhaustion/wet/etc.\n"
        "  inventory      — full list of items they carry (replaces previous)\n"
        "  recent_bullet  — ONE short sentence, IN YOUR OWN WORDS, naming the "
        'event (e.g. "Traded three bullets for bread at the river '
        'camp"). NEVER copy or quote the DM narration.\n'
        "  session_so_far — rolling prose summary, ≤150 words\n\n"
        "Rules:\n"
        "- Output JSON ONLY. Start with `{{` and end with `}}`.\n"
        "- `recent_bullet` is REQUIRED on every turn where this NPC acted, "
        "spoke, was spoken to, or was observed. Dialogue counts. An "
        "introduction counts. Learning or revealing a name or fact counts. "
        "A bare `{{}}` is allowed ONLY when the NPC was genuinely absent "
        "from this turn's events.\n"
        "- Do NOT include any field that already matches the current state.\n"
        "- Every field you write is about {subject} and no one else. Other "
        "characters may appear only as what {subject} did to/with/learned "
        "about them. If the turn centered on someone else, record only "
        "{subject}'s part of it.\n"
        "- `recent_bullet`: NOVELTY is mandatory. If this turn just repeats a "
        "prior beat (same fixation, same demand), say so plainly and tersely; "
        "don't pad it into a near-duplicate of an existing recent bullet.\n"
        "- `session_so_far`: include ONLY when the turn meaningfully changed "
        "the situation (new development, shift in tone, decision made). When "
        "you do include it, rewrite the WHOLE summary, absorbing the new turn. "
        "Omit on filler turns."
    ),
    "hu": (
        "EGYETLEN karakter kör-állapotát frissíted: {subject}. Adj vissza "
        "egyetlen JSON objektumot — próza nélkül, code fence nélkül — CSAK "
        "azokkal a kulcsokkal, amik ebben a körben változnak. Hagyd ki azt, "
        "ami nem mozdult.\n\n"
        "Jelenlegi állapot:\n```yaml\n{state_fm}\n```\n\n"
        "Az NPC közelmúltbeli eseményei (legújabb a végén):\n{recent}\n\n"
        "Mi történt ebben a körben:\n"
        "Játékos: {user_text}\n"
        "DM: {assistant_text}\n\n"
        "Engedélyezett kulcsok (mind opcionális):\n"
        "  mood           — rövid kifejezés, az NPC érzelmi olvasata ebben a körben\n"
        "  scene_goal     — rövid kifejezés, mit akar MOST elérni\n"
        "  social_mask    — rövid kifejezés, milyen arcot mutat\n"
        "  attitude       — egy ezek közül: hostile, wary, neutral, friendly, loyal\n"
        "  status         — egy ezek közül: active, wounded, dead, deceased, missing\n"
        "  location       — rövid kifejezés, hol van most\n"
        "  physical_state — rövid kifejezés, sérülés/kimerültség/stb.\n"
        "  inventory      — TELJES lista a nála lévő tárgyakról (felülírja a régit)\n"
        "  recent_bullet  — EGY rövid mondat, SAJÁT szavaiddal, az esemény "
        'megnevezésével (pl. "Három töltényt cserélt kenyérre a folyóparti '
        'tábornál"). SOHA ne másold vagy idézd a DM narrációt.\n'
        "  session_so_far — gördülő prózai összefoglaló, ≤150 szó\n\n"
        "Szabályok:\n"
        "- CSAK JSON. `{{`-vel kezdődik, `}}`-vel végződik.\n"
        "- A `recent_bullet` KÖTELEZŐ minden olyan körben, ahol az NPC "
        "cselekedett, beszélt, hozzá beszéltek, vagy megfigyelték. A "
        "párbeszéd esemény. A bemutatkozás esemény. Egy név vagy tény "
        "kiderülése esemény. Üres `{{}}` CSAK akkor, ha az NPC ténylegesen "
        "nem szerepelt a kör eseményeiben.\n"
        "- NE adj olyan mezőt, ami már egyezik a jelenlegi állapottal.\n"
        "- Minden mező, amit írsz, {subject} karakterről szól, senki másról. "
        "Más karakter csak úgy jelenhet meg, mint akivel {subject} tett "
        "valamit vagy akiről megtudott valamit. Ha a kör másról szólt, csak "
        "{subject} részét rögzítsd.\n"
        "- `recent_bullet`: ÚJ esemény kötelező. Ha a kör csak egy korábbi beat "
        "ismétlése, mondd meg tömören; ne tölts be majdnem duplikátumot.\n"
        "- `session_so_far`: CSAK akkor add meg, ha a kör érdemben változtatott "
        "a helyzeten (új fejlemény, hangulatváltás, döntés). Amikor megadod, "
        "írd át a TELJES összefoglalót, beleépítve a kört. Filler körön hagyd ki."
    ),
}


_PC_PATCH_PROMPT: dict[str, str] = {
    "en": (
        "You are the DM recording what happened to the player character "
        "{subject} — and only to {subject} — after one turn of play. "
        "Return a single JSON object — no prose, no "
        "code fences. Mechanical keys reflect ONLY what the DM narration "
        "clearly adjudicated; the narrative keys (`recent_bullet`, "
        "`session_so_far`) record events even when no mechanics moved.\n\n"
        "Current state:\n```yaml\n{state_fm}\n```\n\n"
        "Recent events (newest last):\n{recent}\n\n"
        "What happened this turn:\n"
        "Player: {user_text}\n"
        "DM: {assistant_text}\n\n"
        "Allowed keys (all optional):\n"
        "  status         — one of: active, wounded, dead, missing\n"
        "  location       — short phrase, where they are now\n"
        "  hp_current     — integer current HP\n"
        "  conditions     — full list of active conditions (replaces previous)\n"
        "  inventory      — full list of items carried (replaces previous)\n"
        "  physical_state — short phrase, wounds/exhaustion/etc.\n"
        "  recent_bullet  — ONE short sentence, IN YOUR OWN WORDS, naming the "
        'event (e.g. "Traded three bullets for bread at the river '
        'camp"). NEVER copy or quote the DM narration.\n'
        "  session_so_far — rolling prose summary, ≤150 words\n"
        "  time_advance   — ONLY if the narration clearly moved in-fiction "
        "time: next_segment (hours passed, e.g. a scene ended or the party "
        "traveled) or next_day (they slept / camped for the night). Omit "
        "otherwise.\n"
        "  with_pc        — list of names of the characters physically with "
        "{subject} at the END of this turn: same place, able to see and talk "
        "directly. Not someone only mentioned, remembered, heard over a radio, "
        "or elsewhere. Include it every turn; [] when {subject} is alone.\n\n"
        "Rules:\n"
        "- Output JSON ONLY. Start with `{{` and end with `}}`.\n"
        "- `recent_bullet` is REQUIRED on every turn something happened: "
        "meeting someone, an introduction, dialogue, learning a name or a "
        "fact, a decision, moving somewhere. A bare `{{}}` is allowed ONLY "
        "for a turn of literal nothing (empty input, pure repetition).\n"
        "- Mechanical keys (status, hp_current, conditions, inventory, "
        "location): include ONLY what the DM narration clearly adjudicated "
        "to change. Do not speculate.\n"
        "- Do NOT include any field that already matches the current state.\n"
        "- `session_so_far`: include ONLY when the turn meaningfully changed "
        "the situation. When you do include it, rewrite the WHOLE summary, "
        "absorbing the new turn. Omit on filler turns."
    ),
    "hu": (
        "DM-ként azt rögzíted, mi történt a(z) {subject} játékos karakterrel "
        "— és CSAK vele — egy kör alatt. Adj vissza egyetlen JSON "
        "objektumot — próza nélkül, code "
        "fence nélkül. A mechanikai kulcsok CSAK azt tükrözik, amit a DM "
        "narráció egyértelműen eldöntött; a narratív kulcsok "
        "(`recent_bullet`, `session_so_far`) akkor is rögzítik az "
        "eseményeket, ha mechanika nem változott.\n\n"
        "Jelenlegi állapot:\n```yaml\n{state_fm}\n```\n\n"
        "Közelmúltbeli események (legújabb a végén):\n{recent}\n\n"
        "Mi történt ebben a körben:\n"
        "Játékos: {user_text}\n"
        "DM: {assistant_text}\n\n"
        "Engedélyezett kulcsok (mind opcionális):\n"
        "  status         — egy ezek közül: active, wounded, dead, missing\n"
        "  location       — rövid kifejezés, hol van most\n"
        "  hp_current     — egész szám aktuális HP\n"
        "  conditions     — TELJES lista az aktív állapotokról (felülírja a régit)\n"
        "  inventory      — TELJES lista a nála lévő tárgyakról (felülírja a régit)\n"
        "  physical_state — rövid kifejezés, sérülés/kimerültség/stb.\n"
        "  recent_bullet  — EGY rövid mondat, SAJÁT szavaiddal, az esemény "
        'megnevezésével (pl. "Három töltényt cserélt kenyérre a folyóparti '
        'tábornál"). SOHA ne másold vagy idézd a DM narrációt.\n'
        "  session_so_far — gördülő prózai összefoglaló, ≤150 szó\n\n"
        "Szabályok:\n"
        "- CSAK JSON. `{{`-vel kezdődik, `}}`-vel végződik.\n"
        "- A `recent_bullet` KÖTELEZŐ minden körben, ahol bármi történt: "
        "találkozás, bemutatkozás, párbeszéd, egy név vagy tény kiderülése, "
        "döntés, helyváltoztatás. Üres `{{}}` CSAK szó szerinti semmi-körre "
        "(üres input, puszta ismétlés).\n"
        "- Mechanikai kulcsok (status, hp_current, conditions, inventory, "
        "location): CSAK azt add meg, amit a DM narráció egyértelműen "
        "eldöntött. Ne találgass.\n"
        "- NE adj olyan mezőt, ami már egyezik a jelenlegi állapottal.\n"
        "- `session_so_far`: CSAK akkor add meg, ha a kör érdemben változtatott "
        "a helyzeten. Amikor megadod, írd át a TELJES összefoglalót, beleépítve "
        "a kört. Filler körön hagyd ki."
    ),
}


_CONSOLIDATE_PROMPT: dict[str, str] = {
    "en": (
        "You are distilling a recurring NPC's identity from several turns "
        "of play. Return a single JSON object — no prose, no code fences — "
        "with EXACTLY these keys:\n"
        "  gloss — ONE sentence, in your own words: who they are\n"
        "  speech_register — ONE line: how they talk (cadence, vocabulary), "
        "optionally with a short example quote\n"
        "  hexaco_honesty_humility, hexaco_emotionality, hexaco_extraversion, "
        "hexaco_agreeableness, hexaco_conscientiousness, hexaco_openness — "
        "integers 1-5 estimated from observed behavior\n\n"
        "NPC name: {name}\n\n"
        "Evidence from play (current state, recent events, running summary):\n"
        "{evidence}\n\n"
        "Latest exchange:\n"
        "Player: {user_text}\n"
        "DM: {assistant_text}\n\n"
        "Output JSON ONLY. Start with `{{` and end with `}}`."
    ),
    "hu": (
        "Egy visszatérő NPC identitását desztillálod több játszott kör "
        "alapján. Adj vissza egyetlen JSON objektumot — próza nélkül, code "
        "fence nélkül — PONTOSAN ezekkel a kulcsokkal:\n"
        "  gloss — EGY mondat, saját szavaiddal: ki ő\n"
        "  speech_register — EGY sor: hogyan beszél (ritmus, szókincs), "
        "opcionálisan egy rövid példamondattal\n"
        "  hexaco_honesty_humility, hexaco_emotionality, hexaco_extraversion, "
        "hexaco_agreeableness, hexaco_conscientiousness, hexaco_openness — "
        "1-5 egész számok a megfigyelt viselkedés alapján\n\n"
        "NPC neve: {name}\n\n"
        "Bizonyíték a játékból (aktuális állapot, események, összefoglaló):\n"
        "{evidence}\n\n"
        "Legutóbbi kör:\n"
        "Játékos: {user_text}\n"
        "DM: {assistant_text}\n\n"
        "CSAK JSON. `{{`-vel kezdődik, `}}`-vel végződik."
    ),
}


def _validate_consolidation(data: Any) -> tuple[str, str, dict[str, int]] | None:
    """Validate a consolidation payload; None rejects the whole attempt.

    Unlike the per-turn patches (silently drop bad fields), identity
    consolidation is all-or-nothing: a half-valid profile would replace
    a working stub with a broken sheet.
    """
    if not isinstance(data, dict):
        return None
    gloss = str(data.get("gloss") or "").strip()
    speech = str(data.get("speech_register") or "").strip()
    if not gloss or not speech or _STUB_GLOSS_MARKER in gloss:
        return None
    hexaco: dict[str, int] = {}
    for key in _HEXACO_KEYS:
        try:
            value = int(data.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not 1 <= value <= 5:
            return None
        hexaco[key] = value
    return gloss, speech, hexaco


def _consolidated_identity_body(gloss: str, speech: str, hexaco: dict[str, int]) -> str:
    """Identity sheet body from a validated profile.

    Same skeleton as ``_stub_identity_sheet`` so `make vault-lint` holds:
    Gloss + Speech register + Psychology section with parenthesised
    scores matching the frontmatter.
    """
    lines = [
        f"**Gloss:** {gloss}",
        "",
        f"**Speech register:** {speech}",
        "",
        "## Psychology (HEXACO)",
        "",
    ]
    for key in _HEXACO_KEYS:
        trait = key.removeprefix("hexaco_")
        lines.append(f"- {trait} ({hexaco[key]}): estimated from observed play")
    return "\n".join(lines) + "\n"


# --- DM-introduced NPC backstop ---------------------------------------
#
# bench-tools (2026-07-09) showed the 14B model reliably *retrieves*
# with vault tools but fails 10/10 at *authoring* a full NPC sheet in
# tool arguments — so a DM-invented NPC like "Name's Lyra" never gets a
# sheet and evaporates next session. The backstop detects explicit name
# introductions in the DM's reply, seeds a minimal schema-valid stub +
# state file, and enters the NPC into the scene registry. The (working)
# per-turn patch pass then fleshes out mood/goal/mask from play.
#
# Simplification: introduction-phrase heuristic, not NER. Narrative naming without
# an introduction ("the vendor, Talwen, waves") is missed — upgrade path
# is a cheap LLM extraction call if play shows it matters.

# Capitalized token (incl. Hungarian accents), 2-24 chars.
_NAME_RE = r"([A-ZÁÉÍÓÖŐÚÜŰ][\w'’-]{1,23})"  # noqa: RUF001 — the model emits typographic apostrophes
_INTRO_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        rf"\b[Nn]ame\s*['’]s\s+{_NAME_RE}",  # "Name's Lyra"  # noqa: RUF001 — the model emits typographic apostrophes
        rf"\b[Mm]y name is\s+{_NAME_RE}",
        rf"\b[Nn]ame is\s+{_NAME_RE}",
        rf"\b[Cc]all me\s+{_NAME_RE}",
        rf"\b[Tt]hey call me\s+{_NAME_RE}",
        rf"\bI\s*['’]m\s+{_NAME_RE}",  # noqa: RUF001 — the model emits typographic apostrophes
        rf"\bI am\s+{_NAME_RE}",
        rf"\b[Aa] nevem\s+{_NAME_RE}",  # hu: "A nevem Vika"
        rf"{_NAME_RE}\s+a nevem\b",  # hu: "Vika a nevem"
        rf"\b[Ss]zólíts(?:atok)?\s+{_NAME_RE}",  # hu: "Szólíts Vikának"
    )
)


def detect_introduced_names(text: str) -> list[str]:
    """Names explicitly introduced in ``text``, deduped, in order."""
    seen: set[str] = set()
    out: list[str] = []
    for pattern in _INTRO_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1).strip("'’-")  # noqa: RUF001 — the model emits typographic apostrophes
            key = name.casefold()
            if key in NAME_STOPWORDS or key in seen:
                continue
            seen.add(key)
            out.append(name)
    return out


# --- recurring unnamed referents ----------------------------------------
#
# 2026-09-25 marathon (replay-002): "the figure" recurred T5-T9 and
# T35-T50 with no sheet, and the DM re-imagined it each time — "too
# tall, too still" at T37, a man in a coat at T46. Every continuity
# mechanism keys on a name, so an unnamed referent the DM keeps using
# is seeded as a stub like a named introduction (see
# ``_seed_introduced_npcs``) once it appears in two consecutive replies.
#
# Simplification: noun whitelist + two consecutive replies + no "of" construction.
# Measured over 23 marathon transcripts: 12 seeds, all real unnamed
# entities. Extend the whitelist when play shows another noun recurring.
_REFERENT_NOUNS = ("figure", "stranger", "creature", "intruder", "shape", "silhouette")
_REFERENT_RE = re.compile(rf"\bthe\s+({'|'.join(_REFERENT_NOUNS)})\b(?!\s+of\b)", re.IGNORECASE)


def detect_unnamed_referents(text: str) -> list[str]:
    """Display names ("The figure") of whitelisted definite referents, deduped, in order."""
    out: list[str] = []
    for match in _REFERENT_RE.finditer(text):
        name = f"The {match.group(1).lower()}"
        if name not in out:
            out.append(name)
    return out


# --- voiced-NPC auto-enter ---------------------------------------------
#
# Both first live loop runs (2026-07-11) showed the same systemic
# failure: the 14B DM voices *known vault NPCs* by name without ever
# calling `npc_scene_enter`. The introduction-seeder above only fires on
# self-introductions ("Name's Lyra") — a DM that already knows the name
# never produces one. The consequence: no sheet injected → the NPC's
# register drifts off-HEXACO and their state is never patched.
#
# This detector distinguishes PRESENCE from MENTION deterministically:
# a name counts as voiced only when it is followed (same clause) by a
# speech/gesture verb, or forms a possessive with a body-language noun
# ("Marta's expression..."). "Marta's brother is sick" or "ask Marta
# tomorrow" do not trigger.
#
# Simplification: clause-window heuristic, not NER. Presence narrated without
# dialogue or body language ("Marta stands in the corner, silent") is
# missed — upgrade path is a small extraction call if the sweep shows
# it matters.

_SPEECH_GESTURE_VERBS = (
    "says",
    "said",
    "asks",
    "asked",
    "replies",
    "replied",
    "answers",
    "answered",
    "murmurs",
    "mutters",
    "whispers",
    "hisses",
    "snaps",
    "adds",
    "warns",
    "calls",
    "sighs",
    "nods",
    "shrugs",
    "frowns",
    "smiles",
    "grins",
    "steps",
    "leans",
    "turns",
    "exhales",
    "swallows",
)
_BODY_NOUNS = (
    "expression",
    "face",
    "eyes",
    "gaze",
    "jaw",
    "voice",
    "tone",
    "breath",
    "fingers",
    "hand",
    "hands",
    "shoulder",
    "shoulders",
    "posture",
    "lips",
    "mouth",
    "brow",
)


def detect_voiced_names(text: str, candidates: Iterable[str]) -> list[str]:
    """Candidate names physically present/speaking in ``text``, in order.

    Presence = name followed by a speech/gesture verb within the same
    clause (≤40 chars, no sentence break), or ``Name's <body noun>``.
    """
    verbs = "|".join(_SPEECH_GESTURE_VERBS)
    nouns = "|".join(_BODY_NOUNS)
    out: list[str] = []
    for name in candidates:
        clean = name.strip()
        if not clean:
            continue
        escaped = re.escape(clean)
        speaking = re.compile(rf"\b{escaped}\b[^.!?\n]{{0,40}}?\b(?:{verbs})\b", re.IGNORECASE)
        embodied = re.compile(
            rf"\b{escaped}[’']s\s+(?:{nouns})\b",  # noqa: RUF001 — typographic apostrophes
            re.IGNORECASE,
        )
        if speaking.search(text) or embodied.search(text):
            out.append(clean)
    return out


# --- addressed-NPC auto-enter -------------------------------------------
#
# 2026-09-25 marathon (replay-003/004): the player escorted Marta across
# the river T10-T16 ("With me, Marta — match my pace") but the DM never
# gave HER a speech verb, so voiced auto-enter never fired, her sheet
# stayed at the shelter, and an authored "Marta left alone" event fired
# against that stale location while she walked beside the player. Being
# spoken TO is presence too.
#
# Simplification: vocative only, inside a quoted span — name opening the utterance
# before a comma/dash, ", Name" closing a sentence, or "with me, Name".
# Plain mentions and possessives never count. Unquoted narration and
# single-quoted speech are missed; widen only if play shows it matters.
_QUOTED_RE = re.compile(r"“([^”]*)”|\"([^\"]*)\"")


def detect_addressed_names(text: str, candidates: Iterable[str]) -> list[str]:
    """Candidate names spoken TO inside a quoted span of ``text``, in order."""
    spans = [curly or straight for curly, straight in _QUOTED_RE.findall(text)]
    out: list[str] = []
    for name in candidates:
        clean = name.strip()
        if not clean or not spans:
            continue
        n = re.escape(clean)
        vocative = re.compile(
            rf"^\s*{n}\s*[,—–-]|,\s*{n}\s*[.!?…]|\bwith me,\s*{n}\b",  # noqa: RUF001 — en dash
            re.IGNORECASE,
        )
        if any(vocative.search(span) for span in spans):
            out.append(clean)
    return out


def _stub_identity_sheet(name: str, locale: str, location: str) -> str:
    """Minimal schema-valid NPC identity sheet for a just-introduced NPC.

    All-3 HEXACO placeholders — the DM should treat them as unknown, and
    a later `vault_update_npc` (or a human edit) replaces the stub. Body
    carries the Psychology section with matching parenthesised scores so
    the stub passes `make vault-lint`.
    """
    fm: dict[str, Any] = {
        "type": "npc",
        "lang": locale,
        "name": name,
        "status": "active",
        "location": location,
        "hexaco_honesty_humility": 3,
        "hexaco_emotionality": 3,
        "hexaco_extraversion": 3,
        "hexaco_agreeableness": 3,
        "hexaco_conscientiousness": 3,
        "hexaco_openness": 3,
    }
    body = (
        f"**Gloss:** {name} — {_STUB_GLOSS_MARKER} them; "
        "nothing is canon beyond the narration so far.\n\n"
        "**Speech register:** not yet established — stay consistent with "
        "the introduction scene.\n\n"
        "## Psychology (HEXACO)\n\n"
        "- honesty_humility (3): unknown — placeholder until play reveals it\n"
        "- emotionality (3): unknown — placeholder until play reveals it\n"
        "- extraversion (3): unknown — placeholder until play reveals it\n"
        "- agreeableness (3): unknown — placeholder until play reveals it\n"
        "- conscientiousness (3): unknown — placeholder until play reveals it\n"
        "- openness (3): unknown — placeholder until play reveals it\n"
    )
    return frontmatter.serialize(fm, body)


def _stub_state_sheet(location: str) -> str:
    """Fresh state file for a seeded NPC, co-located with the PC.

    Location matters: ``_auto_exit_diverged`` evicts NPCs whose location
    differs from the PC's, so a stub seeded at "unknown" would be thrown
    out of the scene on the very turn it entered.
    """
    fm: dict[str, Any] = {
        "type": "npc_state",
        "status": "active",
        "location": location,
        "in_scene": "true",
    }
    body = f"{_RECENT_HEADER}\n\n_(no events yet)_\n\n{_SESSION_HEADER}\n\n_(no summary yet)_\n"
    return frontmatter.serialize(fm, body)


@dataclass
class SheetRewriteOutcome:
    kind: str  # "npc" | "pc"
    slug: str
    written: bool = False
    skipped_reason: str | None = None
    latency_ms: int = 0
    # PC-only story-clock hint extracted from the patch (never written
    # to the state file). None when absent or invalid.
    time_advance: str | None = None
    # State keys whose value actually changed this turn (vs the pre-patch
    # frontmatter). Feeds the director's scene-movement signal.
    changed_keys: tuple[str, ...] = ()
    # This turn's ``recent_bullet`` as the model sent it (before the
    # Recent-block dedupe): the condensed stand-in for the DM reply once
    # it ages out of the verbatim window. None when absent.
    bullet: str | None = None
    # PC-only presence judgment: character names (as the model wrote
    # them) physically with the PC at the end of the turn. Never written.
    with_pc: tuple[str, ...] = ()


@dataclass
class WriterTurnOutcome:
    rewrites: list[SheetRewriteOutcome] = field(default_factory=list)
    total_latency_ms: int = 0


async def rewrite_sheet(  # noqa: PLR0912, PLR0913, PLR0915 — orchestration is naturally wide
    *,
    client: AsyncOpenAI,
    model: str,
    vault: Vault,
    kind: str,
    slug: str,
    locale: str,
    user_text: str,
    assistant_text: str,
    turn: int,
    timeout_s: float,
    max_tokens: int = 700,
    goal_static: bool = False,
) -> SheetRewriteOutcome:
    """Ask the LLM for a JSON patch, apply it to the state file.

    Returns the outcome; never raises. Missing state file → seed an empty
    one before patching. LLM error / bad JSON → log and skip (old state
    survives).
    """
    out = SheetRewriteOutcome(kind=kind, slug=slug)
    started = time.perf_counter()

    old_state = vault.read_state(kind, slug) or _seed_state(kind)
    fm, body = frontmatter.split(old_state)
    if not fm:
        out.skipped_reason = "no_state_frontmatter"
        logger.info("sheet_writer: {} {} SKIPPED — state file missing frontmatter", kind, slug)
        return out

    recent_list = _parse_recent_block(body)
    template = (_PC_PATCH_PROMPT if kind == "pc" else _NPC_PATCH_PROMPT).get(
        locale,
        _PC_PATCH_PROMPT["en"] if kind == "pc" else _NPC_PATCH_PROMPT["en"],
    )
    user_msg = template.format(
        subject=slug,
        state_fm=_render_state_fm(fm),
        recent="\n".join(recent_list) if recent_list else "(none)",
        user_text=user_text.strip() or "(no user message)",
        assistant_text=assistant_text.strip() or "(no DM reply)",
    )
    if goal_static and kind == "npc":
        # Director-measured: this NPC's scene_goal has been static for
        # GOAL_STATIC_STREAK+ turns. The DM-side "change tack" nudge only
        # changes narration — the sheet moves only if THIS call moves it.
        user_msg += (
            "\n- `scene_goal` check: this value has not changed for several "
            "turns while the scene kept developing. Re-derive it from this "
            "turn's events — if the character's objective has genuinely "
            "shifted, update the key instead of carrying the old value."
        )

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
            temperature=0.3,
            timeout=timeout_s,
        )
    except Exception as exc:
        out.skipped_reason = f"llm_error: {type(exc).__name__}: {exc}"
        logger.warning("sheet_writer: {} {} — {}", kind, slug, out.skipped_reason)
        out.latency_ms = int((time.perf_counter() - started) * 1000)
        return out

    raw = _extract_text(completion)
    if not raw:
        logger.info(
            "sheet_writer: {} {} empty completion — {}", kind, slug, _empty_reply_diag(completion)
        )
    patch = _parse_patch(raw)
    if patch is None:
        # One malformed key (observed: a bare `session_so_far: ...` line)
        # must not cost the well-formed ones, e.g. a valid recent_bullet.
        patch = _salvage_string_keys(raw) or None
        if patch is not None:
            logger.info(
                "sheet_writer: {} {} salvaged {} from malformed JSON", kind, slug, sorted(patch)
            )
    if patch is None:
        out.skipped_reason = "bad_json"
        logger.info(
            "sheet_writer: {} {} SKIPPED — bad_json (reply head: {!r})",
            kind,
            slug,
            raw[:200],
        )
        out.latency_ms = int((time.perf_counter() - started) * 1000)
        return out

    cleaned = _clean_patch(patch, kind)
    # The clock hint is not sheet state — pull it out before the patch
    # is applied so it can never land in frontmatter. story_events
    # clamps and applies it after all patches finish.
    hint = cleaned.pop("time_advance", None)
    if isinstance(hint, str) and hint in _TIME_ADVANCE_VALUES:
        out.time_advance = hint
    out.with_pc = tuple(cleaned.pop("with_pc", ()))
    # session_so_far throttle: the model ignores the "only on meaningful
    # change" rule and rewrites a near-identical summary every turn —
    # ~2-3s of decode for zero information. Drop it when it barely moved.
    new_session = cleaned.get("session_so_far")
    if isinstance(new_session, str):
        prior_session = _parse_session_block(body)
        if prior_session and similarity(new_session, prior_session) > _SESSION_THROTTLE:
            logger.info("sheet_writer: {} {} session_so_far throttled (unchanged)", kind, slug)
            cleaned.pop("session_so_far", None)
    if not cleaned:
        out.skipped_reason = "empty_patch"
        logger.info("sheet_writer: {} {} SKIPPED — empty_patch (no-op turn)", kind, slug)
        out.latency_ms = int((time.perf_counter() - started) * 1000)
        # Still a successful turn from the LLM's POV — no writer error.
        return out

    new_fm, new_body, new_recent_bullet = _apply_patch(fm, body, cleaned, turn)
    # An NPC reaching the writer is in scene this turn by definition.
    # Stamp on the same write to avoid a second read/serialize/write
    # on every NPC turn — the migrated vault starts at ``in_scene: false``.
    if kind == "npc":
        new_fm["in_scene"] = "true"
    new_content = frontmatter.serialize(new_fm, new_body)

    try:
        if kind == "pc":
            vault.write_character_state(slug, new_content)
        else:
            vault.write_npc_state(slug, new_content)
    except (VaultPathError, OSError) as exc:
        out.skipped_reason = f"write_error: {type(exc).__name__}: {exc}"
        logger.warning("sheet_writer: {} {} write FAILED — {}", kind, slug, out.skipped_reason)
        out.latency_ms = int((time.perf_counter() - started) * 1000)
        return out

    out.written = True
    # Movement signal: keys whose value genuinely differs from the
    # pre-patch frontmatter. The model routinely re-sends an unchanged
    # location despite the "omit unchanged" rule — an echoed value must
    # not read as movement. recent_bullet/session_so_far are body, not
    # fm keys, so they never appear here (the scene-movement check only
    # inspects scene_goal/location anyway).
    out.changed_keys = tuple(k for k, v in cleaned.items() if fm.get(k) != v)
    out.bullet = str(cleaned.get("recent_bullet") or "").strip() or None
    out.latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "sheet_writer: patched {} {} ({} keys, bullet={}, {}ms)",
        kind,
        slug,
        len(cleaned),
        bool(new_recent_bullet),
        out.latency_ms,
    )
    return out


def _seed_state(kind: str) -> str:
    """A blank state file used when no ``.state.md`` exists yet.

    First-time `npc_scene_enter` for a freshly-canonicalized NPC has no
    state sibling on disk; rather than skip the rewrite, seed defaults
    and let the patch flow populate them.
    """
    type_name = "character_state" if kind == "pc" else "npc_state"
    fm: dict[str, Any] = {
        "type": type_name,
        "status": "active",
        "location": "unknown",
    }
    if kind == "npc":
        fm["in_scene"] = "true"
    body = f"{_RECENT_HEADER}\n\n_(no events yet)_\n\n{_SESSION_HEADER}\n\n_(no summary yet)_\n"
    return frontmatter.serialize(fm, body)


def _render_state_fm(fm: dict[str, Any]) -> str:
    """Compact key: value lines for the prompt context.

    Drops server-owned fields (``type``, ``last_updated``, ``in_scene``)
    that the model has no business setting; keeps the rest in insertion
    order so the prompt mirrors the file on disk.
    """
    skip = {"type", "last_updated", "in_scene"}
    lines: list[str] = []
    for key, value in fm.items():
        if key in skip:
            continue
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "(empty)"


def _parse_recent_block(body: str) -> list[str]:
    """Extract the bullet lines under ``## Recent (last 5 events)``.

    Returns the raw bullet text (without the leading ``- ``) so we can
    rebuild the section consistently. Placeholder lines (italicised
    "_(no events yet)_" etc.) are dropped.
    """
    if _RECENT_HEADER not in body:
        return []
    after = body.split(_RECENT_HEADER, 1)[1]
    chunk = after.split(_SESSION_HEADER, 1)[0]
    bullets: list[str] = []
    for raw in chunk.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        bullets.append(line[2:].strip())
    return bullets


def _parse_session_block(body: str) -> str:
    if _SESSION_HEADER not in body:
        return ""
    return body.split(_SESSION_HEADER, 1)[1].strip()


def _parse_patch(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON parse: strip code fences and surrounding prose, then load.

    Returns ``None`` if no object can be recovered. Empty ``{}`` is a
    valid patch (means "nothing changed").
    """
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    return result


def _clean_patch(patch: dict[str, Any], kind: str) -> dict[str, Any]:  # noqa: PLR0912 — one branch per key family
    """Drop unknown keys, coerce types, validate enums. Silent on bad fields."""
    allowed = _PC_PATCH_KEYS if kind == "pc" else _NPC_PATCH_KEYS
    schema_type = "character_state" if kind == "pc" else "npc_state"
    schema = SCHEMAS[schema_type]
    out: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in allowed:
            continue
        if value is None:
            continue
        if key in ("inventory", "conditions"):
            if isinstance(value, list):
                out[key] = [str(v).strip() for v in value if str(v).strip()]
            continue
        if key == "with_pc":
            if isinstance(value, list):
                out[key] = [v.strip() for v in value if isinstance(v, str) and v.strip()]
            continue
        if key == "hp_current":
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                continue
            continue
        if key in schema.enums:
            text = str(value).strip()
            if text in schema.enums[key]:
                out[key] = text
            continue
        if key in ("recent_bullet", "session_so_far"):
            text = str(value).strip()
            if text:
                out[key] = text
            continue
        text = str(value).strip()
        if text:
            out[key] = text
    return out


def _apply_patch(
    fm: dict[str, Any],
    body: str,
    patch: dict[str, Any],
    turn: int,
) -> tuple[dict[str, Any], str, str | None]:
    """Apply ``patch`` to ``(fm, body)``, return new pair + the bullet (or None).

    Frontmatter scalars overwrite. ``inventory``/``conditions`` lists
    replace. ``recent_bullet`` is prefixed with ``- T{turn}: `` and the
    Recent block is rotated to keep only the last 5. ``session_so_far``
    replaces the whole prose section.
    """
    new_fm: dict[str, Any] = dict(fm)
    recent_bullet: str | None = None
    new_session: str | None = None
    for key, value in patch.items():
        if key == "recent_bullet":
            recent_bullet = str(value)
            continue
        if key == "session_so_far":
            new_session = str(value)
            continue
        new_fm[key] = value

    recent = _parse_recent_block(body)
    if recent_bullet:
        # Dedupe: the model ignores the "novelty is mandatory" rule and
        # re-emits the previous turn's bullet on repetitive turns
        # (observed live: identical T17/T18 entries). A near-duplicate
        # adds no memory — drop it in code.
        stripped = [re.sub(r"^T\d+:\s*", "", b) for b in recent]
        if any(similarity(recent_bullet, old) > NEAR_DUPLICATE_THRESHOLD for old in stripped):
            logger.info("sheet_writer: bullet deduped — near-duplicate of an existing Recent entry")
            recent_bullet = None
        else:
            recent.append(f"T{turn}: {recent_bullet}")
            recent = recent[-_RECENT_MAX:]

    session = new_session if new_session is not None else _parse_session_block(body)

    parts: list[str] = [_RECENT_HEADER, ""]
    if recent:
        parts.extend(f"- {b}" for b in recent)
    else:
        parts.append("_(no events yet)_")
    parts.extend(["", _SESSION_HEADER, ""])
    parts.append(session if session else "_(no summary yet)_")
    new_body = "\n".join(parts) + "\n"
    return new_fm, new_body, recent_bullet


def _extract_text(completion: Any) -> str:
    try:
        return str(completion.choices[0].message.content or "")
    except (AttributeError, IndexError):
        return ""


def _empty_reply_diag(completion: Any) -> str:
    """Why a completion has no content: finish_reason + reasoning presence.

    llama.cpp puts thinking-mode output in ``reasoning_content``; a patch
    that went there (or hit ``length`` mid-thought) arrives as ``''``.
    """
    try:
        choice = completion.choices[0]
    except (AttributeError, IndexError):
        return "no choices"
    reasoning = getattr(getattr(choice, "message", None), "reasoning_content", None)
    return (
        f"finish_reason={getattr(choice, 'finish_reason', None)!r}, "
        f"reasoning_content={'present' if reasoning else 'absent'}"
    )


# A complete ``"key": "string"`` pair, closed before a delimiter.
_STRING_PAIR_RE = re.compile(r'"(\w+)"\s*:\s*("(?:[^"\\\n]|\\.)*")\s*(?=[,}\n])')


def _salvage_string_keys(raw: str) -> dict[str, Any]:
    """Well-formed string-valued pairs from a reply ``json.loads`` rejected.

    Simplification: strings only — a broken list/int key is dropped, not guessed at.
    ``_clean_patch`` still whitelists and validates whatever survives.
    """
    out: dict[str, Any] = {}
    for key, literal in _STRING_PAIR_RE.findall(raw):
        try:
            out[key] = json.loads(literal)
        except json.JSONDecodeError:
            continue
    return out


class SceneSheetWriter(FrameProcessor):
    """End-of-turn processor that patches every in-scene state file.

    Blocks ``LLMContextAssistantTimestampFrame`` until every patch
    completes so the next user turn's ``SceneSheetInjector`` reads
    fresh state. Worst-case latency: ``n_sheets * per_sheet_timeout`` —
    but patch output is ~5x smaller than the prior full-sheet rewrite,
    so the per-sheet cost is much lower in practice.
    """

    def __init__(  # noqa: PLR0913 — wiring needs all 8 deps
        self,
        *,
        context: LLMContext,
        registry: SceneRegistry,
        vault: Vault,
        client: AsyncOpenAI,
        model: str,
        locale: str,
        timeout_s: float,
        emitter: FrameProcessor,
        director: DirectorState | None = None,
        verbatim_replies: int = 0,
        advance_judge: bool = False,
        scene_end_judge: bool = False,
        nightmare: bool = False,
    ) -> None:
        super().__init__()
        self._context = context
        # DM replies older than the last ``verbatim_replies`` are swapped
        # for their turn's writer bullets (0 = off). Keyed by reply text.
        self._verbatim_replies = verbatim_replies
        # Content-repetition verdict on this turn's reply (advance_judge),
        # computed alongside the patches; feeds director.reply_stalled.
        self._advance_judge = advance_judge
        # Did this turn leave/arrive somewhere or skip time (judge_scene_ends)?
        # Yes = a scene change for the story clock; the DM rarely calls
        # vault_set_scene, so without this the clock froze.
        self._scene_end_judge = scene_end_judge
        # First Nightmare judges + plan (aidm.first_nightmare); the campaign
        # still has to opt in with ``features: [first-nightmare]``.
        self._nightmare = nightmare
        self._digests: dict[str, str] = {}
        self._registry = registry
        self._vault = vault
        self._client = client
        self._model = model
        self._locale = locale
        self._timeout_s = timeout_s
        # Turn-director channel: this writer is the measuring end
        # (repetition, repeated input, goal fixation); the injector
        # renders the findings next turn. None = channel disabled.
        self._director = director
        # Consolidation bookkeeping: writer turns each stub NPC has seen.
        self._stub_turns: dict[str, int] = {}
        # Side-band events must be pushed from a processor upstream of
        # TextSink / transport.output() — the writer is the last
        # processor in the chain, so its own push_frame never reaches
        # the SSE queue or the WebRTC data channel. The LLM service is
        # a safe upstream emitter in both pipelines.
        self._emitter = emitter
        self._turn: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, LLMContextAssistantTimestampFrame
        ):
            try:
                await self._run()
            except Exception as exc:
                logger.warning("sheet_writer crashed end-of-turn: {}: {}", type(exc).__name__, exc)
            finally:
                # Every exit, including a barge-in's CancelledError (not an
                # Exception — it re-raises after this): the web client gates
                # input on turn.status: idle and would otherwise wait forever.
                await self._emit_phase("idle")

        await self.push_frame(frame, direction)

    def snapshot(self) -> object:
        """Per-session bookkeeping a text regenerate rolls back (``aidm.turn_rollback``)."""
        return (self._turn, dict(self._digests), dict(self._stub_turns))

    def restore(self, state: object) -> None:
        turn, digests, stub_turns = cast("tuple[int, dict[str, str], dict[str, int]]", state)
        self._turn, self._digests, self._stub_turns = turn, dict(digests), dict(stub_turns)

    async def _run(self) -> WriterTurnOutcome:  # noqa: PLR0915 — end-of-turn orchestration is naturally wide
        outcome = WriterTurnOutcome()
        # Clear the directive channel up front: every exit path from this
        # method must leave no stale directive behind, or a skipped turn
        # (empty reply, no targets) would re-render turn N's world update
        # in turn N+2. The clock line is state, not a directive — it stays,
        # so a skipped turn's prompt still carries its [Time] line.
        if self._director is not None:
            self._director.event_directives = []
            self._director.reply_stalled = False
        exchanges = _extract_recent_exchanges(self._context.messages, count=2)
        user_text, assistant_text = exchanges[0] if exchanges else ("", "")
        prev_user, prev_assistant = exchanges[1] if len(exchanges) > 1 else ("", "")
        if self._director is not None:
            self._director.observe_turn(
                user_text=user_text,
                prev_user_text=prev_user,
                assistant_text=assistant_text,
                prev_assistant_text=prev_assistant,
            )
        if not assistant_text:
            # No DM reply landed this turn (process_frame still emits idle).
            return outcome
        before = set(self._registry.in_scene)
        self._run_presence_backstops(assistant_text, prev_assistant, user_text)
        entered = self._registry.in_scene - before
        targets: list[tuple[str, str]] = [("npc", s) for s in self._registry.active()]
        pc = self._registry.current_pc_slug
        if pc:
            targets.append(("pc", pc))
        if not targets:
            # No sheets to write (no NPC in scene, no PC slug).
            return outcome

        self._turn += 1
        started = time.perf_counter()
        await self._emit_phase("writing_sheets")
        time_advance_hint: str | None = None
        judge = self._start_advance_judge(prev_assistant, user_text, assistant_text)
        scene_end = self._start_scene_end_judge(user_text, assistant_text)
        trial, asleep, resolved = self._start_nightmare_judges(pc, user_text, assistant_text)
        for kind, slug in targets:
            result = await rewrite_sheet(
                client=self._client,
                model=self._model,
                vault=self._vault,
                kind=kind,
                slug=slug,
                locale=self._locale,
                user_text=user_text,
                assistant_text=assistant_text,
                turn=self._turn,
                timeout_s=self._timeout_s,
                goal_static=(
                    kind == "npc"
                    and self._director is not None
                    and slug in self._director.static_goal_slugs
                ),
            )
            outcome.rewrites.append(result)
            if kind == "pc" and result.time_advance:
                time_advance_hint = result.time_advance
        await self._finish_advance_judge(judge)
        await self._finish_scene_end_judge(scene_end)
        frozen = await self._advance_nightmare(
            trial, asleep, resolved, pc, user_text, assistant_text
        )
        with_pc = self._enter_with_pc(
            next((r.with_pc for r in outcome.rewrites if r.kind == "pc"), ())
        )
        entered = self._registry.in_scene - before
        outcome.total_latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "sheet_writer: turn {} — {} patch(es), {} written, {}ms",
            self._turn,
            len(outcome.rewrites),
            sum(1 for r in outcome.rewrites if r.written),
            outcome.total_latency_ms,
        )
        self._condense_old_replies([r.bullet for r in outcome.rewrites if r.bullet])
        # Backstop entries are settled against the PC's POST-patch
        # location and stamped in_scene regardless of patch outcome.
        pc_loc = _read_state_field(self._vault, "pc", pc, "location") if pc else ""
        for slug in sorted(entered):
            _settle_entered_npc(self._vault, slug, pc_loc)
        moved = {(r.kind, r.slug) for r in outcome.rewrites if "location" in r.changed_keys}
        self._auto_exit_diverged(pc, moved, assistant_text)
        self._auto_exit_gone()
        engine_fired = await self._run_story_engine(
            time_advance_hint, assistant_text, user_text, with_pc, frozen=frozen
        )
        if self._director is not None:
            # Scene-movement signal for the stall escalation: a story-event
            # fired, or some sheet's scene_goal changed. NOT location —
            # incremental steps ("down one flight", "down another") churn
            # location every turn while the scene is narratively static, so
            # location as a movement proxy hid a 20-turn stairwell stall
            # (2026-07-18 replay: D6 read healthy while the judge saw no
            # narrative advance).
            self._director.observe_movement(
                engine_fired or any("scene_goal" in r.changed_keys for r in outcome.rewrites)
            )
            # Post-exit view: goals/tenure are only tracked while in scene.
            active = self._registry.active()
            self._director.prune(active)
            for slug in active:
                goal = _read_state_field(self._vault, "npc", slug, "scene_goal")
                if goal:
                    self._director.observe_goal(slug, goal)
                # Long-scene voice drift: count tenure and carry the
                # identity sheet's register for the voice-check line.
                self._director.observe_scene_tenure(
                    slug, _identity_field(self._vault, slug, "speech_register")
                )
        try:
            await self._maybe_consolidate_stubs(user_text, assistant_text)
        except Exception as exc:
            logger.warning(
                "sheet_writer: stub consolidation failed — {}: {}", type(exc).__name__, exc
            )
        await self._emit_npc_state()
        return outcome

    def _start_advance_judge(
        self, previous: str, player: str, current: str
    ) -> asyncio.Task[bool | None] | None:
        """Run the content-repetition verdict alongside the patches (same gateway)."""
        if not self._advance_judge or self._director is None:
            return None
        return asyncio.create_task(
            judge_repeats(
                self._client, self._model, previous=previous, player=player, current=current
            )
        )

    async def _finish_advance_judge(self, judge: asyncio.Task[bool | None] | None) -> None:
        if judge is None or self._director is None:
            return
        self._director.reply_stalled = await judge is True
        if self._director.reply_stalled:
            logger.info("sheet_writer: advance judge — reply re-told the previous beat")

    def _start_scene_end_judge(self, player: str, current: str) -> asyncio.Task[bool | None] | None:
        if not self._scene_end_judge:
            return None
        return asyncio.create_task(
            judge_scene_ends(self._client, self._model, player=player, current=current)
        )

    async def _finish_scene_end_judge(self, judge: asyncio.Task[bool | None] | None) -> None:
        if judge is not None and await judge is True:
            logger.info("sheet_writer: scene-end judge — the turn moved place or time")
            self._registry.note_scene_change()

    def _start_nightmare_judges(
        self, pc: str | None, player: str, current: str
    ) -> tuple[
        first_nightmare.Trial | None,
        asyncio.Task[bool | None] | None,
        asyncio.Task[bool | None] | None,
    ]:
        """(trial, falls-asleep judge, trial-resolved judge), run alongside the patches.

        The asleep judge runs only before any trial (and never in session
        zero: the PC sheet must exist); the resolved judge only from
        ``MIN_TRIAL_TURNS`` turns into an active trial.
        """
        if not (self._nightmare and pc and first_nightmare.enabled(self._vault)):
            return None, None, None
        trial = first_nightmare.read_trial(self._vault)
        if trial is None:
            try:
                self._vault.read_file(f"characters/{pc}.md")
            except VaultPathError:
                return None, None, None
            asleep = judge_falls_asleep(self._client, self._model, player=player, current=current)
            return None, asyncio.create_task(asleep), None
        plan = trial.plan
        if trial.status == "active" and plan and trial.turns + 1 >= first_nightmare.MIN_TRIAL_TURNS:
            resolved = judge_trial_resolved(
                self._client,
                self._model,
                win=plan.win,
                lose=plan.lose,
                player=player,
                current=current,
            )
            return trial, None, asyncio.create_task(resolved)
        return trial, None, None

    async def _advance_nightmare(  # noqa: PLR0913 — the turn's verdicts + exchange
        self,
        trial: first_nightmare.Trial | None,
        asleep: asyncio.Task[bool | None] | None,
        resolved: asyncio.Task[bool | None] | None,
        pc: str | None,
        player: str,
        current: str,
    ) -> bool:
        """Move the First Nightmare on by this turn's verdicts; True = the clock stays frozen.

        Fails open: an error logs and leaves the trial where disk has it.
        """
        running = trial is not None and trial.status != "resolved"
        try:
            if asleep is not None:
                if await asleep is not True:
                    return False
                first_nightmare.start(self._vault)
                await self._plan_nightmare(pc or "")
                return True
            if trial is None or not running:
                return False
            if trial.status == "dormant" or trial.plan is None:
                await self._plan_nightmare(pc or "")
                return True
            trial = first_nightmare.count_turn(self._vault, trial)
            if resolved is not None and await resolved is True:
                logger.info("first_nightmare: end judge — the trial's conflict resolved")
                block = await first_nightmare.finish(
                    self._vault,
                    self._client,
                    self._model,
                    trial,
                    pc_slug=pc or "",
                    player=player,
                    reply=current,
                    timeout_s=self._timeout_s,
                )
                if block and self._director is not None:
                    self._director.spell_line = block
        except Exception as exc:
            logger.warning("first_nightmare: turn step failed — {}: {}", type(exc).__name__, exc)
        return running

    async def _plan_nightmare(self, pc: str) -> None:
        plan = await first_nightmare.write_plan(
            self._vault, self._client, self._model, pc_slug=pc, timeout_s=self._timeout_s
        )
        if plan is not None and self._director is not None:
            self._director.spell_line = first_nightmare.greeting(self._vault, plan, pc)

    def _condense_old_replies(self, bullets: list[str]) -> None:
        """Record this turn's digest, then condense replies past the verbatim window.

        A model copies what sits verbatim in its context (whole replies
        0.97-0.99 from 3-5 turns back, 2026-09-26 replay-003); the older
        replies' facts already live on the sheets, so only their bullets
        stay. Player turns and tool-call replies are never touched; a
        reply with no bullets stays verbatim.
        """
        if self._verbatim_replies <= 0:
            return
        replies = [
            m
            for m in self._context.messages
            if _role_of(m) == "assistant" and _content_text(m) and not _get(m, "tool_calls")
        ]
        if not replies:
            return
        if bullets:
            self._digests[_content_text(replies[-1])] = " ".join(bullets)
        for message in replies[: -self._verbatim_replies]:
            digest = self._digests.pop(_content_text(message), None)
            if digest is not None:
                _set_content(message, f"{CONDENSED_REPLY_TAG} {digest}")

    async def _run_story_engine(
        self,
        time_advance_hint: str | None,
        assistant_text: str,
        user_text: str = "",
        with_pc: frozenset[str] = frozenset(),
        *,
        frozen: bool = False,
    ) -> bool:
        """One story-engine pass after patches + auto-exits.

        Runs post-writer so this turn's player consequences are on disk
        when triggers are evaluated (player pre-emption is structural).
        No-op for runs without a clock.md. Feeds the director's clock
        line + event directives and emits ``world.event`` side-band
        events for every status change. ``assistant_text`` lets the
        engine confirm which previously rendered directives the DM
        actually narrated before retiring them; with ``user_text`` it also
        holds events that would move an NPC this turn's exchange names, or
        one in ``with_pc`` (slugs the PC writer placed with the PC).

        Returns True when a story-event changed status this turn (the
        world moved) — feeds the director's scene-movement signal.
        """
        outcome: story_events.EngineOutcome | None = None
        try:
            outcome = story_events.run_end_of_turn(
                self._vault,
                self._registry,
                time_advance_hint=time_advance_hint,
                assistant_text=assistant_text,
                user_text=user_text,
                with_pc=with_pc,
                frozen=frozen,
            )
        except Exception as exc:
            logger.warning("sheet_writer: story engine failed — {}: {}", type(exc).__name__, exc)
        if outcome is None:
            # _run cleared the director channel at turn start; nothing to do.
            return False
        if self._director is not None:
            self._director.clock_line = outcome.clock_line
            self._director.event_directives = list(outcome.directives)
        for slug, title, status in outcome.changed_events:
            await self._emitter.push_frame(
                OutputTransportMessageFrame(
                    message={"type": "world.event", "slug": slug, "title": title, "status": status}
                ),
                FrameDirection.DOWNSTREAM,
            )
        return bool(outcome.changed_events)

    async def _maybe_consolidate_stubs(self, user_text: str, assistant_text: str) -> None:
        """Distill a real identity for stub NPCs that have earned one.

        Sheet authoring inside tool arguments fails 10/10 on the 14B
        (bench-tools 2026-07-09), so stubs are consolidated the way the
        per-turn patches work: one dedicated completion with a small
        structured JSON output, validated and rendered through a
        deterministic template. Runs every _CONSOLIDATE_AFTER writer
        turns a stub stays in scene; a failed attempt retries at the
        next multiple, and success removes the stub marker so counting
        stops naturally.
        """
        for slug in self._registry.active():
            try:
                identity = self._vault.read_file(f"npcs/{slug}.md")
            except VaultPathError:
                continue
            if _STUB_GLOSS_MARKER not in identity:
                continue
            count = self._stub_turns.get(slug, 0) + 1
            self._stub_turns[slug] = count
            if count % _CONSOLIDATE_AFTER != 0:
                continue
            await self._consolidate_stub(slug, identity, user_text, assistant_text)

    async def _consolidate_stub(
        self, slug: str, identity: str, user_text: str, assistant_text: str
    ) -> None:
        fm, _body = frontmatter.split(identity)
        name = str((fm or {}).get("name") or slug)
        evidence = self._vault.read_state("npc", slug) or "(no state on file)"
        template = _CONSOLIDATE_PROMPT.get(self._locale, _CONSOLIDATE_PROMPT["en"])
        prompt = template.format(
            name=name,
            evidence=evidence,
            user_text=user_text.strip() or "(none)",
            assistant_text=assistant_text.strip() or "(none)",
        )
        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=450,
                temperature=0.3,
                timeout=self._timeout_s,
            )
        except Exception as exc:
            logger.warning(
                "sheet_writer: consolidation LLM error for {} — {}: {}",
                slug,
                type(exc).__name__,
                exc,
            )
            return
        profile = _validate_consolidation(_parse_patch(_extract_text(completion)))
        if profile is None:
            logger.info("sheet_writer: consolidation for {} rejected (bad payload)", slug)
            return
        gloss, speech, hexaco = profile
        new_fm = dict(fm or {})
        new_fm.update(hexaco)
        content = frontmatter.serialize(new_fm, _consolidated_identity_body(gloss, speech, hexaco))
        try:
            self._vault.write_npc_sheet(slug, content)
        except (VaultPathError, OSError) as exc:
            logger.warning(
                "sheet_writer: consolidation write FAILED for {} — {}: {}",
                slug,
                type(exc).__name__,
                exc,
            )
            return
        logger.info(
            "sheet_writer: consolidated stub identity for {} ({}ms)",
            slug,
            int((time.perf_counter() - started) * 1000),
        )

    def _seed_introduced_npcs(self, assistant_text: str, prev_assistant_text: str = "") -> None:
        """Canonicalize DM-introduced NPCs the model didn't tool-call for.

        Detects explicit name introductions in the reply, plus unnamed
        referents ("the figure") recurring from the previous reply. A
        name that matches an existing vault NPC is simply (re-)entered
        into the scene — the model voiced them without ``npc_scene_enter``.
        An unknown name gets a stub identity + state sheet and enters the
        scene, so the injector and this writer track them from now on.
        """
        previous = detect_unnamed_referents(prev_assistant_text)
        names = detect_introduced_names(assistant_text) + [
            r for r in detect_unnamed_referents(assistant_text) if r in previous
        ]
        if not names:
            return
        pc_slug = self._registry.current_pc_slug
        pc_name = _identity_name(self._vault, "pc", pc_slug) if pc_slug else ""
        pc_loc = _read_state_field(self._vault, "pc", pc_slug, "location") if pc_slug else ""
        roster = _collect_offscene_npc_names(self._vault, [])  # every vault NPC
        seeded = 0
        for name in names:
            # Simplification: max 2 stubs/turn — a misfiring pattern can't flood canon.
            if seeded >= 2:
                break
            # The PC's own name ("I am Rowan" for rowan-blackthorn) never
            # becomes an NPC — same full/first-name/slug-prefix matching.
            if pc_slug and _vault_npc_matches({pc_name or pc_slug: pc_slug}, name):
                continue
            matches = _vault_npc_matches(roster, name)
            if len(matches) > 1:
                logger.info("sheet_writer: {!r} is ambiguous ({}); not seeded", name, matches)
                continue
            slug = matches[0] if matches else _slugify(name)
            if not slug:
                continue
            if slug in self._registry.in_scene:
                continue
            if matches:
                # Existing NPC voiced without npc_scene_enter: enter only
                # (co-located after the patch pass) unless they're gone.
                if _read_state_field(self._vault, "npc", slug, "status") in _GONE_STATUSES:
                    continue
                self._registry.enter(slug)
                logger.info("sheet_writer: auto-entered existing npc {} (introduced)", slug)
                continue
            location = pc_loc or "unknown"
            try:
                self._vault.write_npc_sheet(
                    slug, _stub_identity_sheet(name, self._locale, location)
                )
                self._vault.write_npc_state(slug, _stub_state_sheet(location))
            except (VaultPathError, OSError) as exc:
                logger.warning(
                    "sheet_writer: stub seed FAILED for {} — {}: {}",
                    slug,
                    type(exc).__name__,
                    exc,
                )
                continue
            self._registry.enter(slug)
            seeded += 1
            logger.info("sheet_writer: seeded stub npc {} ({!r} introduced by DM)", slug, name)

    def _run_presence_backstops(
        self, assistant_text: str, prev_assistant_text: str = "", user_text: str = ""
    ) -> None:
        """Both scene-entry backstops, before target collection.

        Order matters: an NPC seeded/entered here gets its first state
        patch from the very turn that introduced them. Failures are
        contained — a misfiring heuristic must not cost the turn's patches.
        """
        try:
            self._seed_introduced_npcs(assistant_text, prev_assistant_text)
        except Exception as exc:
            logger.warning("sheet_writer: npc seeding failed — {}: {}", type(exc).__name__, exc)
        try:
            self._auto_enter_voiced_npcs(assistant_text, user_text)
        except Exception as exc:
            logger.warning(
                "sheet_writer: voiced auto-enter failed — {}: {}", type(exc).__name__, exc
            )

    def _auto_enter_voiced_npcs(self, assistant_text: str, user_text: str = "") -> None:
        """Enter known vault NPCs the DM voiced without ``npc_scene_enter``.

        Sibling backstop of ``_seed_introduced_npcs`` for the case the
        intro-seeder can't see: the DM already knows the name (beat,
        canon) and starts voicing the NPC cold. Presence-vs-mention is
        decided by ``detect_voiced_names`` — plus ``detect_addressed_names``
        over the player's message and the reply: an NPC spoken TO is
        present even when the DM never voices her. ``_run`` co-locates entered
        NPCs with the PC after the patch pass. Gone/missing NPCs are
        never pulled in. Max 2 per turn.

        A first name ("Lyra nods" for Lyra Moonwhisper) counts only when
        it resolves to exactly that NPC across the whole vault roster —
        in-scene NPCs included, so an in-scene Lyra Stone blocks it.
        """
        roster = _collect_offscene_npc_names(self._vault, [])  # every vault NPC
        offscene = {n: s for n, s in roster.items() if s not in self._registry.in_scene}
        if not offscene:
            return
        for full, npc_slug in list(offscene.items()):
            first = full.split()[0]
            # Simplification: stopword/length guard — "The Warden" must not make "the"
            # a presence trigger (detect_voiced_names is case-insensitive).
            if (
                first != full
                and len(first) > 2
                and first.casefold() not in NAME_STOPWORDS
                and _vault_npc_matches(roster, first) == [npc_slug]
            ):
                offscene.setdefault(first, npc_slug)
        voiced = detect_voiced_names(assistant_text, offscene.keys())
        # Per source: an odd straight quote in one text must not shift the
        # quoted spans of the other.
        addressed = detect_addressed_names(user_text, offscene.keys()) + detect_addressed_names(
            assistant_text, offscene.keys()
        )
        entered = 0
        for name, how in [(n, "voiced") for n in voiced] + [(n, "addressed") for n in addressed]:
            # Simplification: max 2/turn — mirrors the stub-seed cap for misfires.
            if entered >= 2:
                break
            slug = offscene[name]
            if slug in self._registry.in_scene:
                continue  # matched by full AND first name, or by both sources
            if _read_state_field(self._vault, "npc", slug, "status") in _GONE_STATUSES:
                continue
            self._registry.enter(slug)
            entered += 1
            if how == "addressed":
                logger.info("presence: addressed {} entered", slug)
            else:
                logger.info(
                    "sheet_writer: auto-entered voiced npc {} ({!r} spoke/acted)", slug, name
                )

    def _enter_with_pc(self, names: Iterable[str]) -> frozenset[str]:
        """Resolve the PC patch's ``with_pc`` names; enter the unscened ones.

        The semantic sibling of the voiced/addressed backstops: the PC
        writer already reads the whole exchange, so it judges presence
        that narration alone carries ("I fall into step beside Marta").
        Names resolve like introductions (full name, unambiguous first
        name, slug prefix); ambiguous or unknown names are ignored — this
        never seeds. Same refusal (dead/missing) and cap (2/turn) as the
        other backstops. Returns every resolved slug, in scene or not, for
        the story engine's subject hold. Never used for exits.
        """
        if not names:
            return frozenset()
        try:
            roster = _collect_offscene_npc_names(self._vault, [])  # every vault NPC
            pc_slug = self._registry.current_pc_slug
            pc_name = _identity_name(self._vault, "pc", pc_slug) if pc_slug else ""
            resolved: list[str] = []
            for name in names:
                if pc_slug and _vault_npc_matches({pc_name or pc_slug: pc_slug}, name):
                    continue
                matches = _vault_npc_matches(roster, name)
                if len(matches) > 1:
                    logger.info("presence: with_pc {!r} is ambiguous ({}); ignored", name, matches)
                if len(matches) == 1 and matches[0] not in resolved:
                    resolved.append(matches[0])
            entered = 0
            for slug in resolved:
                # Simplification: max 2/turn — mirrors the voiced/addressed cap.
                if entered >= 2:
                    break
                if slug in self._registry.in_scene:
                    continue
                if _read_state_field(self._vault, "npc", slug, "status") in _GONE_STATUSES:
                    continue
                self._registry.enter(slug)
                entered += 1
                logger.info("presence: with_pc {} entered", slug)
        except Exception as exc:
            logger.warning("sheet_writer: with_pc entry failed — {}: {}", type(exc).__name__, exc)
            return frozenset()
        return frozenset(resolved)

    def _auto_exit_gone(self) -> None:
        """Drop in-scene NPCs whose status says they're gone.

        Backstop sibling of ``_auto_exit_diverged``: a dead or missing
        NPC at the PC's own location passes the location check and stays
        glued to the scene if the model skipped ``npc_scene_exit``. The
        patch pass just ran, so the status field is this turn's truth.
        """
        for slug in self._registry.active():
            status = _read_state_field(self._vault, "npc", slug, "status")
            if status not in _GONE_STATUSES:
                continue
            self._registry.exit(slug)
            _stamp_in_scene_false(self._vault, slug)
            logger.info("sheet_writer: auto-exit {} — status {!r}", slug, status)

    def _auto_exit_diverged(
        self, pc_slug: str | None, moved: set[tuple[str, str]], reply: str
    ) -> None:
        """Drop in-scene NPCs who moved to a different place than the PC.

        Backstop for missing `npc_scene_exit` calls — the model often
        narrates departure without firing the tool, which leaves the
        NPC glued to every subsequent turn's injection. Evicts only when
        the locations are a different place (``same_place``: the writer's
        free text varies — "hall of X" vs "infirmary of X") AND the PC's
        or that NPC's location changed in this turn's patches
        (``moved`` holds ``(kind, slug)``) — a departure is an event, a
        stale mismatch nobody acted on is not. Never evicts an NPC this
        turn's reply names: the PC writer often misses the PC's own move
        (2026-09-25: Marta, escorting the player across the river, was
        evicted 4x as 'river gate' vs a stale 'ridgeway fallout shelter').
        """
        if not pc_slug:
            return
        pc_loc = _read_state_field(self._vault, "pc", pc_slug, "location")
        if not pc_loc:
            return
        pc_moved = ("pc", pc_slug) in moved
        for slug in self._registry.active():
            npc_loc = _read_state_field(self._vault, "npc", slug, "location")
            if not npc_loc or same_place(npc_loc, pc_loc):
                continue
            if not pc_moved and ("npc", slug) not in moved:
                continue
            if npc_named(reply, _identity_name(self._vault, "npc", slug), slug):
                continue
            self._registry.exit(slug)
            _stamp_in_scene_false(self._vault, slug)
            logger.info(
                "sheet_writer: auto-exit {} — location diverged ({!r} vs pc {!r})",
                slug,
                npc_loc,
                pc_loc,
            )

    async def _emit_phase(self, phase: str) -> None:
        """Tell the browser whether end-of-turn work is still running."""
        await self._emitter.push_frame(
            OutputTransportMessageFrame(message={"type": "turn.status", "phase": phase}),
            FrameDirection.DOWNSTREAM,
        )

    async def _emit_npc_state(self) -> None:
        """Push a fresh ``npc.state`` event for the NPCs-in-scene panel."""
        payload = {
            "type": "npc.state",
            "npcs": collect_npc_state(self._vault, self._registry.active()),
        }
        await self._emitter.push_frame(
            OutputTransportMessageFrame(message=payload),
            FrameDirection.DOWNSTREAM,
        )


def collect_npc_state(vault: Vault, in_scene: list[str]) -> list[dict[str, str]]:
    """Project the in-scene NPCs' state files to the side-channel payload."""
    out: list[dict[str, str]] = []
    for slug in in_scene:
        mood = goal = mask = ""
        state = vault.read_state("npc", slug)
        if state:
            fm, _ = frontmatter.split(state)
            if fm:
                mood = str(fm.get("mood", "") or "")
                goal = str(fm.get("scene_goal", "") or "")
                mask = str(fm.get("social_mask", "") or "")
        out.append({"slug": slug, "mood": mood, "scene_goal": goal, "social_mask": mask})
    return out


def _collect_offscene_npc_names(vault: Vault, in_scene: list[str]) -> dict[str, str]:
    """Display name → slug for every vault NPC not currently in scene.

    Candidate list for the voiced auto-enter. Uses the identity sheet's
    ``name:`` (what the DM actually writes in prose); slugs are included
    as a fallback key when no name field exists.
    """
    try:
        entries = vault.list_dir("npcs")
    except VaultPathError:
        return {}
    in_scene_set = set(in_scene)
    out: dict[str, str] = {}
    for entry in entries:
        if not entry.endswith(".md") or entry.endswith(".state.md"):
            continue
        slug = entry[len("npcs/") : -len(".md")] if entry.startswith("npcs/") else entry[:-3]
        if slug in in_scene_set:
            continue
        name = _identity_name(vault, "npc", slug)
        out[name or slug] = slug
    return out


def _vault_npc_matches(roster: dict[str, str], name: str) -> list[str]:
    """Vault NPC slugs an introduced (usually single-word) name refers to.

    ``roster`` is display name → slug. Exact slug wins; otherwise full
    name, first name, or slug prefix ("Lyra" → ``lyra-moonwhisper``), so
    "I'm Lyra" never seeds a duplicate stub beside the canon sheet. More
    than one hit is ambiguous — the caller neither enters nor seeds.
    """
    slug = _slugify(name)
    if slug in roster.values():
        return [slug]
    key = name.casefold()
    return sorted(
        {
            npc_slug
            for full, npc_slug in roster.items()
            if key in (full.casefold(), full.casefold().split()[0])
            or (slug and npc_slug.startswith(f"{slug}-"))
        }
    )


def _settle_entered_npc(vault: Vault, slug: str, pc_location: str) -> None:
    """Record a backstop entry on disk: ``in_scene: true`` + co-location.

    Runs after the patch pass, whatever its outcome — a skipped patch
    must not leave ``in_scene: false`` behind, since resume rebuilds
    the registry from disk. The narration placed the NPC in the present
    scene, so they adopt the PC's post-patch location, but only when
    theirs is empty or a different place (``same_place``). Missing state
    file → seed a stub state at the PC's location.
    """
    state = vault.read_state("npc", slug)
    if not state:
        try:
            vault.write_npc_state(slug, _stub_state_sheet(pc_location or "unknown"))
        except (VaultPathError, OSError) as exc:
            logger.warning("sheet_writer: entry seed failed for {} — {}", slug, exc)
        return
    fm, body = frontmatter.split(state)
    if not fm:
        return
    new_fm = dict(fm)
    new_fm["in_scene"] = "true"
    npc_loc = str(fm.get("location") or "").strip()
    if pc_location and (not npc_loc or not same_place(npc_loc, pc_location)):
        new_fm["location"] = pc_location
    if new_fm == fm:
        return
    try:
        vault.write_npc_state(slug, frontmatter.serialize(new_fm, body))
    except (VaultPathError, OSError) as exc:
        logger.warning("sheet_writer: entry write failed for {} — {}", slug, exc)


def _identity_name(vault: Vault, kind: str, slug: str) -> str:
    """The ``name:`` field from an identity sheet, or empty string."""
    subdir = "characters" if kind == "pc" else "npcs"
    try:
        fm, _ = frontmatter.split(vault.read_file(f"{subdir}/{slug}.md"))
    except VaultPathError:
        return ""
    if not fm:
        return ""
    return str(fm.get("name") or "").strip()


def _identity_field(vault: Vault, slug: str, key: str) -> str:
    """Any frontmatter field from an NPC identity sheet, or empty string."""
    try:
        fm, _ = frontmatter.split(vault.read_file(f"npcs/{slug}.md"))
    except VaultPathError:
        return ""
    if not fm:
        return ""
    return str(fm.get(key) or "").strip()


def _read_state_field(vault: Vault, kind: str, slug: str, key: str) -> str:
    """Normalized frontmatter value from the state file, or empty string."""
    text = vault.read_state(kind, slug)
    if not text:
        return ""
    fm, _ = frontmatter.split(text)
    if not fm:
        return ""
    return str(fm.get(key) or "").strip().lower()


def _stamp_in_scene_false(vault: Vault, slug: str) -> None:
    """Best-effort write of ``in_scene: false`` on the NPC's state file.

    Keeps the on-disk frontmatter honest after an exit so a later direct
    read doesn't show stale presence. Failures swallowed — the
    SceneRegistry is the source of truth for *current* presence.
    """
    text = vault.read_state("npc", slug)
    if not text:
        return
    fm, body = frontmatter.split(text)
    if not fm or str(fm.get("in_scene", "")) == "false":
        return
    fm["in_scene"] = "false"
    try:
        vault.write_npc_state(slug, frontmatter.serialize(fm, body))
    except (VaultPathError, OSError):
        return


def _extract_recent_exchanges(messages: list[Any], count: int = 2) -> list[tuple[str, str]]:
    """Newest-first ``(user_text, assistant_text)`` pairs, up to ``count``.

    A pair is an assistant reply plus the closest user message that
    precedes it (tool/system messages in between are skipped); a latest
    user message with no reply after it yields ``(user, "")``. Feeds
    both the per-turn patch texts (pair 0) and the turn-director's
    repetition/repeated-input detectors (pair 1).
    """
    flat: list[tuple[str, str]] = []
    for message in messages:
        role = _role_of(message)
        if role not in ("user", "assistant"):
            continue
        text = _content_text(message)
        if text:
            flat.append((role, text))
    pairs: list[tuple[str, str]] = []
    i = len(flat) - 1
    if i >= 0 and flat[i][0] == "user":
        # Nothing answered the latest user message: this turn produced no
        # DM text. Pairing it with the previous reply would re-apply that
        # reply's patches as if it were this turn's.
        pairs.append((flat[i][1], ""))
        i -= 1
    while i >= 0 and len(pairs) < count:
        while i >= 0 and flat[i][0] != "assistant":
            i -= 1
        if i < 0:
            break
        # The code-emitted "*Day N, seg.*" lead line is not narration; left in,
        # the PC patch reads it as time passing and hints another advance.
        assistant = story_events.strip_time_shown(flat[i][1])
        i -= 1
        user = ""
        while i >= 0:
            role, text = flat[i]
            i -= 1
            if role == "user":
                user = text
                break
        pairs.append((user, assistant))
    return pairs


CONDENSED_REPLY_TAG = "[Earlier narration, condensed]"


def _get(message: Any, key: str) -> Any:
    return message.get(key) if isinstance(message, dict) else getattr(message, key, None)


def _set_content(message: Any, text: str) -> None:
    if isinstance(message, dict):
        message["content"] = text
    else:
        message.content = text


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
