"""Deterministic turn director — measured signals become prompt notes.

The 14B DM-LLM demonstrably ignores standing rules buried in a 7k-token
system prompt (anti-repetition, goal-evolution, agency). Instead of
repeating those rules harder, the director *measures* the failure at
end-of-turn with plain code and injects a short, specific instruction at
the very tail of the next prompt — maximum recency, derived from what
actually happened, present only when triggered.

Signal flow per turn:

1. ``SceneSheetWriter`` (end of turn N) calls ``DirectorState.observe_turn``
   with the last two user/assistant exchanges and ``observe_goal`` with
   each NPC's post-patch ``scene_goal``.
2. ``SceneSheetInjector`` (start of turn N+1) calls ``build_notes`` and
   splices the result as a tagged system message appended after the
   latest user message.

The agency contract line is unconditional: the observed failure mode
(session bab180be) was the DM replacing the player's declared action
with tamer narration — a violation no measurement can catch reliably,
so the contract rides every turn at the position the model attends to
most.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import pairwise

TURN_CONTRACT_TAG = "<!--aidm:turn-contract-->"

# Two consecutive DM replies more similar than this are a loop.
# 0.55 is far above natural same-scene prose overlap (~0.2-0.35) and far
# below the observed verbatim repeats (0.95+).
REPLY_SIMILARITY_THRESHOLD = 0.55
# A reply whose OPENING is this similar to the SAME turn's player input
# is an echo — the model mirroring the player back before (or instead
# of) responding (Kimi K2.6 T43-45, marathon replay-004: verbatim
# first-person copies as the reply's first paragraph). 0.8 sits above
# legitimate partial dialogue quoting, below the observed ~1.0 heads.
PLAYER_ECHO_THRESHOLD = 0.8
# Near-duplicate bar shared by bullet dedupe and goal comparison.
NEAR_DUPLICATE_THRESHOLD = 0.85
# A goal streak of 2 unchanged comparisons = the same goal 3 turns running.
GOAL_STATIC_STREAK = 2
# Consecutive turns where nothing moved (no scene_goal/location change, no
# story-event fired) before the plan carries a hard "advance NOW" order.
# The 30t/50t runs stalled into multi-turn standoffs the prompt couldn't
# break; this is the code-owned escalation the persona rule never was.
STALL_ESCALATE_AFTER = 3
# Consecutive in-scene turns before an NPC gets a voice-check line —
# the 30t judge runs showed register drift sets in on LONG scenes
# (hexaco_faithfulness median 6/15), not short exchanges.
REGISTER_REMINDER_AFTER = 6
# A 4-word phrase recurring in this many DISTINCT replies is a tic —
# "boots crunching against gravel" appeared across many turns while the
# consecutive-reply similarity check stayed silent. Replaces the work
# the retired DRY sampler was supposed to do, but targeted.
PHRASE_SHINGLE_WORDS = 4
PHRASE_REPLY_COUNT = 3
MAX_BANNED_PHRASES = 3
# Recurrence is counted over the last N replies, not all time: all-time
# counts let phrases unseen for 10+ turns hold the ban slots (15% of
# ban-slot-turns across 16 real transcripts). The window doubles as the
# anti-flicker hysteresis — a banned phrase the model now avoids stays
# banned until its occurrences age out, not the first turn it's absent.
PHRASE_WINDOW_REPLIES = 10
# Paragraph-level (8-word) shingle tracking — lower threshold (2 replies)
# catches Pantheon-class verbatim paragraph recycling before the 4-gram
# tic detector fires (which needs 3 replies and shorter windows).
PARA_SHINGLE_WORDS = 8
PARA_REPLY_COUNT = 2
MAX_PARA_BANNED = 2

_SHINGLE_WORD_RE = re.compile(r"[a-z']+")


def similarity(a: str, b: str) -> float:
    """Normalized difflib ratio in [0, 1]; whitespace-collapsed, casefolded."""
    a_norm = " ".join(a.split()).casefold()
    b_norm = " ".join(b.split()).casefold()
    if not a_norm or not b_norm:
        return 0.0
    # autojunk off: on texts ≥200 chars it junks common characters and scored
    # 0.83-0.92 verbatim replays as 0.40-0.64 (2026-09-26 marathon).
    return SequenceMatcher(None, a_norm, b_norm, autojunk=False).ratio()


def player_echo_ratio(user_text: str, assistant_text: str) -> float:
    """How strongly the reply's OPENING restates the player's input.

    The observed echoes (Kimi K2.6, marathon replay-004 T43-45) copy the
    player's message verbatim as the reply's first paragraph and then
    continue with thousands of chars of narration — whole-text similarity
    sits at ~0.16 and never trips. Compare the player text against a
    same-length head window of the reply instead.
    """
    user_norm = " ".join(user_text.split())
    # Head window = player length + small slack. A generous slack dilutes
    # short inputs: 80 extra chars on a 57-char input caps the ratio at
    # ~0.59 even for a perfect echo.
    head = " ".join(assistant_text.split())[: len(user_norm) + 10]
    return similarity(head, user_norm)


@dataclass
class DirectorState:
    """Per-session measurements taken at end-of-turn, consumed next prompt."""

    reply_similarity: float = 0.0
    user_repeated: bool = False
    # Last DM reply was a near-copy of the same turn's player input.
    player_echo: bool = False
    # The last reply re-told the previous beat (advance_judge, run by the
    # sheet writer). Renders a Must line; arms the pre-release check.
    reply_stalled: bool = False
    # Consecutive turns where the scene did not move (fed by the writer).
    stall_turns: int = 0
    # Slugs whose scene_goal has been static for GOAL_STATIC_STREAK+ turns.
    static_goal_slugs: list[str] = field(default_factory=list)
    # Story-engine channel (aidm.story_events), refreshed every turn the
    # engine runs: the in-fiction clock line plus the fired-event
    # directives the DM must narrate. Both render in the scene-sheets
    # block (SceneSheetInjector), not in the turn contract. A change of
    # clock_line is what TimeShownProcessor shows at the next reply start.
    # Empty when the run has no clock.md.
    clock_line: str = ""
    event_directives: list[str] = field(default_factory=list)
    # A code-written Spell line (aidm.first_nightmare: the trial greeting or
    # the appraisal block) waiting to open the next reply; TimeShownProcessor
    # takes it. Here, not in the processor, so the regenerate snapshot of
    # the director carries it.
    spell_line: str = ""
    # Emergent narrative commitments ("cause → pending consequence", ≤5),
    # replaced by SessionSummarizer on each summary; rendered in the
    # scene-sheets block every turn.
    open_threads: list[str] = field(default_factory=list)
    # Recycled 4-word phrases (3+ distinct replies) → banned next turn.
    banned_phrases: list[str] = field(default_factory=list)
    # Recycled 8-word phrases (2+ distinct replies) → paragraph-loop guard.
    paragraph_banned: list[str] = field(default_factory=list)
    _goal_streak: dict[str, int] = field(default_factory=dict)
    _last_goal: dict[str, str] = field(default_factory=dict)
    # Prose-tic tracking: per-reply shingle sets, last PHRASE_WINDOW_REPLIES.
    _recent_shingles: deque[set[str]] = field(
        default_factory=lambda: deque(maxlen=PHRASE_WINDOW_REPLIES)
    )
    # Paragraph-loop tracking: per-reply 8-gram sets, same window.
    _recent_para_shingles: deque[set[str]] = field(
        default_factory=lambda: deque(maxlen=PHRASE_WINDOW_REPLIES)
    )
    # Voice-check tracking: consecutive in-scene turns + speech register.
    _scene_tenure: dict[str, int] = field(default_factory=dict)
    _registers: dict[str, str] = field(default_factory=dict)

    def observe_turn(
        self,
        *,
        user_text: str,
        prev_user_text: str,
        assistant_text: str,
        prev_assistant_text: str,
    ) -> None:
        """Measure this turn's exchange against the previous one."""
        self.reply_similarity = (
            similarity(assistant_text, prev_assistant_text) if prev_assistant_text else 0.0
        )
        self.user_repeated = bool(
            user_text.strip()
            and " ".join(user_text.split()).casefold()
            == " ".join(prev_user_text.split()).casefold()
        )
        self.player_echo = player_echo_ratio(user_text, assistant_text) >= PLAYER_ECHO_THRESHOLD
        self._observe_prose(assistant_text)

    def _observe_prose(self, assistant_text: str) -> None:
        """Count phrase shingles per reply; surface the recycled ones.

        Each shingle counts at most once per reply, so a phrase repeated
        inside one purple paragraph doesn't inflate — only cross-reply
        recycling does. All-function-word shingles ("you don't have to")
        are skipped via the ≥5-char-word requirement.
        """
        if not assistant_text:
            return
        self.banned_phrases = _recycled_shingles(
            self._recent_shingles,
            _shingles(assistant_text),
            min_replies=PHRASE_REPLY_COUNT,
            max_banned=MAX_BANNED_PHRASES,
        )
        self.paragraph_banned = _recycled_shingles(
            self._recent_para_shingles,
            _shingles(assistant_text, PARA_SHINGLE_WORDS),
            min_replies=PARA_REPLY_COUNT,
            max_banned=MAX_PARA_BANNED,
        )

    def observe_scene_tenure(self, slug: str, speech_register: str) -> None:
        """Count consecutive in-scene turns; long tenure → voice check."""
        self._scene_tenure[slug] = self._scene_tenure.get(slug, 0) + 1
        if speech_register.strip():
            self._registers[slug] = speech_register.strip()

    def voice_checks(self) -> list[tuple[str, str]]:
        """(slug, register) for NPCs in scene long enough to drift."""
        return [
            (slug, self._registers[slug])
            for slug in sorted(self._scene_tenure)
            if self._scene_tenure[slug] >= REGISTER_REMINDER_AFTER and slug in self._registers
        ]

    def observe_goal(self, slug: str, goal: str) -> None:
        """Track how long ``slug`` has held (essentially) the same scene_goal."""
        prev = self._last_goal.get(slug)
        if prev is not None and similarity(goal, prev) > NEAR_DUPLICATE_THRESHOLD:
            self._goal_streak[slug] = self._goal_streak.get(slug, 0) + 1
        else:
            self._goal_streak[slug] = 0
        self._last_goal[slug] = goal
        self.static_goal_slugs = sorted(
            s for s, streak in self._goal_streak.items() if streak >= GOAL_STATIC_STREAK
        )

    def observe_movement(self, moved: bool) -> None:
        """Track consecutive non-moving turns (scene-level, fed by the writer).

        ``moved`` is True when any sheet's ``scene_goal`` or ``location``
        changed this turn, or a story-event fired. A streak of non-moving
        turns escalates into a hard "advance the scene" order.
        """
        self.stall_turns = 0 if moved else self.stall_turns + 1

    def prune(self, keep: Iterable[str]) -> None:
        """Drop goal/tenure tracking for NPCs no longer in scene.

        Tenure resets on exit by design — a returning NPC starts a fresh
        scene, and the voice-check line only belongs to long scenes.
        """
        keep_set = set(keep)
        for slug in list(self._goal_streak):
            if slug not in keep_set:
                self._goal_streak.pop(slug, None)
                self._last_goal.pop(slug, None)
        for slug in list(self._scene_tenure):
            if slug not in keep_set:
                self._scene_tenure.pop(slug, None)
                self._registers.pop(slug, None)
        self.static_goal_slugs = [s for s in self.static_goal_slugs if s in keep_set]


def phrase_text(text: str) -> str:
    """Canonical word-stream form used for phrase matching.

    Shared with the private-LLM guarded retry (providers/llm_local.py)
    so "did the reply contain a banned shingle" tokenizes exactly like
    the shingle detector — punctuation and case can't cause a miss.
    """
    return " ".join(_SHINGLE_WORD_RE.findall(text.casefold()))


def _shingles(text: str, n: int = PHRASE_SHINGLE_WORDS) -> set[str]:
    """Normalized n-word windows with at least one substantive word."""
    words = _SHINGLE_WORD_RE.findall(text.casefold())
    out: set[str] = set()
    for i in range(len(words) - (n - 1)):
        window = words[i : i + n]
        # Alpha count, not len() — "don't" is 4 letters, not 5.
        if max(sum(c.isalpha() for c in w) for w in window) >= 5:
            out.add(" ".join(window))
    return out


def _recycled_shingles(
    recent: deque[set[str]],
    seen: set[str],
    *,
    min_replies: int,
    max_banned: int,
) -> list[str]:
    """Push one reply's shingles into the ``recent`` window; return the ban list.

    ``recent`` is mutated in place (bounded deque — the oldest reply
    falls out). A shingle is recycled when it appears in ``min_replies``
    distinct replies inside the window.
    """
    recent.append(seen)
    counts = Counter(s for reply in recent for s in reply)
    recycled = sorted(
        (s for s, c in counts.items() if c >= min_replies),
        key=lambda s: (-counts[s], s),
    )
    return _dedupe_overlapping(recycled)[:max_banned]


def _dedupe_overlapping(shingles: list[str]) -> list[str]:
    """Greedy pick that skips shingles overlapping an already-picked one.

    One long tic produces several overlapping 4-word windows; without
    this, a single phrase would occupy every banned-phrase slot.
    """
    # Cluster shingles that share an ordered bigram (windows of one tic
    # chain transitively through their intermediates, regardless of the
    # order they arrive in), then keep one representative per cluster —
    # the first in the caller's count-then-alpha ranking.
    parent: dict[str, str] = {s: s for s in shingles}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    bigram_owner: dict[tuple[str, str], str] = {}
    for shingle in shingles:
        words = shingle.split()
        for bigram in pairwise(words):
            owner = bigram_owner.setdefault(bigram, shingle)
            if owner != shingle:
                parent[find(shingle)] = find(owner)
    picked: list[str] = []
    seen_clusters: set[str] = set()
    for shingle in shingles:
        cluster = find(shingle)
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        picked.append(shingle)
    return picked


_AGENCY = {
    "en": (
        "The player's message {pos} is what their character does and says — "
        "it is canon. Narrate its execution and the world's/NPC's response. "
        "Never replace, soften, reverse, or skip the player's declared action."
    ),
    "hu": (
        "A játékos {pos} üzenete az, amit a karaktere tesz és mond — ez kánon. "
        "Narráld a végrehajtását és a világ/NPC reakcióját. Soha ne cseréld "
        "le, ne szelídítsd, ne fordítsd meg és ne hagyd ki a játékos kimondott "
        "cselekvését."
    ),
}

# The contract's reference to the player message flips with placement:
# tail (after the user message) says "above", lead (spliced before it,
# the public/API-DM placement) says "below".
_AGENCY_POS = {
    "en": {"tail": "above", "lead": "below"},
    "hu": {"tail": "fenti", "lead": "lenti"},
}

_USER_REPEATED = {
    "en": (
        "The player repeated their previous input because your last reply did "
        "not resolve it. Resolve the stated action explicitly this turn."
    ),
    "hu": (
        "A játékos megismételte az előző üzenetét, mert az előző válaszod nem "
        "oldotta fel. Ebben a körben explicit módon játszd le a kimondott "
        "cselekvést."
    ),
}

_REPLY_LOOP = {
    "en": (
        "Your last two replies were nearly identical. Reuse no sentence or "
        "image from them; change what happens, not just the wording."
    ),
    "hu": (
        "Az utolsó két válaszod majdnem azonos volt. Egyetlen mondatot vagy "
        "képet se használj újra belőlük; az történjen másképp, ne csak a "
        "megfogalmazás."
    ),
}

_STATIC_GOAL = {
    "en": (
        "{names} has held the same scene goal for 3+ turns — they change "
        "tack this turn: back off, offer something, escalate differently, "
        "or shift focus."
    ),
    "hu": (
        "{names} 3+ köre ugyanazt a jelenetcélt hajtja — ebben a körben vált: "
        "visszalép, felajánl valamit, másképp eszkalál, vagy fókuszt vált."
    ),
}


# English-only by decision (docs/decisions/english-only.md). The full
# directive text rides the scene-sheets block (the channel the model
# demonstrably obeys — the prompt-tail world block was ignored 9/9 in
# the first 30-turn run); the contract carries only this binding pointer.
_WORLD_POINTER = (
    "A [World events] block rides the scene sheets this turn — narrating "
    "those facts into your reply is part of this contract."
)


# English-only by decision (docs/decisions/english-only.md). "Within the
# scene": moving forward must never mean a time skip (playtest, 2026-09-27).
_STALLED_REPLY = (
    "Your last reply re-told the same beat as the one before it. This reply must move the "
    "scene forward WITHIN the scene: an NPC decides or acts, the outcome of the player's "
    "action lands, or something new is revealed. Do not skip time or cut away."
)


# A "pass" turn: the player lets the scene run instead of acting. The
# 2026-09-27 playtest crawled through them ("Continue", "I listen
# patiently…") one sentence of news at a time.
_PASS_ALONE_RE = re.compile(
    r"^(?:continue|go on|keep going|carry on|and|then|\.{2,}|…)[\s.!?…]*$", re.IGNORECASE
)
_PASS_WATCH_RE = re.compile(r"^i (?:keep |just )?(?:listen|wait|watch)(?:ing)?\b", re.IGNORECASE)
_PASS_MAX_WORDS = 8
_LET_IT_RUN = (
    "The player is letting the scene run. Advance it by a full exchange: the NPCs keep "
    "talking and acting until something needs the player's input or decision. Do not skip time."
)


def player_passes(text: str) -> bool:
    """Whether the player's message only lets the scene run (short, no action of their own)."""
    clean = text.strip().strip("\"'").strip()
    if _PASS_ALONE_RE.match(clean):
        return True
    return len(clean.split()) <= _PASS_MAX_WORDS and bool(_PASS_WATCH_RE.match(clean))


def _scene_line(npc_slugs: Sequence[str], pc_slug: str | None) -> str:
    """Roster line: who may act this turn. Empty string when nobody is set."""
    who = ", ".join(npc_slugs)
    if pc_slug:
        who = f"{who} + pc {pc_slug}" if who else f"pc {pc_slug}"
    if not who:
        return ""
    return f"Scene: {who} — no other named character may act or speak."


def build_notes(  # noqa: PLR0913 — one keyword per contract input
    state: DirectorState,
    locale: str,
    *,
    tail: bool = True,
    npc_slugs: Sequence[str] = (),
    pc_slug: str | None = None,
    player_passes: bool = False,
) -> str:
    """Render the structured turn plan (tag + plan) for the next prompt.

    ``tail`` names where the caller will splice it: after the latest
    user message (private default) or before it (``tail=False``, the
    public/API-DM placement) — only the agency line's positional word
    changes. ``npc_slugs``/``pc_slug`` name the roster for the Scene line.

    Notes are bucketed into Must / Must not / Voice. Must always carries
    the unconditional agency line, so it is never empty; the other two
    sections are omitted when they have no notes.
    """
    loc = locale if locale in _AGENCY else "en"
    must: list[str] = [_AGENCY[loc].format(pos=_AGENCY_POS[loc]["tail" if tail else "lead"])]
    # The clock line rides the scene-sheets block, not this note — the
    # 14B ignores the contract for world info (2026-09-25 replay-002).
    if state.event_directives:
        must.append(_WORLD_POINTER)
    if state.reply_stalled:
        must.append(_STALLED_REPLY)
    if player_passes:
        must.append(_LET_IT_RUN)
    if state.user_repeated:
        must.append(_USER_REPEATED[loc])
    if state.static_goal_slugs:
        must.append(_STATIC_GOAL[loc].format(names=", ".join(state.static_goal_slugs)))
    # English-only by decision (docs/decisions/english-only.md): stall order.
    if state.stall_turns >= STALL_ESCALATE_AFTER:
        must.append(
            f"The scene has not moved for {state.stall_turns} turns — introduce a "
            "concrete development NOW: an arrival, a discovery, an interruption, or a "
            "deadline landing; do not re-describe the standoff."
        )

    must_not: list[str] = []
    # English-only by decision (docs/decisions/english-only.md): new note.
    if state.player_echo:
        must_not.append(
            "Do not repeat or restate the player's own message back at them; "
            "narrate what happens NEXT — the world's and the NPCs' response."
        )
    if state.reply_similarity > REPLY_SIMILARITY_THRESHOLD:
        must_not.append(_REPLY_LOOP[loc])
    if state.banned_phrases:
        phrases = "; ".join(f"“{p}”" for p in state.banned_phrases)
        must_not.append(
            f"Reusing these exact phrases across turns: {phrases}. "
            "Do not use them or close variants this turn — describe something else."
        )

    # English-only by decision (docs/decisions/english-only.md): voice checks.
    voice: list[str] = [
        f"Voice check — {slug} has been in this scene a while and voices drift: "
        f"their register is “{register[:140]}”. Keep their next lines in it."
        for slug, register in state.voice_checks()
    ]

    lines = [TURN_CONTRACT_TAG, "[Turn plan — non-negotiable this turn]"]
    scene = _scene_line(npc_slugs, pc_slug)
    if scene:
        lines.append(scene)
    for header, bucket in (("Must:", must), ("Must not:", must_not), ("Voice:", voice)):
        if bucket:
            lines.append(header)
            lines.extend(f"- {note}" for note in bucket)
    return "\n".join(lines)
