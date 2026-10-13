"""Deterministic story clock + event engine.

The DM-LLM must not invent the plot. Time and story events are owned by
code: a run-scoped ``clock.md`` advances by deterministic rules, and
authored event files under ``runs/<slug>/events/`` fire when their time
and state conditions hold. Fired events mutate the world through the
existing vault write paths and hand the DM a short *directive* — prose
describing what has already happened, rendered into the next turn's
scene-sheets block by the injector — along with the in-fiction time. A
clock advance is shown to the player by code (``aidm.time_shown``), not
left to the model. The model narrates the world's motion; it never
decides it.

Runs without a ``clock.md`` opt out entirely — the engine is a no-op
there, so existing runs are untouched until a clock file appears.

Called from ``SceneSheetWriter`` at end-of-turn, *after* the per-sheet
patches and auto-exits have landed: this turn's player consequences are
already on disk when trigger conditions are read, so player pre-emption
(killing the scout, escorting the medic, admitting the stranger) is
handled structurally rather than by rules.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from aidm import frontmatter
from aidm.scene_state import SceneRegistry, npc_named, place_matches
from aidm.vault import Vault, VaultPathError
from aidm.vault_schema import CLOCK_SEGMENTS, NPC_STATE_FIELDS, SCHEMAS

SEGMENTS = CLOCK_SEGMENTS
# Minimum player turns between two honored writer time hints.
BASELINE_TURNS_PER_SEGMENT = 4
# Fired-event directives rendered into one prompt. Overflow stays
# queued on disk (``narrated: false``) for the following turns.
MAX_DIRECTIVES_PER_TURN = 2
# Narration states that retire a directive ("pre-empted": an arrival the
# player already lived through, see ``_satisfy``).
_NARRATION_DONE = ("true", "abandoned", "pre-empted")
# A rendered directive stays live until the DM's reply shares at least
# this fraction of its content words (per-turn window, so the bar can be
# meaningful — the scorer's whole-transcript 0.2 proved toothless), or
# until it has been shown this many times, whichever comes first.
NARRATION_OVERLAP_MIN = 0.25
MAX_DIRECTIVE_RENDERS = 3
# Time-log bullets kept in clock.md's body.
_TIME_LOG_MAX = 12
_TIME_LOG_HEADER = "## Time log"

DIRECTIVE_HEADER = "## Directive"
ON_FIRE_HEADER = "## On fire"
# Optional one-line imperative rendered to the DM INSTEAD of the full
# Directive prose. The 60-90-word atmospheric paragraphs read as
# skippable color to the 14B (three 30-turn runs, ≤3/9 narrated); a
# terse order is the last prompt-shaped compliance lever.
TELL_HEADER = "## Tell"

# NPC state fields an event mutation may set. Server-owned fields are
# excluded; ``in_scene`` stays registry/writer territory.
EVENT_MUTABLE_NPC_FIELDS: frozenset[str] = NPC_STATE_FIELDS - frozenset(
    {"type", "last_updated", "in_scene"}
)

# Synonyms: the persona prompts write ``deceased``, authored branches
# target ``dead`` — any gone status satisfies any gone requirement.
_GONE_STATUSES = frozenset({"dead", "deceased"})


@dataclass
class ClockState:
    """Parsed ``clock.md`` — day/segment plus the absolute turn counter."""

    day: int = 1
    segment: str = "morning"
    turn: int = 0
    last_advance_turn: int = 0
    # Turn of the last honored writer hint (0 = never). Hints are
    # rate-limited to one per ``turns_per_segment`` turns: 07-12 replay-003
    # let a hint drive 12 of the last 12 advances and ended on day 12.
    last_hint_turn: int = 0
    # Baseline-tick pace, overridable per run via clock.md frontmatter.
    # Default 4 = 20 quiet turns per fiction-day; the eval-30t run pins
    # 2 so its day-2/3 event chains are reachable inside 30 turns.
    turns_per_segment: int = BASELINE_TURNS_PER_SEGMENT
    log: list[str] = field(default_factory=list)

    @property
    def abs_segment(self) -> int:
        return self.day * len(SEGMENTS) + SEGMENTS.index(self.segment)

    def line(self) -> str:
        return f"Day {self.day}, {self.segment}."


@dataclass
class Event:
    """One authored event file: trigger frontmatter + directive + mutations.

    ``body`` is the raw markdown body, preserved verbatim on write-back —
    only the frontmatter mutates at runtime, so authored prose (including
    sections beyond Directive/On fire) survives every status flip.
    """

    slug: str
    fm: dict[str, Any]
    body: str
    directive: str
    mutations: list[tuple[str, ...]]
    tell: str = ""
    # A real status transition this pass (feeds ``changed_events``: the
    # side-band and the director's "world moved" signal).
    status_changed: bool = False
    # Frontmatter differs from disk (status flips, render counts,
    # narration stamps) — persisted, but not a world change.
    dirty: bool = False

    @property
    def rendered_directive(self) -> str:
        """What the DM actually sees: the terse Tell line when authored,
        the full Directive prose otherwise. Confirmation matches against
        this same text — the contract is what was shown."""
        return self.tell or self.directive

    @property
    def status(self) -> str:
        return str(self.fm.get("status") or "pending")

    def set_status(self, value: str) -> None:
        self.fm["status"] = value
        self.status_changed = self.dirty = True


@dataclass
class EngineOutcome:
    """What one end-of-turn engine pass produced."""

    clock_line: str
    directives: list[str] = field(default_factory=list)
    # (slug, title, new status) for every event whose status changed —
    # feeds the ``world.event`` side-band emission. Re-renders and
    # narration stamps are not status changes and never appear here.
    changed_events: list[tuple[str, str, str]] = field(default_factory=list)


def run_end_of_turn(  # noqa: PLR0913 — keyword-only turn inputs
    vault: Vault,
    registry: SceneRegistry,
    *,
    time_advance_hint: str | None,
    assistant_text: str = "",
    user_text: str = "",
    with_pc: Collection[str] = frozenset(),
    frozen: bool = False,
) -> EngineOutcome | None:
    """One engine pass. Returns ``None`` when the run has no clock.md.

    ``frozen`` (the PC is inside their First Nightmare, aidm.first_nightmare):
    the clock does not move — scene changes are consumed and dropped, the
    writer's time hint ignored, the turn not counted — and no event is
    fired, confirmed or rendered: the waking world waits.

    ``assistant_text`` is the turn's DM reply — used to *confirm* that
    previously rendered directives were actually narrated before they
    are retired (see ``confirm_narrated``). Together with ``user_text``
    (the player's message) it is also this turn's exchange: an event that
    would move or change the status of an NPC it names is held. NPC slugs
    in ``with_pc`` (the PC writer's presence judgment) count as named.
    """
    clock = read_clock(vault)
    if clock is None:
        return None
    if frozen:
        dropped = registry.consume_scene_changes()
        logger.info(
            "story_events: clock frozen (First Nightmare) — {} scene change(s), hint {!r} dropped",
            dropped,
            time_advance_hint,
        )
        return EngineOutcome(clock_line=clock.line())
    advance_clock(clock, hint=time_advance_hint, scene_changed=registry.consume_scene_changes() > 0)
    events = load_events(vault)
    confirm_narrated(events, assistant_text)
    cancel_dead_subjects(events, vault)
    fire_due_events(
        events,
        clock,
        vault,
        registry,
        turn_text=f"{user_text}\n{assistant_text}",
        with_pc=with_pc,
    )
    directives = render_directives(events)
    persist(events, clock, vault)
    return EngineOutcome(
        clock_line=clock.line(),
        directives=directives,
        changed_events=[
            (e.slug, str(e.fm.get("title") or e.slug), e.status) for e in events if e.status_changed
        ],
    )


# --- clock ---------------------------------------------------------------


def read_clock(vault: Vault) -> ClockState | None:
    """Parse the run's ``clock.md``; None when absent (engine disabled)."""
    raw = vault.read_clock()
    if raw is None:
        return None
    fm, body = frontmatter.split(raw)
    segment = str(fm.get("segment") or "morning")
    if segment not in SEGMENTS:
        logger.warning("story_events: invalid clock segment {!r}, resetting to morning", segment)
        segment = "morning"
    clock = ClockState(
        day=max(1, _as_int(fm.get("day"), 1)),
        segment=segment,
        turn=max(0, _as_int(fm.get("turn"), 0)),
        last_advance_turn=max(0, _as_int(fm.get("last_advance_turn"), 0)),
        last_hint_turn=max(0, _as_int(fm.get("last_hint_turn"), 0)),
        turns_per_segment=max(1, _as_int(fm.get("turns_per_segment"), BASELINE_TURNS_PER_SEGMENT)),
        log=_parse_time_log(body),
    )
    # Legacy ``transition_*`` keys (the retired narration hold) are
    # ignored here and dropped on the next write.
    return clock


def advance_clock(clock: ClockState, *, hint: str | None, scene_changed: bool) -> None:
    """Apply at most one advance for this turn; the model never owns the value.

    Precedence: the writer-patch hint (clamped to one day, at most one
    per ``turns_per_segment`` turns) beats the scene-change tick (also at
    most one per ``turns_per_segment`` turns since the last advance). There
    is no baseline tick: time moves only when the narration moved it or
    the scene changed. A quiet-turns tick cut into scenes still playing
    out (playtest 2026-09-27: no forced time skip before a scene ends).
    The advance itself is shown to the player by code at the next
    reply's start (``aidm.time_shown``).
    """
    clock.turn += 1
    if (
        hint
        and clock.last_hint_turn
        and clock.turn - clock.last_hint_turn < clock.turns_per_segment
    ):
        logger.info("story_events: time hint {!r} suppressed (rate limit)", hint)
        hint = None
    if hint in ("next_day", "next_segment"):
        clock.last_hint_turn = clock.turn
    if (
        scene_changed
        and clock.last_advance_turn
        and clock.turn - clock.last_advance_turn < clock.turns_per_segment
    ):
        # A walk from gate to hall to yard is one evening, not three.
        logger.info("story_events: scene-change tick suppressed (rate limit)")
        scene_changed = False
    before_day, before_segment = clock.day, clock.segment
    if hint == "next_day":
        clock.day += 1
        clock.segment = SEGMENTS[0]
        reason = "narrated time skip"
    elif hint == "next_segment" or scene_changed:
        _step_segment(clock)
        reason = "narrated time passing" if hint == "next_segment" else "scene change"
    else:
        return
    clock.last_advance_turn = clock.turn
    _log_time(
        clock,
        f"T{clock.turn}: day {before_day} {before_segment} → day {clock.day} {clock.segment} "
        f"({reason})",
    )


def _log_time(clock: ClockState, entry: str) -> None:
    clock.log.append(entry)
    clock.log = clock.log[-_TIME_LOG_MAX:]


_CLOCK_LINE_RE = re.compile(r"^Day (\d+), (\w+)\.$")
# The line code shows the player (and the DM's context) at the start of
# the reply after a clock advance. Per-locale templates keyed like the
# other locale dicts; a new locale adds its own template (and segment
# words) here.
# Simplification: en only (docs/decisions/english-only.md); unknown locales fall back to en.
_TIME_SHOWN: dict[str, str] = {"en": "*Day {day}, {segment}.*"}


def time_shown_line(clock_line: str, locale: str) -> str:
    """The shown-time line for a ``ClockState.line()`` value; "" if unparseable."""
    match = _CLOCK_LINE_RE.match(clock_line)
    if match is None:
        return ""
    template = _TIME_SHOWN.get(locale, _TIME_SHOWN["en"])
    return template.format(day=match[1], segment=match[2])


_TIME_SHOWN_PREFIX_RE = re.compile(r"^\s*\*Day \d+, [a-z]+\.\*\s*")


def strip_time_shown(reply: str) -> str:
    """A DM reply without the code-emitted leading time line (not model prose)."""
    return _TIME_SHOWN_PREFIX_RE.sub("", reply, count=1)


# Simplification: the time-transition directive is retired; this recognizer stays only
# because providers/llm_local.py still imports it — delete with that call site.
_TRANSITION_RE = re.compile(r"^Time has passed: it is now day \d+, (\w+) ")


def is_time_transition(directive_text: str) -> bool:
    """True for a legacy time-transition directive (no longer produced)."""
    return bool(_TRANSITION_RE.match(directive_text))


def _step_segment(clock: ClockState) -> None:
    idx = SEGMENTS.index(clock.segment)
    if idx + 1 < len(SEGMENTS):
        clock.segment = SEGMENTS[idx + 1]
    else:
        clock.day += 1
        clock.segment = SEGMENTS[0]


# --- events --------------------------------------------------------------


def load_events(vault: Vault) -> list[Event]:
    """Parse every event file under the run's ``events/``; bad files skipped."""
    out: list[Event] = []
    for slug, raw in vault.list_events():
        fm, body = frontmatter.split(raw)
        if str(fm.get("type") or "") != "event":
            logger.warning("story_events: {} is not type: event — skipped", slug)
            continue
        out.append(
            Event(
                slug=slug,
                fm=fm,
                body=body,
                directive=_section(body, DIRECTIVE_HEADER),
                tell=_section(body, TELL_HEADER),
                mutations=parse_mutations(_section(body, ON_FIRE_HEADER)),
            )
        )
    return out


def cancel_dead_subjects(events: list[Event], vault: Vault) -> None:
    """Cancel pending events whose subject NPC is permanently gone.

    Exemption: an event that *targets* a gone status (an authored
    alternate branch like "the raid comes anyway, leaderless") survives.
    Chains cancel as a unit through the authored convention that every
    step of a chain carries the same ``subject_npc``.
    """
    for event in events:
        if event.status != "pending":
            continue
        subject = str(event.fm.get("subject_npc") or "").strip()
        if not subject:
            continue
        required = str(event.fm.get("trigger_npc_status") or "").strip().lower()
        if required in _GONE_STATUSES:
            continue
        status = _npc_state_field(vault, subject, "status")
        if status in _GONE_STATUSES:
            event.set_status("cancelled")
            logger.info(
                "story_events: cancelled {} — subject {} is {}", event.slug, subject, status
            )


def fire_due_events(  # noqa: PLR0913 — keyword-only turn inputs
    events: list[Event],
    clock: ClockState,
    vault: Vault,
    registry: SceneRegistry,
    *,
    turn_text: str = "",
    with_pc: Collection[str] = frozenset(),
) -> None:
    """Fire, defer, or expire every pending event against the fresh state.

    At most one fire per chain per pass: a large clock jump could
    otherwise cascade a whole chain in a single turn (step 1's mutation
    satisfying step 2's condition), handing the DM contradictory
    directives in one contract. Later steps stay pending and re-evaluate
    next turn.

    An event that moves its subject (location/status mutation) is treated
    as conditions-false while ``turn_text`` — this turn's player message +
    DM reply — names the subject: she is evidently around the player, and
    a stale sheet location must not send her off-screen (2026-09-25:
    "Marta left alone before dawn" fired while she walked beside him).
    A subject in ``with_pc`` (slugs physically with the PC this turn, per
    the PC writer) is held the same way — escort narration often never
    names her.
    """
    fired_chains: set[str] = set()
    for event in _pending_in_order(events):
        if not _time_eligible(event, clock):
            continue
        chain = str(event.fm.get("chain") or "").strip()
        if chain and chain in fired_chains:
            continue
        subject = str(event.fm.get("subject_npc") or "").strip()
        if (
            subject
            and _npc_state_field(vault, subject, "status") == "missing"
            and not _resolves_subject_status(event, subject)
        ):
            # A missing subject holds the event regardless of policy;
            # expiry below still reaps it eventually. An event that rewrites
            # its subject's status IS the return path ("Marta staggers back"),
            # so it must not wait on the very state it resolves.
            logger.info("story_events: waiting {} — subject {} is missing", event.slug, subject)
            _maybe_expire_held(event, clock)
            continue
        if _conditions_hold(event, vault, registry):
            if _moves_subject(event, subject) and (
                subject in with_pc or npc_named(turn_text, _npc_name(vault, subject), subject)
            ):
                if str(event.fm.get("if_subject_present") or "hold") == "satisfied":
                    # An arrival the player already lived through: firing it
                    # would re-stage it, holding it would block it forever
                    # (2026-09-27 playtest 3: lyra-at-the-gate held 8 turns).
                    if _satisfy(event, clock, vault) and chain:
                        fired_chains.add(chain)
                    continue
                logger.info(
                    "story_events: held {} — subject {} this turn",
                    event.slug,
                    "with PC" if subject in with_pc else "named",
                )
                _defer_or_expire(event, clock)
                continue
            if not _fire(event, clock, vault):
                continue
            _cancel_fired_siblings(event, events)
            if chain:
                fired_chains.add(chain)
            continue
        # Silent before 2026-09-27: playtest 3 could not tell why an eligible
        # event never fired.
        logger.info("story_events: waiting {} — trigger conditions not met", event.slug)
        _defer_or_expire(event, clock)


def confirm_narrated(events: list[Event], assistant_text: str) -> None:
    """Retire rendered directives only once the DM demonstrably told them.

    The first full 30-turn run (2026-07-11) proved a single render is
    worthless: the model ignored 9/9 directives while pursuing its own
    improvised plot, and the old mark-on-render policy silently retired
    every one. A directive now stays live — re-rendered each turn —
    until the turn's reply demonstrably contains it, or the render cap
    is hit, which retires it LOUDLY as unnarrated so the scorer and
    judge see the failure on the event file itself.

    Confirmation must survive paraphrase (replay-003: the DM *did*
    narrate Lyra at the fence, scored 0.04 on exact-word overlap):
    a reply confirms when it names one of the directive's proper-name
    anchors AND echoes ≥3 of its other content words (5-char prefix
    stems, so "staggers"≈"staggered"); the plain overlap fraction stays
    as the fallback for anchor-less directives.
    """
    if not assistant_text:
        return
    reply_stems = _word_stems(assistant_text)
    for event in events:
        if event.status != "fired":
            continue
        if str(event.fm.get("narrated") or "false") in _NARRATION_DONE:
            continue  # settled — do not re-warn every turn
        renders = _as_int(event.fm.get("renders"), 0)
        if renders <= 0:
            continue  # never shown to the model yet
        confirmed, detail = _narration_matches(event.rendered_directive, reply_stems)
        if confirmed:
            event.fm["narrated"] = "true"
            event.dirty = True
            logger.info("story_events: {} narration confirmed ({})", event.slug, detail)
        elif renders >= MAX_DIRECTIVE_RENDERS:
            event.fm["narrated"] = "abandoned"
            event.dirty = True
            logger.warning(
                "story_events: {} directive ABANDONED after {} renders — "
                "the DM never narrated it ({})",
                event.slug,
                renders,
                detail,
            )


# Proper-name anchors: capitalized words that are not sentence-initial.
_ANCHOR_RE = re.compile(r"(?<=[a-z0-9,;:— ] )([A-Z][a-z]{2,})\b")
_STEM_LEN = 5
# Non-anchor content-word echoes required alongside an anchor hit.
_ANCHOR_SUPPORT_WORDS = 3


def _word_stems(text: str) -> set[str]:
    return {w[:_STEM_LEN] for w in _content_words(text)}


def directive_anchor_words(directive_text: str) -> list[str]:
    """Proper-noun anchors in a directive, raw and deduped (order-preserving).

    Same regex ``confirm_narrated``/``directive_covered`` use internally
    (there, stemmed and casefolded for matching) — exposed raw here for
    the logit-bias prototype, which needs the actual capitalized word to
    tokenize, not a matching key. Inherits ``_ANCHOR_RE``'s sentence-
    initial blind spot (a name as the very first word of the directive
    won't match — same anti-noise tradeoff as the narration check);
    later mentions of the same name in a multi-sentence directive still
    catch it.
    """
    seen: dict[str, None] = {}
    for word in _ANCHOR_RE.findall(directive_text):
        seen.setdefault(word, None)
    return list(seen)


# Authored Tell lines open with this imperative; it is an order to the
# DM, not directive content, so it must not dilute the overlap fraction.
_TELL_PREFIX_RE = re.compile(r"^\s*narrate now:\s*", re.IGNORECASE)
# Rendered event directives lead with their in-fiction time — staging,
# not content, so it must not dilute the overlap fraction either.
_EVENT_TIME_PREFIX_RE = re.compile(r"^\s*\(Day \d+, \w+\)\s*")


def _narration_matches(directive: str, reply_stems: set[str]) -> tuple[bool, str]:
    """Paraphrase-tolerant check that a reply narrates the directive."""
    directive = _TELL_PREFIX_RE.sub("", _EVENT_TIME_PREFIX_RE.sub("", directive))
    anchors = {a.casefold()[:_STEM_LEN] for a in _ANCHOR_RE.findall(directive)}
    words = _word_stems(directive)
    plain = words - anchors
    anchor_hits = sum(1 for a in anchors if a in reply_stems)
    plain_hits = sum(1 for w in plain if w in reply_stems)
    if anchors and anchor_hits >= 1 and plain_hits >= _ANCHOR_SUPPORT_WORDS:
        return True, f"anchor+{plain_hits} words"
    overlap = (anchor_hits + plain_hits) / len(words) if words else 1.0
    return overlap >= NARRATION_OVERLAP_MIN, f"overlap {overlap:.2f}"


def render_directives(events: list[Event]) -> list[str]:
    """Up to MAX_DIRECTIVES_PER_TURN live fired directives, render-counted.

    Rendering no longer retires a directive — ``confirm_narrated``
    does, next turn, against the actual reply. ``renders`` is stamped
    on the event file so the retry budget survives restart. Each line
    leads with the event's in-fiction time ("(Day 5, night) …") so the
    DM stages it in time instead of pasting it into the current moment.
    """
    queue = [
        e
        for e in events
        if e.status == "fired"
        and str(e.fm.get("narrated") or "false") not in _NARRATION_DONE
        and e.directive
    ]
    queue.sort(key=lambda e: (-_as_int(e.fm.get("priority"), 0), e.slug))
    rendered = queue[:MAX_DIRECTIVES_PER_TURN]
    for event in rendered:
        event.fm["renders"] = _as_int(event.fm.get("renders"), 0) + 1
        event.fm.setdefault("narrated", "false")
        event.dirty = True
    return [_event_time_prefix(e) + e.rendered_directive for e in rendered]


def _event_time_prefix(event: Event) -> str:
    day = _as_int(event.fm.get("fired_day"), 0)
    segment = str(event.fm.get("fired_segment") or "")
    return f"(Day {day}, {segment}) " if day > 0 and segment in SEGMENTS else ""


def directive_covered(reply: str, directive_text: str) -> bool:
    """True if the reply appears to have narrated the directive.

    Same paraphrase-tolerant match as ``confirm_narrated`` — called
    from the guarded retry before the player sees the draft, for event
    directives only (the guard filters time transitions out with
    ``is_time_transition``).
    """
    if not reply or not directive_text:
        return False
    confirmed, _ = _narration_matches(directive_text, _word_stems(reply))
    return confirmed


# Latin-1 Supplement + Latin Extended-A cover Hungarian's accented
# letters — without them accented content words fragment and the
# guarded retry (providers/llm_local.py) false-negatives on Hungarian
# replies, forcing a wasted extra completion on every event turn.
# No apostrophe in the class: "Lyra's" must stem like "Lyra\u2019s" (\u2192 lyra), not "lyra'".
_WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u017F]{4,}")
_OVERLAP_STOPWORDS = frozenset(
    {
        "have",
        "that",
        "this",
        "with",
        "from",
        "your",
        "them",
        "they",
        "their",
        "there",
        "been",
        "before",
        "toward",
        "into",
        "over",
        "under",
        "about",
        "what",
        "when",
        "where",
        "will",
        "would",
        "gone",
        "went",
        "never",
        "still",
        "just",
    }
)


def _content_words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RE.findall(text) if w.casefold() not in _OVERLAP_STOPWORDS}


def persist(events: list[Event], clock: ClockState, vault: Vault) -> None:
    """Write back the clock and every event whose frontmatter moved."""
    vault.write_clock(_serialize_clock(clock))
    for event in events:
        if not event.dirty:
            continue
        try:
            vault.write_event(event.slug, _serialize_event(event))
        except (VaultPathError, OSError) as exc:
            logger.warning("story_events: write failed for {} — {}", event.slug, exc)


# --- trigger evaluation ---------------------------------------------------


def _pending_in_order(events: list[Event]) -> list[Event]:
    def key(e: Event) -> tuple[int, int, str]:
        return (_trigger_abs(e), -_as_int(e.fm.get("priority"), 0), e.slug)

    return sorted((e for e in events if e.status == "pending"), key=key)


def _trigger_seg_idx(event: Event) -> int:
    seg = str(event.fm.get("trigger_segment") or "")
    return SEGMENTS.index(seg) if seg in SEGMENTS else 0


def _trigger_abs(event: Event) -> int:
    return _as_int(event.fm.get("trigger_day"), 1) * len(SEGMENTS) + _trigger_seg_idx(event)


def _time_eligible(event: Event, clock: ClockState) -> bool:
    return clock.abs_segment >= _trigger_abs(event)


def _resolves_subject_status(event: Event, subject: str) -> bool:
    return any(m[0] == "npc" and m[1] == subject and m[2] == "status" for m in event.mutations)


def _moves_subject(event: Event, subject: str) -> bool:
    return bool(subject) and any(
        m[0] == "npc" and m[1] == subject and m[2] in ("location", "status")
        for m in event.mutations
    )


def _conditions_hold(event: Event, vault: Vault, registry: SceneRegistry) -> bool:
    subject = str(event.fm.get("subject_npc") or "").strip()
    if subject:
        if not _status_satisfies(
            str(event.fm.get("trigger_npc_status") or "").strip().lower(),
            _npc_state_field(vault, subject, "status"),
        ):
            return False
        # The writer records free text ('hall of ridgeway fallout shelter');
        # the authored place must be contained in it, not equal to it.
        required_loc = str(event.fm.get("trigger_npc_location") or "").strip()
        if required_loc and not place_matches(
            required_loc, _npc_state_field(vault, subject, "location")
        ):
            return False
        # "Only if (not) engaged with the player" — an off-screen event
        # like "she gives up and leaves" must not fire mid-conversation.
        # First live smoke run (2026-07-11) fired lyra-leaves while the
        # player was negotiating with her; this condition is the fix.
        required_in_scene = str(event.fm.get("trigger_npc_in_scene") or "").strip().lower()
        if required_in_scene:
            actual = _npc_state_field(vault, subject, "in_scene") or "false"
            if actual != required_in_scene:
                return False
    required_pc_loc = str(event.fm.get("trigger_pc_location") or "").strip()
    if required_pc_loc:
        pc = registry.current_pc_slug
        if not pc:
            return False
        pc_state = vault.read_state("pc", pc) or ""
        fm, _ = frontmatter.split(pc_state)
        if not place_matches(required_pc_loc, str((fm or {}).get("location") or "")):
            return False
    return True


def _status_satisfies(required: str, actual: str) -> bool:
    """Gone statuses are synonyms; no requirement means "subject not gone"."""
    if required in _GONE_STATUSES:
        return actual in _GONE_STATUSES
    if required:
        return actual == required
    return actual not in _GONE_STATUSES


def _defer_or_expire(event: Event, clock: ClockState) -> None:
    policy = str(event.fm.get("defer_policy") or "hold")
    if policy == "expire":
        event.set_status("expired")
        logger.info("story_events: expired {} — conditions false at trigger (one-shot)", event.slug)
        return
    if policy == "next_segment" and clock.abs_segment > _trigger_abs(event) + 1:
        event.set_status("expired")
        logger.info("story_events: expired {} — next_segment recheck window passed", event.slug)
        return
    _maybe_expire_held(event, clock)


def _maybe_expire_held(event: Event, clock: ClockState) -> None:
    days = _as_int(event.fm.get("expires_after_days"), 0)
    if days > 0 and clock.day > _as_int(event.fm.get("trigger_day"), 1) + days:
        event.set_status("expired")
        logger.info("story_events: expired {} — held past expiry window", event.slug)


def _fire(event: Event, clock: ClockState, vault: Vault) -> bool:
    """Persist ``status: fired`` first, then apply mutations.

    Order is the idempotency guard: mutations (beat/notes appends) are
    not re-runnable, so they only land once the event can no longer read
    as pending on disk. A failed status write leaves the event untouched
    to fire cleanly on a later turn. Returns whether it fired.
    """
    fired_fm = {
        **event.fm,
        "status": "fired",
        "fired_turn": clock.turn,
        "fired_day": clock.day,
        "fired_segment": clock.segment,
    }
    fired_fm.setdefault("narrated", "false")
    try:
        vault.write_event(event.slug, frontmatter.serialize(fired_fm, event.body))
    except (VaultPathError, OSError) as exc:
        logger.warning("story_events: not firing {} — status write failed: {}", event.slug, exc)
        return False
    event.fm = fired_fm
    event.status_changed = True
    for mutation in event.mutations:
        try:
            _apply_mutation(mutation, vault)
        except (VaultPathError, OSError) as exc:
            logger.warning(
                "story_events: mutation {} failed for {} — {}", mutation, event.slug, exc
            )
    logger.info("story_events: FIRED {} (day {} {})", event.slug, clock.day, clock.segment)
    return True


def _satisfy(event: Event, clock: ClockState, vault: Vault) -> bool:
    """Mark an arrival pre-empted: fired, no mutations, no directive."""
    done_fm = {
        **event.fm,
        "status": "fired",
        "fired_turn": clock.turn,
        "fired_day": clock.day,
        "fired_segment": clock.segment,
        "narrated": "pre-empted",
    }
    try:
        vault.write_event(event.slug, frontmatter.serialize(done_fm, event.body))
    except (VaultPathError, OSError) as exc:
        logger.warning("story_events: not satisfying {} — status write failed: {}", event.slug, exc)
        return False
    event.fm = done_fm
    event.status_changed = True
    logger.info("story_events: satisfied {} — subject already with the player", event.slug)
    return True


def _cancel_fired_siblings(fired: Event, events: list[Event]) -> None:
    """Authored alternate branches are exclusive per (chain, chain_step):
    the winning branch fires, the losers are cancelled."""
    chain = str(fired.fm.get("chain") or "").strip()
    step = fired.fm.get("chain_step")
    if not chain or step is None:
        return
    for event in events:
        if event is fired or event.status != "pending":
            continue
        if str(event.fm.get("chain") or "").strip() != chain:
            continue
        if event.fm.get("chain_step") != step:
            continue
        event.set_status("cancelled")
        logger.info("story_events: cancelled {} — sibling branch {} fired", event.slug, fired.slug)


# --- mutations ------------------------------------------------------------


def parse_mutations(section: str) -> list[tuple[str, ...]]:
    """Parse ``## On fire`` bullets into typed mutations.

    Grammar (one bullet each): ``- <npc-slug>.<state-field>: <value>``,
    ``- beat: <text>``, ``- note: <text>``. Unknown shapes are dropped
    with a warning — a typo must not stop the event from firing.
    """
    out: list[tuple[str, ...]] = []
    for raw in section.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        body = line[2:].strip()
        key, sep, value = body.partition(":")
        if not sep:
            logger.warning("story_events: unparseable mutation line {!r}", line)
            continue
        key = key.strip()
        value = value.strip()
        if key == "beat":
            out.append(("beat", value))
            continue
        if key == "note":
            out.append(("note", value))
            continue
        slug, dot, fld = key.partition(".")
        if not dot or not slug or fld not in EVENT_MUTABLE_NPC_FIELDS:
            logger.warning("story_events: invalid mutation target {!r}", key)
            continue
        enums = SCHEMAS["npc_state"].enums
        if fld in enums and value not in enums[fld]:
            logger.warning("story_events: mutation {} → {!r} fails enum, dropped", key, value)
            continue
        out.append(("npc", slug.strip(), fld, value))
    return out


def _apply_mutation(mutation: tuple[str, ...], vault: Vault) -> None:
    kind = mutation[0]
    if kind == "beat":
        current = _read_optional(vault, "beat.md")
        vault.write_beat(current.rstrip() + "\n\n" + mutation[1] + "\n" if current else mutation[1])
        return
    if kind == "note":
        vault.append_note("world-events", mutation[1])
        return
    _, slug, fld, value = mutation
    state = vault.read_state("npc", slug)
    if state is None:
        logger.warning("story_events: no state file for npc {!r}, mutation skipped", slug)
        return
    fm, body = frontmatter.split(state)
    if not fm:
        logger.warning("story_events: npc {!r} state has no frontmatter, mutation skipped", slug)
        return
    fm[fld] = value
    vault.write_npc_state(slug, frontmatter.serialize(fm, body))


# --- serialization ---------------------------------------------------------


def _serialize_clock(clock: ClockState) -> str:
    fm: dict[str, Any] = {
        "type": "clock",
        "day": clock.day,
        "segment": clock.segment,
        "turn": clock.turn,
        "last_advance_turn": clock.last_advance_turn,
        "last_hint_turn": clock.last_hint_turn,
        "turns_per_segment": clock.turns_per_segment,
    }
    lines = [_TIME_LOG_HEADER, ""]
    if clock.log:
        lines.extend(f"- {entry}" for entry in clock.log)
    else:
        lines.append("_(no advances yet)_")
    return frontmatter.serialize(fm, "\n".join(lines) + "\n")


def _serialize_event(event: Event) -> str:
    """Frontmatter mutates at runtime; the authored body is preserved verbatim."""
    return frontmatter.serialize(event.fm, event.body)


def _parse_time_log(body: str) -> list[str]:
    if _TIME_LOG_HEADER not in body:
        return []
    chunk = body.split(_TIME_LOG_HEADER, 1)[1]
    return [
        line.strip()[2:].strip() for line in chunk.splitlines() if line.strip().startswith("- ")
    ]


def _section(body: str, header: str) -> str:
    """Text of one ``## ...`` section, up to the next ``## `` header."""
    if header not in body:
        return ""
    after = body.split(header, 1)[1]
    for next_header in ("\n## ", "\n# "):
        if next_header in after:
            after = after.split(next_header, 1)[0]
    return after.strip()


# --- small helpers ----------------------------------------------------------


def _npc_state_field(vault: Vault, slug: str, key: str) -> str:
    state = vault.read_state("npc", slug)
    if not state:
        return ""
    fm, _ = frontmatter.split(state)
    return str((fm or {}).get(key) or "").strip().lower()


def _npc_name(vault: Vault, slug: str) -> str:
    fm, _ = frontmatter.split(_read_optional(vault, f"npcs/{slug}.md"))
    return str((fm or {}).get("name") or "").strip()


def _read_optional(vault: Vault, path: str) -> str:
    try:
        text = vault.read_file(path)
    except VaultPathError:
        return ""
    marker = "\n\n# Referenced by\n"
    return text[: text.index(marker)] if marker in text else text


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
