"""Deterministic scorer for a closed-loop replay bundle.

Consumes one ``loop_sim.py`` replay directory (transcript.json,
events.jsonl, vault-snapshot/) plus the checked-in campaign canon, and
produces the code-measured half of the hybrid scorecard — everything a
regex/difflib pass can catch without a judge:

- D1 duplication (20) — DM replies fuzzy-matching an earlier reply in
  the 12-turn context window at the calibrated 0.60 ratio (same detector as ``eval_llm.py``).
- D2 assistant-voice tells (10) — the shared tell list.
- D3 entity grounding (15) — mid-sentence capitalized tokens in DM
  prose that appear nowhere in canon or the run snapshot.
- D4 tool discipline (15) — tool-result errors in the final
  ``debug.context`` snapshot.
- D5 reply-length trend (10) — mode-collapse slope across the run.
- D6 director health (10) — static ``scene_goal`` streaks from the
  ``npc.state`` event feed.
- D7 event-engine assertions (20) — fired events narrated, directive
  content reflected in DM prose, beat mutations on disk, clock sanity.

Thresholds are first-pass values to be calibrated against real runs
(the same method that produced eval_llm's 0.60 duplication ratio); the
per-bucket detail strings exist so calibration has something to read.

Usage::

    uv run python services/orchestrator/scripts/loop_score.py \\
        docs/bench/<date>/loop/<scenario>/replay-001

Writes ``score.json`` + ``score.md`` into the replay dir and prints the
one-line total. ``judge_run.py`` imports ``score_replay`` directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from aidm import frontmatter  # noqa: E402
from aidm.turn_director import player_echo_ratio  # noqa: E402
from eval_llm import _ASSISTANT_VOICE_TELLS, _FUZZY_DUP_RATIO  # noqa: E402

# Sentence-initial capitalized words are noise; only tokens preceded by
# a lowercase word count as entity mentions.
_MID_SENTENCE_CAP_RE = re.compile(r"(?<=[a-z] )([A-Z][a-z]{2,})\b")
_CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_WORD_RE = re.compile(r"[a-zA-Z']{4,}")
_STOPWORDS = frozenset(
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
    }
)
# Directive content-word overlap below this = the DM ignored the event.
_NARRATION_OVERLAP_MIN = 0.2
_STATIC_GOAL_STREAK_WARN = 4
_STATIC_GOAL_STREAK_FAIL = 6

# D-bucket penalty weights — tune against the 30t/50t replay corpus.
# Each constant is the deduction per unit of failure; bucket max lives
# in the individual score_* functions.
_D1_PENALTY_PER_DUP = (
    2  # D1: per fuzzy-duplicate pair; 10 pairs zero it (7 saturated every 2026-09 run at 3)
)
_D1_WINDOW = 12  # D1: HistoryTrimmer's max_recent_turns
_PLAYER_ECHO_RATIO = 0.8  # D1: DM reply vs same-turn player input (echo)
_D2_PENALTY_PER_TELL = 3  # D2: per assistant-voice tell hit (max 10)
_D3_PENALTY_PER_ENTITY = 2  # D3: per unfamiliar mid-sentence name (max 15)
_D4_PENALTY_PER_ERROR = 3  # D4: per tool-call error result (max 15)
_TOOL_CALL_RE = re.compile(r"Calling function \[")  # one per dispatched tool call
_D7_PENALTY_PER_ABANDONED = 4  # D7: per directive stamped narrated:abandoned (max 20)


@dataclass
class Bucket:
    label: str
    awarded: int
    maximum: int
    detail: str = ""


@dataclass
class DeterministicScore:
    buckets: list[Bucket] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(b.awarded for b in self.buckets)

    @property
    def maximum(self) -> int:
        return sum(b.maximum for b in self.buckets)


# --- bundle loading ---------------------------------------------------------


@dataclass
class ReplayBundle:
    transcript: list[dict[str, Any]]
    events: list[dict[str, Any]]
    snapshot_dir: Path
    canon_dir: Path
    server_log: str = ""

    @property
    def dm_replies(self) -> list[str]:
        return [str(t.get("dm") or "") for t in self.transcript if t.get("dm")]


def load_bundle(replay_dir: Path, canon_dir: Path) -> ReplayBundle:
    transcript_path = replay_dir / "transcript.json"
    transcript = (
        json.loads(transcript_path.read_text(encoding="utf-8")) if transcript_path.is_file() else []
    )
    events: list[dict[str, Any]] = []
    events_path = replay_dir / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
    log_path = replay_dir / "server.log"
    return ReplayBundle(
        transcript=transcript if isinstance(transcript, list) else [],
        events=events,
        snapshot_dir=replay_dir / "vault-snapshot",
        canon_dir=canon_dir,
        server_log=log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else "",
    )


# --- buckets ------------------------------------------------------------------


def score_duplication(replies: list[str], transcript: list[dict[str, Any]] | None = None) -> Bucket:
    # Each reply vs every earlier reply in the DM's 12-turn context window,
    # counted once: consecutive-only pairs missed 0.97-0.99 copies from 3-5
    # turns back (2026-09-26 replay-003, T11=T6, T41=T38).
    dups = 0
    pairs: list[str] = []
    for i, curr in enumerate(replies):
        if not curr:
            continue
        earlier = [p for p in replies[max(0, i - _D1_WINDOW) : i] if p]
        ratio = max((SequenceMatcher(None, p, curr).ratio() for p in earlier), default=0.0)
        if ratio >= _FUZZY_DUP_RATIO:
            dups += 1
            pairs.append(f"#{i + 1}={ratio:.2f}")
    # Player-echo blind spot (Kimi K2.6 T43-45, marathon replay-004): a
    # reply that opens by restating the same turn's player input
    # duplicates the PLAYER, so reply-to-reply pairs never see it. Same
    # per-hit penalty — an echoed turn is a lost turn like a looped one.
    echoes = 0
    for t in transcript or []:
        player, dm = str(t.get("player") or ""), str(t.get("dm") or "")
        ratio = player_echo_ratio(player, dm)
        if ratio >= _PLAYER_ECHO_RATIO:
            echoes += 1
            pairs.append(f"echo T{t.get('turn')}={ratio:.2f}")
    return Bucket(
        label=(
            f"D1 duplication ({dups} fuzzy pair(s) ≥ {_FUZZY_DUP_RATIO}, "
            f"{echoes} player-echo turn(s) ≥ {_PLAYER_ECHO_RATIO})"
        ),
        awarded=max(0, 20 - _D1_PENALTY_PER_DUP * (dups + echoes)),
        maximum=20,
        detail="; ".join(pairs),
    )


def score_tells(transcript: list[dict[str, Any]]) -> Bucket:
    hits: list[str] = []
    for entry in transcript:
        body = str(entry.get("dm") or "").lower()
        for phrase in _ASSISTANT_VOICE_TELLS:
            if phrase in body:
                hits.append(f"T{entry.get('turn')}:{phrase!r}")
                break
    return Bucket(
        label=f"D2 assistant-voice tells ({len(hits)} turn(s))",
        awarded=max(0, 10 - _D2_PENALTY_PER_TELL * len(hits)),
        maximum=10,
        detail="; ".join(hits),
    )


def _allowed_entity_bag(bundle: ReplayBundle) -> set[str]:
    bag: set[str] = set()
    for root in (bundle.canon_dir, bundle.snapshot_dir):
        if not root.is_dir():
            continue
        for path in root.rglob("*.md"):
            if any(part.startswith(".") for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            bag.update(_CAP_TOKEN_RE.findall(text))
    # Player inputs are canon too — the simulator may name things.
    for entry in bundle.transcript:
        bag.update(_CAP_TOKEN_RE.findall(str(entry.get("player") or "")))
    return bag


def score_grounding(bundle: ReplayBundle) -> Bucket:
    allowed = _allowed_entity_bag(bundle)
    drift: set[str] = set()
    for reply in bundle.dm_replies:
        for token in _MID_SENTENCE_CAP_RE.findall(reply):
            if token not in allowed:
                drift.add(token)
    listed = ", ".join(sorted(drift)[:12])
    return Bucket(
        label=f"D3 entity grounding ({len(drift)} unfamiliar mid-sentence name(s))",
        awarded=max(0, 15 - _D3_PENALTY_PER_ENTITY * len(drift)),
        maximum=15,
        detail=listed,
    )


def score_tool_discipline(events: list[dict[str, Any]], server_log: str = "") -> Bucket:
    """Tool calls from server.log, error results from every debug.context.

    The last context alone saw none of them: HistoryTrimmer drops old tool
    turns and the closing session-log call comes after the last context
    (0 calls on every 2026-09 marathon while server.log had 1-2). Tool
    messages carry their result under ``text``, JSON-quoted; identical
    results repeat across contexts, so they are deduplicated by text.
    """
    contexts = [e for e in events if e.get("type") == "debug.context"]
    if not contexts and not server_log:
        return Bucket(
            label="D4 tool discipline (no debug.context captured)",
            awarded=15,
            maximum=15,
            detail="no data — bucket vacuous",
        )
    results: set[str] = set()
    for context in contexts:
        for message in context.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "tool":
                results.add(str(message.get("text") or message.get("content") or ""))
    errors = [r for r in results if r.lstrip('"').lstrip().lower().startswith("error")]
    tool_calls = len(_TOOL_CALL_RE.findall(server_log)) if server_log else len(results)
    return Bucket(
        label=f"D4 tool discipline ({len(errors)} error result(s) / {tool_calls} tool call(s))",
        awarded=max(0, 15 - _D4_PENALTY_PER_ERROR * len(errors)),
        maximum=15,
        detail="; ".join(e[:80] for e in errors),
    )


def score_length_trend(replies: list[str]) -> Bucket:
    counts = [len(r.split()) for r in replies if r]
    if len(counts) < 4:
        return Bucket(
            label="D5 reply-length trend (too few turns to measure)",
            awarded=10,
            maximum=10,
            detail=f"counts={counts}",
        )
    half = len(counts) // 2
    first = sum(counts[:half]) / half
    second = sum(counts[half:]) / (len(counts) - half)
    ratio = second / first if first else 1.0
    if 0.5 <= ratio <= 2.0:
        awarded = 10
    elif 0.35 <= ratio <= 2.8:
        awarded = 5
    else:
        awarded = 0
    return Bucket(
        label=f"D5 reply-length trend (2nd/1st half ratio {ratio:.2f})",
        awarded=awarded,
        maximum=10,
        detail=f"first_half_mean={first:.0f}w second_half_mean={second:.0f}w",
    )


def score_director_health(events: list[dict[str, Any]]) -> Bucket:
    streaks: dict[str, int] = {}
    last_goal: dict[str, str] = {}
    worst: tuple[str, int] = ("", 0)
    for event in events:
        if event.get("type") != "npc.state":
            continue
        for npc in event.get("npcs") or []:
            if not isinstance(npc, dict):
                continue
            slug = str(npc.get("slug") or "")
            goal = str(npc.get("scene_goal") or "")
            if not slug or not goal:
                continue
            if last_goal.get(slug) == goal:
                streaks[slug] = streaks.get(slug, 1) + 1
            else:
                streaks[slug] = 1
            last_goal[slug] = goal
            if streaks[slug] > worst[1]:
                worst = (slug, streaks[slug])
    slug, streak = worst
    if streak >= _STATIC_GOAL_STREAK_FAIL:
        awarded = 0
    elif streak >= _STATIC_GOAL_STREAK_WARN:
        awarded = 5
    else:
        awarded = 10
    return Bucket(
        label=f"D6 director health (longest static scene_goal streak: {streak})",
        awarded=awarded,
        maximum=10,
        detail=f"slug={slug}" if slug else "no npc.state events",
    )


def _content_words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RE.findall(text) if w.casefold() not in _STOPWORDS}


def _load_snapshot_events(snapshot_dir: Path) -> list[tuple[str, dict[str, Any], str]]:
    events_dir = snapshot_dir / "events"
    out: list[tuple[str, dict[str, Any], str]] = []
    if not events_dir.is_dir():
        return out
    for path in sorted(events_dir.glob("*.md")):
        fm, body = frontmatter.split(path.read_text(encoding="utf-8"))
        out.append((path.stem, fm, body))
    return out


def score_event_engine(bundle: ReplayBundle) -> Bucket:  # noqa: PLR0912 — four sub-checks
    """Fired events narrated + reflected; beat mutations landed; clock sane."""
    snapshot_events = _load_snapshot_events(bundle.snapshot_dir)
    fired = [(slug, fm, body) for slug, fm, body in snapshot_events if fm.get("status") == "fired"]
    notes: list[str] = []
    awarded = 20

    if not snapshot_events:
        return Bucket(
            label="D7 event engine (no events in snapshot — engine idle or run opted out)",
            awarded=20,
            maximum=20,
            detail="vacuous",
        )

    # (a) every fired event was rendered to the DM (narrated flag).
    # "abandoned" is handled separately below — it WAS rendered, the DM
    # just never narrated it within MAX_DIRECTIVE_RENDERS.
    unrendered = [
        slug for slug, fm, _ in fired if str(fm.get("narrated")) not in ("true", "abandoned")
    ]
    if unrendered:
        awarded -= 5
        notes.append(f"unrendered fired event(s): {unrendered}")

    abandoned = [slug for slug, fm, _ in fired if str(fm.get("narrated")) == "abandoned"]
    if abandoned:
        awarded -= _D7_PENALTY_PER_ABANDONED * len(abandoned)
        notes.append(f"abandoned directive(s): {abandoned}")

    # (b) directive content reflected in DM prose after the fire turn.
    all_dm = " ".join(bundle.dm_replies).casefold()
    ignored: list[str] = []
    for slug, _fm, body in fired:
        directive = body.split("## On fire", 1)[0].replace("## Directive", "")
        words = _content_words(directive)
        if not words:
            continue
        overlap = sum(1 for w in words if w in all_dm) / len(words)
        if overlap < _NARRATION_OVERLAP_MIN:
            ignored.append(f"{slug} (overlap {overlap:.2f})")
    if ignored:
        awarded -= min(10, 5 * len(ignored))
        notes.append(f"directive not woven into prose: {ignored}")

    # (c) beat mutations landed on disk.
    beat_path = bundle.snapshot_dir / "beat.md"
    beat_text = beat_path.read_text(encoding="utf-8") if beat_path.is_file() else ""
    missing_beat: list[str] = []
    for slug, _fm, body in fired:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("- beat:"):
                text = stripped[len("- beat:") :].strip()
                if text and text not in beat_text:
                    missing_beat.append(slug)
    if missing_beat:
        awarded -= 5
        notes.append(f"beat mutation missing from beat.md: {sorted(set(missing_beat))}")

    # (d) clock sanity: parseable, valid segment, turn count > 0.
    clock_path = bundle.snapshot_dir / "clock.md"
    if clock_path.is_file():
        clock_fm, _ = frontmatter.split(clock_path.read_text(encoding="utf-8"))
        segment = str(clock_fm.get("segment") or "")
        turn = clock_fm.get("turn")
        if segment not in ("morning", "afternoon", "evening", "night") or not isinstance(turn, int):
            awarded -= 5
            notes.append(f"clock.md unsound: segment={segment!r} turn={turn!r}")
    else:
        awarded -= 5
        notes.append("clock.md missing from snapshot")

    return Bucket(
        label=f"D7 event engine ({len(fired)} fired / {len(snapshot_events)} authored)",
        awarded=max(0, awarded),
        maximum=20,
        detail="; ".join(notes),
    )


# --- entry --------------------------------------------------------------------


def score_replay(replay_dir: Path, canon_dir: Path | None = None) -> DeterministicScore:
    canon = canon_dir or (_REPO_ROOT / "vault" / "ember-hollow")
    bundle = load_bundle(replay_dir, canon)
    score = DeterministicScore()
    score.buckets.append(score_duplication(bundle.dm_replies, bundle.transcript))
    score.buckets.append(score_tells(bundle.transcript))
    score.buckets.append(score_grounding(bundle))
    score.buckets.append(score_tool_discipline(bundle.events, bundle.server_log))
    score.buckets.append(score_length_trend(bundle.dm_replies))
    score.buckets.append(score_director_health(bundle.events))
    score.buckets.append(score_event_engine(bundle))
    return score


def render_markdown(score: DeterministicScore, replay_dir: Path) -> str:
    lines = [f"# Deterministic score — `{replay_dir.name}`", ""]
    lines.append(f"**D = {score.total}/{score.maximum}**")
    lines.append("")
    for bucket in score.buckets:
        mark = "✅" if bucket.awarded == bucket.maximum else ("⚠️" if bucket.awarded else "❌")
        lines.append(f"- {mark} {bucket.label}: {bucket.awarded}/{bucket.maximum}")
        if bucket.detail:
            lines.append(f"  - {bucket.detail}")
    return "\n".join(lines) + "\n"


def write_score(score: DeterministicScore, replay_dir: Path) -> None:
    payload = {
        "total": score.total,
        "maximum": score.maximum,
        "buckets": [
            {"label": b.label, "awarded": b.awarded, "maximum": b.maximum, "detail": b.detail}
            for b in score.buckets
        ],
    }
    (replay_dir / "score.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (replay_dir / "score.md").write_text(render_markdown(score, replay_dir), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path, help="A loop_sim replay-NNN directory")
    parser.add_argument(
        "--canon",
        type=Path,
        default=None,
        help="Campaign canon dir for the grounding bag (default vault/ember-hollow)",
    )
    args = parser.parse_args()
    if not args.replay_dir.is_dir():
        print(f"not a directory: {args.replay_dir}", file=sys.stderr)
        return 2
    score = score_replay(args.replay_dir, args.canon)
    write_score(score, args.replay_dir)
    print(f"D = {score.total}/{score.maximum}  ({args.replay_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
