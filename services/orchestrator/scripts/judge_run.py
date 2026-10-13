"""Opus judge for a closed-loop replay bundle — the hybrid scorecard's other half.

Runs 2-3 structured-output passes of ``claude-opus-4-8`` over a
distilled replay bundle (transcript, final sheets, canon, event log —
never raw DEBUG logs) and takes the per-axis median, because judges are
noisy. Every deduction must carry quoted evidence and turn citations,
and every continuity violation must be attributed:

- ``model``        — the contradicted fact was still inside the recent
                     context window → the DM model failed.
- ``carry``        — the fact had aged out of the window and lived only
                     on sheets/beat → the sheet-writer/injector carry
                     path failed.
- ``event_engine`` — the DM contradicted a fired event directive → the
                     engine/contract path failed.

The context window is deterministic: ``HistoryTrimmer`` keeps the last
12 user turns, so at turn N everything before ``max(1, N-11)`` is
carried only by sheets/beat/notes. The judge receives that formula.

Deterministic metrics come from ``loop_score.py`` (imported; run first
if ``score.json`` is missing) and the two halves land in one
``scorecard.md`` row: ``D <det>/100 · J <judge>/100 · combined /200``.

Dev-only tooling — the ``anthropic`` SDK lives in the root dev
dependency group and is never imported from ``aidm/`` runtime.

Usage::

    uv run python services/orchestrator/scripts/judge_run.py \\
        docs/bench/<date>/loop/<scenario>/replay-001 [--passes 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from dotenv import load_dotenv  # noqa: E402

# Same .env the orchestrator loads — ANTHROPIC_API_KEY can live there.
load_dotenv(_REPO_ROOT / ".env", override=False)

from aidm import frontmatter  # noqa: E402
from loop_score import score_replay, write_score  # noqa: E402

JUDGE_MODEL = "claude-opus-4-8"
# Retries per pass on a transient Anthropic error (5xx/429/timeout). The
# judge is 3 long Opus calls; an unretried flap discards the whole run.
JUDGE_API_RETRIES = 4
# Adaptive thinking spends from the same budget as the answer; a
# 30-turn bundle's claims ledger can think past 8k and leave no room
# for the JSON (observed live: empty text block on the first 22-turn
# judge pass). Keep headroom generous — the schema caps the output.
JUDGE_MAX_TOKENS = 32000
_WINDOW_TURNS = 12  # HistoryTrimmer's max_recent_turns

_AXES: tuple[tuple[str, int], ...] = (
    ("fact_continuity", 25),
    ("plot_progression", 15),
    ("hexaco_faithfulness", 15),
    ("player_agency", 15),
    ("sheet_narrative_agreement", 10),
    ("event_integration", 10),
    ("prose_variety", 10),
)

# Flat schema by necessity: the structured-output grammar compiler
# rejects the naturally-nested version ("compiled grammar is too large"
# — verified live 2026-07-11). One shared violations array with an
# `axis` tag replaces three nested per-axis copies; per-NPC/per-event
# detail rides sibling top-level arrays. No numerical constraints
# either (also unsupported) — score ranges live in the prompt and are
# clamped by ``median_axes``.
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {name: {"type": "integer"} for name, _ in _AXES},
            "required": [name for name, _ in _AXES],
            "additionalProperties": False,
        },
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "axis": {"type": "string", "enum": [name for name, _ in _AXES]},
                    "turn": {"type": "integer"},
                    "claim": {"type": "string"},
                    "contradicts_turn": {"type": "integer"},
                    "evidence_quote": {"type": "string"},
                    "attribution": {
                        "type": "string",
                        "enum": ["model", "carry", "event_engine"],
                    },
                },
                "required": ["axis", "turn", "claim", "evidence_quote", "attribution"],
                "additionalProperties": False,
            },
        },
        "per_npc_hexaco": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string"},
                    "score": {"type": "integer"},
                    "evidence": {"type": "string"},
                },
                "required": ["slug", "score", "evidence"],
                "additionalProperties": False,
            },
        },
        "per_event": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["woven", "pasted", "missed"]},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["slug", "verdict"],
                "additionalProperties": False,
            },
        },
        "recycled_fragments": {"type": "array", "items": {"type": "string"}},
        "claims_ledger": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"turn": {"type": "integer"}, "claim": {"type": "string"}},
                "required": ["turn", "claim"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["scores", "violations", "claims_ledger", "summary"],
    "additionalProperties": False,
}


# --- bundle distillation ------------------------------------------------------


def _read(path: Path, cap: int = 8000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[:cap] + ("\n[... truncated]" if len(text) > cap else "")


def _sheets_section(snapshot_dir: Path, canon_dir: Path) -> str:
    parts: list[str] = []
    for state_path in sorted(snapshot_dir.glob("npcs/*.state.md")):
        slug = state_path.name.removesuffix(".state.md")
        identity = _read(canon_dir / "npcs" / f"{slug}.md", cap=3000)
        parts.append(f"### npc: {slug}\n{identity}\n\n{_read(state_path, cap=2000)}")
    for state_path in sorted(snapshot_dir.glob("characters/*.state.md")):
        slug = state_path.name.removesuffix(".state.md")
        identity = _read(snapshot_dir / "characters" / f"{slug}.md", cap=3000)
        parts.append(f"### pc: {slug}\n{identity}\n\n{_read(state_path, cap=2000)}")
    return "\n\n---\n\n".join(parts) or "(no sheets in snapshot)"


def _events_section(snapshot_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(snapshot_dir.glob("events/*.md")):
        fm, body = frontmatter.split(path.read_text(encoding="utf-8"))
        directive = body.split("## On fire", 1)[0].replace("## Directive", "").strip()
        parts.append(
            f"- {path.stem}: status={fm.get('status')} trigger=day {fm.get('trigger_day')} "
            f"{fm.get('trigger_segment') or ''} fired_turn={fm.get('fired_turn')}\n"
            f"  directive: {directive[:400]}"
        )
    return "\n".join(parts) or "(run has no authored events)"


def _transcript_section(replay_dir: Path) -> tuple[str, int]:
    raw = _read(replay_dir / "transcript.json", cap=400_000)
    try:
        transcript = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        transcript = []
    lines: list[str] = []
    for entry in transcript:
        turn = entry.get("turn")
        window_start = max(1, int(turn or 1) - (_WINDOW_TURNS - 1))
        lines.append(f"[T{turn} | in-context window starts at T{window_start}]")
        lines.append(f"Player: {entry.get('player')}")
        lines.append(f"DM: {entry.get('dm')}")
        lines.append("")
    return "\n".join(lines) or "(empty transcript)", len(transcript)


def build_judge_prompt(replay_dir: Path, canon_dir: Path) -> str:
    transcript, n_turns = _transcript_section(replay_dir)
    snapshot = replay_dir / "vault-snapshot"
    axes_list = "\n".join(f"- {name} (0-{mx})" for name, mx in _AXES)
    return (
        "You are auditing one session of an AI Dungeon Master (a local "
        "14B model) that ran against a deterministic story engine. Score "
        "it on these axes:\n"
        f"{axes_list}\n\n"
        "Report per-axis integer scores in `scores` (within each axis's "
        "0-max range above) and every deduction as an entry in the flat "
        "`violations` array tagged with its `axis`.\n\n"
        "Hard rules:\n"
        "- Every deduction needs a QUOTED piece of evidence and turn "
        "citation(s) in `violations`. No quote, no deduction.\n"
        "- Build the claims ledger first: notable world-state claims per "
        "turn. Flag contradictions with both turn numbers.\n"
        f"- Attribution is mandatory for every violation. The DM's context "
        f"window keeps only the last {_WINDOW_TURNS} player turns: at turn N, "
        f"anything before turn max(1, N-{_WINDOW_TURNS - 1}) is carried ONLY by "
        "the sheets/beat below. If the contradicted fact was still inside "
        "the window → attribution 'model'. If it had aged out and lived "
        "only on sheets/beat → 'carry'. If the DM contradicted a fired "
        "event directive (event list below) → 'event_engine'.\n"
        "- event_integration: for each FIRED event, add a `per_event` "
        "entry — 'woven' (integrated into the fiction), 'pasted' (dumped "
        "verbatim/mechanically), or 'missed' (never narrated).\n"
        "- hexaco_faithfulness: judge each NPC's dialogue/behavior against "
        "the HEXACO scores on their sheet (1=very low, 5=very high); add a "
        "`per_npc_hexaco` entry per judged NPC.\n"
        "- player_agency: the player's declared action is canon; flag any "
        "turn where the DM replaced, softened, reversed, or skipped it.\n"
        "- prose_variety: list recycled fragments (phrases recurring "
        "across 3+ turns).\n\n"
        f"## Campaign canon (grounding truth)\n\n{_read(canon_dir / 'campaign.md', cap=4000)}\n\n"
        f"## Final sheets (identity + end-of-run state)\n\n"
        f"{_sheets_section(snapshot, canon_dir)}\n\n"
        f"## Story events (authored; status after the run)\n\n{_events_section(snapshot)}\n\n"
        f"## Transcript ({n_turns} turns)\n\n{transcript}\n"
    )


# --- judge passes ---------------------------------------------------------------


def _is_transient(exc: Exception) -> bool:
    """A 5xx / 429 / timeout worth retrying — Anthropic flaps intermittently."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status == 429):
        return True
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connection", "InternalServer", "Overloaded"))


async def _judge_pass_with_retry(client: Any, prompt: str) -> Any:
    """One judge pass, retrying transient Anthropic errors; None if it gives up."""
    for attempt in range(JUDGE_API_RETRIES + 1):
        try:
            async with client.messages.stream(
                model=JUDGE_MODEL,
                max_tokens=JUDGE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                return await stream.get_final_message()
        except Exception as exc:
            if attempt >= JUDGE_API_RETRIES or not _is_transient(exc):
                raise
            print(
                f"[judge] transient API error ({type(exc).__name__}); "
                f"retry {attempt + 1}/{JUDGE_API_RETRIES}",
                file=sys.stderr,
            )
            await asyncio.sleep(3.0 * (attempt + 1))
    return None


async def run_judge_passes(
    prompt: str, passes: int, *, debug_dir: Path | None = None
) -> list[dict[str, Any]]:
    from anthropic import AsyncAnthropic  # noqa: PLC0415

    client = AsyncAnthropic()
    results: list[dict[str, Any]] = []
    for index in range(passes):
        print(f"[judge] pass {index + 1}/{passes} ({JUDGE_MODEL})", file=sys.stderr)
        # Streamed because the SDK requires it once max_tokens implies a
        # possibly->10-min request; we only consume the final message.
        response = await _judge_pass_with_retry(client, prompt)
        if response is None:
            print(f"[judge] pass {index + 1} gave up after retries", file=sys.stderr)
            continue
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            print(
                f"[judge] pass {index + 1} returned bad JSON ({exc}); "
                f"stop_reason={response.stop_reason}",
                file=sys.stderr,
            )
            if debug_dir is not None:
                dump = debug_dir / f"judge-pass-{index + 1}-error.txt"
                dump.write_text(
                    f"stop_reason: {response.stop_reason}\nusage: {response.usage}\n\n{text}",
                    encoding="utf-8",
                )
                print(f"[judge] raw response dumped to {dump}", file=sys.stderr)
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def median_axes(passes: list[dict[str, Any]]) -> dict[str, int]:
    """Per-axis median score across passes (judges are noisy)."""
    out: dict[str, int] = {}
    for name, maximum in _AXES:
        scores = []
        for run in passes:
            score = (run.get("scores") or {}).get(name)
            if isinstance(score, int):
                scores.append(max(0, min(maximum, score)))
        out[name] = int(statistics.median(scores)) if scores else 0
    return out


def representative_pass(passes: list[dict[str, Any]], medians: dict[str, int]) -> dict[str, Any]:
    """The pass whose axis scores sit closest to the medians — its
    violations/evidence become the headline detail."""
    if not passes:
        return {}

    def distance(run: dict[str, Any]) -> int:
        total = 0
        for name, _ in _AXES:
            score = (run.get("scores") or {}).get(name)
            if isinstance(score, int):
                total += abs(score - medians[name])
        return total

    return min(passes, key=distance)


# --- scorecard -------------------------------------------------------------------


def render_scorecard(  # noqa: PLR0913 — scorecard context is naturally wide
    *,
    replay_dir: Path,
    det_total: int,
    det_max: int,
    medians: dict[str, int],
    rep: dict[str, Any],
    n_passes: int,
) -> str:
    judge_total = sum(medians.values())
    judge_max = sum(mx for _, mx in _AXES)
    lines = [f"# Scorecard — `{replay_dir}`", ""]
    lines.append(
        f"**D = {det_total}/{det_max} · J = {judge_total}/{judge_max} "
        f"(median of {n_passes}) · combined {det_total + judge_total}/{det_max + judge_max}**"
    )
    lines.append("")
    lines.append("## Judge axes (median)")
    lines.append("")
    for name, maximum in _AXES:
        lines.append(f"- {name}: {medians[name]}/{maximum}")
    violations = rep.get("violations") or []
    if violations:
        lines.append("")
        lines.append("## Violations (representative pass)")
        lines.append("")
        for v in violations:
            lines.append(
                f"- T{v.get('turn')} [{v.get('axis')}/{v.get('attribution')}] "
                f'{v.get("claim")} — "{v.get("evidence_quote")}"'
            )
    summary = str(rep.get("summary") or "")
    if summary:
        lines.append("")
        lines.append("## Judge summary (representative pass)")
        lines.append("")
        lines.append(summary)
    lines.append("")
    lines.append("_Deterministic breakdown: `score.md`; all judge passes: `judge.json`._")
    return "\n".join(lines) + "\n"


async def _main_async(args: argparse.Namespace) -> int:
    replay_dir: Path = args.replay_dir
    if not replay_dir.is_dir():
        print(f"not a directory: {replay_dir}", file=sys.stderr)
        return 2
    canon_dir = args.canon or (_REPO_ROOT / "vault" / "ember-hollow")

    det = score_replay(replay_dir, canon_dir)
    write_score(det, replay_dir)
    print(f"D = {det.total}/{det.maximum}", file=sys.stderr)

    prompt = build_judge_prompt(replay_dir, canon_dir)
    if args.export_prompt:
        # Keyless judging: an external judge (a person or another model) answers
        # this exact prompt into judge-pass-N.json, then --offline-passes scores it.
        target = replay_dir / "judge_prompt.md"
        schema = json.dumps(JUDGE_SCHEMA, indent=2)
        target.write_text(
            f"{prompt}\n\n## Output: one JSON object matching this schema\n\n```json\n{schema}\n```\n",
            encoding="utf-8",
        )
        print(f"judge prompt → {target}", file=sys.stderr)
        return 0
    if args.offline_passes:
        passes = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(replay_dir.glob("judge-pass-[0-9].json"))
        ]
        model = "offline (judge-pass-*.json)"
    else:
        passes = await run_judge_passes(prompt, args.passes, debug_dir=replay_dir)
        model = JUDGE_MODEL
    if not passes:
        print("no judge pass returned valid JSON", file=sys.stderr)
        return 1
    medians = median_axes(passes)
    rep = representative_pass(passes, medians)

    (replay_dir / "judge.json").write_text(
        json.dumps(
            {"model": model, "passes": passes, "medians": medians},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scorecard = render_scorecard(
        replay_dir=replay_dir,
        det_total=det.total,
        det_max=det.maximum,
        medians=medians,
        rep=rep,
        n_passes=len(passes),
    )
    (replay_dir / "scorecard.md").write_text(scorecard, encoding="utf-8")
    judge_total = sum(medians.values())
    print(
        f"D = {det.total}/{det.maximum} · J = {judge_total}/{sum(mx for _, mx in _AXES)} "
        f"· combined {det.total + judge_total}/{det.maximum + sum(mx for _, mx in _AXES)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path, help="A loop_sim replay-NNN directory")
    parser.add_argument("--passes", type=int, default=3, help="Judge passes (median wins)")
    parser.add_argument("--canon", type=Path, default=None, help="Campaign canon dir")
    parser.add_argument(
        "--export-prompt",
        action="store_true",
        help="Write judge_prompt.md (prompt + JSON schema) and stop — no API call",
    )
    parser.add_argument(
        "--offline-passes",
        action="store_true",
        help="Score from replay_dir/judge-pass-N.json written by an external judge",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
