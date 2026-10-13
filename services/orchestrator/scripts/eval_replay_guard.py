"""Offline check for the local provider's replay guard.

Finds the turns of a loop_sim replay whose DM reply is a replay of the
previous one (``turn_director.similarity`` >= the guard bar), rebuilds each
such turn's prompt from its ``debug.context`` event, and runs it through
the REAL guarded-generation path (``build_local_llm`` →
``get_chat_completions``) against the local gateway (``AIDM_LOCAL_LLM_URL``
from ``.env``). Reports, per turn: the recorded similarity, the fresh
draft's similarity to the prior replies the guard sees, the retry's
similarity, and whether the guard cleared the replay.

Fidelity: the per-turn message list is exact (summary, history, sheets,
turn contract). The system prompt is the private-mode persona WITHOUT
the vault map / campaign titles, and no tool schema is sent. By default
only the replay guard is armed; ``--live-guards`` also arms the guards
the live turn had — banned/paragraph phrases rebuilt by replaying
``DirectorState.observe_turn`` over the transcript (exactly what the
writer fed it), event/transition directives parsed from the sheets
block — the path that actually released the live loops (the old retry
copied the prior reply). The critic pass stays off (default-off live).

``--all`` runs every turn (not only the replays) and ends with a summary:
retries, residual banned/paragraph hits and 8-gram repeats (vs the
context's assistant replies) in the released text, total generation
seconds. ``--no-excise`` restores the old retry-only behaviour for an
apples-to-apples baseline through the same harness.

``--summary-json PATH`` writes one comparable record per (model, profile)
run: replay-guard fires/cleared, residual banned/paragraph/8-gram hits,
directive misses, excisions, retries, generation seconds (mean/median/p90
per turn), mean reasoning tokens (thinking profiles) and the profile name
(``AIDM_PROFILE``). The residual/miss counts are measured on the released
text against the live turn's bans and directives whether or not those
guards are armed, so a guards-off profile stays comparable. The guards
themselves follow the profile's ``AIDM_GUARD_*`` switches.

Usage::

    uv run python services/orchestrator/scripts/eval_replay_guard.py \\
        docs/bench/<date>/loop/marathon-50t/replay-001 [--turns 13,14] [--live-guards]
        [--all] [--no-excise] [--dump out.jsonl] [--summary-json out.json]

    # one JSON per (model, profile): same 50 prompts, gateway on that profile
    PROFILE=qwen3-14b-thinking make gateway-profile
    set -a; . services/llm-gateway/profiles/qwen3-14b-thinking.env; set +a
    uv run python services/orchestrator/scripts/eval_replay_guard.py <replay> \\
        --all --live-guards --summary-json docs/bench/<date>/profiles/qwen3-14b-thinking.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

from aidm.config import load_config  # noqa: E402
import aidm.providers.llm_local as llm_local  # noqa: E402
from aidm.providers.llm_local import (  # noqa: E402
    _REPLAY_SIMILARITY_MIN,
    _assistant_texts,
    _chunks_text,
    _first_banned_phrase,
    _missed_directives,
    _recent_replies,
    build_local_llm,
)
from aidm.turn_director import (  # noqa: E402
    PARA_SHINGLE_WORDS,
    DirectorState,
    _shingles,
    similarity,
)
from loguru import logger  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402


def _contexts_by_player(events_path: Path) -> dict[str, list[dict[str, str]]]:
    """First ``debug.context`` per latest-user text (a tool follow-up repeats it)."""
    found: dict[str, list[dict[str, str]]] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("type") != "debug.context":
            continue
        messages = event["messages"]
        last_user = next((m["text"] for m in reversed(messages) if m["role"] == "user"), "")
        found.setdefault(last_user, messages)
    return found


def _sheet_directives(messages: list[dict[str, str]]) -> list[str]:
    """The ``- `` bullets under the sheets block's ``[World events`` header."""
    for m in messages:
        if m["role"] != "system" or not m["text"].startswith("<!--aidm:scene-sheets-->"):
            continue
        lines = m["text"].splitlines()
        start = next((i for i, ln in enumerate(lines) if ln.startswith("[World events")), None)
        if start is None:
            return []
        out: list[str] = []
        for ln in lines[start + 1 :]:
            if not ln.startswith("- "):
                break
            out.append(ln[2:])
        return out
    return []


def _director_bans(transcript: list[dict[str, Any]], turn: int) -> tuple[list[str], list[str]]:
    """(banned, paragraph_banned) as the live director held them entering ``turn``."""
    state = DirectorState()
    prev: dict[str, Any] = {"player": "", "dm": ""}
    for t in transcript:
        if t["turn"] >= turn:
            break
        state.observe_turn(
            user_text=t["player"],
            prev_user_text=prev["player"],
            assistant_text=t["dm"],
            prev_assistant_text=prev["dm"],
        )
        prev = t
    return list(state.banned_phrases), list(state.paragraph_banned)


def _reasoning_tokens(chunks: list[Any]) -> int:
    """Reasoning tokens of one completion: usage when reported, else reasoning deltas.

    Simplification: llama.cpp streams ~one token per delta, so the delta count is the
    fallback estimate when ``completion_tokens_details`` is absent.
    """
    for chunk in reversed(chunks):
        details = getattr(getattr(chunk, "usage", None), "completion_tokens_details", None)
        if details is not None and details.reasoning_tokens:
            return int(details.reasoning_tokens)
    return sum(
        1
        for chunk in chunks
        for choice in chunk.choices or []
        if getattr(choice.delta, "reasoning_content", None)
    )


# Unparsed thinking markers (Qwen <think>, Gemma 4 channel) in released content.
_THINK_MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>")


def _p90(values: list[float]) -> float:
    return statistics.quantiles(values, n=10)[-1] if len(values) > 1 else sum(values)


async def _run_turn(svc: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    context = LLMContext(messages=[{"role": m["role"], "content": m["text"]} for m in messages])
    prior = [text for _, text in _recent_replies(context.messages)]
    texts: list[str] = []
    seconds: list[float] = []
    reasoning: list[int] = []
    original = svc._dispatch_logged

    async def recording(ctx: Any) -> Any:
        started = time.perf_counter()
        chunks = [c async for c in await original(ctx)]
        seconds.append(time.perf_counter() - started)
        texts.append(_chunks_text(chunks))
        reasoning.append(_reasoning_tokens(chunks))

        async def replay() -> Any:
            for chunk in chunks:
                yield chunk

        return replay()

    svc._dispatch_logged = recording
    try:
        released = _chunks_text([c async for c in await svc.get_chat_completions(context)])
    finally:
        svc._dispatch_logged = original
    after = [m.get("content") for m in context.messages]
    assert after == [m["text"] for m in messages], "retry leaked into context history"

    def sim(text: str) -> float:
        return max((similarity(text, p) for p in prior), default=0.0)

    repeats = set().union(
        *(_shingles(t, PARA_SHINGLE_WORDS) for t in _assistant_texts(context.messages))
    )
    return {
        "texts": texts,
        "released": released,
        "repeated_8grams": len(_shingles(released, PARA_SHINGLE_WORDS) & repeats),
        "draft_sim": sim(texts[0]),
        "retry_sim": sim(texts[1]) if len(texts) > 1 else None,
        "released_sim": sim(released),
        "seconds": seconds,
        "reasoning_tokens": reasoning,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay", type=Path)
    parser.add_argument("--turns", default="", help="comma-separated turn numbers (default: all)")
    parser.add_argument("--live-guards", action="store_true", help="arm the live turn's guards")
    parser.add_argument("--dump", type=Path, help="write draft/retry texts as JSONL here")
    parser.add_argument("--all", action="store_true", help="every turn, not only the replays")
    parser.add_argument("--no-excise", action="store_true", help="old retry-only behaviour")
    parser.add_argument("--summary-json", type=Path, help="write the comparison record here")
    args = parser.parse_args()
    if args.no_excise:
        llm_local._EXCISE_MAX_CUT = -1.0  # every excision falls back to the retry
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    transcript = json.loads((args.replay / "transcript.json").read_text(encoding="utf-8"))
    contexts = _contexts_by_player(args.replay / "events.jsonl")
    wanted = {int(t) for t in args.turns.split(",") if t}

    config = load_config()
    session = SimpleNamespace(locale="en", campaign_title="", run_title="")
    svc = build_local_llm(config, session)  # type: ignore[arg-type]

    print(f"gateway {config.local_llm_base_url} · model {config.local_llm_model}")
    print(f"profile: {config.profile or '(none)'} · dm thinking {config.dm_thinking}")
    print(f"guards: {'live (banned/para/directives)' if args.live_guards else 'replay only'}")
    print(f"excision: {'off (retry only)' if args.no_excise else 'on'}")
    print("| turn | original | draft | retry | released | outcome | draft s | retry s |")
    print("|---|---|---|---|---|---|---|---|")
    totals = {"turns": 0, "retries": 0, "excised": 0, "seconds": 0.0}
    residual = {"banned": 0, "paragraph": 0, "turns with 8-gram repeats": 0}
    replay = {"drafts": 0, "fired": 0, "cleared": 0}
    misses = {"directive_misses": 0, "reasoning_leaks": 0}
    turn_seconds: list[float] = []
    reasoning_tokens: list[int] = []
    for i, cur in enumerate(transcript):
        original = similarity(cur["dm"], transcript[i - 1]["dm"]) if i else 0.0
        if wanted and cur["turn"] not in wanted:
            continue
        if not args.all and not wanted and original < _REPLAY_SIMILARITY_MIN:
            continue
        messages = contexts.get(cur["player"])
        if messages is None:
            print(f"| {cur['turn']} | {original:.2f} | — no debug.context for this turn |")
            continue
        # Measured on every run; armed as guards only with --live-guards.
        banned, para = _director_bans(transcript, cur["turn"])
        directives = _sheet_directives(messages)
        if args.live_guards:
            svc.set_banned_phrases_provider(lambda b=banned: b)
            svc.set_para_banned_provider(lambda p=para: p)
            svc.set_event_directives_provider(lambda d=directives: d)
        r = await _run_turn(svc, messages)
        excised = len(r["texts"]) == 1 and r["released"] != r["texts"][0]
        totals["turns"] += 1
        totals["retries"] += len(r["texts"]) - 1
        totals["excised"] += excised
        totals["seconds"] += sum(r["seconds"])
        residual["banned"] += _first_banned_phrase(r["released"], banned) is not None
        residual["paragraph"] += _first_banned_phrase(r["released"], para) is not None
        residual["turns with 8-gram repeats"] += r["repeated_8grams"] > 0
        misses["directive_misses"] += bool(_missed_directives(r["released"], directives))
        misses["reasoning_leaks"] += any(m in r["released"] for m in _THINK_MARKERS)
        turn_seconds.append(sum(r["seconds"]))
        reasoning_tokens += r["reasoning_tokens"]
        if args.dump:
            with args.dump.open("a", encoding="utf-8") as fh:
                row = {
                    "turn": cur["turn"],
                    "player": cur["player"],
                    "texts": r["texts"],
                    "released": r["released"],
                    "banned": banned,
                    "para": para,
                }
                fh.write(json.dumps(row) + "\n")
        replayed = r["draft_sim"] >= _REPLAY_SIMILARITY_MIN
        replay["drafts"] += replayed
        if replayed and r["retry_sim"] is not None:
            replay["fired"] += 1
            replay["cleared"] += r["retry_sim"] < _REPLAY_SIMILARITY_MIN
        if r["retry_sim"] is None:
            verdict = "excised, no retry" if excised else "draft fresh, no retry"
        elif replayed:
            cleared = r["retry_sim"] < _REPLAY_SIMILARITY_MIN
            verdict = "replay guard → cleared" if cleared else "replay guard → STILL similar"
        else:
            verdict = "other guard retry"
        retry = f"{r['retry_sim']:.2f}" if r["retry_sim"] is not None else "—"
        secs = r["seconds"] + [0.0]
        print(
            f"| {cur['turn']} | {original:.2f} | {r['draft_sim']:.2f} | {retry} "
            f"| {r['released_sim']:.2f} | {verdict} | {secs[0]:.1f} | {secs[1]:.1f} |",
            flush=True,
        )
    print(f"\nturns {totals['turns']} · retries {totals['retries']} · excised {totals['excised']}")
    print(f"generation seconds total {totals['seconds']:.0f}")
    print("residual in released text: " + ", ".join(f"{k} {v}" for k, v in residual.items()))
    if args.summary_json:
        summary = {
            "profile": config.profile,
            "model": config.local_llm_model,
            "replay": str(args.replay),
            "dm_thinking": config.dm_thinking,
            "guards": {
                "tics": config.guard_tics,
                "paragraph": config.guard_paragraph,
                "replay": config.guard_replay,
                "directives": config.guard_directives,
                "armed": args.live_guards,
                "excise": not args.no_excise,
            },
            "turns": totals["turns"],
            "replay_drafts": replay["drafts"],
            "replay_guard_fired": replay["fired"],
            "replay_guard_cleared": replay["cleared"],
            "residual_banned": residual["banned"],
            "residual_paragraph": residual["paragraph"],
            "residual_8gram_turns": residual["turns with 8-gram repeats"],
            **misses,
            "excisions": totals["excised"],
            "retries": totals["retries"],
            "gen_seconds_mean": statistics.fmean(turn_seconds) if turn_seconds else None,
            "gen_seconds_median": statistics.median(turn_seconds) if turn_seconds else None,
            "gen_seconds_p90": _p90(turn_seconds) if turn_seconds else None,
            "reasoning_tokens_mean": (
                statistics.fmean(reasoning_tokens)
                if config.dm_thinking and reasoning_tokens
                else None
            ),
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"summary → {args.summary_json}")


if __name__ == "__main__":
    asyncio.run(main())
