"""Offline evaluator for the PC writer's ``with_pc`` presence signal.

Replays each turn of a loop_sim replay through the SAME code the live
writer runs — ``rewrite_sheet`` for the PC (prompt, call parameters,
JSON parse/salvage, key validation, extraction) and
``SceneSheetWriter._enter_with_pc`` for name → slug resolution — against
the local llama.cpp gateway, then scores it against hand labels. The
current rule-based signals (voiced + addressed detectors, via
``SceneSheetWriter._auto_enter_voiced_npcs``) are scored on the same
labels as the baseline.

Input reconstruction per turn:
- player/DM text: ``transcript.json`` (what the live writer read).
- PC state: the ``character_state`` block injected into that turn's
  ``debug.context`` prompt (the state on disk when the turn started);
  a ``scene.sheets`` event carrying sheet text wins if a replay has one;
  last resort is ``vault-snapshot`` (end-of-run state — unfaithful for
  every turn but the last, flagged in the report).
- NPC roster: identity sheets of the campaign canon (``--canon``).

Labels: JSONL, one ``{"turn", "with_pc": [slugs], "unsure": [slugs]}``
per turn; unsure (turn, slug) pairs are excluded from every metric.

Usage::

    uv run python services/orchestrator/scripts/eval_presence.py \\
        docs/bench/<date>/loop/marathon-50t/replay-004 labels/0925-r004.jsonl \\
        [--turns 1-5] [--canon vault/ember-hollow]

Prints a markdown table; writes ``<labels>.eval.json`` next to the labels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_THIS.parents[1]))

from openai import AsyncOpenAI  # noqa: E402

from aidm import frontmatter  # noqa: E402
from aidm.scene_sheet_writer import SceneSheetWriter, rewrite_sheet  # noqa: E402
from aidm.scene_state import SceneRegistry  # noqa: E402
from aidm.vault import VaultPathError  # noqa: E402

_STATE_MARK = "---\ntype: character_state"


class _ReplayVault:
    """Read-only vault duck: canon NPC identities + one PC state per turn."""

    def __init__(self, canon: Path, snapshot: Path, pc_slug: str) -> None:
        self.files: dict[str, str] = {}
        for path in sorted((canon / "npcs").glob("*.md")):
            if not path.name.endswith(".state.md"):
                self.files[f"npcs/{path.name}"] = path.read_text(encoding="utf-8")
        pc_identity = snapshot / "characters" / f"{pc_slug}.md"
        if pc_identity.is_file():
            self.files[f"characters/{pc_slug}.md"] = pc_identity.read_text(encoding="utf-8")
        self.pc_state = ""

    def read_file(self, path: str, *, max_bytes: int = 200_000) -> str:
        if path not in self.files:
            raise VaultPathError(f"not a file: {path}")
        return self.files[path]

    def list_dir(self, path: str) -> list[str]:
        return [p for p in self.files if p.startswith(f"{path}/")]

    def read_state(self, kind: str, slug: str) -> str | None:
        # NPC states are unknown mid-run: never "gone", so no refusal skews recall.
        return self.pc_state if kind == "pc" else None

    def write_character_state(self, slug: str, content: str) -> str:
        return f"characters/{slug}.state.md"  # Simplification: evaluation never writes

    def write_npc_state(self, slug: str, content: str) -> str:
        return f"npcs/{slug}.state.md"


def _pc_state_from_sheets_block(text: str, pc_slug: str) -> str:
    """The PC ``character_state`` file as the injector rendered it, or ''."""
    head = f"### pc: {pc_slug}\n\n"
    if head not in text:
        return ""
    section = text.split(head, 1)[1].split("\n\n---\n\n### ", 1)[0]
    return section[section.index(_STATE_MARK) :] if _STATE_MARK in section else ""


def load_turn_inputs(
    replay: Path, pc_slug: str
) -> tuple[list[dict[str, Any]], dict[int, tuple[str, str]]]:
    """Transcript rows + ``turn -> (pc_state_text, source)``."""
    transcript: list[dict[str, Any]] = json.loads((replay / "transcript.json").read_text())
    by_player = {str(row["player"]).strip(): int(row["turn"]) for row in transcript}
    states: dict[int, tuple[str, str]] = {}
    turn: int | None = None
    for line in (replay / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event.get("type")
        if kind == "debug.context":
            messages = event.get("messages") or []
            users = [m.get("text", "") for m in messages if m.get("role") == "user"]
            turn = by_player.get(users[-1].strip()) if users else None
            if turn is None:
                continue
            for message in messages:
                state = _pc_state_from_sheets_block(str(message.get("text", "")), pc_slug)
                if state:
                    # Last context of the turn wins (tool rounds re-send it).
                    states[turn] = (state, "debug.context")
        elif kind == "scene.sheets" and turn is not None:
            for sheet in event.get("sheets") or []:
                blob = next(
                    (v for v in sheet.values() if isinstance(v, str) and _STATE_MARK in v), ""
                )
                if sheet.get("kind") == "pc" and blob:
                    states[turn] = (blob[blob.index(_STATE_MARK) :], "scene.sheets")
    snapshot = replay / "vault-snapshot" / "characters" / f"{pc_slug}.state.md"
    fallback = snapshot.read_text(encoding="utf-8") if snapshot.is_file() else ""
    for row in transcript:
        states.setdefault(int(row["turn"]), (fallback, "vault-snapshot"))
    return transcript, states


def _pc_slug(replay: Path) -> str:
    for line in (replay / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("type") == "scene.sheets":
            for sheet in event.get("sheets") or []:
                if sheet.get("kind") == "pc":
                    return str(sheet["slug"])
    states = sorted((replay / "vault-snapshot" / "characters").glob("*.state.md"))
    return states[0].name.removesuffix(".state.md")


def _writer(vault: _ReplayVault, pc_slug: str, locale: str) -> SceneSheetWriter:
    registry = SceneRegistry()
    registry.set_pc(pc_slug)
    return SceneSheetWriter(
        context=SimpleNamespace(messages=[]),  # type: ignore[arg-type]
        registry=registry,
        vault=vault,  # type: ignore[arg-type]
        client=None,  # type: ignore[arg-type]
        model="",
        locale=locale,
        timeout_s=1.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),  # type: ignore[arg-type]
    )


def baseline_present(vault: _ReplayVault, pc_slug: str, player: str, dm: str) -> set[str]:
    """Slugs the voiced + addressed detectors would enter (fresh, empty scene)."""
    writer = _writer(vault, pc_slug, "en")
    writer._auto_enter_voiced_npcs(dm, player)
    return set(writer._registry.in_scene)


def score(
    predictions: dict[int, set[str]], labels: dict[int, dict[str, set[str]]], roster: set[str]
) -> dict[str, Any]:
    """Per-NPC precision/recall + pair accuracy + exact-turn match, unsure excluded."""
    per: dict[str, dict[str, int]] = {s: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for s in roster}
    exact = 0
    for turn, predicted in predictions.items():
        gold, unsure = labels[turn]["with_pc"], labels[turn]["unsure"]
        if predicted - unsure == gold - unsure:
            exact += 1
        for slug in roster - unsure:
            hit, real = slug in predicted, slug in gold
            per[slug]["tp" if hit and real else "fp" if hit else "fn" if real else "tn"] += 1
    npcs: dict[str, dict[str, Any]] = {}
    for slug, c in sorted(per.items()):
        if c["tp"] + c["fp"] + c["fn"] == 0:
            continue
        npcs[slug] = {
            **c,
            "precision": round(c["tp"] / (c["tp"] + c["fp"]), 3) if c["tp"] + c["fp"] else None,
            "recall": round(c["tp"] / (c["tp"] + c["fn"]), 3) if c["tp"] + c["fn"] else None,
        }
    pairs = sum(sum(c.values()) for c in per.values())
    correct = sum(c["tp"] + c["tn"] for c in per.values())
    return {
        "npcs": npcs,
        "pair_accuracy": round(correct / pairs, 3) if pairs else None,
        "exact_turn_match": round(exact / len(predictions), 3) if predictions else None,
    }


def _table(results: dict[str, dict[str, Any]]) -> str:
    lines = ["| signal | npc | P | R | tp | fp | fn |", "|---|---|---|---|---|---|---|"]
    for signal, res in results.items():
        for slug, c in res["npcs"].items():
            lines.append(
                f"| {signal} | {slug} | {c['precision']} | {c['recall']} "
                f"| {c['tp']} | {c['fp']} | {c['fn']} |"
            )
        lines.append(
            f"| {signal} | **overall** | pair acc {res['pair_accuracy']} "
            f"| exact-turn {res['exact_turn_match']} | | | |"
        )
    return "\n".join(lines)


def _parse_turns(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    lo, _, hi = spec.partition("-")
    return int(lo), int(hi or lo)


async def run(args: argparse.Namespace) -> int:
    replay: Path = args.replay_dir
    pc_slug = _pc_slug(replay)
    vault = _ReplayVault(args.canon, replay / "vault-snapshot", pc_slug)
    roster = {p.removeprefix("npcs/").removesuffix(".md") for p in vault.list_dir("npcs")}
    labels = {
        int(row["turn"]): {"with_pc": set(row["with_pc"]), "unsure": set(row.get("unsure") or [])}
        for row in map(json.loads, filter(str.strip, args.labels.read_text().splitlines()))
    }
    # Labelled slugs outside canon (run-seeded stubs) stay in: unreachable FNs, not hidden.
    roster |= {s for label in labels.values() for s in label["with_pc"]}
    transcript, states = load_turn_inputs(replay, pc_slug)
    window = _parse_turns(args.turns)
    client = AsyncOpenAI(
        base_url=os.environ.get("AIDM_LOCAL_LLM_URL", "http://localhost:8002/v1"),
        api_key="local-llama-cpp",  # gateway ignores it, the SDK refuses empty
    )
    model = os.environ.get("AIDM_LOCAL_LLM_MODEL", "qwen3-14b")
    writer = _writer(vault, pc_slug, args.locale)
    llm: dict[int, set[str]] = {}
    base: dict[int, set[str]] = {}
    per_turn: list[dict[str, Any]] = []
    sources: dict[str, int] = {}
    for row in transcript:
        turn = int(row["turn"])
        if turn not in labels or (window and not window[0] <= turn <= window[1]):
            continue
        vault.pc_state, source = states[turn]
        sources[source] = sources.get(source, 0) + 1
        out = await rewrite_sheet(
            client=client,
            model=model,
            vault=vault,  # type: ignore[arg-type]
            kind="pc",
            slug=pc_slug,
            locale=args.locale,
            user_text=str(row["player"]),
            assistant_text=str(row["dm"]),
            turn=turn,
            timeout_s=args.timeout,
        )
        llm[turn] = set(writer._enter_with_pc(out.with_pc))
        base[turn] = baseline_present(vault, pc_slug, str(row["player"]), str(row["dm"]))
        per_turn.append(
            {
                "turn": turn,
                "label": sorted(labels[turn]["with_pc"]),
                "unsure": sorted(labels[turn]["unsure"]),
                "with_pc_raw": list(out.with_pc),
                "with_pc": sorted(llm[turn]),
                "baseline": sorted(base[turn]),
                "writer_skip": out.skipped_reason,
                "pc_state_source": source,
            }
        )
        print(
            f"T{turn}: label={sorted(labels[turn]['with_pc'])} with_pc={sorted(llm[turn])} "
            f"baseline={sorted(base[turn])} skip={out.skipped_reason}",
            file=sys.stderr,
        )
    results = {"with_pc": score(llm, labels, roster), "baseline": score(base, labels, roster)}
    failed = sum(
        1
        for t in per_turn
        if t["writer_skip"] and not t["writer_skip"].startswith(("empty_patch", "write_error"))
    )
    print(f"## Presence eval — {replay} ({len(per_turn)} turns)\n")
    print(_table(results))
    print(f"\nPC state source: {sources}; writer call/parse failures: {failed}")
    out_path = args.labels.with_suffix(".eval.json")
    out_path.write_text(
        json.dumps(
            {
                "replay": str(replay),
                "labels": str(args.labels),
                "model": model,
                "turns": len(per_turn),
                "pc_state_source": sources,
                "writer_failures": failed,
                "results": results,
                "per_turn": per_turn,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--turns", help="inclusive range, e.g. 1-5 (default: all labelled)")
    parser.add_argument("--canon", type=Path, default=_REPO_ROOT / "vault" / "ember-hollow")
    parser.add_argument("--locale", default="en")
    parser.add_argument("--timeout", type=float, default=120.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
