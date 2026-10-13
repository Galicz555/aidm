"""Deterministic replay scorer — bucket behavior on a synthetic bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.loop_score import (
    _D1_PENALTY_PER_DUP,
    _D4_PENALTY_PER_ERROR,
    score_director_health,
    score_duplication,
    score_length_trend,
    score_replay,
    score_tells,
    score_tool_discipline,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def replay(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal synthetic replay dir + canon dir."""
    replay_dir = tmp_path / "replay-001"
    canon = tmp_path / "canon"
    _write(canon / "campaign.md", "# OSR\nThe shelter of Ridgeway holds; Marta heals.\n")
    _write(
        replay_dir / "transcript.json",
        json.dumps(
            [
                {
                    "turn": 1,
                    "player": "I look around.",
                    "dm": "The gate yard is quiet; Marta waves.",
                },
                {
                    "turn": 2,
                    "player": "I talk to her.",
                    "dm": "She lays out the pharmacy plan calmly.",
                },
                {
                    "turn": 3,
                    "player": "I agree.",
                    "dm": "Relief crosses her face; she packs the med bag.",
                },
                {"turn": 4, "player": "We leave.", "dm": "The road east is grey and silent."},
            ]
        ),
    )
    events = [
        {"type": "npc.state", "npcs": [{"slug": "marta", "scene_goal": "recruit an escort"}]},
        {"type": "npc.state", "npcs": [{"slug": "marta", "scene_goal": "reach the pharmacy"}]},
        {"type": "turn.status", "phase": "idle"},
    ]
    _write(replay_dir / "events.jsonl", "\n".join(json.dumps(e) for e in events))
    snapshot = replay_dir / "vault-snapshot"
    _write(
        snapshot / "clock.md",
        "---\ntype: clock\nday: 2\nsegment: morning\nturn: 4\nlast_updated: 2026-07-11\n---\n\n"
        "## Time log\n\n- T4: day 1 night → day 2 morning (baseline tick)\n",
    )
    _write(snapshot / "beat.md", "---\ntype: beat\n---\n\nMarta recruited an escort.\n")
    _write(
        snapshot / "events/marta-asks.md",
        "---\ntype: event\nlang: en\ntitle: Marta asks\nstatus: fired\ntrigger_day: 2\n"
        "fired_turn: 2\nnarrated: true\nlast_updated: 2026-07-11\n---\n\n"
        "## Directive\n\nMarta lays out the pharmacy plan and asks for an escort.\n\n"
        "## On fire\n\n- beat: Marta recruited an escort.\n",
    )
    _write(
        snapshot / "npcs/marta.state.md",
        "---\ntype: npc_state\nstatus: active\nlocation: shelter\nin_scene: true\n"
        "last_updated: 2026-07-11\n---\n\n## Recent (last 5 events)\n\n- T2: asked for escort\n",
    )
    return replay_dir, canon


def test_clean_replay_scores_full_marks(replay: tuple[Path, Path]) -> None:
    replay_dir, canon = replay
    score = score_replay(replay_dir, canon)
    assert score.maximum == 100
    assert score.total == 100, [
        f"{b.label}: {b.awarded}/{b.maximum} {b.detail}" for b in score.buckets
    ]


def test_duplication_bucket_penalizes_fuzzy_repeats() -> None:
    same = "The shelter hums around you, warm and steady, alive with quiet voices."
    bucket = score_duplication(
        [same, same + " Again.", "Something entirely different happens now."]
    )
    assert bucket.awarded < 20


def test_duplication_bucket_sees_copies_beyond_the_previous_reply() -> None:
    same = "The shelter hums around you, warm and steady, alive with quiet voices."
    fillers = [f"Unrelated beat number {n}: {'xyz' * n} happens elsewhere." for n in range(4)]
    bucket = score_duplication([same, *fillers, same])
    assert bucket.awarded < 20


def test_tells_bucket_flags_assistant_voice() -> None:
    bucket = score_tells([{"turn": 3, "dm": "I'd be happy to help you with that quest!"}])
    assert bucket.awarded < 10
    assert "T3" in bucket.detail


def test_tells_bucket_flags_leaked_turn_contract_tag() -> None:
    bucket = score_tells([{"turn": 1, "dm": "<!--aidm:turn-executed-->\n\nThe gate creaks open."}])
    assert bucket.awarded < 10
    assert "T1" in bucket.detail


def test_tells_bucket_flags_leaked_dice_roll() -> None:
    bucket = score_tells([{"turn": 2, "dm": "[ROLL d20+3 = 7 (4+3)]\n\nYou miss the jump."}])
    assert bucket.awarded < 10
    assert "T2" in bucket.detail


def test_tells_bucket_flags_leaked_tool_syntax() -> None:
    bucket = score_tells([{"turn": 4, "dm": "`vault_set_beat` called with:\n- Dusk at the gate"}])
    assert bucket.awarded < 10
    assert "T4" in bucket.detail


def test_tells_bucket_clean_reply_scores_full() -> None:
    bucket = score_tells([{"turn": 1, "dm": "The gate creaks open under the weight of night."}])
    assert bucket.awarded == 10


def test_grounding_bucket_flags_unknown_midsentence_names(replay: tuple[Path, Path]) -> None:
    replay_dir, canon = replay
    transcript = json.loads((replay_dir / "transcript.json").read_text(encoding="utf-8"))
    transcript[0]["dm"] = "You spot the merchant Zebulon near the wall."
    (replay_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    score = score_replay(replay_dir, canon)
    grounding = next(b for b in score.buckets if b.label.startswith("D3"))
    assert grounding.awarded < 15
    assert "Zebulon" in grounding.detail


def test_length_trend_flags_collapse() -> None:
    long_replies = ["word " * 120] * 3
    short_replies = ["word " * 10] * 3
    bucket = score_length_trend(long_replies + short_replies)
    assert bucket.awarded == 0


def test_director_health_flags_frozen_goal() -> None:
    events = [
        {"type": "npc.state", "npcs": [{"slug": "marta", "scene_goal": "same goal"}]}
        for _ in range(6)
    ]
    bucket = score_director_health(events)
    assert bucket.awarded == 0


def test_event_engine_flags_missed_directive(replay: tuple[Path, Path]) -> None:
    replay_dir, canon = replay
    # DM prose that shares nothing with the directive content words.
    transcript = [
        {"turn": n, "player": "x", "dm": "Rain falls. Nothing else moves tonight."}
        for n in range(1, 5)
    ]
    (replay_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    score = score_replay(replay_dir, canon)
    engine = next(b for b in score.buckets if b.label.startswith("D7"))
    assert engine.awarded < 20
    assert "not woven" in engine.detail


def test_event_engine_vacuous_without_events(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay-001"
    _write(replay_dir / "transcript.json", "[]")
    (replay_dir / "vault-snapshot").mkdir(parents=True)
    score = score_replay(replay_dir, tmp_path)
    engine = next(b for b in score.buckets if b.label.startswith("D7"))
    assert engine.awarded == 20


def test_grounding_allows_player_named_entities(replay: tuple[Path, Path]) -> None:
    replay_dir, canon = replay
    transcript = json.loads((replay_dir / "transcript.json").read_text(encoding="utf-8"))
    transcript[0]["player"] = "I ask about Zebulon."
    transcript[0]["dm"] = "You spot the merchant Zebulon near the wall."
    (replay_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    score = score_replay(replay_dir, canon)
    grounding = next(b for b in score.buckets if b.label.startswith("D3"))
    assert grounding.awarded == 15


def test_duplication_flags_player_echo_turns() -> None:
    # D1 blind spot closed after Kimi replay-004: an echoed reply
    # duplicates the PLAYER, so reply-to-reply pairs never saw it.
    transcript = [
        {
            "turn": 1,
            "player": "I hold the gate shut and watch the treeline for movement.",
            "dm": "I hold the gate shut and watch the treeline for movement.",
        },
        {
            "turn": 2,
            "player": "I ask her name.",
            "dm": "She answers slowly, giving a name and a road and nothing else she can spare.",
        },
    ]
    bucket = score_duplication([t["dm"] for t in transcript], transcript)
    assert "1 player-echo turn(s)" in bucket.label
    assert bucket.awarded == 20 - _D1_PENALTY_PER_DUP


def test_duplication_flags_prefix_echo_with_long_tail() -> None:
    player = "I hold the gate shut and watch the treeline for movement."
    transcript = [
        {"turn": 1, "player": player, "dm": player + " " + ("The night thickens. " * 200)},
    ]
    bucket = score_duplication([transcript[0]["dm"]], transcript)
    assert "1 player-echo turn(s)" in bucket.label


def test_tool_discipline_counts_calls_from_the_log_and_errors_from_every_context() -> None:
    # Old tool turns are trimmed out of later contexts; the closing call
    # comes after the last one. Results are JSON-quoted under "text".
    early = {"role": "tool", "text": '"error: not a file: npcs/ghost.md"'}
    ok = {"role": "tool", "text": '"---\\ntype: location"'}
    events = [
        {"type": "debug.context", "messages": [early, ok]},
        {"type": "debug.context", "messages": [ok]},
        {"type": "debug.context", "messages": []},
    ]
    log = "\n".join(
        f"DEBUG _LoggingLocalLLMService#0 Calling function [{name}:x] with arguments {{}}"
        for name in ("vault_read_file", "vault_read_file", "vault_write_session_log")
    )
    bucket = score_tool_discipline(events, log)
    assert bucket.label == "D4 tool discipline (1 error result(s) / 3 tool call(s))"
    assert bucket.awarded == 15 - _D4_PENALTY_PER_ERROR
