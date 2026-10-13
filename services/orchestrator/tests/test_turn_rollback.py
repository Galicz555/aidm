"""Turn snapshot / rollback for the text transport's "regenerate last reply"."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any, cast

import pytest
from aidm.scene_state import SceneRegistry
from aidm.session_summary import SessionSummarizer
from aidm.time_shown import TimeShownProcessor
from aidm.turn_director import DirectorState
from aidm.turn_rollback import TurnRollback
from aidm.vault import Vault
from pipecat.processors.aggregators.llm_context import LLMContext


class _PendingSummary:
    """Summarizer stand-in whose in-flight summary lands only when settled."""

    def __init__(self, context: LLMContext) -> None:
        self._context = context
        self.settled = False

    async def settle(self) -> None:
        await asyncio.sleep(0)
        self._context.add_message({"role": "system", "content": "summary of turns 1-5"})
        self.settled = True


def _world(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "runs" / "r1" / "characters").mkdir(parents=True)
    vault = Vault(tmp_path, run_slug="r1")
    vault.write_character_state("lyra", "---\ntype: character_state\n---\nat the gate\n")
    context = LLMContext(
        messages=[
            {"role": "user", "content": "I knock."},
            {"role": "assistant", "content": "The gate opens a crack."},
        ]
    )
    transcript = [
        {"role": "user", "text": "I knock."},
        {"role": "assistant", "text": "The gate opens a crack."},
    ]
    registry = SceneRegistry(in_scene={"mara"}, current_pc_slug="lyra")
    director = DirectorState(clock_line="Day 1, morning.", open_threads=["gate → who opened it"])
    director.observe_turn(
        user_text="I knock.",
        prev_user_text="",
        assistant_text="The gate opens a crack.",
        prev_assistant_text="",
    )
    time_shown = TimeShownProcessor(director, "en", initial_clock_line="Day 1, morning.")
    inlined: set[str] = {"npcs/mara.md"}
    summary = _PendingSummary(context)
    rollback = TurnRollback(
        context=context,
        transcript=transcript,
        registry=registry,
        director=director,
        vault=vault,
        processors=[time_shown],
        summarizer=cast(SessionSummarizer, summary),
        inlined_sheets=inlined,
    )
    return {
        "vault": vault,
        "context": context,
        "transcript": transcript,
        "registry": registry,
        "director": director,
        "time_shown": time_shown,
        "inlined": inlined,
        "summary": summary,
        "rollback": rollback,
    }


def _play_a_turn(w: dict[str, Any], reply: str) -> None:
    """Everything a turn touches: history, transcript, scene, director, disk."""
    w["transcript"].append({"role": "user", "text": "I step inside."})
    w["context"].add_message({"role": "user", "content": "I step inside."})
    w["context"].add_message({"role": "assistant", "content": reply})
    w["context"].messages[1]["content"] = "[condensed] gate opened"  # writer, in place
    w["context"].set_tool_choice("required")
    w["transcript"].append({"role": "assistant", "text": reply})
    w["registry"].enter("stranger")
    w["registry"].note_scene_change()
    w["director"].observe_turn(
        user_text="I step inside.",
        prev_user_text="I knock.",
        assistant_text=reply,
        prev_assistant_text="The gate opens a crack.",
    )
    w["director"].clock_line = "Day 1, afternoon."
    w["director"].open_threads = ["stranger → wants the key"]
    w["time_shown"]._shown = "Day 1, afternoon."
    w["inlined"].add("npcs/stranger.md")
    w["vault"].write_character_state("lyra", "---\ntype: character_state\n---\ninside\n")
    w["vault"].write_npc_state("stranger", "---\ntype: npc_state\n---\n")


@pytest.mark.asyncio
async def test_restore_returns_every_piece_of_turn_state_to_the_snapshot(tmp_path: Path) -> None:
    w = _world(tmp_path)
    rollback: TurnRollback = w["rollback"]
    director: DirectorState = w["director"]
    banned_provider = lambda: director.banned_phrases  # noqa: E731 — how the LLM service reads it

    await rollback.capture("I step inside.")
    assert w["summary"].settled  # the pending summary landed before the snapshot
    before = {
        "messages": copy.deepcopy(w["context"].messages),
        "tools": w["context"].tools,
        "tool_choice": w["context"].tool_choice,
        "transcript": copy.deepcopy(w["transcript"]),
        "registry": copy.deepcopy(w["registry"]),
        "director": copy.deepcopy(director),
        "state": (w["vault"].run_root / "characters" / "lyra.state.md").read_bytes(),
    }

    for reply in ("A stranger waits in the hall.", "The hall is empty but for a lamp."):
        _play_a_turn(w, reply)
        assert rollback.discarded_reply() == reply
        assert rollback.turn_number() == 2
        rollback.restore()

        assert w["context"].messages == before["messages"]
        assert w["context"].tools is before["tools"]
        assert w["context"].tool_choice == before["tool_choice"]
        assert w["transcript"] == before["transcript"]
        assert w["registry"] == before["registry"]
        assert w["director"] is director  # restored in place: providers keep working
        assert director == before["director"]
        assert list(director._recent_shingles) == list(before["director"]._recent_shingles)
        assert banned_provider() == before["director"].banned_phrases
        assert w["time_shown"]._shown == "Day 1, morning."
        assert w["inlined"] == {"npcs/mara.md"}
        run = w["vault"].run_root
        assert (run / "characters" / "lyra.state.md").read_bytes() == before["state"]
        assert not (run / "npcs" / "stranger.state.md").exists()


@pytest.mark.asyncio
async def test_discard_drops_the_snapshot_and_stops_journaling(tmp_path: Path) -> None:
    w = _world(tmp_path)
    rollback: TurnRollback = w["rollback"]
    await rollback.capture("I step inside.")
    rollback.discard()
    assert rollback.snapshot is None
    w["vault"].write_beat("farewell beat")
    w["vault"].restore_journal()
    assert (w["vault"].run_root / "beat.md").exists()
