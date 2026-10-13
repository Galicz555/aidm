"""Closed-loop harness — pure-function coverage (no network, no server)."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from scripts.loop_sim import (
    Beat,
    PortBusyError,
    _sim_complete,
    _wait_port_free,
    build_sim_prompt,
    collect_turn,
    current_beat,
    is_degenerate,
    load_scenario,
    parse_sse_line,
    start_server,
)

_SCENARIO_YAML = """\
scenario:
  id: smoke-5t
  run_slug: eval-5t
  turns: 5
beats:
  - intent: "Size up the stranger."
    until_turn: 2
  - intent: "Decide the gate."
    until_turn: 5
"""


def test_load_scenario_parses_beats_and_meta(tmp_path: Path) -> None:
    path = tmp_path / "smoke.yaml"
    path.write_text(_SCENARIO_YAML, encoding="utf-8")
    scenario = load_scenario(path)
    assert scenario.id == "smoke-5t"
    assert scenario.run_slug == "eval-5t"
    assert scenario.turns == 5
    assert [b.intent for b in scenario.beats] == ["Size up the stranger.", "Decide the gate."]


def test_load_scenario_rejects_missing_beats(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("scenario: {id: x, run_slug: r, turns: 3}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenario(path)


def test_current_beat_advances_on_schedule_and_clamps() -> None:
    beats = (Beat("first", 2), Beat("second", 5))
    assert current_beat(beats, 1) == "first"
    assert current_beat(beats, 2) == "first"
    assert current_beat(beats, 3) == "second"
    assert current_beat(beats, 99) == "second"  # overflow rides the last beat


def test_is_degenerate_flags_near_duplicates_only() -> None:
    priors = ["I step toward the wire and ask her name."]
    assert is_degenerate("I step toward the wire and ask her name.", priors)
    assert is_degenerate("I step towards the wire and ask her name?", priors)
    assert not is_degenerate('I call up to the platform: "Hold the light steady."', priors)
    assert is_degenerate("   ", priors)  # empty output is always degenerate


def test_build_sim_prompt_carries_sheet_beat_history_and_priors() -> None:
    prompt = build_sim_prompt(
        pc_sheet="name: Rowan",
        beat_intent="Decide the gate.",
        history=[("hello", "the wire hums")],
        priors=["I nod."],
    )
    assert "name: Rowan" in prompt
    assert "Decide the gate." in prompt
    assert "Player: hello" in prompt
    assert "DM: the wire hums" in prompt
    assert "- I nod." in prompt
    assert "Output ONLY the player input" in prompt


def test_parse_sse_line_accepts_data_lines_only() -> None:
    assert parse_sse_line('data: {"type": "turn.status", "phase": "idle"}') == {
        "type": "turn.status",
        "phase": "idle",
    }
    assert parse_sse_line(": keepalive") is None
    assert parse_sse_line("data: not-json") is None
    assert parse_sse_line("") is None
    assert parse_sse_line('data: ["not", "a", "dict"]') is None


def _sse_with(events: list[dict[str, object]]) -> object:
    """Minimal SseCapture stand-in: collect_turn only touches .queue."""

    class _Stub:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    stub = _Stub()
    for event in events:
        stub.queue.put_nowait(event)
    return stub


async def test_collect_turn_survives_tool_call_continuation() -> None:
    # Real public-mode shape: narration → tool call → continuation gives
    # two debug.bot payloads and two writer cycles (two idles).
    sse = _sse_with(
        [
            {"type": "debug.bot", "text": "The shadow swallows you."},
            {"type": "turn.status", "phase": "writing_sheets"},
            {"type": "debug.bot", "text": "The wind shifts."},
            {"type": "turn.status", "phase": "idle"},
            {"type": "turn.status", "phase": "writing_sheets"},
            {"type": "turn.status", "phase": "idle"},
        ]
    )
    dm_text, done = await collect_turn(sse, timeout_s=1.0, grace_s=0.05)  # type: ignore[arg-type]
    assert done
    assert dm_text == "The shadow swallows you.\n\nThe wind shifts."


async def test_collect_turn_times_out_without_idle() -> None:
    sse = _sse_with([{"type": "turn.status", "phase": "writing_sheets"}])
    dm_text, done = await collect_turn(sse, timeout_s=0.1, grace_s=0.05)  # type: ignore[arg-type]
    assert not done
    assert dm_text == ""


async def test_collect_turn_falls_back_to_segments() -> None:
    sse = _sse_with(
        [
            {"type": "transcript.segment", "text": "The wind "},
            {"type": "transcript.segment", "text": "shifts."},
            {"type": "turn.status", "phase": "idle"},
        ]
    )
    dm_text, done = await collect_turn(sse, timeout_s=1.0, grace_s=0.05)  # type: ignore[arg-type]
    assert done
    assert dm_text == "The wind shifts."


async def test_collect_turn_bails_early_when_no_text_ever_arrives() -> None:
    # Empty LLM completion: a scene.sheets event but never any DM text —
    # bail at first_text_s so the caller can re-post, don't burn timeout_s.
    sse = _sse_with([{"type": "scene.sheets", "sheets": []}])
    start = asyncio.get_event_loop().time()
    dm_text, done = await collect_turn(  # type: ignore[arg-type]
        sse, timeout_s=30.0, grace_s=0.05, first_text_s=0.1
    )
    elapsed = asyncio.get_event_loop().time() - start
    assert not done
    assert dm_text == ""
    assert elapsed < 5.0


async def test_sim_complete_retries_transient_then_succeeds() -> None:
    class _Boom(Exception):
        status_code = 500

    calls = {"n": 0}

    async def _create(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Boom("flap")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" hi "))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    # OpenRouter slug path (has "/"); backoff sleeps are real but tiny, few.
    text = await _sim_complete(client, "x/y", "prompt")
    assert text == "hi"
    assert calls["n"] == 3


async def test_sim_complete_reraises_non_transient() -> None:
    class _Bad(Exception):
        status_code = 400

    async def _create(**kwargs: object) -> object:
        raise _Bad("bad request")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    with pytest.raises(_Bad):
        await _sim_complete(client, "x/y", "prompt")


def test_start_server_refuses_a_busy_port(tmp_path: Path) -> None:
    # 2026-09-27: a stale server on the port answered /health for the new
    # one, which died on "address already in use"; a marathon ran on it.
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        with (
            patch("scripts.loop_sim._PORT_FREE_TIMEOUT_S", 0.0),
            pytest.raises(PortBusyError),
        ):
            start_server(
                sandbox_campaign=tmp_path, run_slug="r", port=port, log_path=tmp_path / "log"
            )


def test_port_probe_passes_a_closed_listener() -> None:
    # A server stopped a moment ago leaves its port in TIME_WAIT at most.
    with socket.socket() as gone:
        gone.bind(("127.0.0.1", 0))
        port = gone.getsockname()[1]
    _wait_port_free(port, timeout_s=0.0)
