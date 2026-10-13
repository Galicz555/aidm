"""Tests for HistoryTrimmer — drop old turns past the recency threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from aidm.history_trimmer import HistoryTrimmer
from pipecat.frames.frames import LLMContextAssistantTimestampFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection


def _end_frame() -> LLMContextAssistantTimestampFrame:
    return LLMContextAssistantTimestampFrame(timestamp="2026-06-28T16:00:00Z")


@dataclass
class _FakeContext:
    messages: list[dict[str, Any]]


@pytest.mark.asyncio
async def test_trims_old_turns_past_threshold() -> None:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "persona"}]
    for i in range(15):
        msgs.append({"role": "user", "content": f"user_{i}"})
        msgs.append({"role": "assistant", "content": f"asst_{i}"})
    trimmer = HistoryTrimmer(_FakeContext(messages=msgs), max_recent_turns=12)
    await trimmer.process_frame(_end_frame(), FrameDirection.DOWNSTREAM)
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 12
    assert user_msgs[0]["content"] == "user_3"
    assert user_msgs[-1]["content"] == "user_14"
    assert any("trimmed" in m["content"] for m in msgs if m["role"] == "system")


@pytest.mark.asyncio
async def test_no_trim_under_threshold() -> None:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "persona"}]
    for i in range(8):
        msgs.append({"role": "user", "content": f"user_{i}"})
        msgs.append({"role": "assistant", "content": f"asst_{i}"})
    before = len(msgs)
    trimmer = HistoryTrimmer(_FakeContext(messages=msgs), max_recent_turns=12)
    await trimmer.process_frame(_end_frame(), FrameDirection.DOWNSTREAM)
    assert len(msgs) == before


@pytest.mark.asyncio
async def test_preserves_leading_system_block() -> None:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": "scene-sheets"},
        {"role": "system", "content": "resume"},
    ]
    for i in range(15):
        msgs.append({"role": "user", "content": f"user_{i}"})
        msgs.append({"role": "assistant", "content": f"asst_{i}"})
    trimmer = HistoryTrimmer(_FakeContext(messages=msgs), max_recent_turns=12)
    await trimmer.process_frame(_end_frame(), FrameDirection.DOWNSTREAM)
    # First three system messages survive at the head.
    assert msgs[0]["content"] == "persona"
    assert msgs[1]["content"] == "scene-sheets"
    assert msgs[2]["content"] == "resume"
    # Placeholder takes the slot where old turns were.
    assert "trimmed" in msgs[3]["content"]


@pytest.mark.asyncio
async def test_ignores_other_frames() -> None:
    msgs = [{"role": "system", "content": "persona"}]
    trimmer = HistoryTrimmer(_FakeContext(messages=msgs), max_recent_turns=12)
    await trimmer.process_frame(TextFrame(text="ignore me"), FrameDirection.DOWNSTREAM)
    assert msgs == [{"role": "system", "content": "persona"}]


def _long_session(turns: int, reply_chars: int) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "persona"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"user_{i}"})
        msgs.append({"role": "assistant", "content": f"asst_{i} " + "x" * reply_chars})
    return msgs


@pytest.mark.asyncio
async def test_token_budget_trims_below_the_turn_count() -> None:
    # 2026-09-27: 11 turns of ~350-word replies overflowed a 20k window
    # while the 12-turn count never fired. 4000-char replies ≈ 1000 tokens;
    # 5600 fixed chars (persona + tools) ≈ 1400 tokens; budget 4500 fits 3.
    msgs = _long_session(turns=10, reply_chars=4000)
    trimmer = HistoryTrimmer(
        _FakeContext(messages=msgs), max_recent_turns=12, token_budget=4500, fixed_chars=5600
    )
    await trimmer.process_frame(_end_frame(), FrameDirection.DOWNSTREAM)
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert users == ["user_7", "user_8", "user_9"]


@pytest.mark.asyncio
async def test_token_budget_never_drops_below_two_turns() -> None:
    msgs = _long_session(turns=6, reply_chars=20000)
    trimmer = HistoryTrimmer(_FakeContext(messages=msgs), token_budget=1000)
    await trimmer.process_frame(_end_frame(), FrameDirection.DOWNSTREAM)
    assert [m["content"] for m in msgs if m["role"] == "user"] == ["user_4", "user_5"]
