"""TimeShownProcessor — code shows a clock advance at the next reply's start."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aidm.character_splitter import CharacterStreamSplitter
from aidm.text_sink import TextSink
from aidm.time_shown import TimeShownProcessor
from aidm.turn_director import DirectorState
from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.tests.utils import run_test


def _reply(*chunks: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(text=c) for c in chunks),
        LLMFullResponseEndFrame(),
    ]


async def _texts(proc: TimeShownProcessor, frames: list[Frame]) -> list[str]:
    down, _ = await run_test(proc, frames_to_send=frames)
    return [f.text for f in down if isinstance(f, LLMTextFrame)]


def _proc(director: DirectorState, initial: str = "Day 1, night.") -> TimeShownProcessor:
    return TimeShownProcessor(director, "en", initial_clock_line=initial)


@pytest.mark.asyncio
async def test_advance_is_shown_once_at_the_start_of_the_next_reply() -> None:
    director = DirectorState(clock_line="Day 1, night.")
    proc = _proc(director)
    assert await _texts(proc, _reply("Quiet.")) == ["Quiet."]  # no advance yet

    director.clock_line = "Day 2, morning."  # end-of-turn engine pass advanced
    shown = await _texts(proc, _reply("Grey ", "light."))
    assert shown == ["*Day 2, morning.*\n\n", "Grey ", "light."]
    # The following turn, with no further advance, carries no line.
    assert await _texts(proc, _reply("Later.")) == ["Later."]


@pytest.mark.asyncio
async def test_first_clock_line_is_adopted_silently_without_a_baseline() -> None:
    director = DirectorState(clock_line="Day 3, evening.")
    proc = _proc(director, initial="")  # no clock seen yet: position, not advance
    assert await _texts(proc, _reply("Dusk.")) == ["Dusk."]
    director.clock_line = "Day 3, night."
    assert (await _texts(proc, _reply("Dark.")))[0] == "*Day 3, night.*\n\n"


@pytest.mark.asyncio
async def test_tool_only_leg_defers_the_line_to_the_leg_that_speaks() -> None:
    director = DirectorState(clock_line="Day 2, morning.")
    proc = _proc(director)
    tool_leg: list[Frame] = [
        LLMFullResponseStartFrame(),
        FunctionCallInProgressFrame(function_name="f", tool_call_id="t", arguments={}),
        LLMFullResponseEndFrame(),
    ]
    assert await _texts(proc, [*tool_leg, *_reply("Sunlight.")]) == [
        "*Day 2, morning.*\n\n",
        "Sunlight.",
    ]


@pytest.mark.asyncio
async def test_shown_line_lands_in_the_transcript_and_llm_context_exactly_once() -> None:
    director = DirectorState(clock_line="Day 1, night.")
    context = LLMContext(messages=[{"role": "user", "content": "I sleep."}])
    sink: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    director_proc = _proc(director)
    director.clock_line = "Day 2, morning."
    pipeline = Pipeline(
        [
            director_proc,
            CharacterStreamSplitter(),
            TextSink(sink),
            LLMContextAggregatorPair(context).assistant(),
        ]
    )
    await run_test(pipeline, frames_to_send=_reply("You wake ", "to birdsong."))

    assistant = [m for m in context.messages if m.get("role") == "assistant"]
    assert [m["content"] for m in assistant] == ["*Day 2, morning.*\n\nYou wake to birdsong."]
    segments: list[str] = []
    while not sink.empty():
        item = sink.get_nowait()
        if item and item.get("type") == "transcript.segment":
            segments.append(str(item["text"]))
    assert "".join(segments).count("*Day 2, morning.*") == 1
