"""Small pipeline processors that carry session-lifecycle behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aidm import log
from aidm.pipeline import (
    _build_sheet_writer,
    _EndAfterFarewell,
    _IdleStatus,
    _TranscriptRecorder,
    _TurnCounter,
)
from aidm.tool_router import ToolSubsetSelector
from pipecat.frames.frames import (
    EndTaskFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    OutputTransportMessageFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection


def _end_of_turn() -> LLMContextAssistantTimestampFrame:
    return LLMContextAssistantTimestampFrame(timestamp="2026-09-25T10:00:00Z")


def _pushed_types(proc: Any) -> list[tuple[type, FrameDirection]]:
    return [(type(c.args[0]), c.args[1]) for c in proc.push_frame.await_args_list]


@pytest.mark.asyncio
async def test_farewell_turn_requests_pipeline_end_once_end_session_is_set() -> None:
    selector = ToolSubsetSelector()
    proc = _EndAfterFarewell(selector, cast(Any, SimpleNamespace()))
    proc.push_frame = AsyncMock()  # type: ignore[method-assign]

    await proc.process_frame(_end_of_turn(), FrameDirection.DOWNSTREAM)
    assert EndTaskFrame not in [t for t, _ in _pushed_types(proc)]  # ordinary turn

    selector.end_session = True
    await proc.process_frame(_end_of_turn(), FrameDirection.DOWNSTREAM)
    assert (EndTaskFrame, FrameDirection.UPSTREAM) in _pushed_types(proc)


@pytest.mark.asyncio
async def test_farewell_text_leg_with_pending_tool_call_does_not_end() -> None:
    selector = ToolSubsetSelector()
    selector.end_session = True
    assistant = SimpleNamespace(has_function_calls_in_progress=True)
    proc = _EndAfterFarewell(selector, cast(Any, assistant))
    proc.push_frame = AsyncMock()  # type: ignore[method-assign]

    await proc.process_frame(_end_of_turn(), FrameDirection.DOWNSTREAM)
    assert EndTaskFrame not in [t for t, _ in _pushed_types(proc)]


@pytest.mark.asyncio
async def test_transcript_recorder_appends_the_landed_reply() -> None:
    context = LLMContext(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The innkeeper looks up."},
        ]
    )
    transcript: list[dict[str, str]] = []
    proc = _TranscriptRecorder(context, transcript)
    proc.push_frame = AsyncMock()  # type: ignore[method-assign]

    await proc.process_frame(_end_of_turn(), FrameDirection.DOWNSTREAM)
    assert transcript == [{"role": "assistant", "text": "The innkeeper looks up."}]


@pytest.mark.asyncio
async def test_turn_id_set_in_one_task_is_seen_by_sibling_pipeline_tasks() -> None:
    # Pipecat runs each processor in its own task; they are all spawned
    # after set_session. A turn bump in one must reach the others' logs.
    async def run_session() -> int:
        log.set_session("s1")
        bumped = asyncio.Event()

        async def counter() -> None:
            proc = _TurnCounter()
            proc.push_frame = AsyncMock()  # type: ignore[method-assign]
            await proc.process_frame(
                LLMContextFrame(context=LLMContext()), FrameDirection.DOWNSTREAM
            )
            bumped.set()

        async def sibling() -> int:
            await bumped.wait()
            return log.current_turn()

        seen = asyncio.create_task(sibling())
        await asyncio.create_task(counter())
        return await seen

    assert await asyncio.create_task(run_session()) == 1


@pytest.mark.asyncio
async def test_disabled_sheet_writer_still_ungates_the_ui() -> None:
    emitter = SimpleNamespace(push_frame=AsyncMock())
    config = SimpleNamespace(npc_failsafe_enabled=False)
    proc = _build_sheet_writer(
        cast(Any, config),
        cast(Any, None),
        cast(Any, None),
        cast(Any, None),
        "en",
        emitter=cast(Any, emitter),
    )
    assert isinstance(proc, _IdleStatus)
    proc.push_frame = AsyncMock()  # type: ignore[method-assign]

    await proc.process_frame(_end_of_turn(), FrameDirection.DOWNSTREAM)
    frame = emitter.push_frame.await_args.args[0]
    assert isinstance(frame, OutputTransportMessageFrame)
    assert frame.message == {"type": "turn.status", "phase": "idle"}


@pytest.mark.asyncio
async def test_summarizer_writes_open_threads_into_the_shared_director_state() -> None:
    from aidm.pipeline import _build_summarizer  # noqa: PLC0415
    from aidm.session import session_for_locale  # noqa: PLC0415
    from aidm.turn_director import DirectorState  # noqa: PLC0415

    context = LLMContext(messages=[])
    for n in range(1, 6):
        context.messages.append({"role": "user", "content": f"u{n}"})
        context.messages.append({"role": "assistant", "content": f"a{n}"})
    parts = SimpleNamespace(context=context, director=DirectorState())
    config = SimpleNamespace(mode="private", local_llm_base_url="http://x/v1", local_llm_model="m")
    vault = SimpleNamespace(append_note=lambda *_: "notes/session-t.md")
    summ = _build_summarizer(
        cast(Any, parts), cast(Any, config), session_for_locale("en"), cast(Any, vault)
    )

    async def fake_llm(head: list[Any], prior: str, threads: list[str]) -> str:
        return "Prose.\n\nOpen threads:\n- the ridge watcher → identity unknown"

    summ._ask_llm_to_summarize = fake_llm  # type: ignore[method-assign]
    summ._turns_before = 1
    summ._turns_to_keep = 2
    await summ.maybe_summarize(context)
    assert parts.director.open_threads == ["the ridge watcher → identity unknown"]


def _tracker() -> tuple[Any, list[dict[str, Any]]]:
    from aidm.text_sink import TurnActivity  # noqa: PLC0415

    emitted: list[dict[str, Any]] = []
    return TurnActivity(emit=emitted.append), emitted


def _leg(activity: Any, *, text: str = "", call: str = "") -> None:
    from pipecat.frames.frames import (  # noqa: PLC0415
        FunctionCallFromLLM,
        FunctionCallsStartedFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )

    activity.observe(LLMFullResponseStartFrame())
    if text:
        activity.observe(LLMTextFrame(text=text))
    if call:
        fc = FunctionCallFromLLM(function_name="t", tool_call_id=call, arguments={}, context=None)
        activity.observe(FunctionCallsStartedFrame(function_calls=[fc]))
    activity.observe(LLMFullResponseEndFrame())


def _result(activity: Any, call: str, *, run_llm: bool | None = None) -> None:
    from pipecat.frames.frames import FunctionCallResultFrame  # noqa: PLC0415

    activity.observe(
        FunctionCallResultFrame(
            function_name="t", tool_call_id=call, arguments={}, result="ok", run_llm=run_llm
        )
    )


def test_a_text_leg_with_a_tool_call_does_not_settle_before_the_rerun_lands() -> None:
    # 2026-09-28 review #2: leg 1's writer idle ungated a regenerate while leg 2 streamed.
    activity, emitted = _tracker()
    activity.begin()
    _leg(activity, text="Let me check the ledger.", call="c1")
    _result(activity, "c1")  # the result re-runs the LLM — leg 2 not started yet
    activity.response_landed()  # leg 1 passed the writer + summarizer
    assert activity.busy and emitted == []

    _leg(activity, text="The ledger names you.")
    assert activity.busy  # leg 2 streamed but its writer has not run yet
    activity.response_landed()
    assert not activity.busy and emitted == [{"type": "turn.status", "phase": "idle"}]


def test_a_tool_only_leg_and_its_rerun_settle_once() -> None:
    activity, emitted = _tracker()
    activity.begin()
    _leg(activity, call="c1")  # no text: no timestamp, no writer pass
    assert activity.busy
    _result(activity, "c1")
    _leg(activity, text="Done.")
    activity.response_landed()
    assert not activity.busy and len(emitted) == 1


def test_an_empty_completion_settles_at_its_end() -> None:
    # Review #10: no text → no timestamp → no writer idle; busy used to stick.
    activity, emitted = _tracker()
    activity.begin()
    _leg(activity)
    assert not activity.busy and len(emitted) == 1


def test_a_result_that_does_not_rerun_the_llm_settles() -> None:
    activity, emitted = _tracker()
    activity.begin()
    _leg(activity, text="I note it.", call="c1")
    activity.response_landed()
    assert activity.busy  # the call is still pending
    _result(activity, "c1", run_llm=False)
    assert not activity.busy and len(emitted) == 1


def test_an_interruption_resets_the_counts_for_the_next_turn() -> None:
    from pipecat.frames.frames import InterruptionFrame  # noqa: PLC0415

    activity, emitted = _tracker()
    activity.begin()
    _leg(activity, text="Half a reply", call="c1")  # its timestamp never comes
    activity.observe(InterruptionFrame())
    activity.begin()  # the interrupting turn
    _leg(activity, text="New reply.")
    activity.response_landed()
    assert not activity.busy and len(emitted) == 1


@pytest.mark.asyncio
async def test_the_sink_drops_writer_idles_when_it_tracks_the_turn() -> None:
    from aidm.text_sink import TextSink  # noqa: PLC0415

    activity, emitted = _tracker()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    sink = TextSink(queue, activity=activity)
    sink.push_frame = AsyncMock()  # type: ignore[method-assign]
    idle = OutputTransportMessageFrame(message={"type": "turn.status", "phase": "idle"})
    writing = OutputTransportMessageFrame(
        message={"type": "turn.status", "phase": "writing_sheets"}
    )

    await sink.process_frame(writing, FrameDirection.DOWNSTREAM)
    await sink.process_frame(idle, FrameDirection.DOWNSTREAM)

    assert queue.qsize() == 1  # writing_sheets passes, the writer's idle does not
    assert await activity.wait_settled(0.01)  # nothing in flight yet
    assert emitted == []
