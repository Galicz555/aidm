"""Tests for the session summarizer: split/render helpers, the rolling
summary + open-threads register, and the end-of-turn processor (own task,
live-list splice). The LLM call is replaced by a recorder, never mocked
at the OpenAI-client level.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aidm.session import session_for_locale
from aidm.session_summary import (
    SessionSummarizer,
    _age_threads,
    _carried_threads,
    _count_user_messages,
    _find_split_index,
    _is_summary,
    _render_summary,
    _render_transcript,
    _split_threads,
    _summary_prompt,
    _take_leading_system_messages,
    _wrap_summary_for_context,
)
from aidm.turn_director import DirectorState
from loguru import logger
from pipecat.frames.frames import EndFrame, LLMContextAssistantTimestampFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def test_split_returns_none_when_history_too_short() -> None:
    # 2 user turns, keep last 4 — nothing to summarize yet.
    messages = [
        _msg("system", "persona"),
        _msg("user", "hi"),
        _msg("assistant", "hello"),
        _msg("user", "go"),
        _msg("assistant", "yes"),
    ]
    assert _find_split_index(messages, keep_last_user_turns=4) is None


def test_split_lands_before_kth_last_user_message() -> None:
    messages = [
        _msg("system", "persona"),  # 0
        _msg("user", "u1"),  # 1
        _msg("assistant", "a1"),  # 2
        _msg("user", "u2"),  # 3
        _msg("assistant", "a2"),  # 4
        _msg("user", "u3"),  # 5  <- keep starts here (last 2 user turns)
        _msg("assistant", "a3"),  # 6
        _msg("user", "u4"),  # 7
        _msg("assistant", "a4"),  # 8
    ]
    split = _find_split_index(messages, keep_last_user_turns=2)
    assert split == 5
    head = messages[:split]
    tail = messages[split:]
    assert _count_user_messages(head) == 2
    assert _count_user_messages(tail) == 2


def test_split_returns_none_when_keep_window_covers_all() -> None:
    # Keep 3 user turns but the history only has 3 — nothing older to
    # compress; do not split.
    messages = [
        _msg("system", "persona"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
        _msg("assistant", "a2"),
        _msg("user", "u3"),
        _msg("assistant", "a3"),
    ]
    assert _find_split_index(messages, keep_last_user_turns=3) is None


def test_take_leading_system_messages_stops_at_first_non_system() -> None:
    messages = [
        _msg("system", "persona"),
        _msg("system", "tools"),
        _msg("user", "hi"),
        _msg("system", "should_be_ignored"),  # not at the head; skipped
    ]
    leading = _take_leading_system_messages(messages)
    assert len(leading) == 2
    assert leading[0]["content"] == "persona"
    assert leading[1]["content"] == "tools"


def test_render_transcript_skips_system_and_formats_labels() -> None:
    messages = [
        _msg("system", "persona"),
        _msg("user", "Belépek a kocsmába"),
        _msg("assistant", "A pultos felnéz."),
        _msg("tool", "vault_grep(elara)=found"),
    ]
    rendered = _render_transcript(messages)
    assert "PLAYER: Belépek a kocsmába" in rendered
    assert "DM: A pultos felnéz." in rendered
    assert "TOOL: vault_grep(elara)=found" in rendered
    assert "persona" not in rendered  # system role excluded


def test_render_transcript_handles_list_content() -> None:
    # OpenAI/Anthropic tool-using messages sometimes carry content as a
    # list of typed parts. We flatten to text.
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}],
        }
    ]
    rendered = _render_transcript(messages)
    assert "part one" in rendered
    assert "part two" in rendered


def test_wrap_summary_uses_locale_specific_preamble() -> None:
    hu = _wrap_summary_for_context("kis összegzés", "hu")
    en = _wrap_summary_for_context("short summary", "en")
    assert "automatikus összefoglalója" in hu
    assert "kis összegzés" in hu
    assert "Automated summary" in en
    assert "short summary" in en


# --- pressure-based trigger ---


class _Ctx:
    """Stand-in for LLMContext with a settable messages list."""

    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages


def test_pressure_trigger_fires_before_turn_threshold(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # ctx=1000 tokens, 70% threshold => 700 tokens (~2450 chars).
    # Stuff one message past that and verify the pressure check trips
    # immediately, regardless of turn count.
    from aidm.config import Config  # noqa: PLC0415
    from aidm.session import session_for_locale  # noqa: PLC0415
    from aidm.session_summary import SessionSummarizer  # noqa: PLC0415
    from aidm.vault import Vault  # noqa: PLC0415

    monkeypatch.setenv("AIDM_LLM_CTX_SIZE", "1000")
    monkeypatch.setenv("AIDM_SUMMARY_PRESSURE_RATIO", "0.7")
    monkeypatch.setenv("AIDM_SUMMARY_EVERY_N_TURNS", "100")

    # Minimal Config + Vault stub: the summarizer only touches them
    # if we let it call the LLM, which we don't here.
    cfg = Config(
        mode="private",
        transport="text",
        locale="hu",
        audio_input_device=None,
        audio_output_device=None,
        vault_root=__import__("pathlib").Path("/tmp"),
        vault_path=__import__("pathlib").Path("/tmp"),
        groq_api_key="x",
        groq_stt_model="x",
        openrouter_api_key="x",
        openrouter_model="x",
        elevenlabs_api_key="x",
        elevenlabs_voice_id="x",
        elevenlabs_voice_id_hu="",
        elevenlabs_voice_id_en="",
        elevenlabs_model="x",
        elevenlabs_stt_model="x",
        elevenlabs_tts_transport="ws",
        stt_provider="groq",
        local_llm_base_url="http://localhost:8002/v1",
        local_llm_model="x",
        local_stt_base_url="x",
        local_tts_ws_url="x",
        local_tts_speaker_wav="x",
    )
    vault = Vault(__import__("pathlib").Path("/tmp"))
    summ = SessionSummarizer(
        context=LLMContext(messages=[]), config=cfg, session=session_for_locale("hu"), vault=vault
    )

    # Empty messages → not under pressure.
    assert summ._under_pressure(_Ctx([])) is False  # type: ignore[arg-type]
    # ~3000 chars ≈ 857 tokens > 700 threshold → pressure trips.
    big = _Ctx([{"role": "user", "content": "x" * 3000}])
    assert summ._under_pressure(big) is True  # type: ignore[arg-type]

    # Public (API) mode: the local ctx size doesn't apply; pressure is off.
    public = SessionSummarizer(
        context=LLMContext(messages=[]),
        config=dataclasses.replace(cfg, mode="public"),
        session=session_for_locale("hu"),
        vault=vault,
    )
    assert public._under_pressure(big) is False  # type: ignore[arg-type]


# --- end-of-turn processor: wiring, rolling summary, concurrent turns ---


class _NoteVault:
    def __init__(self) -> None:
        self.notes: list[str] = []

    def append_note(self, _slug: str, content: str) -> str:
        self.notes.append(content)
        return "notes/session-t.md"


def _processor(
    context: LLMContext,
    prompts: list[str],
    *,
    every: int = 1,
    replies: list[str] | None = None,
) -> SessionSummarizer:
    """Summarizer with the LLM call replaced by a recorder; no Config needed."""
    summ = object.__new__(SessionSummarizer)
    FrameProcessor.__init__(summ)
    summ._context = context
    summ._session = session_for_locale("en")
    summ._vault = _NoteVault()  # type: ignore[assignment]
    summ._director = DirectorState()
    summ._thread_ages = {}
    summ._retired_threads = set()
    summ._on_event = None
    summ._turns_since_summary = 0
    summ._summary_index = 0
    summ._task = None
    summ._turns_before = every
    summ._turns_to_keep = 2
    summ._ctx_size = 0
    summ._pressure_ratio = 0.7

    async def fake_llm(head: list[Any], prior: str, threads: list[str]) -> str:
        prompts.append(_summary_prompt("en", _render_transcript(head), prior, threads))
        return replies.pop(0) if replies else f"SUMMARY-{len(prompts)}"

    summ._ask_llm_to_summarize = fake_llm  # type: ignore[method-assign]
    return summ


def _add_turns(context: LLMContext, first: int, last: int) -> None:
    for n in range(first, last + 1):
        context.messages.append(_msg("user", f"u{n}"))
        context.messages.append(_msg("assistant", f"a{n}"))


def _summaries(context: LLMContext) -> list[str]:
    return [str(m["content"]) for m in context.messages if _is_summary(m)]


@pytest.mark.asyncio
async def test_end_of_turn_frame_triggers_summary_and_is_forwarded() -> None:
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    prompts: list[str] = []
    summ = _processor(context, prompts)
    summ.push_frame = AsyncMock()  # type: ignore[method-assign]

    frame = LLMContextAssistantTimestampFrame(timestamp="2026-09-25T10:00:00Z")
    await summ.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert summ._task is not None
    await summ._task

    assert len(prompts) == 1
    assert _summaries(context) == [_wrap_summary_for_context("SUMMARY-1", "en")]
    summ.push_frame.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)


@pytest.mark.asyncio
async def test_summary_survives_cancellation_of_the_frame_task() -> None:
    # Text mode: the next user turn broadcasts an interruption that cancels the
    # processor's frame task while the slow summary call is still running.
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    prompts: list[str] = []
    summ = _processor(context, prompts)
    summ.push_frame = AsyncMock()  # type: ignore[method-assign]
    release = asyncio.Event()
    inner = summ._ask_llm_to_summarize

    async def slow_llm(head: list[Any], prior: str, threads: list[str]) -> str:
        await release.wait()
        return await inner(head, prior, threads)

    summ._ask_llm_to_summarize = slow_llm  # type: ignore[method-assign]
    frame = LLMContextAssistantTimestampFrame(timestamp="2026-09-25T10:00:00Z")
    frame_task = asyncio.create_task(summ.process_frame(frame, FrameDirection.DOWNSTREAM))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    frame_task.cancel()  # the interruption
    summ.push_frame.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    release.set()
    assert summ._task is not None
    await summ._task
    assert _summaries(context) == [_wrap_summary_for_context("SUMMARY-1", "en")]


@pytest.mark.asyncio
async def test_end_frame_waits_for_the_in_flight_summary() -> None:
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    prompts: list[str] = []
    summ = _processor(context, prompts)
    summ.push_frame = AsyncMock()  # type: ignore[method-assign]
    await summ.process_frame(
        LLMContextAssistantTimestampFrame(timestamp="2026-09-25T10:00:00Z"),
        FrameDirection.DOWNSTREAM,
    )
    await summ.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert _summaries(context) == [_wrap_summary_for_context("SUMMARY-1", "en")]


@pytest.mark.asyncio
async def test_rolling_summary_keeps_one_message_and_feeds_prior_forward() -> None:
    context = LLMContext(messages=[])
    prompts: list[str] = []
    summ = _processor(context, prompts)
    _add_turns(context, 1, 5)
    await summ.maybe_summarize(context)
    _add_turns(context, 6, 10)
    await summ.maybe_summarize(context)

    assert _summaries(context) == [_wrap_summary_for_context("SUMMARY-2", "en")]
    assert "SUMMARY-1" in prompts[1]
    assert "PLAYER: u1\n" not in prompts[1]  # folded into SUMMARY-1, not re-fed raw
    assert [m["content"] for m in context.messages[1:]] == ["u9", "a9", "u10", "a10"]


@pytest.mark.asyncio
async def test_turn_appended_during_summary_call_is_kept() -> None:
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    prompts: list[str] = []
    summ = _processor(context, prompts)
    inner = summ._ask_llm_to_summarize

    async def barge_in(head: list[Any], prior: str, threads: list[str]) -> str:
        context.messages.append(_msg("user", "barge-in"))
        return await inner(head, prior, threads)

    summ._ask_llm_to_summarize = barge_in  # type: ignore[method-assign]
    await summ.maybe_summarize(context)

    assert context.messages[-1]["content"] == "barge-in"
    assert len(_summaries(context)) == 1


@pytest.mark.asyncio
async def test_first_summary_of_resumed_session_folds_in_the_resume_memory() -> None:
    from aidm.session_resume import _wrap_for_context  # noqa: PLC0415

    resume = _wrap_for_context("Mira fled Ashford with the ledger.", "log", "s0", "en")
    context = LLMContext(messages=[_msg("system", resume)])
    _add_turns(context, 1, 5)
    prompts: list[str] = []
    summ = _processor(context, prompts)
    await summ.maybe_summarize(context)
    await summ.maybe_summarize(context)  # next summary: prior is the summary, not the memory

    # Otherwise this session's notes file (the summary) drops session s0.
    assert "=== Earlier summary ===\nMira fled Ashford with the ledger.\n" in prompts[0]
    assert "Memory of a previous session" not in prompts[0]  # unwrapped body only
    # The resume message itself stays in the context untouched.
    assert context.messages[0]["content"] == resume


# --- open threads: emergent commitments carried beside the prose summary ---


def test_summary_output_splits_into_prose_and_thread_bullets() -> None:
    raw = (
        "Mira fled Ashford with the ledger.\n\n"
        "**Open threads:**\n"
        "- Kerek claimed dead → body never found\n"
        "* the ridge watcher -> identity unknown\n"
        "1. water margin: nine days → caravan must turn back\n"
    )
    prose, threads = _split_threads(raw)
    assert prose == "Mira fled Ashford with the ledger."
    assert threads == [
        "Kerek claimed dead → body never found",
        "the ridge watcher → identity unknown",
        "water margin: nine days → caravan must turn back",
    ]
    assert _split_threads("Just prose.") == ("Just prose.", None)


@pytest.mark.asyncio
async def test_resolved_thread_is_dropped_and_unresolved_one_kept() -> None:
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    prompts: list[str] = []
    reply = "Prose.\n\nOpen threads:\n- the ridge watcher → identity unknown\n"
    summ = _processor(context, prompts, replies=[reply])
    summ._director.open_threads = [
        "Kerek claimed dead → body never found",
        "the ridge watcher → identity unknown",
    ]
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="INFO")
    try:
        await summ.maybe_summarize(context)
    finally:
        logger.remove(sink)

    # The prior list rides the prompt so the model can drop / keep against it.
    assert "=== Earlier open threads ===\n- Kerek claimed dead → body never found\n" in prompts[0]
    assert summ._director.open_threads == ["the ridge watcher → identity unknown"]
    assert "open_threads: 1 open (1 dropped since last, 0 aged out)" in lines
    # Carried verbatim → one summary older, persisted beside the bullet.
    aged = reply.strip() + " <!--aidm:age=1-->"
    assert _summaries(context) == [_wrap_summary_for_context(aged, "en")]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["Prose only, no section.", "Prose.\n\nOpen threads:\nnone"])
async def test_bad_or_empty_thread_section_keeps_the_previous_list(reply: str) -> None:
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    summ = _processor(context, [], replies=[reply])
    summ._director.open_threads = ["the ridge watcher → identity unknown"]
    await summ.maybe_summarize(context)

    assert summ._director.open_threads == ["the ridge watcher → identity unknown"]
    # The kept list is persisted too (carried verbatim, so one summary older).
    assert summ._vault.notes[-1].endswith(  # type: ignore[attr-defined]
        "Open threads:\n- the ridge watcher → identity unknown <!--aidm:age=1-->"
    )


@pytest.mark.asyncio
async def test_open_threads_are_capped_at_five() -> None:
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    bullets = "\n".join(f"- cause {n} → consequence {n}" for n in range(1, 8))
    summ = _processor(context, [], replies=[f"Prose.\n\nOpen threads:\n{bullets}"])
    await summ.maybe_summarize(context)
    assert summ._director.open_threads == [f"cause {n} → consequence {n}" for n in range(1, 6)]


@pytest.mark.asyncio
async def test_resumed_session_inherits_the_persisted_threads() -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from aidm.session_resume import _wrap_for_context  # noqa: PLC0415

    # What the previous session's notes file holds as its last block.
    note = _render_summary("Mira fled Ashford.", ["the ledger → Ashford guard hunts Mira"])
    resume = _wrap_for_context(note, "running", "s0", "en")
    context = LLMContext(messages=[_msg("system", resume)])
    _add_turns(context, 1, 5)
    director = DirectorState()
    config = SimpleNamespace(mode="private", local_llm_base_url="http://x/v1", local_llm_model="m")
    summ = SessionSummarizer(
        context=context,
        config=config,  # type: ignore[arg-type]
        session=session_for_locale("en"),
        vault=_NoteVault(),  # type: ignore[arg-type]
        director=director,
    )
    # Seeded at build time: the injector renders them from the first turn.
    assert director.open_threads == ["the ledger → Ashford guard hunts Mira"]

    prompts: list[str] = []
    summ._ask_llm_to_summarize = _processor(context, prompts)._ask_llm_to_summarize  # type: ignore[method-assign]
    summ._turns_before = 1
    summ._turns_to_keep = 2
    await summ.maybe_summarize(context)
    assert "=== Earlier summary ===\nMira fled Ashford.\n===" in prompts[0]
    assert "=== Earlier open threads ===\n- the ledger → Ashford guard hunts Mira\n" in prompts[0]


# --- thread aging: verbatim-carried threads go stale after THREAD_MAX_AGE summaries ---

_LYRA = "Lyra\u2019s approach → who she is and why she is coming closer"


@pytest.mark.asyncio
async def test_verbatim_carried_thread_ages_out_at_the_third_summary() -> None:
    context = LLMContext(messages=[])
    _add_turns(context, 1, 5)
    reply = f"Prose.\n\nOpen threads:\n- {_LYRA}\n"
    summ = _processor(context, [], replies=[reply] * 4)
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="INFO")
    try:
        for n in range(4):
            if n:
                _add_turns(context, 4 + 2 * n, 5 + 2 * n)
            await summ.maybe_summarize(context)
            if n < 3:
                assert summ._director.open_threads == [_LYRA]
    finally:
        logger.remove(sink)

    assert summ._director.open_threads == []
    assert [line for line in lines if line.startswith("open_threads:")] == [
        "open_threads: 1 open (0 dropped since last, 0 aged out)",
        "open_threads: 1 open (0 dropped since last, 0 aged out)",
        "open_threads: 1 open (0 dropped since last, 0 aged out)",
        "open_threads: 0 open (0 dropped since last, 1 aged out)",
    ]


def test_reworded_thread_resets_age_and_trivial_edits_do_not() -> None:
    prev = {"lyra's approach → who she is and why she is coming closer": 2}
    kept, ages, _ = _age_threads(["Lyra's approach → she asked for shelter"], prev, "")
    assert ages == {"lyra's approach → she asked for shelter": 0}
    # Case, curly apostrophe, spacing and a final period are the same wording.
    kept, ages, out = _age_threads([_LYRA.upper().replace("\u2019", "'") + "."], prev, "")
    assert (kept, out) == ([], [_LYRA.upper().replace("\u2019", "'") + "."])


def test_thread_still_mentioned_in_the_excerpt_survives_past_max_age() -> None:
    prev = {"lyra's approach → who she is and why she is coming closer": 2}
    excerpt = "Lyra waited at the fence. Her approach had been slow and careful."
    kept, ages, out = _age_threads([_LYRA], prev, excerpt)
    assert (kept, out) == ([_LYRA], [])
    assert ages == {"lyra's approach → who she is and why she is coming closer": 3}
    # Replay 2026-09-26: generic prose ("closer", "coming") kept a departed NPC alive.
    _, _, out = _age_threads([_LYRA], prev, "Kerek stepped closer; dawn was coming.")
    assert out == [_LYRA]


def test_retired_thread_relisted_verbatim_is_dropped_unless_mentioned() -> None:
    # The 14B re-lists an aged-out thread word for word; without a memory of
    # retired keys it would restart at age 0 and cycle forever.
    retired = {"lyra's approach → who she is and why she is coming closer"}
    kept, _, out = _age_threads([_LYRA, "new → fresh"], {}, "", retired)
    assert (kept, out) == (["new → fresh"], [_LYRA])
    kept, _, _ = _age_threads([_LYRA], {}, "Lyra's approach slows; she stops at the wire.", retired)
    assert kept == [_LYRA]


def test_aging_never_refills_past_the_cap() -> None:
    threads = [f"cause {n} → consequence {n}" for n in range(1, 6)]
    kept, _, out = _age_threads(threads, {"cause 1 → consequence 1": 2}, "")
    assert (kept, out) == (threads[1:], threads[:1])


@pytest.mark.asyncio
async def test_thread_age_survives_a_rebuild_from_the_carried_summary() -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    # What a WebRTC grace rebuild / resume carries: the rendered summary.
    note = _render_summary(
        "Prose.",
        [_LYRA, "new → fresh"],
        {"new → fresh": 0} | {"lyra's approach → who she is and why she is coming closer": 2},
    )
    assert _carried_threads([_msg("system", _wrap_summary_for_context(note, "en"))]) == [
        (_LYRA, 2),
        ("new → fresh", 0),
    ]
    context = LLMContext(messages=[_msg("system", _wrap_summary_for_context(note, "en"))])
    _add_turns(context, 1, 5)
    director = DirectorState()
    config = SimpleNamespace(mode="private", local_llm_base_url="http://x/v1", local_llm_model="m")
    summ = SessionSummarizer(
        context=context,
        config=config,  # type: ignore[arg-type]
        session=session_for_locale("en"),
        vault=_NoteVault(),  # type: ignore[arg-type]
        director=director,
    )
    assert director.open_threads == [_LYRA, "new → fresh"]  # the marker never reaches the DM
    reply = f"Prose.\n\nOpen threads:\n- {_LYRA}\n- new → fresh\n"
    summ._ask_llm_to_summarize = _processor(context, [], replies=[reply])._ask_llm_to_summarize  # type: ignore[method-assign]
    summ._turns_before = 1
    summ._turns_to_keep = 2
    await summ.maybe_summarize(context)
    assert director.open_threads == ["new → fresh"]  # Lyra was at age 2, not reset to 0
