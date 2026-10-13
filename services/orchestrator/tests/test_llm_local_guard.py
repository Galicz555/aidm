"""Guarded-completion retry paths in the local LLM provider.

Tests the three guard conditions (phrase tic, paragraph loop, directive miss),
the draft->critic pass, the directive logit-bias prototype, and the
retry-is-dirty passthrough — all without a real LLM gateway.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from aidm.providers.llm_local import (
    _REPLAY_LOOKBACK,
    _REPLAY_SIMILARITY_MIN,
    EVENT_NOTE_MARKER,
    _cut_echo_opening,
    _earliest_phrase,
    _excise,
    _excise_player_echo,
    _extract_violations,
    _find_sheets_text,
    _first_token_id,
    _LoggingLocalLLMService,
    _missed_directives,
    _npc_count,
    _replay_score,
    _sentence_units,
    _tokenize_url,
)
from aidm.scene_sheet_injector import SCENE_SHEETS_TAG
from aidm.turn_director import similarity
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


def _chunk(text: str, model: str = "test") -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="x",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
    )


async def _stream(*texts: str) -> AsyncIterator[ChatCompletionChunk]:
    for t in texts:
        yield _chunk(t)


def _make_service() -> _LoggingLocalLLMService:
    with patch(
        "aidm.providers.llm_local._LoggingLocalLLMService.create_client",
        return_value=None,
    ):
        svc = _LoggingLocalLLMService.__new__(_LoggingLocalLLMService)
    return svc


class _FakeContext:
    def __init__(self) -> None:
        self.messages: list[Any] = []


@pytest.mark.asyncio
async def test_clean_turn_streams_through_untouched() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    dispatch_calls = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal dispatch_calls
        dispatch_calls += 1
        return _stream("The gate stands open under grey skies.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert dispatch_calls == 1
    assert any("gate" in (c.choices[0].delta.content or "") for c in chunks)


@pytest.mark.asyncio
async def test_banned_phrase_triggers_retry_with_note() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    notes_seen: list[str] = []
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream("The dust stirs faintly across the floor.")
        # On retry, capture the appended system note.
        notes_seen.extend(m["content"] for m in context.messages if m.get("role") == "system")
        return _stream("Grit settles in the corridor behind you.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, ["dust stirs faintly"], [], [])]

    assert call_n == 2
    assert any("DISCARDED" in n and "dust stirs faintly" in n for n in notes_seen)
    # Note must not remain in context after retry.
    assert all("DISCARDED" not in (m.get("content") or "") for m in ctx.messages)
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert "Grit" in text


@pytest.mark.asyncio
async def test_directive_miss_triggers_retry_with_world_events_note() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    notes_seen: list[str] = []
    call_n = 0

    # First reply omits any mention of Marta; second includes her.
    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream("The sun sets quietly.")
        notes_seen.extend(m["content"] for m in context.messages if m.get("role") == "system")
        return _stream("Marta limped past the gate, heading for the pharmacy at last.")

    directive = "Marta leaves the shelter heading for the pharmacy district wounded."
    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [directive], [])]

    assert call_n == 2
    assert any("MUST be woven" in n for n in notes_seen)
    assert not ctx.messages  # note cleaned up


@pytest.mark.asyncio
async def test_paragraph_loop_triggers_para_loop_note() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    notes_seen: list[str] = []
    call_n = 0

    para = "dust stirs faintly beneath the collapsed shelter settling again as silence reclaims"

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(f"{para} the corridor around you.")
        notes_seen.extend(m["content"] for m in context.messages if m.get("role") == "system")
        return _stream("You move on. The air is different now.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [], [para])]

    assert call_n == 2
    assert any("verbatim repeated" in n for n in notes_seen)


@pytest.mark.asyncio
async def test_retry_still_dirty_releases_with_warning() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    call_n = 0

    tic = "boots crunching against gravel down the path"

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream(f"You walk on, {tic}, toward the ridge.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [tic], [], [])]

    # Both completions fired, and the dirty second draft was released
    # (not silently dropped) — the player sees the reply even if still tic.
    assert call_n == 2
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert "crunching" in text


class _FakeCompletion:
    """Minimal stand-in for a non-streaming chat completion response."""

    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


def _sheets_message(*npc_slugs: str) -> dict[str, str]:
    body = "\n\n".join(f"### npc: {slug}\n\n{slug}, some sheet text." for slug in npc_slugs)
    return {"role": "system", "content": f"{SCENE_SHEETS_TAG}\n{body}"}


def test_missed_directives_empty_on_match() -> None:
    reply = "Marta limped past the gate heading for the pharmacy district."
    directive = "Marta leaves the shelter heading for the pharmacy district wounded."
    assert _missed_directives(reply, [directive]) == []


def test_missed_directives_returns_miss() -> None:
    reply = "The sun rose quietly over empty streets."
    directive = "Marta leaves the shelter heading for the pharmacy district wounded."
    assert _missed_directives(reply, [directive]) == [directive]


@pytest.mark.asyncio
async def test_combined_violations_compose_one_note() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    notes_seen: list[str] = []
    call_n = 0

    tic = "dust stirs faintly across the floor"
    directive = "Marta leaves the shelter heading for the pharmacy district wounded."

    # First draft reuses the tic AND misses the directive; the single
    # retry note must name both failures.
    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(f"The {tic}, nothing else moves.")
        notes_seen.extend(m["content"] for m in context.messages if m.get("role") == "system")
        return _stream("Marta limped past the gate, heading for the pharmacy at last.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [tic], [directive], [])]

    assert call_n == 2
    assert any(tic in n and "MUST be woven" in n for n in notes_seen)
    assert not ctx.messages  # note cleaned up


def test_extract_violations_parses_plain_json() -> None:
    assert _extract_violations('{"violations": ["Lyra called Kerek her brother"]}') == [
        "Lyra called Kerek her brother"
    ]


def test_extract_violations_parses_fenced_json() -> None:
    raw = '```json\n{"violations": []}\n```'
    assert _extract_violations(raw) == []


def test_extract_violations_returns_none_on_bad_json() -> None:
    assert _extract_violations("not json at all") is None


def test_find_sheets_text_and_npc_count() -> None:
    ctx = _FakeContext()
    ctx.messages.append({"role": "system", "content": "persona"})
    ctx.messages.append(_sheets_message("lyra", "marta"))
    ctx.messages.append({"role": "user", "content": "hi"})
    sheets = _find_sheets_text(ctx.messages)
    assert sheets is not None
    assert _npc_count(sheets) == 2


@pytest.mark.asyncio
async def test_critic_disabled_by_default_skips_llm_call() -> None:
    svc = _make_service()
    svc._client = AsyncMock()
    ctx = _FakeContext()
    ctx.messages.append(_sheets_message("lyra", "marta"))

    async def fake_dispatch(context: Any) -> Any:
        return _stream("Lyra and Marta talk quietly by the gate.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    svc._client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_critic_skips_llm_call_below_npc_threshold() -> None:
    svc = _make_service()
    svc._critic_enabled = True
    svc._client = AsyncMock()
    ctx = _FakeContext()
    ctx.messages.append(_sheets_message("lyra"))  # only one NPC — not selective

    async def fake_dispatch(context: Any) -> Any:
        return _stream("Lyra waits quietly by the gate.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    svc._client.chat.completions.create.assert_not_called()
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert "waits" in text


@pytest.mark.asyncio
async def test_critic_violation_triggers_retry_with_note() -> None:
    svc = _make_service()
    svc._critic_enabled = True
    svc._client = AsyncMock()
    svc._client.chat.completions.create.return_value = _FakeCompletion(
        '{"violations": ["invents a sibling bond between Lyra and Kerek"]}'
    )
    ctx = _FakeContext()
    ctx.messages.append(_sheets_message("lyra", "kerek"))
    notes_seen: list[str] = []
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream("Lyra reminds Kerek their sibling bond runs deep.")
        notes_seen.extend(m["content"] for m in context.messages if m.get("role") == "system")
        return _stream("Lyra nods once at Kerek, saying nothing.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert call_n == 2
    assert any("sibling bond" in n and "continuity check" in n for n in notes_seen)
    assert not any(m.get("content", "").startswith("Your previous draft") for m in ctx.messages)


@pytest.mark.asyncio
async def test_critic_clean_verdict_streams_through_untouched() -> None:
    svc = _make_service()
    svc._critic_enabled = True
    svc._client = AsyncMock()
    svc._client.chat.completions.create.return_value = _FakeCompletion('{"violations": []}')
    ctx = _FakeContext()
    ctx.messages.append(_sheets_message("lyra", "kerek"))
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream("Lyra and Kerek exchange a wary glance.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert call_n == 1  # critic found nothing — no regenerate
    svc._client.chat.completions.create.assert_awaited_once()
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert "wary glance" in text


@pytest.mark.asyncio
async def test_critic_failure_fails_open() -> None:
    svc = _make_service()
    svc._critic_enabled = True
    svc._client = AsyncMock()
    svc._client.chat.completions.create.side_effect = RuntimeError("gateway down")
    ctx = _FakeContext()
    ctx.messages.append(_sheets_message("lyra", "kerek"))
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream("Lyra and Kerek exchange a wary glance.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert call_n == 1  # critic error treated as no-violation, not a crash
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert "wary glance" in text


def test_tokenize_url_strips_v1_suffix() -> None:
    assert _tokenize_url("http://llm-gateway:8002/v1") == "http://llm-gateway:8002/tokenize"


def test_tokenize_url_handles_trailing_slash() -> None:
    assert _tokenize_url("http://llm-gateway:8002/v1/") == "http://llm-gateway:8002/tokenize"


def test_tokenize_url_empty_base_is_empty() -> None:
    assert _tokenize_url("") == "/tokenize"


@pytest.mark.asyncio
async def test_first_token_id_returns_first_token() -> None:
    fake_response = AsyncMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {"tokens": [8127, 64]}
    fake_client = AsyncMock()
    fake_client.post.return_value = fake_response
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=fake_client):
        token_id = await _first_token_id("http://x/tokenize", " Marta")

    assert token_id == 8127


@pytest.mark.asyncio
async def test_first_token_id_fails_open_on_error() -> None:
    with patch("httpx.AsyncClient", side_effect=RuntimeError("gateway down")):
        token_id = await _first_token_id("http://x/tokenize", " Marta")
    assert token_id is None


@pytest.mark.asyncio
async def test_resolve_logit_bias_empty_without_directives() -> None:
    svc = _make_service()
    svc._logit_bias_strength = 6.0
    assert await svc._resolve_logit_bias([]) is None


@pytest.mark.asyncio
async def test_resolve_logit_bias_maps_anchor_to_token_id() -> None:
    svc = _make_service()
    svc._logit_bias_strength = 6.0

    async def fake_first_token_id(url: str, text: str) -> int | None:
        return 8127 if text == " Marta" else None

    directive = "The plan falls apart when Marta leaves the shelter wounded."
    with patch("aidm.providers.llm_local._first_token_id", side_effect=fake_first_token_id):
        bias = await svc._resolve_logit_bias([directive])

    assert bias == {"8127": 6}


@pytest.mark.asyncio
async def test_resolve_logit_bias_drops_words_that_fail_to_tokenize() -> None:
    svc = _make_service()
    svc._logit_bias_strength = 6.0

    async def fake_first_token_id(url: str, text: str) -> int | None:
        return None

    directive = "The plan falls apart when Marta leaves the shelter wounded."
    with patch("aidm.providers.llm_local._first_token_id", side_effect=fake_first_token_id):
        bias = await svc._resolve_logit_bias([directive])

    assert bias is None


def test_build_chat_completion_params_injects_pending_bias() -> None:
    svc = _make_service()
    svc._pending_logit_bias = {"8127": 6}
    with patch(
        "pipecat.services.openai.base_llm.BaseOpenAILLMService.build_chat_completion_params",
        return_value={"model": "qwen3-14b"},
    ):
        params = svc.build_chat_completion_params({})
    assert params["logit_bias"] == {"8127": 6}


def test_build_chat_completion_params_no_bias_when_unset() -> None:
    svc = _make_service()
    svc._pending_logit_bias = None
    with patch(
        "pipecat.services.openai.base_llm.BaseOpenAILLMService.build_chat_completion_params",
        return_value={"model": "qwen3-14b"},
    ):
        params = svc.build_chat_completion_params({})
    assert "logit_bias" not in params


# --- replay guard: a draft that copies a prior reply --------------------------

_PREV_REPLY = (
    "Kerek's jaw tightens as you grab your curved blade and head for the door, "
    "his eyes flicking between you and the wall where Lyra had stood. He doesn't "
    "move for a long moment, fingers curled against the edge of the bench."
)
_REPLAY_DRAFT = (
    "Kerek's jaw tightens as you adjust your backpack straps and head for the door, "
    "his eyes flicking between you and the wall where Lyra had stood. He doesn't "
    "move for a long moment, fingers curled against the edge of the bench."
)
_FRESH_REPLY = "The river path is slick with runoff; Tomi points out the collapsed pylon ahead."
_LATEST_USER = "The river's going to be worse than the depot, I mutter. Let's move."


def _replay_context() -> _FakeContext:
    ctx = _FakeContext()
    ctx.messages = [
        {"role": "system", "content": "<!--aidm:session-summary-->\nsummary"},
        {"role": "user", "content": "Let's go, I say, grabbing my curved blade."},
        {"role": "assistant", "content": _PREV_REPLY},
        {"role": "system", "content": f"{SCENE_SHEETS_TAG}\nsheets"},
        {"role": "user", "content": _LATEST_USER},
        {"role": "system", "content": "<!--aidm:turn-contract-->\ncontract"},
    ]
    return ctx


def _log_sink() -> tuple[list[str], int]:
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="INFO")
    return lines, sink_id


@pytest.mark.asyncio
async def test_replay_draft_fails_guard_and_retry_hides_the_copied_reply() -> None:
    svc = _make_service()
    ctx = _replay_context()
    before = list(ctx.messages)
    retry_messages: list[Any] = []
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(_REPLAY_DRAFT)
        retry_messages.extend(context.messages)
        return _stream(_FRESH_REPLY)

    lines, sink_id = _log_sink()
    try:
        with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
            chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]
    finally:
        logger.remove(sink_id)

    assert call_n == 2
    # The copied assistant reply is hidden from the retry request only.
    assert all(m.get("content") != _PREV_REPLY for m in retry_messages)
    note = retry_messages[-1]["content"]
    assert "respond to THIS" in note and _LATEST_USER in note
    # History is byte-identical afterwards: same objects, same order.
    assert ctx.messages == before
    assert all(a is b for a, b in zip(ctx.messages, before, strict=True))
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == _FRESH_REPLY
    assert any("replay guard (sim" in ln and "retry cleared" in ln for ln in lines)


@pytest.mark.asyncio
async def test_replay_guard_checks_replies_five_turns_back() -> None:
    svc = _make_service()
    ctx = _replay_context()
    other = "Marta hands you the med kit without a word; the infirmary smells of bleach."
    ctx.messages[4:4] = [
        m
        for n in range(5)
        for m in (
            {"role": "user", "content": f"I nod and follow her out ({n})."},
            {"role": "assistant", "content": f"{other} Step {n}."},
        )
    ]
    before = list(ctx.messages)
    retry_messages: list[Any] = []
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(_REPLAY_DRAFT)
        retry_messages.extend(context.messages)
        return _stream(_FRESH_REPLY)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert call_n == 2
    contents = [m.get("content") for m in retry_messages]
    assert _PREV_REPLY not in contents  # the copied one is hidden
    assert f"{other} Step 4." in contents  # the uncopied ones stay
    assert ctx.messages == before


@pytest.mark.asyncio
async def test_replay_retry_hides_every_copy_in_context_not_just_the_last_two() -> None:
    svc = _make_service()
    ctx = _replay_context()
    # A loop leaves a run of copies in history: three earlier copies.
    ctx.messages[1:1] = [
        m
        for n in range(3)
        for m in (
            {"role": "user", "content": f"Earlier player line {n}."},
            {"role": "assistant", "content": _PREV_REPLY},
        )
    ]
    before = list(ctx.messages)
    retry_messages: list[Any] = []
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(_REPLAY_DRAFT)
        retry_messages.extend(context.messages)
        return _stream(_FRESH_REPLY)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert all(m.get("content") != _PREV_REPLY for m in retry_messages)
    assert ctx.messages == before
    assert all(a is b for a, b in zip(ctx.messages, before, strict=True))


@pytest.mark.asyncio
async def test_replay_retry_still_similar_is_released_after_one_retry() -> None:
    svc = _make_service()
    ctx = _replay_context()
    before = list(ctx.messages)
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream(_REPLAY_DRAFT)

    lines, sink_id = _log_sink()
    try:
        with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
            chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]
    finally:
        logger.remove(sink_id)

    assert call_n == 2  # max one retry
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == _REPLAY_DRAFT
    assert any("replay guard (sim" in ln and "still similar" in ln for ln in lines)
    assert ctx.messages == before


@pytest.mark.asyncio
async def test_fresh_draft_is_released_when_its_retry_replays_a_prior_reply() -> None:
    # 2026-09-26 live-guard replay: T15/T28/T38 drafts were fresh but hit a
    # banned phrase; the retry came back as a 0.84-0.97 replay and shipped.
    svc = _make_service()
    ctx = _replay_context()
    fresh = "The pump coughs twice, then catches, and a moment longer the lights hold."
    outputs = iter([fresh, _REPLAY_DRAFT])

    async def fake_dispatch(context: Any) -> Any:
        return _stream(next(outputs))

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, ["a moment longer"], [], [])]

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == fresh


@pytest.mark.asyncio
async def test_replay_retry_samples_hotter_for_that_call_only() -> None:
    svc = _make_service()
    ctx = _replay_context()
    temps: list[Any] = []
    call_n = 0

    def params() -> Any:
        with patch(
            "pipecat.services.openai.base_llm.BaseOpenAILLMService.build_chat_completion_params",
            return_value={"model": "qwen3-14b", "temperature": 0.65},
        ):
            return svc.build_chat_completion_params({})["temperature"]

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        temps.append(params())
        return _stream(_REPLAY_DRAFT if call_n == 1 else _FRESH_REPLY)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert temps[0] == 0.65
    assert temps[1] > 0.65
    assert params() == 0.65  # reset after the retry


@pytest.mark.asyncio
async def test_prior_reply_alone_arms_the_guard_through_get_chat_completions() -> None:
    svc = _make_service()
    ctx = _replay_context()
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream(_REPLAY_DRAFT if call_n == 1 else _FRESH_REPLY)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        stream = await svc.get_chat_completions(ctx)  # type: ignore[arg-type]
        _ = [c async for c in stream]

    assert call_n == 2


# --- time transitions are not the pre-release guard's business ----------------

_TRANSITION = (
    "Time has passed: it is now day 2, morning (was day 1, night). "
    "Show the passage of time before continuing."
)
_EVENT = "Marta leaves the shelter heading for the pharmacy district wounded."


@pytest.mark.asyncio
async def test_transition_directive_miss_does_not_trigger_retry() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream("Marta limped past the gate, heading for the pharmacy at last.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [_TRANSITION, _EVENT], [])]

    assert call_n == 1


@pytest.mark.asyncio
async def test_event_directive_miss_still_retries_alongside_a_transition() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    notes_seen: list[str] = []
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream("The sun sets quietly.")
        notes_seen.extend(m["content"] for m in context.messages if m.get("role") == "system")
        return _stream("Marta limped past the gate, heading for the pharmacy at last.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [_TRANSITION, _EVENT], [])]

    assert call_n == 2
    assert any(_EVENT in n and "Time has passed" not in n for n in notes_seen)


# --- deterministic excision instead of a regenerate for tics/loops ------------

_TIC = "dust stirs faintly across"
_EXCISE_DRAFT = (
    "Kerek sets the lantern on the crate and counts the shells twice. "
    f"The {_TIC} the floor as the wind shifts. "
    '"We leave at first light. Not a minute later," he says, not looking up. '
    "Outside, the generator coughs and settles into its uneven rhythm."
)


def _text(chunks: list[ChatCompletionChunk]) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks)


@pytest.mark.asyncio
async def test_banned_phrase_sentence_is_excised_without_a_second_dispatch() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream(_EXCISE_DRAFT)

    lines, sink_id = _log_sink()
    try:
        with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
            chunks = [c async for c in svc._guarded_completion(ctx, [_TIC], [], [])]
    finally:
        logger.remove(sink_id)

    assert call_n == 1
    text = _text(chunks)
    assert "dust stirs" not in text
    assert text.startswith("Kerek sets the lantern")
    assert '"We leave at first light. Not a minute later," he says' in text
    assert text.endswith("uneven rhythm.")
    assert any("excised 1 sentence(s) (banned) — no retry" in ln for ln in lines)


@pytest.mark.asyncio
async def test_paragraph_loop_excises_every_sentence_repeated_from_a_prior_reply() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    looped = "The rain drums on the corrugated roof of the old pump station above you."
    also_copied = "Somewhere below, water gurgles through pipes nobody has maintained in years."
    ctx.messages = [
        {"role": "user", "content": "I wait."},
        {"role": "assistant", "content": f"{looped} {also_copied} Tomi yawns."},
        {"role": "user", "content": "I check the valve."},
    ]
    para = "drums on the corrugated roof of the old pump"
    fresh_tail = (
        "Tomi swears under his breath and hands you the pipe wrench from his belt. "
        "The second try moves it a full turn, and somewhere deep in the station a "
        "pressure gauge needle starts to climb toward the red."
    )
    draft = (
        f"You crouch by the valve and test the wheel. {looped} "
        f"It gives a quarter turn, then locks. {also_copied} {fresh_tail}"
    )
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream(draft)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [para])]

    assert call_n == 1
    assert _text(chunks) == (
        "You crouch by the valve and test the wheel. "
        f"It gives a quarter turn, then locks. {fresh_tail}"
    )


def test_sentence_units_never_split_inside_a_quote_pair() -> None:
    text = (
        '"Stop. Don\'t move," she says. The rain falls.\n\n'
        '“Why? Tell me.” He shrugs. "Fine." She turns away!'
    )
    units = [u for u, _ in _sentence_units(text)]
    assert units == [
        '"Stop. Don\'t move," she says.',
        "The rain falls.",
        "“Why? Tell me.”",
        "He shrugs.",
        '"Fine."',
        "She turns away!",
    ]


def test_excision_drops_the_whole_quote_and_keeps_paragraph_breaks() -> None:
    text = (
        "The hall is cold.\n\n"
        '"The dust stirs faintly across the hall. Go," she says. '
        "Nobody moves.\n\nA bell rings somewhere far off."
    )
    cut = _excise(text, [_TIC], [], [])
    assert cut is not None
    assert cut.text == "The hall is cold.\n\nNobody moves.\n\nA bell rings somewhere far off."
    assert cut.dropped == 1


@pytest.mark.asyncio
async def test_excision_over_40_percent_falls_back_to_one_retry() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    call_n = 0
    long_tic_sentence = (
        f"The {_TIC} the floor, curling around the broken crates and the rusted "
        "bedframes stacked against the far wall of the dormitory."
    )

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(f"You wait. {long_tic_sentence} Tomi nods.")
        return _stream("Grit settles in the corridor behind you. Tomi nods.")

    lines, sink_id = _log_sink()
    try:
        with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
            chunks = [c async for c in svc._guarded_completion(ctx, [_TIC], [], [])]
    finally:
        logger.remove(sink_id)

    assert call_n == 2
    assert _text(chunks) == "Grit settles in the corridor behind you. Tomi nods."
    assert any("excised 1 sentence(s) (banned) — fallback retry (would cut" in ln for ln in lines)


@pytest.mark.asyncio
async def test_replay_with_a_banned_phrase_still_retries_with_hiding() -> None:
    svc = _make_service()
    ctx = _replay_context()
    retry_messages: list[Any] = []
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(_REPLAY_DRAFT)
        retry_messages.extend(context.messages)
        return _stream(_FRESH_REPLY)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, ["the edge of the bench"], [], [])]

    assert call_n == 2
    assert all(m.get("content") != _PREV_REPLY for m in retry_messages)
    assert _text(chunks) == _FRESH_REPLY


@pytest.mark.asyncio
async def test_missed_directive_retries_then_the_released_retry_is_excised() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _stream(f"The {_TIC} the floor. The sun sets quietly. Nobody speaks.")
        return _stream(
            "Marta limped past the gate, heading for the pharmacy at last. "
            f"The {_TIC} the floor behind her. You watch until she is gone. "
            "The shelter feels larger without her."
        )

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [_TIC], [_EVENT], [])]

    assert call_n == 2
    assert _text(chunks) == (
        "Marta limped past the gate, heading for the pharmacy at last. "
        "You watch until she is gone. The shelter feels larger without her."
    )


def test_excision_drops_both_sentences_only_when_a_phrase_spans_their_boundary() -> None:
    text = (
        "You wait by the door. The wind howls. Dust stirs faintly. "
        "Across the yard a dog barks twice. Tomi shrugs."
    )
    cut = _excise(text, [_TIC], [], [])
    assert cut is not None
    assert cut.text == "You wait by the door. The wind howls. Tomi shrugs."
    assert (cut.dropped, cut.kinds, cut.kept) == (2, "banned", 3)


@pytest.mark.asyncio
async def test_unsafe_full_repeat_cut_narrows_to_the_flagged_phrase_not_a_retry() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    recycled = [
        "The generator hums behind the plywood wall of the old clinic.",
        "Somewhere below, water gurgles through pipes nobody has maintained in years.",
        "Tomi leans on the doorframe and counts the cartridges in his palm.",
    ]
    # Older than the replay guard's lookback, still in context.
    ctx.messages = [
        {"role": "assistant", "content": " ".join(recycled)},
        *(
            {"role": "assistant", "content": f"Rain again ({n}). Marta sleeps."}
            for n in range(_REPLAY_LOOKBACK)
        ),
    ]
    draft = f"You nod. {' '.join(recycled)} The {_TIC} the step."
    call_n = 0

    async def fake_dispatch(context: Any) -> Any:
        nonlocal call_n
        call_n += 1
        return _stream(draft)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [_TIC], [], [])]

    assert call_n == 1
    assert _text(chunks) == f"You nod. {' '.join(recycled)}"


# --- backtracking phrase ban --------------------------------------------------


def test_earliest_phrase_offsets_into_the_raw_text() -> None:
    text = "He waits. The DUST, stirs faintly — across the floor. It wasn’t sealed."  # noqa: RUF001 — the model emits typographic apostrophes
    assert _earliest_phrase(text, ["wasn t sealed", _TIC]) == (text.index("DUST"), _TIC)
    assert _earliest_phrase(text, ["dusty stirs"]) is None


@pytest.mark.asyncio
async def test_banned_phrase_is_rewritten_in_place_by_backtracking() -> None:
    svc = _make_service()
    svc._guard_backtrack = True
    svc._tokenize_url = "http://gateway/tokenize"
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "I wait."}]
    before = list(ctx.messages)
    seen: list[tuple[Any, Any, Any]] = []

    async def fake_dispatch(context: Any) -> Any:
        last = context.messages[-1]
        seen.append((last.get("content"), svc._step_logit_ban, svc._step_max_tokens))
        if last["role"] == "user":
            return _stream(_EXCISE_DRAFT)
        if svc._step_logit_ban:
            return _stream(" Grit hisses")
        # llama-server echoes the prefill ahead of the continuation.
        return _stream(
            f"{last['content']} over the floor as the wind shifts. Outside, all is still."
        )

    async def fake_token_id(_url: str, text: str) -> int:
        return 7 if text.strip().lower() == "dust" else 9

    with (
        patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch),
        patch("aidm.providers.llm_local._first_token_id", side_effect=fake_token_id),
    ):
        chunks = [c async for c in svc._guarded_completion(ctx, [_TIC], [], [])]

    prefix = _EXCISE_DRAFT[: _EXCISE_DRAFT.index("dust")].rstrip(" ")
    assert _text(chunks) == (
        f"{prefix} Grit hisses over the floor as the wind shifts. Outside, all is still."
    )
    # Draft, then the banned few-token step, then the unbanned rest.
    assert seen[1] == (prefix, {"7": -100}, 4)
    assert seen[2] == (f"{prefix} Grit hisses", None, None)
    assert len(seen) == 3
    assert ctx.messages == before


@pytest.mark.asyncio
async def test_backtrack_at_reply_start_strips_the_empty_think_closer() -> None:
    # 2026-09-26 replay-005 T24/T41: the tic opened the reply, the prefill
    # was empty, and the continuation came back led by "</think>".
    svc = _make_service()
    svc._guard_backtrack = True
    svc._tokenize_url = "http://gateway/tokenize"
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "I wait."}]
    draft = "Kerek’s jaw tightens as he turns away. The generator coughs."  # noqa: RUF001 — the model emits typographic apostrophes

    async def fake_dispatch(context: Any) -> Any:
        last = context.messages[-1]
        if last["role"] == "user":
            return _stream(draft)
        if svc._step_logit_ban:
            return _stream("</think>\n\nThe old")
        return _stream("</think>\n\nThe old man turns away. The generator coughs.")

    async def fake_token_id(_url: str, _text: str) -> int:
        return 5

    with (
        patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch),
        patch("aidm.providers.llm_local._first_token_id", side_effect=fake_token_id),
    ):
        chunks = [c async for c in svc._guarded_completion(ctx, ["kerek s jaw tightens"], [], [])]

    assert _text(chunks) == "The old man turns away. The generator coughs."


# --- player-echo cut ----------------------------------------------------------

_ECHO_PLAYER = (
    "I ran the multitool along the floor again, a slow, deliberate sweep this time, "
    "focusing on the area just ahead of Marta’s feet."  # noqa: RUF001 — the model emits typographic apostrophes
)
_ECHO_REPLY = (
    "You run the multitool along the floor again, slow and deliberate this time, "
    "focusing on the area just ahead of Marta’s feet. "  # noqa: RUF001 — the model emits typographic apostrophes
    "Kerek lets out a sharp breath through his nose. "
    "Somewhere above, the generator coughs and dies."
)


def test_player_echo_opening_is_cut_to_the_world_response() -> None:
    # 2026-09-26 replay-006 T22: person and tense swapped, content copied.
    assert _excise_player_echo(_ECHO_REPLY, _ECHO_PLAYER) == (
        "Kerek lets out a sharp breath through his nose. "
        "Somewhere above, the generator coughs and dies."
    )


def test_a_reply_that_answers_the_player_is_not_cut() -> None:
    reply = "Marta shakes her head. The floor holds, but only just. Kerek says nothing."
    assert _excise_player_echo(reply, _ECHO_PLAYER) is None


def test_an_echo_with_nothing_after_it_is_left_alone() -> None:
    assert _excise_player_echo(_ECHO_REPLY.split(" Kerek", maxsplit=1)[0], _ECHO_PLAYER) is None


@pytest.mark.asyncio
async def test_guarded_completion_releases_the_reply_without_the_echo() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": _ECHO_PLAYER}]

    async def fake_dispatch(context: Any) -> Any:
        return _stream(_ECHO_REPLY)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert _text(chunks).startswith("Kerek lets out a sharp breath")


# --- world events ride the player's message (request only) -------------------

_RAID_EVENT = "(Day 5, night) Narrate now: an Ash Circle raiding party hits the gate."


@pytest.mark.asyncio
async def test_fired_events_ride_the_players_message_for_this_turn_only() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "I check the gate."}]
    seen: list[str] = []

    async def fake_dispatch(context: Any) -> Any:
        seen.append(context.messages[-1]["content"])
        return _stream("Torches crest the ridge — the Ash Circle raiding party hits the gate.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [_RAID_EVENT], [])]

    assert seen[0].startswith("I check the gate.")
    assert EVENT_NOTE_MARKER in seen[0] and _RAID_EVENT in seen[0]
    # History is the player's own words again once the turn is released.
    assert ctx.messages == [{"role": "user", "content": "I check the gate."}]


@pytest.mark.asyncio
async def test_event_injection_can_be_switched_off() -> None:
    svc = _make_service()
    svc._events_in_user = False
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "I check the gate."}]
    seen: list[str] = []

    async def fake_dispatch(context: Any) -> Any:
        seen.append(context.messages[-1]["content"])
        return _stream("The raiders hit the gate.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        _ = [c async for c in svc._guarded_completion(ctx, [], [_RAID_EVENT], [])]

    assert seen[0] == "I check the gate."


@pytest.mark.asyncio
async def test_echo_guard_compares_against_the_players_text_not_the_event_block() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": _ECHO_PLAYER}]

    async def fake_dispatch(context: Any) -> Any:
        return _stream(_ECHO_REPLY + " The raiders hit the gate.")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], ["raiders hit the gate"], [])]

    assert _text(chunks).startswith("Kerek lets out a sharp breath")


# --- replay coverage: a verbatim slice of a longer prior reply ---------------

_LONG_PRIOR = (
    "Her eyes drop once, flat and brief, checking whether he is actually serious or if this is "
    "another test of her patience, and then she looks back up at his face before it hits her what "
    "that means in practice: no time for ceremony, not even a step through the wire. Just here, "
    "now, standing on opposite sides of chain-link like two people negotiating terms across a "
    "border where neither side has a flag. The light keeps going. Somewhere below, the valley "
    "starts to make the noises it only makes after dark."
)


def test_a_prefix_copy_of_a_longer_reply_scores_as_a_replay() -> None:
    # 2026-09-27 playtest: T15 was T14's first 275 of 756 chars (ratio ~0.53).
    draft = _LONG_PRIOR[:275]
    assert similarity(draft, _LONG_PRIOR) < _REPLAY_SIMILARITY_MIN
    assert _replay_score(draft, _LONG_PRIOR) >= _REPLAY_SIMILARITY_MIN


def test_a_fresh_reply_in_the_same_scene_is_not_a_replay() -> None:
    fresh = (
        "She steps through the gap without waiting for an invitation, one foot then the other, "
        "until she stands on your side of the wire. The fence rattles behind her. Down in the "
        "valley something answers the sound, low and patient, and both of you hear it."
    )
    assert _replay_score(fresh, _LONG_PRIOR) < _REPLAY_SIMILARITY_MIN


# --- second stalled beat: judged before release ------------------------------


@pytest.mark.asyncio
async def test_a_second_stalled_beat_is_judged_and_retried_once() -> None:
    svc = _make_service()
    svc.set_stalled_provider(lambda: True)
    svc._client = SimpleNamespace()  # judge_repeats is patched below
    ctx = _FakeContext()
    ctx.messages = [
        {"role": "assistant", "content": "She hesitates at the wire, weighing it."},
        {"role": "user", "content": "Well?"},
    ]
    notes: list[str] = []

    async def fake_dispatch(context: Any) -> Any:
        last = context.messages[-1]
        if last["role"] == "system":
            notes.append(last["content"])
            return _stream("She steps through the gap and hands you the knife.")
        # Same beat, new words: invisible to the verbatim replay guard.
        return _stream("Her fingers stay on the chain-link; nothing is decided yet.")

    with (
        patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch),
        patch("aidm.providers.llm_local.judge_repeats", AsyncMock(return_value=True)),
    ):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert _text(chunks) == "She steps through the gap and hands you the knife."
    assert len(notes) == 1 and "WITHIN the scene" in notes[0] and "Do not skip time" in notes[0]


@pytest.mark.asyncio
async def test_no_judge_call_unless_the_last_reply_stalled() -> None:
    svc = _make_service()
    svc.set_stalled_provider(lambda: False)
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "Well?"}]
    judge = AsyncMock(return_value=True)

    async def fake_dispatch(context: Any) -> Any:
        return _stream("She hesitates at the wire.")

    with (
        patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch),
        patch("aidm.providers.llm_local.judge_repeats", judge),
    ):
        _ = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    judge.assert_not_awaited()


def test_an_opening_that_italic_quotes_the_player_is_cut() -> None:
    # 2026-09-27 playtest 3 T12: 7 of 18 replies opened by quoting the player.
    reply = (
        "The finality in your voice cuts through the tension like a blade. *Decide.*\n\n"
        "Lyra's jaw clenches. She looks at the gate, then at you."
    )
    assert _cut_echo_opening(reply, '"this really is the last minute. Decide"') == (
        "Lyra's jaw clenches. She looks at the gate, then at you."
    )
    # An italic word that is not the player's own stays.
    kept = "The valley goes *quiet*.\n\nLyra looks at the gate, then at you."
    assert _cut_echo_opening(kept, "Decide now") is None


@pytest.mark.asyncio
async def test_leaked_state_brackets_are_removed_from_the_reply() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "I wait at the gate."}]
    reply = "[State: Lyra's mood is tense.]\n\nThe light fails over the gate. Lyra waits."

    async def fake_dispatch(context: Any) -> Any:
        return _stream(reply)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert _text(chunks) == "The light fails over the gate. Lyra waits."


@pytest.mark.asyncio
async def test_every_draft_is_judged_and_regenerated_up_to_the_retry_cap() -> None:
    # Playtest note (2026-09-27): run the check before every release, several times if needed.
    svc = _make_service()
    svc._judge_every_draft = True
    svc._judge_retries = 2
    svc._client = SimpleNamespace()  # judge_repeats is patched below
    ctx = _FakeContext()
    ctx.messages = [
        {"role": "assistant", "content": "She hesitates at the wire, weighing it."},
        {"role": "user", "content": "Well?"},
    ]
    drafts = iter(["Draft one, same beat.", "Draft two, same beat.", "Draft three, still same."])

    async def fake_dispatch(context: Any) -> Any:
        return _stream(next(drafts))

    judge = AsyncMock(return_value=True)
    with (
        patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch),
        patch("aidm.providers.llm_local.judge_repeats", judge),
    ):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert judge.await_count == 2  # judged before each of the 2 retries
    assert _text(chunks) == "Draft three, still same."  # the cap ships the last draft


# --- /end farewell: unguarded, capped goodbye --------------------------------


@pytest.mark.asyncio
async def test_the_end_session_farewell_skips_the_advance_judge_and_every_retry() -> None:
    # 2026-09-27 playtest: the advance judge's "move the scene forward" retry
    # regenerated the goodbye twice into a 5872-char story continuation.
    svc = _make_service()
    svc._judge_every_draft = True
    svc._judge_retries = 2
    svc.set_stalled_provider(lambda: True)
    svc.set_banned_phrases_provider(lambda: ["boots crunching against gravel"])
    svc.set_end_session_provider(lambda: True)
    ctx = _FakeContext()
    ctx.messages = [
        {"role": "assistant", "content": "Boots crunching against gravel, you reach the gate."},
        {"role": "user", "content": "Thanks, we're done for today."},
        {"role": "tool", "content": "session log written to sessions/x.md"},
    ]
    seen: list[tuple[int | None, bool]] = []
    goodbye = "Boots crunching against gravel, we stop here. Thanks for playing!"

    async def fake_dispatch(context: Any) -> Any:
        seen.append((svc._step_max_tokens, svc._farewell_leg))
        return _stream(goodbye)

    judge = AsyncMock(return_value=True)
    with (
        patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch),
        patch("aidm.providers.llm_local.judge_repeats", judge),
    ):
        stream = await svc.get_chat_completions(ctx)  # type: ignore[arg-type]
        chunks = [c async for c in stream]

    judge.assert_not_awaited()
    assert _text(chunks) == goodbye  # released as drafted: no tic excision, no retry
    assert seen == [(256, True)]  # one dispatch, capped, thinking off
    assert svc._step_max_tokens is None and svc._farewell_leg is False


@pytest.mark.asyncio
async def test_the_end_session_log_leg_keeps_the_full_token_budget() -> None:
    svc = _make_service()
    svc.set_end_session_provider(lambda: True)
    ctx = _FakeContext()
    ctx.tool_choice = "required"  # type: ignore[attr-defined]
    seen: list[int | None] = []

    async def fake_dispatch(context: Any) -> Any:
        seen.append(svc._step_max_tokens)
        return _stream("")

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        await svc.get_chat_completions(ctx)  # type: ignore[arg-type]

    assert seen == [None]  # the session-log JSON is never truncated


def test_the_farewell_leg_turns_thinking_off_without_mutating_the_settings() -> None:
    svc = _make_service()
    svc._farewell_leg = True
    extra_body = {"min_p": 0.05, "chat_template_kwargs": {"enable_thinking": True}}
    with patch(
        "pipecat.services.openai.base_llm.BaseOpenAILLMService.build_chat_completion_params",
        return_value={"model": "qwen3-14b", "extra_body": extra_body},
    ):
        params = svc.build_chat_completion_params({})
    assert params["extra_body"] == {
        "min_p": 0.05,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": True}


# --- contrast-construction tic ("It isn't X. It's Y.") -------------------------


def test_the_contrast_construction_is_counted_in_its_three_shapes() -> None:
    from aidm.providers.llm_local import contrast_count  # noqa: PLC0415

    assert contrast_count("It isn't darkness. It's *water*.") == 1
    assert contrast_count("The light doesn't flood. It *arrives*.") == 1
    assert contrast_count("Not a wall — a mouth.") == 1
    assert contrast_count("He doesn't answer; he turns away.") == 1
    assert contrast_count("The door opens. Marta steps through, and the room goes quiet.") == 0
    assert contrast_count("You don't see it coming. The blade is already there.") == 0


_TWO_CONTRASTS = "It isn't darkness. It's *water*. The light doesn't flood. It *arrives*."


@pytest.mark.asyncio
async def test_a_draft_with_two_contrasts_is_retried_once_with_the_note() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "I open my eyes."}]
    notes: list[str] = []

    async def fake_dispatch(context: Any) -> Any:
        last = context.messages[-1]
        if last["role"] == "system":
            notes.append(last["content"])
            return _stream("Water fills the hall to your knees, cold and black.")
        return _stream(_TWO_CONTRASTS)

    with patch.object(svc, "_dispatch_logged", side_effect=fake_dispatch):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]

    assert _text(chunks) == "Water fills the hall to your knees, cold and black."
    assert len(notes) == 1 and "contrast construction" in notes[0]
    assert ctx.messages == [{"role": "user", "content": "I open my eyes."}]  # note removed


@pytest.mark.asyncio
async def test_the_contrast_guard_fails_open_and_allows_one_use() -> None:
    svc = _make_service()
    ctx = _FakeContext()
    ctx.messages = [{"role": "user", "content": "I open my eyes."}]
    calls = 0

    async def still_contrasting(context: Any) -> Any:
        nonlocal calls
        calls += 1
        return _stream(_TWO_CONTRASTS)

    with patch.object(svc, "_dispatch_logged", side_effect=still_contrasting):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]
    assert calls == 2 and _text(chunks) == _TWO_CONTRASTS  # one retry, then released as-is

    once = "It isn't darkness. It's *water*. You wade in."
    with patch.object(svc, "_dispatch_logged", side_effect=lambda _c: _stream(once)):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]
    assert _text(chunks) == once  # a single use is a choice, not a tic

    svc._guard_tics = False  # AIDM_GUARD_TICS=0 switches it off with the tic guard
    with patch.object(svc, "_dispatch_logged", side_effect=lambda _c: _stream(_TWO_CONTRASTS)):
        chunks = [c async for c in svc._guarded_completion(ctx, [], [], [])]
    assert _text(chunks) == _TWO_CONTRASTS
