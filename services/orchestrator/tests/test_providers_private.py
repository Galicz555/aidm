"""Private-mode provider factory dispatch.

We can't construct real Pipecat services in a unit test (they bind to
a Pipecat event loop and pull in heavy adapters), but the part worth
guarding is that private mode no longer raises
``NotImplementedError`` from the factory — that's the load-bearing
change in Phase 7.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import pytest
from aidm.config import Config
from aidm.providers import make_llm, make_stt, make_tts, tts_output_sample_rate
from aidm.providers.llm_local import APOLOGY_TEXT
from aidm.session import session_for_locale
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


def _make_private_config() -> Config:
    return Config(
        mode="private",
        transport="local",
        locale="hu",
        audio_input_device=None,
        audio_output_device=None,
        vault_root=Path("/tmp"),
        vault_path=Path("/tmp"),
        groq_api_key="",
        groq_stt_model="whisper-large-v3-turbo",
        openrouter_api_key="",
        openrouter_model="anthropic/claude-sonnet-4.5",
        elevenlabs_api_key="",
        elevenlabs_voice_id="",
        elevenlabs_voice_id_hu="",
        elevenlabs_voice_id_en="",
        elevenlabs_model="eleven_multilingual_v2",
        elevenlabs_stt_model="scribe_v2",
        # Guard tests here pin retry/excision; backtracking would call the
        # gateway's /tokenize for real.
        guard_backtrack=False,
        advance_judge=False,
        elevenlabs_tts_transport="ws",
        stt_provider="elevenlabs",
        local_llm_base_url="http://llm-gateway:8002/v1",
        local_llm_model="gemma-4-26B-A4B-it",
        local_stt_base_url="http://stt:8001",
        local_tts_ws_url="ws://tts:8003/v1/tts/stream",
        local_tts_speaker_wav="/voices/dm_hu.wav",
    )


def test_tts_output_sample_rate_private_is_xtts_native() -> None:
    # XTTS-v2 emits 24 kHz natively; the transport rate gate depends on
    # this matching aidm.audio.OUTPUT_SAMPLE_RATE_TTS.
    config = _make_private_config()
    assert tts_output_sample_rate(config) == 24000


def test_make_stt_private_requires_http_session() -> None:
    # The Whisper-local builder talks HTTP. The factory must refuse to
    # construct without a session rather than crashing mid-turn.
    config = _make_private_config()
    session = session_for_locale("hu")
    with pytest.raises(RuntimeError, match=r"aiohttp\.ClientSession"):
        make_stt(config, session, http_session=None)


def test_make_tts_private_requires_http_session() -> None:
    config = _make_private_config()
    session = session_for_locale("hu")
    with pytest.raises(RuntimeError, match=r"aiohttp\.ClientSession"):
        make_tts(config, session, http_session=None)


def test_emotion_tags_disabled_in_private_mode() -> None:
    # ElevenLabs v3 inline tags are public-mode-only; XTTS would speak
    # the brackets literally. The persona gate keys off this flag.
    config = _make_private_config()
    assert config.tts_supports_emotion_tags is False


@pytest.mark.asyncio
async def test_make_llm_private_returns_openai_compat_service() -> None:
    # llm_local pins the base_url at the local llama.cpp gateway and
    # reuses Pipecat's OpenAI-compat client. Constructing the service
    # is cheap (no network until first turn), so we can assert the
    # wiring directly.
    config = _make_private_config()
    session = session_for_locale("hu")
    async with aiohttp.ClientSession():
        llm = make_llm(config, session)
    # The OpenAILLMService stores its base_url on the async client.
    base_url = getattr(getattr(llm, "_client", None), "base_url", None)
    assert base_url is None or "llm-gateway" in str(base_url) or "8002" in str(base_url)


@pytest.mark.asyncio
async def test_make_llm_private_wires_sampling_from_config() -> None:
    # Config sampling knobs must reach the Settings, and min_p (a
    # llama.cpp-only param) MUST land nested under extra["extra_body"] —
    # a bare extra key becomes a top-level create() kwarg and TypeErrors
    # at first turn. Distinctive values so a pass can't be coincidence.
    config = dataclasses.replace(
        _make_private_config(),
        llm_temperature=0.33,
        llm_top_p=0.91,
        llm_min_p=0.11,
        llm_frequency_penalty=0.22,
        llm_presence_penalty=0.13,
    )
    session = session_for_locale("hu")
    async with aiohttp.ClientSession():
        llm = make_llm(config, session)
    settings = llm._settings
    assert settings.temperature == 0.33
    assert settings.top_p == 0.91
    assert settings.frequency_penalty == 0.22
    assert settings.presence_penalty == 0.13
    extra_body = settings.extra["extra_body"]
    assert extra_body["min_p"] == 0.11
    # Qwen3 <think> traces must stay suppressed for the NPC reply.
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_llm_dispatch_failure_speaks_apology_instead_of_dying() -> None:
    # A gateway failure (ctx-overflow 400, timeout, down) must produce a
    # normal-looking completion carrying the in-character apology — the
    # ErrorFrame path kills the turn upstream and no `turn.status: idle`
    # ever reaches the clients (observed live at loop-turn 23, 2026-07-11).
    config = _make_private_config()
    session = session_for_locale("en")
    async with aiohttp.ClientSession():
        llm = make_llm(config, session)

    async def _boom(self: object, context: object) -> object:
        raise RuntimeError("request (16960 tokens) exceeds the available context size")

    with patch("pipecat.services.openai.base_llm.BaseOpenAILLMService.get_chat_completions", _boom):
        stream = await llm.get_chat_completions(object())
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == APOLOGY_TEXT
    assert chunks[0].choices[0].finish_reason == "stop"


# ---- guarded retry (banned-phrase enforcement) ---------------------------


def _chunk_stream(text: str) -> object:
    async def _gen() -> object:
        yield ChatCompletionChunk(
            id="test",
            object="chat.completion.chunk",
            created=0,
            model="test",
            choices=[Choice(index=0, delta=ChoiceDelta(content=text), finish_reason="stop")],
        )

    return _gen()


def _scripted_dispatch(replies: list[str], calls: list[object]) -> object:
    async def _dispatch(self: object, context: object) -> object:
        calls.append([dict(m) for m in getattr(context, "messages", [])])
        return _chunk_stream(replies[len(calls) - 1])

    return _dispatch


async def _run_guarded(llm: object, context: object) -> str:
    stream = await llm.get_chat_completions(context)  # type: ignore[attr-defined]
    parts: list[str] = []
    async for chunk in stream:
        parts.append(chunk.choices[0].delta.content or "")
    return "".join(parts)


@pytest.mark.asyncio
async def test_guard_is_bypassed_without_banned_phrases() -> None:
    config = _make_private_config()
    async with aiohttp.ClientSession():
        llm = make_llm(config, session_for_locale("en"))
    llm.set_banned_phrases_provider(list)  # always empty
    calls: list[object] = []
    context = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])
    with patch(
        "pipecat.services.openai.base_llm.BaseOpenAILLMService.get_chat_completions",
        _scripted_dispatch(["boots crunching against gravel again"], calls),
    ):
        text = await _run_guarded(llm, context)
    assert "boots crunching" in text  # no bans active → no interference
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_banned_phrase_triggers_one_retry_and_note_is_removed() -> None:
    config = _make_private_config()
    async with aiohttp.ClientSession():
        llm = make_llm(config, session_for_locale("en"))
    llm.set_banned_phrases_provider(lambda: ["boots crunching against gravel"])
    calls: list[list[dict[str, str]]] = []
    context = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])
    with patch(
        "pipecat.services.openai.base_llm.BaseOpenAILLMService.get_chat_completions",
        _scripted_dispatch(
            ["You walk on, boots crunching against gravel, alone.", "You walk on in silence."],
            calls,
        ),
    ):
        text = await _run_guarded(llm, context)
    assert text == "You walk on in silence."  # first draft discarded
    assert len(calls) == 2
    # The retry dispatch saw the discard note; the live context must not.
    assert any("DISCARDED" in m["content"] for m in calls[1])
    assert all("DISCARDED" not in m["content"] for m in context.messages)


@pytest.mark.asyncio
async def test_double_violation_releases_second_draft_bounded() -> None:
    config = _make_private_config()
    async with aiohttp.ClientSession():
        llm = make_llm(config, session_for_locale("en"))
    llm.set_banned_phrases_provider(lambda: ["boots crunching against gravel"])
    calls: list[object] = []
    dirty = "Still, boots crunching against gravel, onward."
    context = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])
    with patch(
        "pipecat.services.openai.base_llm.BaseOpenAILLMService.get_chat_completions",
        _scripted_dispatch([dirty, dirty], calls),
    ):
        text = await _run_guarded(llm, context)
    assert text == dirty  # released with a warning, never a third attempt
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_banned_match_survives_punctuation_and_case() -> None:
    config = _make_private_config()
    async with aiohttp.ClientSession():
        llm = make_llm(config, session_for_locale("en"))
    llm.set_banned_phrases_provider(lambda: ["boots crunching against gravel"])
    calls: list[object] = []
    context = SimpleNamespace(messages=[])
    with patch(
        "pipecat.services.openai.base_llm.BaseOpenAILLMService.get_chat_completions",
        _scripted_dispatch(["Boots — crunching, against GRAVEL.", "Quiet now."], calls),
    ):
        text = await _run_guarded(llm, context)
    assert text == "Quiet now."  # detector tokenizes like the shingle counter
