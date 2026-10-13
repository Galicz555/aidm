"""Per-model harness knobs: DM thinking and the four pre-release guard switches.

Defaults must reproduce the baseline harness exactly; each ``guard_*``
off disables exactly its own guard; ``dm_thinking`` reaches the DM request
only; and llama.cpp ``reasoning_content`` deltas never reach the guards,
the transcript/TTS text frames, or ``LLMContext`` history.
"""

from __future__ import annotations

import asyncio
import dataclasses
import shlex
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aidm.character_splitter import CharacterStreamSplitter
from aidm.config import Config, load_config
from aidm.providers import make_llm
from aidm.providers.llm_local import _LoggingLocalLLMService
from aidm.session import session_for_locale
from aidm.text_sink import TextSink
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pipecat.frames.frames import LLMContextFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.tests.utils import run_test

_BASE_DISPATCH = "pipecat.services.openai.base_llm.BaseOpenAILLMService.get_chat_completions"
_GUARDS = ("guard_tics", "guard_paragraph", "guard_replay", "guard_directives")
_KNOB_ENV = (
    "AIDM_DM_THINKING",
    "AIDM_GUARD_TICS",
    "AIDM_GUARD_PARAGRAPH",
    "AIDM_GUARD_REPLAY",
    "AIDM_GUARD_DIRECTIVES",
    "AIDM_PROFILE",
)

# A prior DM reply, a tic inside it, and an 8-word looped passage inside it.
_PRIOR = (
    "The dust stirs faintly across the cracked floor of the depot. "
    "Lyra keeps one hand on the rusted rail and watches the tunnel mouth. "
    "Somewhere below, water drips into a pool you cannot see."
)
_TIC = "dust stirs faintly"
_LOOP = "keeps one hand on the rusted rail and"
_DIRECTIVE = "Marta leaves the shelter heading for the pharmacy district wounded."
_FRESH = "Grey light spills through the vent. A gull cries once, far overhead."
_RETRY = "Marta limped past the gate, heading for the pharmacy at last."


def _config(**overrides: Any) -> Config:
    base = Config(
        mode="private",
        transport="text",
        locale="en",
        audio_input_device=None,
        audio_output_device=None,
        vault_root=Path("/tmp"),
        vault_path=Path("/tmp"),
        groq_api_key="",
        groq_stt_model="",
        openrouter_api_key="",
        openrouter_model="",
        elevenlabs_api_key="",
        elevenlabs_voice_id="",
        elevenlabs_voice_id_hu="",
        elevenlabs_voice_id_en="",
        elevenlabs_model="eleven_multilingual_v2",
        elevenlabs_stt_model="scribe_v2",
        elevenlabs_tts_transport="ws",
        stt_provider="elevenlabs",
        local_llm_base_url="http://llm-gateway:8002/v1",
        local_llm_model="test",
        local_stt_base_url="",
        local_tts_ws_url="",
        local_tts_speaker_wav="",
    )
    return dataclasses.replace(base, **overrides)


async def _make(**overrides: Any) -> _LoggingLocalLLMService:
    # These tests pin the four original guards; backtracking and the
    # advance judge would call the live gateway.
    overrides.setdefault("guard_backtrack", False)
    overrides.setdefault("advance_judge", False)
    async with aiohttp.ClientSession():
        llm = make_llm(_config(**overrides), session_for_locale("en"))
    assert isinstance(llm, _LoggingLocalLLMService)
    return llm


def _chunk(content: str | None = None, reasoning: str | None = None) -> ChatCompletionChunk:
    # llama.cpp's reasoning_content is an extra field the SDK keeps as-is.
    fields: dict[str, Any] = {"content": content, "role": "assistant"}
    if reasoning is not None:
        fields["reasoning_content"] = reasoning
    delta = ChoiceDelta.model_validate(fields)
    return ChatCompletionChunk(
        id="t",
        object="chat.completion.chunk",
        created=0,
        model="test",
        choices=[Choice(index=0, delta=delta, finish_reason=None)],
    )


def _scripted(replies: list[list[ChatCompletionChunk]], calls: list[list[dict[str, Any]]]) -> Any:
    """Patch target for the parent dispatch: one scripted chunk list per call."""

    async def _dispatch(self: object, context: Any) -> AsyncIterator[ChatCompletionChunk]:
        calls.append([dict(m) for m in getattr(context, "messages", [])])
        chunks = replies[min(len(calls), len(replies)) - 1]

        async def _gen() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in chunks:
                yield chunk

        return _gen()

    return _dispatch


def _arm(llm: _LoggingLocalLLMService) -> None:
    llm.set_banned_phrases_provider(lambda: [_TIC])
    llm.set_para_banned_provider(lambda: [_LOOP])
    llm.set_event_directives_provider(lambda: [_DIRECTIVE])


async def _released(llm: _LoggingLocalLLMService, context: Any) -> str:
    stream = await llm.get_chat_completions(context)
    return "".join([c.choices[0].delta.content or "" async for c in stream])


def _retry_note(calls: list[list[dict[str, Any]]]) -> str:
    return str(calls[1][-1]["content"])


# ---- config -------------------------------------------------------------


def _min_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AIDM_MODE", "private")
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "c"))
    for key in _KNOB_ENV:
        monkeypatch.delenv(key, raising=False)


def test_knob_defaults_keep_the_baseline_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _min_env(monkeypatch, tmp_path)
    config = load_config()
    assert config.dm_thinking is False
    assert all(getattr(config, g) is True for g in _GUARDS)
    assert config.profile == ""


def test_each_knob_is_read_from_its_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _min_env(monkeypatch, tmp_path)
    for key in _KNOB_ENV[1:5]:
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("AIDM_DM_THINKING", "1")
    monkeypatch.setenv("AIDM_PROFILE", "qwen3-14b-thinking")
    config = load_config()
    assert config.dm_thinking is True
    assert not any(getattr(config, g) for g in _GUARDS)
    assert config.profile == "qwen3-14b-thinking"


# ---- thinking reaches the DM request only -------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [False, True])
async def test_dm_request_carries_enable_thinking_from_config(thinking: bool) -> None:
    llm = await _make(dm_thinking=thinking)
    params = llm.build_chat_completion_params({"messages": [], "tools": None, "tool_choice": None})
    assert params["extra_body"]["chat_template_kwargs"] == {"enable_thinking": thinking}


@pytest.mark.asyncio
async def test_critic_side_call_does_not_inherit_dm_thinking() -> None:
    # The critic shares the DM service's client — the likeliest leak path.
    llm = await _make(dm_thinking=True, critic_pass_enabled=True)
    message = SimpleNamespace(content='{"violations": []}')
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    llm._client = cast(
        Any, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )
    sheets = "<!--aidm:scene-sheets-->\n### npc: a\n### npc: b\n"
    context = LLMContext(messages=[{"role": "system", "content": sheets}])
    await llm._run_critic_pass(context, "draft")
    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert "extra_body" not in kwargs
    assert "enable_thinking" not in repr(kwargs)


def test_only_the_dm_builder_sends_the_thinking_kwarg() -> None:
    # Writer / summary / minds / suggest calls use bare clients and rely on
    # the gateway's non-thinking default; none may copy the DM kwarg.
    aidm_root = Path(__file__).resolve().parents[1] / "aidm"
    senders = sorted(
        str(p.relative_to(aidm_root))
        for p in aidm_root.rglob("*.py")
        if "chat_template_kwargs" in p.read_text(encoding="utf-8")
    )
    assert senders == ["providers/llm_local.py"]


# ---- guard switches -----------------------------------------------------


# Each case trips exactly one guard: (guard, prior replies, draft, dispatches
# with the guard on). Tic/paragraph hits are excised in place (1 dispatch);
# replay/directive failures regenerate once (2 dispatches).
_SOLO = [
    ("guard_tics", [], f"{_FRESH} The {_TIC} again. Nothing else moves.", 1),
    ("guard_paragraph", [], f"{_FRESH} She {_LOOP} waits. Nothing else moves.", 1),
    ("guard_replay", [_PRIOR], _PRIOR, 2),
    ("guard_directives", [], _FRESH, 2),
]


def _solo_context(prior: list[str]) -> LLMContext:
    messages: list[Any] = [{"role": "user", "content": "I wait."}]
    for reply in prior:
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Go."}]
    return LLMContext(messages=messages)


def _solo_arm(llm: _LoggingLocalLLMService, guard: str) -> None:
    if guard == "guard_tics":
        llm.set_banned_phrases_provider(lambda: [_TIC])
    elif guard == "guard_paragraph":
        llm.set_para_banned_provider(lambda: [_LOOP])
    elif guard == "guard_directives":
        llm.set_event_directives_provider(lambda: [_DIRECTIVE])


@pytest.mark.asyncio
@pytest.mark.parametrize(("guard", "prior", "draft", "on_dispatches"), _SOLO)
async def test_guard_on_by_default_acts_on_its_failure(
    guard: str, prior: list[str], draft: str, on_dispatches: int
) -> None:
    llm = await _make()
    _solo_arm(llm, guard)
    calls: list[list[dict[str, Any]]] = []
    with patch(_BASE_DISPATCH, _scripted([[_chunk(draft)], [_chunk(_RETRY)]], calls)):
        text = await _released(llm, _solo_context(prior))
    assert len(calls) == on_dispatches
    assert text != draft


@pytest.mark.asyncio
@pytest.mark.parametrize(("guard", "prior", "draft", "on_dispatches"), _SOLO)
async def test_guard_off_releases_its_failure_untouched(
    guard: str, prior: list[str], draft: str, on_dispatches: int
) -> None:
    llm = await _make(**{guard: False})
    _solo_arm(llm, guard)
    calls: list[list[dict[str, Any]]] = []
    with patch(_BASE_DISPATCH, _scripted([[_chunk(draft)], [_chunk(_RETRY)]], calls)):
        text = await _released(llm, _solo_context(prior))
    assert len(calls) == 1
    assert text == draft


# Retry-note section per guard (llm_local's _*_SECTION texts).
_SECTION = {
    "guard_tics": "overused phrase",
    "guard_paragraph": "verbatim repeated a passage",
    "guard_replay": "replayed your previous reply",
    "guard_directives": "MUST be woven",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("off", _GUARDS)
async def test_one_guard_off_leaves_the_other_three_armed(off: str) -> None:
    # The draft replays the prior reply, which carries the tic and the
    # looped passage and misses the directive: all four guards trip.
    llm = await _make(**{off: False})
    _arm(llm)
    calls: list[list[dict[str, Any]]] = []
    with patch(_BASE_DISPATCH, _scripted([[_chunk(_PRIOR)], [_chunk(_FRESH)]], calls)):
        await _released(llm, _solo_context([_PRIOR]))
    assert len(calls) == 2
    note = _retry_note(calls)
    for guard, section in _SECTION.items():
        assert (section in note) is (guard != off), guard


# ---- reasoning deltas ---------------------------------------------------


def _thinking_reply(reasoning: str, content: str) -> list[ChatCompletionChunk]:
    half = len(reasoning) // 2
    return [
        _chunk(reasoning=reasoning[:half]),
        _chunk(reasoning=reasoning[half:]),
        _chunk(content=content[:10]),
        _chunk(content=content[10:]),
    ]


@pytest.mark.asyncio
async def test_guards_judge_only_content_not_reasoning() -> None:
    # The reasoning copies the prior reply (tic + loop included) and never
    # names the directive; the content is fresh and names Marta. No guard fires.
    llm = await _make(dm_thinking=True)
    _arm(llm)
    content = "Marta limps out toward the pharmacy district, wounded. " + _FRESH
    calls: list[list[dict[str, Any]]] = []
    reply = _thinking_reply(_PRIOR, content)
    with patch(_BASE_DISPATCH, _scripted([reply], calls)):
        text = await _released(llm, _solo_context([_PRIOR]))
    assert len(calls) == 1
    assert text == content


async def _run_pipeline(
    llm: _LoggingLocalLLMService, context: LLMContext, reply: list[ChatCompletionChunk]
) -> tuple[list[str], list[str], list[Any]]:
    """(LLM text frames, transcript segments, assistant history) for one reply.

    The text frames are what the splitter and TTS consume; they come from a
    run of the LLM alone because the assistant aggregator swallows them.
    """
    calls: list[list[dict[str, Any]]] = []
    with patch(_BASE_DISPATCH, _scripted([reply], calls)):
        down, _ = await run_test(llm, frames_to_send=[LLMContextFrame(context)])
    frames = [f.text for f in down if isinstance(f, LLMTextFrame)]
    sink: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    pipeline = Pipeline(
        [
            llm,
            CharacterStreamSplitter(),
            TextSink(sink),
            LLMContextAggregatorPair(context).assistant(),
        ]
    )
    with patch(_BASE_DISPATCH, _scripted([reply], calls)):
        await run_test(pipeline, frames_to_send=[LLMContextFrame(context)])
    assert len(calls) == 2  # one dispatch per run: no guard retry
    segments: list[str] = []
    while not sink.empty():
        item = sink.get_nowait()
        if item and item.get("type") == "transcript.segment":
            segments.append(str(item["text"]))
    messages = cast(list[dict[str, Any]], context.messages)
    history = [m["content"] for m in messages if m.get("role") == "assistant"]
    return frames, segments, history


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered", [False, True], ids=["streamed", "guarded"])
async def test_reasoning_never_reaches_text_frames_transcript_or_history(buffered: bool) -> None:
    llm = await _make(dm_thinking=True)
    _arm(llm)
    secret = "SECRET-PLAN: the player cannot know Marta is the traitor."
    content = "Marta limps out toward the pharmacy district, wounded."
    prior = [_PRIOR] if buffered else []  # a prior reply arms the buffered path
    context = _solo_context(prior)
    if not buffered:
        llm.set_event_directives_provider(list)
        llm.set_banned_phrases_provider(list)
        llm.set_para_banned_provider(list)
    reply = _thinking_reply(secret, content)
    frames, segments, history = await _run_pipeline(llm, context, reply)
    assert "".join(frames).strip() == content  # what TTS / splitter consume
    assert "".join(segments).strip() == content  # transcript
    assert history[-1] == content  # LLMContext history
    everything = repr(frames) + repr(segments) + repr(context.messages)
    assert "SECRET" not in everything


# ---- shipped profiles ---------------------------------------------------

_PROFILES = Path(__file__).resolve().parents[3] / "services" / "llm-gateway" / "profiles"
# What an "as-is" profile may change relative to the Qwen3-14B baseline.
_ASIS_DELTA = {
    "AIDM_PROFILE",
    "AIDM_LLM_MODEL_PATH",
    "AIDM_LLM_GPU_LAYERS",
    "AIDM_LLM_N_CPU_MOE",
    "AIDM_LOCAL_LLM_MODEL",
    # Stability / capacity, not harness: the sm_120 CUDA-graph hang workaround
    # and a larger window where VRAM allows.
    "AIDM_LLM_CUDA_GRAPHS",
    "AIDM_LLM_CTX_SIZE",
}


def _profile(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (_PROFILES / f"{name}.env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, raw = line.partition("=")
            out[key] = " ".join(shlex.split(raw))
    return out


def test_profiles_share_one_key_set_and_name_themselves() -> None:
    names = sorted(p.stem for p in _PROFILES.glob("*.env"))
    assert names == ["qwen3-14b", "qwen3-14b-thinking"]
    keys = {n: set(_profile(n)) for n in names}
    assert all(k == keys["qwen3-14b"] for k in keys.values())
    assert all(_profile(n)["AIDM_PROFILE"] == n for n in names)


# No "-asis" profile ships today; the invariant holds for any one added.
@pytest.mark.parametrize("name", sorted(p.stem for p in _PROFILES.glob("*-asis.env")))
def test_asis_profiles_change_only_model_and_offload(name: str) -> None:
    base, asis = _profile("qwen3-14b"), _profile(name)
    # A chat-template swap is compatibility, not tuning: the GGUF template
    # rejects the orchestrator's mid-conversation system messages.
    if asis["AIDM_LLM_EXTRA_FLAGS"].startswith("--chat-template-file /templates/"):
        asis["AIDM_LLM_EXTRA_FLAGS"] = base["AIDM_LLM_EXTRA_FLAGS"]
    assert {k for k in base if base[k] != asis[k]} <= _ASIS_DELTA


def test_baseline_profile_keeps_thinking_off_and_every_guard_on() -> None:
    baseline = _profile("qwen3-14b")
    assert baseline["AIDM_DM_THINKING"] == "0"
    assert baseline["AIDM_LLM_REASONING_BUDGET"] == "0"
    guards = ("TICS", "PARAGRAPH", "REPLAY", "DIRECTIVES")
    assert all(baseline[f"AIDM_GUARD_{g}"] == "1" for g in guards)


@pytest.mark.parametrize("name", ["qwen3-14b-thinking"])
def test_tuned_profiles_think_on_the_dm_only_with_reasoning_split_out(name: str) -> None:
    tuned = _profile(name)
    assert tuned["AIDM_DM_THINKING"] == "1"
    # Gateway default stays non-thinking so the side calls never reason.
    assert tuned["AIDM_LLM_CHAT_TEMPLATE_KWARGS"] == ""
    assert "--reasoning-format deepseek" in tuned["AIDM_LLM_EXTRA_FLAGS"]
    assert tuned["AIDM_LLM_REASONING_BUDGET"] != "0"  # 0 would force-close DM thinking
