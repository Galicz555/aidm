"""Provider factories per pipeline slot.

The voice loop is mode-agnostic: it asks for an STT, an LLM, and a TTS,
and these factories hand back whichever backend the configured mode
requires. Public mode wires Groq/Claude/ElevenLabs; private mode wires
local Whisper/Gemma/XTTS over the compose-internal network.

Why factories, not Protocols: every backend is already a Pipecat
FrameProcessor with the same surface (``register_function`` for LLMs,
``set_settings`` for runtime tweaks). An empty Protocol would be noise.
The factory dispatch IS the abstraction.
"""

from __future__ import annotations

from typing import Any

import aiohttp
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from aidm.config import Config
from aidm.providers.llm_local import build_local_llm
from aidm.providers.llm_openrouter import build_openrouter_llm
from aidm.providers.stt_elevenlabs import build_elevenlabs_stt
from aidm.providers.stt_groq import build_groq_stt
from aidm.providers.stt_whisper_local import build_whisper_local_stt
from aidm.providers.tts_elevenlabs import (
    OUTPUT_SAMPLE_RATE as _ELEVENLABS_TTS_RATE,
)
from aidm.providers.tts_elevenlabs import build_elevenlabs_tts
from aidm.providers.tts_moss_nano import OUTPUT_SAMPLE_RATE as _MOSS_NANO_TTS_RATE
from aidm.providers.tts_moss_nano import build_moss_nano_tts
from aidm.session import SessionContext
from aidm.vault import Vault


def make_stt(
    config: Config,
    session: SessionContext,
    *,
    http_session: aiohttp.ClientSession | None = None,
    vault: Vault | None = None,
) -> STTService:
    if config.mode == "private":
        if http_session is None:
            raise RuntimeError(
                "private-mode STT (whisper-local) requires an aiohttp.ClientSession; "
                "the pipeline wiring is missing one."
            )
        return build_whisper_local_stt(config, session, http_session=http_session)
    if config.stt_provider == "elevenlabs":
        if http_session is None:
            raise RuntimeError(
                "STT_PROVIDER=elevenlabs requires an aiohttp.ClientSession; "
                "the pipeline wiring is missing one."
            )
        return build_elevenlabs_stt(config, session, http_session=http_session, vault=vault)
    return build_groq_stt(config, session)


# Return type is the concrete service rather than the generic LLMService base.
# Pipecat's LLMService is Generic[TAdapter] (TAdapter = AnthropicLLMAdapter for
# Claude, GeminiLLMAdapter for Gemini, ...), and the factory can't sensibly pin
# a single TAdapter when it returns different concretes per mode. Pipeline.py
# treats this as ``Any`` for the next-processor link anyway.
def make_llm(
    config: Config,
    session: SessionContext,
    *,
    phase: str = "play",
    vault_summary: str = "",
) -> Any:
    """Build the per-session LLM service.

    ``phase`` selects the session-zero interviewer persona when no
    character sheet exists for the active run; ``vault_summary``
    carries the vault map plus authoritative canon (campaign.md /
    run.md) so the model has the world frame in every turn without
    re-reading. Both go through ``persona_for`` inside the provider.
    """
    if config.mode == "private":
        return build_local_llm(config, session, phase=phase, vault_summary=vault_summary)
    return build_openrouter_llm(config, session, phase=phase, vault_summary=vault_summary)


def make_tts(
    config: Config,
    session: SessionContext,
    *,
    http_session: aiohttp.ClientSession | None = None,
) -> TTSService:
    if config.mode == "private":
        if http_session is None:
            raise RuntimeError(
                "private-mode TTS (moss-nano) requires an aiohttp.ClientSession; "
                "the pipeline wiring is missing one."
            )
        return build_moss_nano_tts(config, session, http_session=http_session)
    return build_elevenlabs_tts(config, session, http_session=http_session)


def tts_output_sample_rate(config: Config) -> int:
    """The sample rate the active TTS will emit.

    The transport needs this at construction time to avoid resampling on
    the output side. Pipecat's TTS services only expose ``sample_rate``
    after pipeline negotiation, so each provider declares it as a module
    constant and we dispatch the same way as ``make_tts``.
    """
    if config.mode == "private":
        return _MOSS_NANO_TTS_RATE
    return _ELEVENLABS_TTS_RATE
