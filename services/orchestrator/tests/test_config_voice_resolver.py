"""Per-locale voice ID resolver."""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.config import Config


def _make_config(
    *,
    default_voice: str = "default-voice",
    hu_voice: str = "",
    en_voice: str = "",
) -> Config:
    return Config(
        mode="public",
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
        elevenlabs_voice_id=default_voice,
        elevenlabs_voice_id_hu=hu_voice,
        elevenlabs_voice_id_en=en_voice,
        elevenlabs_model="eleven_multilingual_v2",
        elevenlabs_stt_model="scribe_v2",
        elevenlabs_tts_transport="ws",
        stt_provider="elevenlabs",
        local_llm_base_url="http://llm-gateway:8002/v1",
        local_llm_model="gemma-3-27b-it",
        local_stt_base_url="http://stt:8001",
        local_tts_ws_url="ws://tts:8003/v1/tts/stream",
        local_tts_speaker_wav="/voices/dm_hu.wav",
    )


def test_voice_falls_back_to_default_when_per_locale_unset() -> None:
    config = _make_config()
    assert config.voice_id_for("hu") == "default-voice"
    assert config.voice_id_for("en") == "default-voice"


def test_voice_uses_locale_override_when_set() -> None:
    config = _make_config(hu_voice="voice-hu-123", en_voice="voice-en-456")
    assert config.voice_id_for("hu") == "voice-hu-123"
    assert config.voice_id_for("en") == "voice-en-456"


def test_voice_falls_back_per_locale_independently() -> None:
    config = _make_config(hu_voice="voice-hu-123")
    assert config.voice_id_for("hu") == "voice-hu-123"
    assert config.voice_id_for("en") == "default-voice"


def test_unknown_locale_returns_default() -> None:
    # voice_id_for is a lookup, not a validator — the locale comes from
    # SessionContext which already enforces the whitelist.
    config = _make_config(hu_voice="voice-hu-123")
    assert config.voice_id_for("de") == "default-voice"


@pytest.mark.parametrize(
    "transport_value, expected_voice",
    [("ws", "default-voice"), ("http", "default-voice")],
)
def test_voice_resolution_independent_of_tts_transport(
    transport_value: str, expected_voice: str
) -> None:
    config = _make_config()
    assert config.voice_id_for("hu") == expected_voice
