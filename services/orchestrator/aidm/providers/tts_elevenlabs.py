"""ElevenLabs TTS — public-mode speech synthesis.

Two transports, picked at startup via ``ELEVENLABS_TTS_TRANSPORT``:

* ``ws`` (default) — ``ElevenLabsTTSService`` over the multi-stream-input
  WebSocket. Lowest latency, supports ``eleven_flash_v2_5``,
  ``eleven_turbo_v2_5``, ``eleven_multilingual_v2``. **Rejects
  ``eleven_v3`` with HTTP 403** — v3 is not on this endpoint.
* ``http`` — ``ElevenLabsHttpTTSService`` over the chunked-streaming
  ``/stream/with-timestamps`` REST endpoint. ~500ms extra TTFB, but
  unlocks ``eleven_v3`` (highest quality + emotional range + multi-speaker
  + inline ``[whispers]``/``[laughs]`` tags) and ``mp3_44100_192`` audio
  on Creator-tier accounts.

The persona prompt gates emotion-tag emission on the model alone via
``Config.tts_supports_emotion_tags``, so switching transport without
switching the model leaves prompt behavior unchanged.
"""

from __future__ import annotations

import aiohttp
from pipecat.services.elevenlabs.tts import (
    ElevenLabsHttpTTSService,
    ElevenLabsTTSService,
)
from pipecat.transcriptions.language import Language

from aidm.audio import OUTPUT_SAMPLE_RATE_TTS as OUTPUT_SAMPLE_RATE
from aidm.config import Config
from aidm.session import SessionContext

# Re-export for `providers/__init__.py`'s `tts_output_sample_rate()` dispatcher.
# The canonical value lives in `aidm.audio`; this alias keeps the import path
# stable for the providers module while the constant moves to a single source
# of truth at the transport boundary.
__all__ = ["OUTPUT_SAMPLE_RATE", "build_elevenlabs_tts"]

# Models that accept the ``language_code`` parameter. Pipecat 1.3.0 emits
# a ``logger.warning`` every connection if we pass a language with any
# other model — including our default ``eleven_multilingual_v2``, which
# auto-detects language from the input text. The set mirrors Pipecat's
# internal ``ELEVENLABS_MULTILINGUAL_MODELS`` constant; keep them in
# sync if Pipecat extends the list.
_LANGUAGE_AWARE_MODELS: frozenset[str] = frozenset({"eleven_flash_v2_5", "eleven_turbo_v2_5"})


def _language_for(config: Config, session: SessionContext) -> Language | None:
    return session.language if config.elevenlabs_model in _LANGUAGE_AWARE_MODELS else None


class _V3SafeElevenLabsHttpTTSService(ElevenLabsHttpTTSService):
    # eleven_v3 rejects `previous_text` with HTTP 422. Pipecat's HTTP service
    # appends every utterance to `self._previous_text` and unconditionally
    # sends it on the next request. Routing that attribute through a property
    # that no-ops writes (and reports empty) under v3 leaves the existing
    # `if self._previous_text:` guard inside run_tts evaluating False, so the
    # payload key is omitted. Independent of the venv-side patch — survives
    # `uv sync`.

    @property
    def _previous_text(self) -> str:
        if self._is_v3():
            return ""
        stored: str = self.__dict__.get("_v3safe_previous_text", "")
        return stored

    @_previous_text.setter
    def _previous_text(self, value: str) -> None:
        self.__dict__["_v3safe_previous_text"] = "" if self._is_v3() else value

    def _is_v3(self) -> bool:
        model = getattr(getattr(self, "_settings", None), "model", None)
        return isinstance(model, str) and model.startswith("eleven_v3")


def build_elevenlabs_tts(
    config: Config,
    session: SessionContext,
    *,
    http_session: aiohttp.ClientSession | None = None,
) -> ElevenLabsTTSService | ElevenLabsHttpTTSService:
    language = _language_for(config, session)
    voice_id = config.voice_id_for(session.locale)
    if config.elevenlabs_tts_transport == "http":
        if http_session is None:
            raise RuntimeError(
                "ELEVENLABS_TTS_TRANSPORT=http requires an aiohttp.ClientSession; "
                "the pipeline wiring is missing one."
            )
        return _V3SafeElevenLabsHttpTTSService(
            api_key=config.elevenlabs_api_key,
            aiohttp_session=http_session,
            settings=ElevenLabsHttpTTSService.Settings(
                voice=voice_id,
                model=config.elevenlabs_model,
                language=language,
            ),
        )
    return ElevenLabsTTSService(
        api_key=config.elevenlabs_api_key,
        settings=ElevenLabsTTSService.Settings(
            voice=voice_id,
            model=config.elevenlabs_model,
            language=language,
        ),
    )
