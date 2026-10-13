"""MOSS-TTS-Nano client — private-mode speech synthesis.

The remote ``services/tts`` container hosts MOSS-TTS-Nano (CPU,
0.1B params, 20 languages including Hungarian) behind a WebSocket
streaming endpoint (``WS /v1/tts/stream``). Per utterance:

* open one WebSocket
* send a JSON open-frame ``{text, language, speaker_wav}``
* drain binary PCM16 chunks at 24 kHz mono until a JSON
  ``{"event": "done"}`` arrives

We open one WebSocket per ``run_tts`` call rather than holding one
for the session lifetime: localhost handshake is sub-millisecond on
the private compose network, and per-call connections let interrupts
(barge-in) cleanly drop in-flight audio by closing the socket — no
mid-utterance protocol to manage.

We strip ElevenLabs-style ``[whispers]`` bracketed tags defensively
before sending — Gemma should not emit them in private mode
(``Config.tts_supports_emotion_tags`` is False), but the small local
model occasionally hallucinates them and the TTS would spell out the
brackets literally.

Event-driven by nature — Code as Story does not apply.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSStoppedFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from aidm.audio import OUTPUT_SAMPLE_RATE_TTS as OUTPUT_SAMPLE_RATE
from aidm.config import Config
from aidm.session import SessionContext

__all__ = ["OUTPUT_SAMPLE_RATE", "build_moss_nano_tts"]

# ElevenLabs v3 emotion tags: ``[whispers]``, ``[excited]``, ``[laughs]``,
# ``[sad]``, ``[angry]``, and the inline ``[pause 500ms]`` style. The
# private persona never asks for them; this strip is defense-in-depth
# against a local-model regression.
_EMOTION_TAG_RE = re.compile(r"\[[a-zA-Z][^\]]{0,40}\]")


class MossNanoWebsocketTTSService(TTSService):
    """Streaming WebSocket client for the local MOSS-TTS-Nano service."""

    def __init__(
        self,
        *,
        ws_url: str,
        http_session: aiohttp.ClientSession,
        language: str,
        speaker_wav: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=speaker_wav, language=None),
            sample_rate=OUTPUT_SAMPLE_RATE,
            **kwargs,
        )
        self._ws_url = ws_url
        self._http = http_session
        self._language = language
        self._speaker_wav = speaker_wav

    def can_generate_metrics(self) -> bool:
        return True

    # See note in stt_whisper_local: mypy mis-sees the base's
    # ``async def -> AsyncGenerator`` as a Coroutine over a generator.
    # The signatures match at runtime.
    async def run_tts(  # type: ignore[override]
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        cleaned = _strip_emotion_tags(text).strip()
        if not cleaned:
            yield TTSStoppedFrame(context_id=context_id)
            return

        logger.debug("moss-nano: synth [{}]", cleaned)
        try:
            await self.start_tts_usage_metrics(cleaned)
            async for frame in self._stream_audio_frames_from_iterator(
                self._stream_pcm(cleaned),
                in_sample_rate=OUTPUT_SAMPLE_RATE,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as exc:
            logger.exception("moss-nano run_tts failed: {}", exc)
            yield ErrorFrame(error=f"moss-nano: {exc}")
        finally:
            await self.stop_ttfb_metrics()

    async def _stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        payload = {
            "text": text,
            "language": self._language,
            "speaker_wav": self._speaker_wav,
        }
        async with self._http.ws_connect(self._ws_url, heartbeat=30) as ws:
            await ws.send_json(payload)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    yield msg.data
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    event = msg.json()
                    kind = event.get("event")
                    if kind == "done":
                        return
                    if kind == "error":
                        raise RuntimeError(event.get("error") or "moss-nano error")
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    return


def _strip_emotion_tags(text: str) -> str:
    return _EMOTION_TAG_RE.sub("", text)


def build_moss_nano_tts(
    config: Config,
    session: SessionContext,
    *,
    http_session: aiohttp.ClientSession,
) -> MossNanoWebsocketTTSService:
    if http_session is None:
        raise RuntimeError(
            "private-mode TTS (moss-nano) requires an aiohttp.ClientSession; "
            "the pipeline wiring is missing one."
        )
    return MossNanoWebsocketTTSService(
        ws_url=config.local_tts_ws_url,
        http_session=http_session,
        language=session.locale,
        speaker_wav=config.local_tts_speaker_wav,
    )
