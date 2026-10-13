"""Faster-Whisper STT client — private-mode speech-to-text.

The remote ``services/stt`` container hosts Faster-Whisper Large-v3
behind a single ``POST /v1/transcribe`` endpoint. The orchestrator
sends a VAD-bounded WAV blob and reads the final transcript back.

We inherit from Pipecat's ``SegmentedSTTService`` to reuse its VAD
plumbing (start-speaking / stop-speaking → WAV buffer assembly) — the
same base ``BaseWhisperSTTService`` uses for Groq's hosted Whisper.
Only ``run_stt`` differs: we POST the WAV instead of calling the OpenAI
SDK against Whisper.

No partial transcripts in Phase 7 — see PLAN.md "Risks". The remote
service is request/response; revisit when latency demands streaming.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from aidm.config import Config
from aidm.session import SessionContext

_LOCALE_PRIMING_PROMPTS: dict[str, str] = {
    "hu": "Magyar nyelvű DM beszélgetés, fantasy szerepjáték. Játékos és mesélő közötti párbeszéd.",
    "en": "Tabletop RPG session. Dialogue between a player and the dungeon master.",
}


class WhisperLocalSTTService(SegmentedSTTService):
    """Client for the local Faster-Whisper FastAPI service.

    ``SegmentedSTTService`` handles VAD-segmented audio buffering and
    WAV framing; we only own the network call and the response parse.
    """

    def __init__(
        self,
        *,
        base_url: str,
        http_session: aiohttp.ClientSession,
        language: Language | None,
        prompt: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._http = http_session
        self._language = language
        self._prompt = prompt or ""

    # mypy treats Pipecat's base ``async def run_stt -> AsyncGenerator`` as
    # ``Coroutine[Any, Any, AsyncGenerator]`` while our ``yield``-bodied
    # override is correctly typed as an async generator function. The
    # signatures match at runtime; the ignore is the same shape Pipecat's
    # own first-party services rely on.
    async def run_stt(  # type: ignore[override]
        self, audio: bytes
    ) -> AsyncGenerator[Frame | None, None]:
        if not audio:
            return

        form = aiohttp.FormData()
        form.add_field("audio", audio, filename="utt.wav", content_type="audio/wav")
        if self._language is not None:
            form.add_field("language", _language_code(self._language))
        if self._prompt:
            form.add_field("prompt", self._prompt)

        try:
            async with self._http.post(
                f"{self._base_url}/v1/transcribe",
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    yield ErrorFrame(error=f"whisper-local {resp.status}: {body[:200]}")
                    return
                payload = await resp.json()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning("whisper-local request failed: {}", exc)
            yield ErrorFrame(error=f"whisper-local unreachable: {exc}")
            return

        text = (payload.get("text") or "").strip()
        if not text:
            logger.debug("whisper-local returned empty transcript")
            return

        logger.debug("whisper-local transcription: [{}]", text)
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601())


def _language_code(language: Language) -> str:
    """Map pipecat ``Language`` to the two-letter code Whisper expects."""
    value = getattr(language, "value", None)
    if isinstance(value, str) and value:
        return value.split("-")[0].lower()
    return str(language).lower()


def build_whisper_local_stt(
    config: Config,
    session: SessionContext,
    *,
    http_session: aiohttp.ClientSession,
) -> WhisperLocalSTTService:
    return WhisperLocalSTTService(
        base_url=config.local_stt_base_url,
        http_session=http_session,
        language=session.language,
        prompt=_LOCALE_PRIMING_PROMPTS.get(session.locale, ""),
    )
