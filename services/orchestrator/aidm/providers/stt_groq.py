"""Groq Whisper STT — public-mode speech-to-text.

Groq runs Whisper Large-v3 (and the Turbo distilled variant) on their LPU
hardware, which gives near-realtime transcription at a fraction of native
Whisper's latency. ~150ms typical for a short utterance, Hungarian
quality is identical to OpenAI's Whisper since it's the same weights.

The base ``BaseWhisperSTTService`` does not stream partial transcripts in
the Pipecat 1.3.0 implementation — it buffers an utterance (VAD-bounded
upstream) and emits one ``TranscriptionFrame`` when the user stops
speaking. That's fine for our turn-based loop; we just need Silero VAD
upstream of this processor.
"""

from __future__ import annotations

from pipecat.services.groq.stt import GroqSTTService

from aidm.config import Config
from aidm.session import SessionContext

# Whisper's ``prompt`` field biases the decoder toward a register. We pick
# one per locale so the priming language matches the spoken language —
# a Hungarian primer on an English utterance would distort transcription.
# Unknown locales send no primer rather than a wrong-language one.
_PRIMING_PROMPTS: dict[str, str] = {
    "hu": "Magyar nyelvű DM beszélgetés, fantasy szerepjáték. Játékos és mesélő közötti párbeszéd.",
    "en": "Tabletop RPG session. Dialogue between a player and the dungeon master.",
}


def build_groq_stt(config: Config, session: SessionContext) -> GroqSTTService:
    return GroqSTTService(
        api_key=config.groq_api_key,
        settings=GroqSTTService.Settings(
            model=config.groq_stt_model,
            language=session.language,
            # temperature=0 forces deterministic decoding and meaningfully
            # reduces Whisper's silence-hallucination failure mode (e.g.
            # the "Köszönöm, hogy megnéztétek" YouTube-outro injection).
            temperature=0.0,
            prompt=_PRIMING_PROMPTS.get(session.locale, ""),
        ),
    )
