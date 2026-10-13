"""Audio format constants for the voice loop.

Single source of truth for sample rates and channel layout across every
transport boundary (local PyAudio, WebRTC, future WebSocket). Pipecat
will resample where needed; these constants make sure each side of the
pipeline agrees on the target so resampling stays at the edges instead
of happening twice.

* ``INPUT_SAMPLE_RATE`` — STT input. 16 kHz is the Nyquist-safe rate for
  speech (energy sits below 8 kHz). Every STT we use (Groq Whisper,
  ElevenLabs Scribe, Faster-Whisper) accepts 16 kHz natively; higher
  rates just add bytes.
* ``OUTPUT_SAMPLE_RATE_TTS`` — TTS provider's native output rate.
  ElevenLabs PCM streams emit 24 kHz on the default ``pcm_24000`` profile.
  Coqui XTTS-v2 (Phase 7) emits 24 kHz too, so the constant covers both
  backends without per-mode resampling at the provider seam.
* ``OUTPUT_SAMPLE_RATE_TRANSPORT`` — rate the transport advertises to the
  pipeline. Forced to 48 kHz on the WebRTC path because aiortc's opus
  encoder produces silent/garbage RTP when fed AudioFrames at any other
  rate (opus's RTP clockrate is fixed at 48 kHz; the encoder's Python
  wrapper does not auto-resample its input). Pipecat's
  ``BaseOutputTransport.handle_audio_frame`` resamples the provider's
  24 kHz frames up to 48 kHz before queuing them, so the provider stays
  on its native rate and only the transport seam carries the upsample.
  Local PyAudio is happy with either rate; we keep them equal there.
* ``CHANNELS`` — mono. Stereo gains nothing for spoken dialogue and
  doubles bandwidth on the WebRTC leg.
"""

from __future__ import annotations

INPUT_SAMPLE_RATE: int = 16000
OUTPUT_SAMPLE_RATE_TTS: int = 24000
OUTPUT_SAMPLE_RATE_TRANSPORT: int = 48000
CHANNELS: int = 1
