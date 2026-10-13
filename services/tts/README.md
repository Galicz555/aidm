# aidm-tts — MOSS-TTS-Nano (private mode)

Phase 7 private-mode TTS. CPU-hosted MOSS-TTS-Nano (0.1B parameters,
20 languages including Hungarian, voice clone via a reference clip).
The orchestrator's ``tts_moss_nano`` provider opens a WebSocket per
utterance and reads back 24 kHz mono PCM16 chunks.

## Why CPU

MOSS-TTS-Nano is small enough to be realtime on a 4-core CPU, and
keeping it off the GPU frees the full 16 GB VRAM for LLM + Whisper
co-residency. On a 5060 Ti 16 GB the only way to make a ~10 GB
Qwen3-14B Q5_K_M (plus its KV cache), Whisper Large-v3-Turbo and any
TTS coexist is to keep the TTS off the card.

## Endpoints

- ``GET /health`` → ``{"ready": bool}``.
- ``WS /v1/tts/stream``:
  - Client open-frame: ``{"text", "language", "speaker_wav"}``.
  - Server streams binary frames: raw 16-bit mono PCM at 24 kHz.
  - Server closes with ``{"event": "done", "synth_ms": int, "audio_ms": int}``
    (or ``{"event": "error", "error": str}`` on failure).

## Voice clone — quality dominates

MOSS-TTS-Nano conditions on a 6–30 s reference clip. Quality of that
clip dominates output naturalness. Default speaker path: ``/voices/dm_hu.wav``
(host-side ``./voices/dm_hu.wav`` mounted read-only into the container).

## Env

- ``MOSS_TTS_CHECKPOINT`` (default ``/models/moss-tts`` — the
  ``OpenMOSS-Team/MOSS-TTS-Nano-100M`` repo snapshot)
- ``MOSS_TTS_AUDIO_TOKENIZER`` (default ``/models/moss-audio-tokenizer``
  — the ``OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`` repo snapshot,
  shared across the MOSS-TTS family)
- ``MOSS_TTS_DEVICE`` (default ``cpu``; ``cuda`` if you do want to
  burn VRAM on the TTS)
- ``MOSS_TTS_DTYPE`` (default ``float32``; ``float16`` on GPU only)
- ``MOSS_TTS_DEFAULT_SPEAKER_WAV`` (default ``/voices/dm_hu.wav``)

## Sample-rate boundary

MOSS-TTS-Nano emits **48 kHz stereo**. The server downmixes to mono
and resamples to 24 kHz before streaming so ``aidm.audio.OUTPUT_SAMPLE_RATE_TTS``
stays the single source of truth for the transport boundary. If you
ever change the target rate, do it there and not here.
