# aidm-stt — Faster-Whisper Large-v3 (private mode)

Phase 7 private-mode STT. FastAPI process wrapping a single
``faster_whisper.WhisperModel``. The orchestrator's
``stt_whisper_local`` provider POSTs a VAD-bounded utterance to
``/v1/transcribe`` and reads the final transcript back.

## Endpoints

- ``GET /health`` → ``{"ready": bool}``. The Docker healthcheck polls
  this; the orchestrator container blocks on it via
  ``depends_on.condition: service_healthy``.
- ``POST /v1/transcribe`` (multipart: ``audio`` WAV 16-bit mono 16 kHz,
  ``language`` optional, ``prompt`` optional priming text) →
  ``{"text", "language", "duration_ms"}``.

## Run locally (no container)

```
pip install faster-whisper==1.0.3 ctranslate2 fastapi 'uvicorn[standard]' python-multipart numpy
WHISPER_DOWNLOAD_ROOT=./models/whisper uvicorn server:app --host 0.0.0.0 --port 8001
```

## Env

- ``WHISPER_MODEL`` (default ``large-v3-turbo`` — distilled Large-v3,
  ~5× faster decoding. Override to ``large-v3`` only if a language
  regresses noticeably; we haven't seen it on Hungarian.)
- ``WHISPER_DEVICE`` (default ``cuda``; set ``cpu`` to evict from GPU
  if VRAM is contested by the other private services)
- ``WHISPER_COMPUTE_TYPE`` (default ``int8_float16``; try ``float16`` if
  quality matters more than VRAM)
- ``WHISPER_DOWNLOAD_ROOT`` (default ``/models``)

## Why no partial transcripts in Phase 7

Faster-Whisper supports streaming, but emitting Pipecat
``InterimTranscriptionFrame``s doubles the integration surface and
requires the STT service to know about chunk boundaries before VAD
closes. Phase 7 trades a few hundred ms of turn-start latency for a
much smaller blast radius. Revisit in Phase 7.5 if the latency budget
demands it.
