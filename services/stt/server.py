"""Faster-Whisper Large-v3 transcription server (private-mode STT).

A thin FastAPI process around a single ``WhisperModel`` instance. The
orchestrator POSTs a complete utterance (WAV bytes, VAD-bounded upstream)
and receives the final transcript in one response. No partial streaming
in Phase 7. The endpoint shape mirrors what
``aidm/providers/stt_whisper_local.py`` expects, so the two halves can
move in lockstep.

Model load is eager and a 1-second silence warm-up runs at startup, so
the first real utterance does not pay the model-init cost. The Docker
healthcheck blocks the orchestrator container until ``/health`` reports
``ready: true``.

Event-driven by nature — Code as Story does not apply.
"""

from __future__ import annotations

import io
import logging
import os
import time
import wave
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

logger = logging.getLogger("aidm.stt")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# Faster-Whisper accepts any sample rate but resamples internally. The
# orchestrator pins 16 kHz at the transport boundary (aidm/audio.py), so
# the WAVs we receive are already at the target rate — declaring it here
# lets us skip the resample step and warm-up with the right shape.
SAMPLE_RATE: int = 16000


_state: dict[str, Any] = {"model": None, "ready": False}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ``large-v3-turbo`` is a distilled Large-v3 with 4 decoder layers
    # instead of 32 — ~5× faster decode, near-identical Hungarian WER.
    # Override to ``large-v3`` for the rare languages where Turbo
    # regresses (none we've measured on Hungarian).
    model_name = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
    download_root = os.environ.get("WHISPER_DOWNLOAD_ROOT", "/models")

    logger.info(
        "loading Whisper model=%s device=%s compute_type=%s root=%s",
        model_name,
        device,
        compute_type,
        download_root,
    )
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )

    # Warm-up: feed one second of silence through the decoder so the
    # first real request does not pay JIT / CUDA-context init.
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    segments, _ = model.transcribe(silence, language="hu", beam_size=1)
    _ = list(segments)
    logger.info("Whisper warm-up complete")

    _state["model"] = model
    _state["ready"] = True
    try:
        yield
    finally:
        _state["ready"] = False
        _state["model"] = None


app = FastAPI(title="aidm-stt (Faster-Whisper)", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ready": _state["ready"]})


@app.post("/v1/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("hu"),
    prompt: str = Form(""),
) -> JSONResponse:
    """Transcribe a WAV-encoded utterance.

    Returns ``{"text", "language", "duration_ms"}``. Empty audio yields
    an empty string rather than 4xx — the orchestrator already discards
    empty TranscriptionFrames, so we match its expectations.
    """
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="model not ready")

    raw = await audio.read()
    samples = _wav_to_float32(raw)
    if samples.size == 0:
        return JSONResponse({"text": "", "language": language, "duration_ms": 0})

    started = time.perf_counter()
    # beam_size=5 matches Faster-Whisper's default; vad_filter=False
    # because Silero VAD already bounded the utterance upstream — a
    # second pass would only chop short syllables.
    segments_iter, info = _state["model"].transcribe(
        samples,
        language=language or None,
        beam_size=5,
        vad_filter=False,
        initial_prompt=prompt or None,
        temperature=0.0,
    )
    text = "".join(seg.text for seg in segments_iter).strip()
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "transcribed lang=%s dur=%dms chars=%d audio_ms=%d",
        info.language,
        elapsed_ms,
        len(text),
        int(info.duration * 1000) if info.duration else 0,
    )
    return JSONResponse(
        {"text": text, "language": info.language or language, "duration_ms": elapsed_ms}
    )


def _wav_to_float32(raw: bytes) -> np.ndarray:
    """Decode a WAV blob to float32 mono in [-1, 1].

    The orchestrator's ``SegmentedSTTService`` produces 16-bit PCM mono
    WAVs at ``self.sample_rate`` (16 kHz here). Anything else (8-bit,
    stereo, foreign sample rate) gets rejected — we'd rather see a
    clear error than silently mistranscribe.
    """
    if not raw:
        return np.zeros(0, dtype=np.float32)
    with wave.open(io.BytesIO(raw), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise HTTPException(status_code=400, detail="expected 16-bit PCM WAV")
        if wav.getnchannels() != 1:
            raise HTTPException(status_code=400, detail="expected mono audio")
        if wav.getframerate() != SAMPLE_RATE:
            raise HTTPException(
                status_code=400,
                detail=f"expected {SAMPLE_RATE} Hz, got {wav.getframerate()} Hz",
            )
        frames = wav.readframes(wav.getnframes())
    pcm16 = np.frombuffer(frames, dtype=np.int16)
    return (pcm16.astype(np.float32) / 32768.0).copy()
