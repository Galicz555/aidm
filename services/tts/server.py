"""MOSS-TTS-Nano streaming server (private-mode TTS, CPU).

A FastAPI process around OpenMOSS's ``NanoTTSService`` — the 0.1B
multilingual TTS model that runs comfortably on a CPU and supports
Hungarian natively. We keep the same WebSocket contract the
orchestrator already speaks against XTTS-v2 so the swap is a
URL + Dockerfile change, not an orchestrator-side refactor.

Per request: client sends one JSON open-frame
``{text, language, speaker_wav}``, server streams binary 16-bit mono
PCM at 24 kHz, then closes the request with ``{"event": "done"}``.

MOSS-TTS-Nano emits **48 kHz stereo** natively; we downmix to mono and
resample to 24 kHz on the way out so the orchestrator's
``aidm.audio.OUTPUT_SAMPLE_RATE_TTS`` constant stays the single source
of truth for the transport boundary.

Event-driven by nature — Code as Story does not apply.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import torchaudio.functional as AF
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from moss_tts_nano_runtime import NanoTTSService

logger = logging.getLogger("aidm.tts")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# Orchestrator-side audio boundary: 24 kHz mono. MOSS-TTS-Nano emits
# 48 kHz stereo, so we downsample + downmix here. Pipecat would also
# resample on the orchestrator side, but doing it once at the source
# saves a round of conversion and keeps the WS frames small.
OUTPUT_SAMPLE_RATE: int = 24000

# Number of PCM samples per binary websocket message after resampling.
# ~40 ms at 24 kHz. Smaller chunks reduce TTFB but stack WS overhead.
WS_CHUNK_SAMPLES: int = 960

_state: dict[str, Any] = {"service": None, "ready": False}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    checkpoint = os.environ.get("MOSS_TTS_CHECKPOINT", "/models/moss-tts")
    audio_tokenizer = os.environ.get(
        "MOSS_TTS_AUDIO_TOKENIZER", "/models/moss-audio-tokenizer"
    )
    device = os.environ.get("MOSS_TTS_DEVICE", "cpu")
    dtype = os.environ.get("MOSS_TTS_DTYPE", "float32")

    logger.info(
        "loading MOSS-TTS-Nano checkpoint=%s audio_tokenizer=%s device=%s dtype=%s",
        checkpoint,
        audio_tokenizer,
        device,
        dtype,
    )
    service = NanoTTSService(
        checkpoint_path=checkpoint,
        audio_tokenizer_path=audio_tokenizer,
        device=device,
        dtype=dtype,
    )
    # Warm-load: instantiating loads weights but the first inference
    # still pays a JIT cost. `preload(load_model=True)` ticks both.
    service.preload(load_model=True)
    logger.info("MOSS-TTS-Nano warm")

    _state["service"] = service
    _state["ready"] = True
    try:
        yield
    finally:
        _state["ready"] = False
        _state["service"] = None


app = FastAPI(title="aidm-tts (MOSS-TTS-Nano)", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ready": _state["ready"]})


@app.websocket("/v1/tts/stream")
async def stream(ws: WebSocket) -> None:
    """Streaming WS endpoint.

    Same protocol as the previous XTTS-v2 server. The connection
    stays open for further requests — the orchestrator opens one
    WS per call today but the contract supports session reuse.
    """
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            await _handle_one(ws, payload)
    except WebSocketDisconnect:
        logger.info("ws client disconnected")
    except Exception as exc:
        logger.exception("ws handler crashed: %s", exc)
        await _safe_close(ws, code=1011)


async def _handle_one(ws: WebSocket, payload: dict[str, Any]) -> None:
    if not _state["ready"]:
        await ws.send_json({"event": "error", "error": "model not ready"})
        return

    text = (payload.get("text") or "").strip()
    if not text:
        await ws.send_json({"event": "done", "synth_ms": 0, "audio_ms": 0})
        return

    speaker_wav = payload.get("speaker_wav") or os.environ.get(
        "MOSS_TTS_DEFAULT_SPEAKER_WAV", "/voices/dm_hu.wav"
    )

    started = time.perf_counter()
    audio_ms = 0
    try:
        async for pcm_bytes, chunk_audio_ms in _synth_stream(text, speaker_wav):
            await ws.send_bytes(pcm_bytes)
            audio_ms += chunk_audio_ms
    except Exception as exc:
        logger.exception("synth failed: %s", exc)
        await ws.send_json({"event": "error", "error": str(exc)})
        return

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    await ws.send_json({"event": "done", "synth_ms": elapsed_ms, "audio_ms": audio_ms})
    logger.info(
        "synth chars=%d synth_ms=%d audio_ms=%d", len(text), elapsed_ms, audio_ms
    )


async def _synth_stream(text: str, speaker_wav: str):
    """Async wrapper around the model's blocking stream generator.

    ``NanoTTSService.synthesize_stream`` is a synchronous generator —
    we hop each iteration to a worker thread so the FastAPI event
    loop stays responsive (other WS connections, healthcheck, etc.).
    """
    loop = asyncio.get_running_loop()
    service: NanoTTSService = _state["service"]

    def _open():
        return service.synthesize_stream(
            text=text,
            mode="voice_clone",
            prompt_audio_path=speaker_wav,
        )

    iterator = await loop.run_in_executor(None, _open)
    while True:
        event = await loop.run_in_executor(None, _next_or_none, iterator)
        if event is None:
            return
        if event.get("type") != "audio":
            continue
        waveform = event["waveform"]  # torch.Tensor on CPU, float32
        in_sr = int(event["sample_rate"])
        pcm_bytes, chunk_audio_ms = _to_pcm16_mono_24k(waveform, in_sr)
        # Slice into WS-sized chunks so the client can start playback
        # while later samples are still being encoded.
        for chunk in _slice_bytes(pcm_bytes, WS_CHUNK_SAMPLES * 2):
            slice_ms = int(len(chunk) / 2 / OUTPUT_SAMPLE_RATE * 1000)
            yield chunk, slice_ms
            chunk_audio_ms -= slice_ms


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def _to_pcm16_mono_24k(waveform: torch.Tensor, in_sr: int) -> tuple[bytes, int]:
    """Downmix stereo→mono, resample to 24 kHz, pack as PCM16 bytes."""
    if waveform.dim() == 2:
        # Shape [channels, samples] — average across channels.
        mono = waveform.mean(dim=0)
    elif waveform.dim() == 1:
        mono = waveform
    else:
        # Unexpected shape; flatten and hope. Logged loudly so we notice.
        logger.warning("unexpected waveform dim=%s, flattening", waveform.dim())
        mono = waveform.reshape(-1)
    if in_sr != OUTPUT_SAMPLE_RATE:
        mono = AF.resample(mono.unsqueeze(0), in_sr, OUTPUT_SAMPLE_RATE).squeeze(0)
    pcm = torch.clamp(mono * 32767.0, -32768.0, 32767.0).to(torch.int16)
    audio_ms = int(pcm.numel() / OUTPUT_SAMPLE_RATE * 1000)
    return pcm.numpy().tobytes(), audio_ms


def _slice_bytes(buf: bytes, size: int):
    if size <= 0:
        yield buf
        return
    view = memoryview(buf)
    for offset in range(0, len(view), size):
        yield bytes(view[offset : offset + size])


async def _safe_close(ws: WebSocket, *, code: int) -> None:
    try:
        await ws.close(code=code)
    except Exception:
        pass


# Silence numpy unused-import warning when type checkers run against
# this file — np is used implicitly via torch.Tensor.numpy().
_ = np
