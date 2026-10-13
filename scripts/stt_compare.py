"""A/B compare Groq Whisper vs ElevenLabs Scribe on the same 5s capture."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import wave
from pathlib import Path

import httpx
import pyaudio
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

RATE = 16000
CHANNELS = 1
SECONDS = 5
CHUNK = 1024


def record() -> bytes:
    pa = pyaudio.PyAudio()
    device = os.environ.get("AUDIO_INPUT_DEVICE")
    device_index = int(device) if device else None
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_index,
    )
    print(f"recording {SECONDS}s — beszélj most...", flush=True)
    frames = []
    for _ in range(int(RATE / CHUNK * SECONDS)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    pa.terminate()
    return b"".join(frames)


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


async def groq_v3(wav: bytes) -> str:
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    r = await client.audio.transcriptions.create(
        file=("audio.wav", wav, "audio/wav"),
        model="whisper-large-v3",
        language="hu",
        temperature=0.0,
        response_format="json",
    )
    return r.text


async def elevenlabs_scribe(wav: bytes, model: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            data={"model_id": model, "language_code": "hu"},
            files={"file": ("audio.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        return r.json().get("text", "<no text>")


async def main() -> int:
    pcm = record()
    wav = wav_bytes(pcm)
    print(f"captured {len(pcm)} bytes pcm, {len(wav)} bytes wav\n")

    results: list[tuple[str, str]] = []
    try:
        results.append(("groq whisper-large-v3 (t=0)", await groq_v3(wav)))
    except Exception as e:
        results.append(("groq whisper-large-v3 (t=0)", f"ERROR: {e}"))

    for model in ("scribe_v1", "scribe_v2"):
        try:
            results.append((f"elevenlabs {model}", await elevenlabs_scribe(wav, model)))
        except Exception as e:
            results.append((f"elevenlabs {model}", f"ERROR: {e}"))

    print()
    for label, text in results:
        print(f"{label:35s} → {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
