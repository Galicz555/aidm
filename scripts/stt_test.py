"""Standalone STT bench — record 5s from mic, transcribe with Groq, print.

Skips ElevenLabs / OpenRouter entirely, so iteration burns no TTS/LLM
credits. Tweak the constants below to compare models and prompts.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pyaudio
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

MODELS = ["whisper-large-v3"]
PROMPTS = [
    None,
    "Magyar nyelvű DM beszélgetés, fantasy szerepjáték. Játékos és mesélő közötti párbeszéd.",
]
TEMPERATURES = [None, 0.0]
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
    import wave
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


async def transcribe(
    client: AsyncGroq,
    wav: bytes,
    model: str,
    prompt: str | None,
    temperature: float | None,
) -> str:
    kwargs = {
        "file": ("audio.wav", wav, "audio/wav"),
        "model": model,
        "language": "hu",
        "response_format": "json",
    }
    if prompt:
        kwargs["prompt"] = prompt
    if temperature is not None:
        kwargs["temperature"] = temperature
    r = await client.audio.transcriptions.create(**kwargs)
    return r.text


async def main() -> int:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set", file=sys.stderr)
        return 1

    pcm = record()
    wav = wav_bytes(pcm)
    print(f"captured {len(pcm)} bytes pcm, {len(wav)} bytes wav\n")

    client = AsyncGroq(api_key=api_key)
    for model in MODELS:
        for prompt in PROMPTS:
            for temp in TEMPERATURES:
                label = (
                    f"{model:24s} "
                    f"prompt={'yes' if prompt else 'no '} "
                    f"temp={temp if temp is not None else 'def'}"
                )
                try:
                    text = await transcribe(client, wav, model, prompt, temp)
                    print(f"{label} → {text!r}")
                except Exception as e:
                    print(f"{label} → ERROR: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
