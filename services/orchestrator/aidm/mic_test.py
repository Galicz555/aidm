"""Direct PyAudio capture test — bypasses Pipecat entirely.

Records 3 seconds from the configured AUDIO_INPUT_DEVICE and reports
the average sample amplitude. If amplitude stays at zero while you talk,
PyAudio is not actually receiving audio from the device, regardless of
what the system audio routing says.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import pyaudio
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

DEVICE = (
    int(os.environ.get("AUDIO_INPUT_DEVICE", "-1"))
    if os.environ.get("AUDIO_INPUT_DEVICE")
    else None
)
RATE = 48000
CHANNELS = 1
SECONDS = 3
CHUNK = 1024


def main() -> None:
    pa = pyaudio.PyAudio()
    try:
        if DEVICE is not None:
            info = pa.get_device_info_by_index(DEVICE)
            print(f"Opening device [{DEVICE}] {info['name']} at {RATE} Hz")
        else:
            info = pa.get_default_input_device_info()
            print(f"Opening default device [{info['index']}] {info['name']} at {RATE} Hz")

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=DEVICE,
            frames_per_buffer=CHUNK,
        )
        print(f"Speak for {SECONDS} seconds...")
        start = time.time()
        max_amp = 0
        total_chunks = 0
        loud_chunks = 0
        while time.time() - start < SECONDS:
            data = stream.read(CHUNK, exception_on_overflow=False)
            total_chunks += 1
            samples = struct.unpack(f"{len(data) // 2}h", data)
            chunk_max = max(abs(s) for s in samples)
            if chunk_max > 500:
                loud_chunks += 1
            max_amp = max(max_amp, chunk_max)
        stream.stop_stream()
        stream.close()

        print()
        print(f"Captured {total_chunks} chunks.")
        print(f"Peak amplitude: {max_amp} / 32767  ({max_amp / 32767 * 100:.1f}%)")
        print(f"'Loud' chunks (>500 amp): {loud_chunks} / {total_chunks}")
        print()
        if max_amp < 100:
            print("VERDICT: silence. PyAudio is not capturing real audio from this device.")
        elif max_amp < 2000:
            print("VERDICT: very quiet. Mic gain low or you weren't speaking loud enough.")
        else:
            print("VERDICT: PyAudio captures audio fine on this device.")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
