"""List local audio devices visible to PyAudio.

Run with `make audio-devices` or `uv run python -m aidm.devices`.
"""

from __future__ import annotations

import pyaudio


def main() -> None:
    pa = pyaudio.PyAudio()
    try:
        try:
            default_in = pa.get_default_input_device_info()
            print(f"Default INPUT : [{default_in['index']}] {default_in['name']}")
        except OSError:
            print("Default INPUT : <none>")
        try:
            default_out = pa.get_default_output_device_info()
            print(f"Default OUTPUT: [{default_out['index']}] {default_out['name']}")
        except OSError:
            print("Default OUTPUT: <none>")
        print()
        print(f"{'idx':>4}  {'name':60s}  in   out  rate")
        print("-" * 90)
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            print(
                f"{i:>4}  {d['name'][:60]:60s}  "
                f"{int(d['maxInputChannels']):>2}   {int(d['maxOutputChannels']):>2}   "
                f"{int(d['defaultSampleRate'])}"
            )
        print()
        print("Set AUDIO_INPUT_DEVICE and AUDIO_OUTPUT_DEVICE in .env to the indices you want.")
        print("On PipeWire systems, the 'pipewire' device follows your system audio settings.")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
