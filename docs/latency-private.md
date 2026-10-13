# Private-mode latency notes

Measurement journal for Phase 7. Fill in once the stack is up and
running on the target hardware (RTX 5060 Ti 16GB, 64GB RAM).

Stack:
- STT: Faster-Whisper Large-v3-Turbo (GPU, int8_float16, ~1.5 GB VRAM)
- LLM: Qwen3-14B instruct GGUF Q5_K_M via llama.cpp — full offload, ~10 GB weights +
  ~2 GB q8_0 KV at 16k ctx, ~37 tok/s decode
- TTS: MOSS-TTS-Nano 0.1B (**CPU**, no VRAM cost)

## Target

- p50 first-byte (mic-stop → first TTS audio) **< 3.5 s**
- p95 first-byte **< 5 s**
- p99 first-byte **< 7 s** (rare; one full GC pause acceptable)

Public-mode baseline today on the same machine sits at ~1.4 s p50 / 2.0
s p95 (Groq STT + Claude 4.5 + ElevenLabs ws). Private mode trades
about 2 s for the privacy + filter guarantees. Prefix/KV caching does
most of the work in text mode (turn-start prefill measured 1825 → 39
tokens once the volatile blocks moved to the prompt tail); Turbo STT
covers the voice leg.

## How to measure

The orchestrator already emits per-turn latency via `attach_turn_logging`
in `pipeline.py`. Run a session through:

```
AIDM_MODE=private AIDM_TRANSPORT=local AIDM_LOG_LEVEL=DEBUG \
  uv run aidm 2> private-latency.log
```

…then grep for `turn` entries and break out p50/p95/p99. Aim for 10
consecutive turns of normal RP play (mix of NPC questions, location
descriptions, character updates).

## Results

| Date | Hardware | Whisper compute | LLM offload | Notes | p50 | p95 |
| ---- | -------- | --------------- | ----------- | ----- | --- | --- |
| TBD  | 5060 Ti 16GB | int8_float16 (Turbo) | 99 (full offload) | initial bring-up | — | — |

## Known failure modes

- **CUDA OOM on the LLM container** mid-session — the KV cache grows
  with ctx-size. Lower `AIDM_LLM_GPU_LAYERS` (or `auto`) or
  `AIDM_LLM_CTX_SIZE`; a desktop game holding VRAM has caused this live.
- **First turn slow (>10 s)** — model load + speaker-latent precompute.
  Expected; the Docker healthcheck blocks the orchestrator on it.
- **MOSS-TTS-Nano clip degradation on Hungarian sibilants** — wrong
  reference clip. Re-record `voices/dm_hu.wav` from a clean source
  and restart the `tts` container.
- **CPU TTS first-byte spike** — if the TTS container is starved by
  another process, MOSS-TTS-Nano's first-chunk latency stretches. The
  container has no CPU pinning today; consider `cpuset` in compose
  if a noisy neighbor on the host becomes a problem.
