# Latency baseline (public mode)

The `UserBotLatencyObserver` measures end-to-end mic→speaker latency per
turn and logs it. The goal is **end-to-end under ~2 seconds** for the
"humanlike" feel.

Phase 5 pipeline (local CLI, `AIDM_TRANSPORT=local`): Groq Whisper Turbo
or ElevenLabs Scribe v2 → OpenRouter (Claude Sonnet 4.5) → ElevenLabs
Multilingual v2.

Phase 6 (browser via WebRTC, `AIDM_TRANSPORT=webrtc`) gets its own
measurement to confirm no regression versus the
local-audio baseline below. Private mode (Phase 7) targets ≤3.5s round-
trip on RTX 5060 Ti 16GB and gets its own measurements at
`latency-private.md` when it lands.

## How to measure

1. Run `uv run aidm`.
2. Speak a short, complete sentence (`"Szia, kezdjük el a játékot."`).
3. The observer logs one line per turn:
   ```
   12:34:56.789 INFO    [ab12cd34/t3] turn latency: 1240ms (mic→speaker)
   12:34:56.790 INFO    [ab12cd34/t3] latency breakdown: stt=420ms, llm=680ms, tts=140ms
   ```
4. Capture three turns of natural play. Paste the lines below.

## Results

| Turn | session | mic→speaker | breakdown                |
| ---- | ------- | ----------- | ------------------------ |
| 1    | TODO    | TODO        | TODO                     |
| 2    | TODO    | TODO        | TODO                     |
| 3    | TODO    | TODO        | TODO                     |

## Notes

- The observer's "latency" counts from `VADUserStoppedSpeakingFrame` to
  the first `BotStartedSpeakingFrame`. It does not include speaker buffer
  delay on the OS side, which is usually <50ms.
- The `turn latency` / `latency breakdown` lines log at `INFO`, so any
  `AIDM_LOG_LEVEL` shows them. (`DEBUG` adds turn narrative; `TRACE` adds
  prompt dumps + per-chunk + wire logs — see `aidm/log.py`.)
- If the total stays under 1s, treat that as a latency win. If it's
  over 3s consistently in public mode, something is wrong — investigate
  the slowest leg in the breakdown before iterating further.
