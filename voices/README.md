# voices/ — XTTS-v2 reference clips

Drop voice clone references here (`dm_hu.wav`, `dm_en.wav`, …). 6–30 s
of clean mono audio per file. Quality of the clip dominates XTTS
output naturalness more than any other knob.

The Docker compose `tts` service mounts this directory read-only at
`/voices`. The default speaker the orchestrator passes is
`/voices/dm_hu.wav` (override via `AIDM_LOCAL_TTS_SPEAKER`).

Clips themselves are gitignored — they're large and often personal.
