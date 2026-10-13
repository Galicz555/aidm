# orchestrator

The voice loop. Owns the Pipecat pipeline (VAD, audio I/O, turn-taking, barge-in), holds session state, carries the DM persona system prompt, gives the LLM filesystem access to the mounted vault, and (under `AIDM_TRANSPORT=webrtc`) hosts the FastAPI server browser clients connect to.

This service is **event-driven**, not pipeline-shaped — Code as Story does not apply to the loop itself, only to its data-preparation helpers (vault navigation, slug sanitization, log writes).

## Architecture

Mode-dispatched provider chain (same graph for both transports):

```
transport.input() → Silero VAD → STT → forwarder.user → ctx.user → LLM → forwarder.bot → TTS → transport.output()
                                                                    ↘     ↗
                                                                     vault tools
```

`AIDM_MODE` selects providers at startup:

| Slot | `public` (default) | `private` |
|------|--------------------|-----------|
| STT  | Groq Whisper Large-v3 Turbo or ElevenLabs Scribe v2 (`STT_PROVIDER`) | Faster-Whisper Large-v3, local |
| LLM  | OpenRouter → Google Gemini 2.5 Flash (default; swap via `OPENROUTER_MODEL`) | llama.cpp → Qwen3-14B (Q5_K_M, 16k ctx) |
| TTS  | ElevenLabs Multilingual v2 / v3 alpha | MOSS-TTS-Nano, local CPU |

The factories in `aidm/providers/` are the only place that knows which backend lives in each slot.

`AIDM_TRANSPORT` selects the input/output side:

| Value | Behavior |
|-------|----------|
| `local` (default) | PyAudio mic + speakers. Single-session CLI loop. |
| `webrtc` | FastAPI server hosting `SmallWebRTCTransport`. Browser clients connect via `POST /api/offer`; data channel carries transcript / scene / character side-channel events. See `aidm/server.py` and `docs/session-lifecycle.md`. |

## Run it

```sh
# one-time
cp .env.example .env
$EDITOR .env       # set GROQ_API_KEY, OPENROUTER_API_KEY, ELEVENLABS_API_KEY

# install (run once, or after dep changes)
make sync

# local CLI loop
uv run aidm

# browser mode (orchestrator + Vite dev server in one terminal)
make start-public-voice
# variants: start-public-voice-debug, start-private-text, start-private-text-debug
```

Speak Hungarian. `Ctrl+C` to stop. The DM persona will introduce itself and ask about your character if you don't open with one.

Per-session and per-turn IDs are attached to every log line via loguru contextvars (`[session/tN]`). Per-turn mic→speaker latency is logged automatically by `UserBotLatencyObserver`.

## Environment knobs

Required (public mode):
- `GROQ_API_KEY` — from [console.groq.com](https://console.groq.com). Free tier covers POC usage.
- `OPENROUTER_API_KEY` — from [openrouter.ai/keys](https://openrouter.ai/keys). $5 credit covers many hours.
- `ELEVENLABS_API_KEY` — from [elevenlabs.io](https://elevenlabs.io).

Optional:
- `AIDM_MODE` — `public` (default) or `private` (Faster-Whisper + llama.cpp/Qwen3-14B + MOSS-TTS-Nano via the `private` compose profile).
- `AIDM_TRANSPORT` — `local` (default, PyAudio CLI) or `webrtc` (FastAPI + browser).
- `AIDM_SERVER_HOST` / `AIDM_SERVER_PORT` — bind address for `AIDM_TRANSPORT=webrtc` (default `127.0.0.1:8000`).
- `AIDM_LOCALE` — `hu` (default) or `en`. Acts as the seed for `SessionContext.locale`; the browser client can override per-session via the start-form dropdown.
- `AIDM_LOG_LEVEL` — `TRACE`/`DEBUG`/`INFO`/`WARNING` (default `INFO`). `INFO` = turn anchors only; `DEBUG` = + turn narrative (no prompt dumps, no per-chunk emits, no HTTP wire logs); `TRACE` = the firehose (full context dumps, per-chunk splitter/sink emits, httpx/httpcore/openai wire logs). `make start-private-text-debug` runs at `DEBUG`; prefix with `AIDM_LOG_LEVEL=TRACE` to restore the firehose.
- `AIDM_PROMPT_FILE` — override the locale-based persona lookup with an absolute path.
- `AIDM_LLM_TEMPERATURE` / `AIDM_LLM_TOP_P` / `AIDM_LLM_MIN_P` / `AIDM_LLM_FREQUENCY_PENALTY` / `AIDM_LLM_PRESENCE_PENALTY` — private-mode sampling, sent per request by `llm_local` (no gateway rebuild). Model-dependent: instruct ~0.35 temp, creative finetunes ~0.65. `min_p` is llama.cpp-only, routed via `extra_body`.
- Gateway-side knobs (`AIDM_LLM_MODEL_PATH`, `AIDM_LLM_DRY_*`, `AIDM_LLM_CACHE_REUSE`, thinking suppression, …) are compose passthroughs — see `services/llm-gateway/README.md`.
- `STT_PROVIDER` — `elevenlabs` (default, Scribe v2) or `groq` (Whisper Large-v3 Turbo).
- `GROQ_STT_MODEL` — `whisper-large-v3-turbo` (default) or `whisper-large-v3`.
- `ELEVENLABS_STT_MODEL` — `scribe_v2` (default).
- `OPENROUTER_MODEL` — `google/gemini-2.5-flash` (current deployment). Repo-default in `config.py` is `anthropic/claude-sonnet-4.5`; override in `.env`. Try `anthropic/claude-sonnet-4.6`, `openai/gpt-5`, `deepseek/deepseek-chat`, etc. — full list at [openrouter.ai/models](https://openrouter.ai/models).
- `ELEVENLABS_VOICE_ID` — default voice, used when no per-locale override is set.
- `ELEVENLABS_VOICE_ID_HU` / `ELEVENLABS_VOICE_ID_EN` — optional per-locale voice overrides. Multilingual v2 keeps voice identity across languages so leaving these blank is the common case; set them when you want a distinct voice per language.
- `ELEVENLABS_MODEL` — `eleven_multilingual_v2` (default), `eleven_turbo_v2_5` for ~150ms-faster first byte, or `eleven_v3` for inline emotion tags (alpha, tier-gated).
- `ELEVENLABS_TTS_TRANSPORT` — `ws` (default, low-latency WebSocket; v2/flash/turbo) or `http` (chunked-streaming REST, needed for `eleven_v3`).
- `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE` — PyAudio device indices (see `make audio-devices`). Only used by `AIDM_TRANSPORT=local`. On PipeWire systems the `pipewire` device transparently handles BT-headset routing.
- `AIDM_VAULT_ROOT` — absolute path to the directory holding every campaign vault; default derives from `VAULT_PATH.parent` (i.e. the repo's `vault/`). The web picker enumerates this directory.
- `VAULT_PATH` — absolute path to the *default* campaign vault under the vault root; used by the local CLI (no picker) and as the web picker's initial selection.
- `RUN_SLUG` — default run slug for sessions that omit one (local CLI; legacy clients). The web picker always sets `run_id` per session in the `/api/offer` requestData.

## Persona prompts

Persona lives in `prompts/dm.<locale>.md` and is read at runtime — edit and restart, no rebuild. A/B-test alternates with `AIDM_PROMPT_FILE=/path/to/draft.md uv run aidm`.

When `config.tts_supports_emotion_tags` is true (currently: `ELEVENLABS_MODEL=eleven_v3`), the persona gets a small trailer instructing it to emit `[whispers]` / `[excited]` / etc. inline. Otherwise the prompt is clean — the model never learns about tags the TTS can't speak.

## Audio device routing

`make audio-devices` lists PyAudio devices with their indices. On Linux with PipeWire:

- For Bluetooth headset routing, set both `AUDIO_INPUT_DEVICE` and `AUDIO_OUTPUT_DEVICE` to the `pipewire` device index. PipeWire then handles the BT codec and resampling transparently.
- Direct ALSA capture of BT hardware at 16 kHz fails with `paInvalidSampleRate` — always go through PipeWire.
- If the headset disconnects, `make audio-devices` will renumber and the PyAudio open will fail with `Errno -9999 Unanticipated host error`. Reconnect the headset and re-check the index.

`make mic-test` records 3 seconds from the configured input device and prints the peak level — fast sanity check that PyAudio actually captures what you think it does.

## Container

`make build` builds the orchestrator image via Compose. Local voice loop continues to use `uv run aidm` because audio passthrough into the container needs PipeWire socket mounting (platform-fiddly, not worth automating yet). Private-mode services live under the `private` compose profile as `stt` (Faster-Whisper), `llm-gateway` (llama.cpp + Qwen3-14B Q5_K_M), and `tts` (MOSS-TTS-Nano); production adds nginx + certbot via `docker-compose.prod.yml`. Browser-mode dev uses `make start-public-voice` (orchestrator + Vite dev server in parallel).
