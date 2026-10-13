# aidm-llm-gateway — llama.cpp + Qwen3 14B (private mode)

Private-mode LLM. Runs ``llama-server`` (from
``ghcr.io/ggml-org/llama.cpp:server-cuda``) against a Qwen3-14B-family
GGUF and exposes an OpenAI-compatible ``/v1/chat/completions`` on port
8002, which the orchestrator's ``llm_local`` provider hits via
``OpenAILLMService``. All aux calls (sheet writer, player suggest,
session summary) hit the same endpoint with bare clients.

The compose default is ``Qwen/Qwen3-14B`` instruct, Q5_K_M GGUF
(~10.5 GB) on an RTX 5060 Ti 16GB: resists confabulation and uses
llama.cpp's native tool-call handler. Any GGUF with a tool-calling chat
template can replace it via ``AIDM_LLM_MODEL_PATH``; higher-temperature
finetunes want temp ~0.65 and the DRY sampler on (see below) or they
drift into mode-collapse loops.

``models/candidates/`` holds Qwen3-30B-A3B MoE builds for the future
"bigger model" step — ``N_CPU_MOE`` exists for that day.

## VRAM budget (16 GB card)

~10 GB weights + ~2 GB q8_0 KV at 16k ctx + ~1.5 GB Whisper Turbo
(int8_float16) ≈ 13.5 GB. TTS (MOSS-TTS-Nano) runs on CPU. Set
``GPU_LAYERS=auto`` to let llama.cpp fit into whatever VRAM a desktop
session left free.

## Chat template + thinking suppression

The GGUF's embedded template is used (``--jinja``); override with
``CHAT_TEMPLATE_FILE`` only when iterating on a finetune. Qwen3 is a
hybrid reasoning model, and thinking is killed **server-side, two
layers** (verified live 2026-07-09: an unguarded sheet-writer call
burned 416 tokens of chain-of-thought before its JSON):

1. ``CHAT_TEMPLATE_KWARGS`` defaults to ``{"enable_thinking":false}``
   — the template injects an empty ``<think></think>`` so the model
   skips reasoning. Per-request kwargs still override.
2. ``REASONING_BUDGET=0`` force-closes any ``<think>`` block opened
   anyway. It cannot catch untagged reasoning in content — that is
   what layer 1 is for.

A thinking DM (``AIDM_DM_THINKING=1``, the ``*-tuned`` profiles) keeps
layer 1 as the server default — side calls stay non-thinking — and
overrides it on the DM request only (llama.cpp merges request
``chat_template_kwargs`` over the server's). Layer 2 becomes a cap
(``REASONING_BUDGET=1024``), and ``--reasoning-format deepseek`` (via
``EXTRA_FLAGS``) returns thoughts as ``reasoning_content``, which the
orchestrator never reads into content.

## Anti-repetition stack

RP failure mode (session bab180be): near-verbatim replies two turns
running, one catchphrase six turns straight. Three server-side layers:

- ``REPEAT_PENALTY=1.15`` — floor against single-register collapse;
  >1.2 starts mangling tool-call JSON.
- ``TOP_K=40`` — breaks degenerate-token spikes.
- **DRY sampler** (``DRY_MULTIPLIER``, default off; dev ``.env`` runs
  0.8) — penalizes re-extending any sequence seen in the last
  ``DRY_PENALTY_LAST_N=2048`` tokens. The window deliberately excludes
  the ~7k-token static persona + scene sheets so they can't trigger
  false penalties. Measured effect: repeat-probe similarity 1.000 →
  0.035.

## Prompt-cache economics

``CACHE_REUSE=256`` enables chunked KV reuse — load-bearing because
the orchestrator splices volatile blocks (scene sheets, turn contract)
into the prompt every turn. KV shifting requires an f16 K cache, so
the entrypoint **drops the flag automatically when K is quantized**
(the default ``KV_CACHE_TYPE=q8_0`` keeps prefix reuse for the
unchanged head of the prompt, which the injector maximizes by keeping
mutations at the prompt tail). Asymmetric f16-K/q8_0-V was tried and
rejected: the CUDA FA kernel refuses the mixed pair.

## Speculative decoding — measured, rejected

``DRAFT_MODEL_PATH`` wires ``--model-draft`` (Qwen3-0.6B for this
target). A/B on 2026-07-09: decode **dropped** 37 → 34.5 tok/s (the
RP sampler stack kills draft acceptance) and draft KV can't be
checkpointed, which breaks cross-prompt cache restore (full re-prefill
on session resume). Left disabled; the knobs remain for other
model/sampler combos.

## Endpoints

- ``GET /health`` — Docker healthcheck; the orchestrator blocks on
  ``depends_on.condition: service_healthy``.
- ``POST /v1/chat/completions`` — OpenAI-compatible, streaming + tool
  calling.

## Env (entrypoint reads these; compose maps ``AIDM_LLM_*`` onto them)

| Var | Default | Notes |
|---|---|---|
| ``MODEL_PATH`` | ``/models/qwen3/Qwen3-14B-Q5_K_M.gguf`` | override with ``AIDM_LLM_MODEL_PATH`` |
| ``CTX_SIZE`` | ``16384`` | fits 16 GB with q8_0 KV |
| ``GPU_LAYERS`` | ``99`` | ``auto`` = fit to free VRAM |
| ``PARALLEL`` | ``1`` | single-session DM; concurrency wrecks KV locality |
| ``KV_CACHE_TYPE`` | ``q8_0`` | symmetric K/V quantization |
| ``CACHE_TYPE_K`` / ``CACHE_TYPE_V`` | empty | explicit pair overrides ``KV_CACHE_TYPE`` |
| ``MLOCK`` / ``NO_MMAP`` | ``1`` / ``1`` | pin weights in RAM; needs ``IPC_LOCK`` cap |
| ``N_CPU_MOE`` | ``0`` | MoE expert CPU-offload; dense models leave 0 |
| ``REPEAT_PENALTY`` | ``1.15`` | see anti-repetition |
| ``TOP_K`` | ``40`` | see anti-repetition |
| ``DRY_MULTIPLIER`` | ``0.0`` (off) | dev runs ``0.8`` |
| ``DRY_BASE`` / ``DRY_ALLOWED_LENGTH`` / ``DRY_PENALTY_LAST_N`` | ``1.75`` / ``2`` / ``2048`` | DRY shape |
| ``REASONING_BUDGET`` | ``0`` | thinking kill, layer 2; ``N>0`` caps a thinking DM (tuned profiles: 1024) |
| ``CHAT_TEMPLATE_KWARGS`` | ``{"enable_thinking":false}`` | thinking kill, layer 1 |
| ``CHAT_TEMPLATE_FILE`` | empty | GGUF-embedded template |
| ``CACHE_REUSE`` | ``256`` | auto-dropped when K cache is quantized |
| ``DRAFT_MODEL_PATH`` | empty (disabled) | speculative decoding — measured net-negative here |
| ``DRAFT_GPU_LAYERS`` / ``DRAFT_N_MAX`` | ``99`` / ``16`` | draft shape |
| ``EXTRA_FLAGS`` | empty | free-form llama-server flags |

Request-side sampling (temperature, top_p, min_p, frequency/presence
penalties) is **not** set here — the orchestrator sends it per request
from ``AIDM_LLM_TEMPERATURE`` & co. (see ``services/orchestrator``).

## Model swap procedure

1. Point ``AIDM_LLM_MODEL_PATH`` at the new GGUF in ``.env``.
2. Adjust ``AIDM_LLM_TEMPERATURE`` (instruct ~0.35, creative finetunes ~0.65).
3. ``docker compose up -d llm-gateway``, confirm boot log shows the
   model + the DRY/thinking flags.
4. Gate: ``make bench-tools`` must hold ≥85% tool / ≥75% args;
   ``make bench-regress-smoke`` for DM-quality drift.

The orchestrator does not need to know which model is on the other
end.

## Model profiles

``profiles/<name>.env`` pins one (model, harness) pair: gateway vars
(``AIDM_LLM_*`` — model path, offload, server samplers, reasoning) and
orchestrator vars (request samplers, ``AIDM_DM_THINKING``,
``AIDM_GUARD_*``, ``AIDM_PROFILE``) in one flat, sh-sourceable file.
Each value carries its source on the line above it. Exported profile
vars beat ``.env`` on both sides. ``qwen3-14b`` is the shipped example;
copy it to add a (model, harness) pair.

```sh
make gateway-profile PROFILE=qwen3-14b             # recreate llm-gateway only
PROFILE=qwen3-14b make bench-tools                 # English tool-call gate
PROFILE=qwen3-14b make loop-sim-50                 # orchestrator side
```

For ``eval_replay_guard.py --summary-json`` (one comparable JSON per
profile over the same replay prompts), source the profile first:
``set -a; . services/llm-gateway/profiles/<name>.env; set +a``.
