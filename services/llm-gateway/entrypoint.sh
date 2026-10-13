#!/usr/bin/env bash
# Boot llama-server with the flags the orchestrator expects.
#
# Tunables are env-overridable so the same image can swap models
# without rebuilding. The default is Qwen/Qwen3-14B instruct
# (Q5_K_M, ~10.5 GB), chosen to resist confabulation and use llama.cpp's
# native tool-call handler. Qwen3 is a hybrid reasoning model; the orchestrator
# passes enable_thinking=False so no <think> traces leak. Fetch with:
#   hf download Qwen/Qwen3-14B-GGUF \
#     Qwen3-14B-Q5_K_M.gguf --local-dir ./models/qwen3
set -euo pipefail

: "${MODEL_PATH:=/models/qwen3/Qwen3-14B-Q5_K_M.gguf}"
: "${PORT:=8002}"
: "${CTX_SIZE:=16384}"
# Server-side anti-repetition. A 14B model mode-collapses
# into a single emotional register without a penalty floor — "guarded X"
# loops and "shelter hums around you" katechism on long Lyra sessions were
# the live tell. 1.15 caught the long-context drift that 1.1 didn't; >1.2
# starts mangling tool-call JSON.
: "${REPEAT_PENALTY:=1.15}"
# Top-k narrows the candidate pool — combined with the penalty stack, this
# breaks degenerate-token spikes that keep dragging the model back into
# the same imagery turn after turn.
: "${TOP_K:=40}"
# **Full GPU offload** by default. Qwen3-14B at
# Q5_K_M weighs ~10.5 GB — combined with Whisper Turbo (~1.5 GB
# int8_float16) and a q8_0 KV cache at 16k ctx (~2 GB) it sits at
# ~13.3 GB on a 5060 Ti. TTS runs on CPU (MOSS-TTS-Nano) so it
# costs no VRAM. Drop to 40-50 if you co-resident another CUDA
# process or swap to a 22B model that needs CPU spill (see the
# Mistral-Small / Magnum entries in docs/bench/<date>/).
# Number of transformer layers to offload to the GPU. ``99`` is the
# upstream convention for "all layers" (clamped to the model's actual
# layer count). Set ``GPU_LAYERS=auto`` to skip the flag entirely and
# let llama.cpp's common_fit_params pick the largest value that fits
# in the currently-free VRAM — useful on a 16 GB card that shares the
# GPU with a desktop/browser, where the free amount varies session
# to session.
: "${GPU_LAYERS:=99}"
: "${PARALLEL:=1}"
# KV cache quantization. Default fp16 ≈ 230 bytes/token on Mistral
# Small 24B (56 layers × 8 KV heads × 128 dim × 2). At ctx=32768 that
# is ~7.5 GB just for the cache — too much for a 16 GB card sharing
# with a desktop. ``q8_0`` halves it to ~3.8 GB with negligible
# quality loss; ``q4_0`` quarters again for ~1.9 GB if you really
# need to push ctx to 64k+.
#
# Symmetric default. To run asymmetric (TurboQuant — DeepMind 2024:
# 4-bit keys, 3-bit values, near-lossless via random rotation),
# set CACHE_TYPE_K and CACHE_TYPE_V explicitly and leave KV_CACHE_TYPE
# at the default — the explicit pair overrides.
: "${KV_CACHE_TYPE:=q8_0}"
: "${CACHE_TYPE_K:=}"
: "${CACHE_TYPE_V:=}"
# Pin all weights in RAM (mlock) and force full RAM load (no-mmap).
# Together they kill the two long-run latency sources we saw on the
# Lyra session: mid-inference page faults (mmap loading experts on
# demand) and OS paging the model out under memory pressure. Default
# ON for the prod-style continuous server. Requires the IPC_LOCK
# Docker capability — set in docker-compose.yml.
: "${MLOCK:=1}"
: "${NO_MMAP:=1}"
# MoE expert offload to CPU RAM. Only meaningful for MoE GGUFs (Qwen3-
# 30B-A3B variants, Qwen3-Next, etc.). For a 30B-A3B Q4_K_M (~18 GB
# weights) on a 16 GB card: set N_CPU_MOE=15 to keep most experts in
# RAM and ~3 GB of active params in VRAM. Dense models: leave at 0.
: "${N_CPU_MOE:=0}"
# Empty by default: let llama.cpp use the chat template embedded in
# the GGUF (Qwen3 / Mistral-Small / Llama 3.1 tokenizers ship correct
# templates). Set CHAT_TEMPLATE_FILE to a path to override — useful
# when iterating on a custom format during a finetune.
: "${CHAT_TEMPLATE_FILE:=}"
# Server-side thinking kill switch, two layers. The orchestrator's
# main DM call passes enable_thinking=False via chat_template_kwargs,
# but the aux call sites (sheet writer, player suggest, session
# summary) use bare clients — and Qwen3 then reasons at will. Verified
# live 2026-07-09: a writer-style patch call burned 416 tokens of
# Chinese chain-of-thought (12s) before the JSON.
#
# Layer 1: template kwarg. enable_thinking=false makes the Qwen3
# template inject an empty <think></think> block — the model skips
# reasoning entirely. Request-level kwargs still override.
if [[ -z "${CHAT_TEMPLATE_KWARGS:-}" ]]; then
    CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi
# Layer 2: reasoning budget 0 force-closes any <think> block a model
# opens anyway (variant templates, per-request overrides). It cannot
# catch untagged reasoning in content — that's what layer 1 is for.
: "${REASONING_BUDGET:=0}"
# DRY sampling — token-level anti-repetition. Penalizes extending a
# sequence that already occurred in the last DRY_PENALTY_LAST_N tokens,
# which is exactly the observed RP failure mode: near-verbatim replies
# two turns running and the same catchphrase six turns straight
# (session bab180be, 2026-07-09). LAST_N=2048 keeps the penalty window
# on recent turns so the ~7k-token static persona + scene sheets can't
# trigger false penalties. 0.0 = disabled (upstream default); the dev
# .env turns it on.
: "${DRY_MULTIPLIER:=0.0}"
: "${DRY_BASE:=1.75}"
: "${DRY_ALLOWED_LENGTH:=2}"
: "${DRY_PENALTY_LAST_N:=2048}"
# Min chunk size for prefix reuse via KV shifting. The scene-sheet
# injector mutates one system message mid-prompt every turn; without
# chunked reuse that invalidates the KV for the entire history behind
# it — a full re-prefill per turn. Requires a shiftable (f16) K cache;
# the entrypoint drops the flag automatically when K is quantized.
: "${CACHE_REUSE:=256}"
# Speculative decoding. Point DRAFT_MODEL_PATH at a small same-tokenizer
# GGUF (Qwen3-0.6B for the Qwen3-14B target) to speculate decode.
# RP prose accepts drafts readily — typical 1.5-2x on the ~38 tok/s
# decode for ~0.7 GB VRAM. Empty = disabled.
: "${DRAFT_MODEL_PATH:=}"
: "${DRAFT_GPU_LAYERS:=99}"
: "${DRAFT_N_MAX:=16}"
: "${EXTRA_FLAGS:=}"

# CUDA graphs: CUDA_GRAPHS=0 exports GGML_CUDA_DISABLE_GRAPHS=1. Workaround
# for the sm_120 (Blackwell) GPU-channel hang — RC watchdog + Xid 8 mid-run,
# llama.cpp issue #27330; hit by both Gemma-4-31B quants on 2026-09-26. ggml
# only checks whether the variable is set, so export it only when asked.
: "${CUDA_GRAPHS:=1}"
if [[ "${CUDA_GRAPHS}" == "0" ]]; then
    export GGML_CUDA_DISABLE_GRAPHS=1
    echo "entrypoint: CUDA graphs disabled (GGML_CUDA_DISABLE_GRAPHS=1)" >&2
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "Fetch the production default with:" >&2
    echo "  hf download Qwen/Qwen3-14B-GGUF \\" >&2
    echo "    Qwen3-14B-Q5_K_M.gguf --local-dir ./models/qwen3" >&2
    echo "Or set AIDM_LLM_MODEL_PATH in .env to a different GGUF on disk." >&2
    exit 1
fi

# The upstream image keeps the binary at /app/llama-server and does
# not put it on PATH. Hardcoding the absolute path is more robust
# than mutating PATH inside the entrypoint.
CHAT_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE_FILE}" ]]; then
    CHAT_TEMPLATE_ARGS=(--chat-template-file "${CHAT_TEMPLATE_FILE}")
fi

GPU_LAYERS_ARGS=(--n-gpu-layers "${GPU_LAYERS}")
if [[ "${GPU_LAYERS}" == "auto" ]]; then
    # Omit the flag: llama.cpp's common_fit_params auto-fits to free
    # VRAM. Required when the GPU is shared with a desktop session and
    # the available memory varies.
    GPU_LAYERS_ARGS=()
fi

KV_CACHE_ARGS=()
if [[ -n "${CACHE_TYPE_K}" && -n "${CACHE_TYPE_V}" ]]; then
    # Asymmetric (TurboQuant-style). The explicit pair wins.
    KV_CACHE_ARGS=(--cache-type-k "${CACHE_TYPE_K}" --cache-type-v "${CACHE_TYPE_V}")
elif [[ "${KV_CACHE_TYPE}" != "f16" ]]; then
    KV_CACHE_ARGS=(--cache-type-k "${KV_CACHE_TYPE}" --cache-type-v "${KV_CACHE_TYPE}")
fi

MEMLOCK_ARGS=()
if [[ "${MLOCK}" == "1" ]]; then
    MEMLOCK_ARGS+=(--mlock)
fi
if [[ "${NO_MMAP}" == "1" ]]; then
    MEMLOCK_ARGS+=(--no-mmap)
fi

MOE_ARGS=()
if [[ "${N_CPU_MOE}" -gt 0 ]]; then
    MOE_ARGS+=(--n-cpu-moe "${N_CPU_MOE}")
fi

# KV shifting cannot rotate a quantized K cache — llama.cpp errors at
# shift time, not boot time. Only pass --cache-reuse when the effective
# K type is f16 so a q8_0 profile still boots clean.
# Mirrors the KV_CACHE_ARGS rule: the explicit pair wins, otherwise
# the symmetric default applies to K too.
if [[ -n "${CACHE_TYPE_K}" && -n "${CACHE_TYPE_V}" ]]; then
    EFFECTIVE_CACHE_TYPE_K="${CACHE_TYPE_K}"
else
    EFFECTIVE_CACHE_TYPE_K="${KV_CACHE_TYPE}"
fi
CACHE_REUSE_ARGS=()
if [[ "${CACHE_REUSE}" -gt 0 && "${EFFECTIVE_CACHE_TYPE_K}" == "f16" ]]; then
    CACHE_REUSE_ARGS=(--cache-reuse "${CACHE_REUSE}")
fi

DRY_ARGS=()
if [[ "${DRY_MULTIPLIER}" != "0.0" && "${DRY_MULTIPLIER}" != "0" ]]; then
    DRY_ARGS=(
        --dry-multiplier "${DRY_MULTIPLIER}"
        --dry-base "${DRY_BASE}"
        --dry-allowed-length "${DRY_ALLOWED_LENGTH}"
        --dry-penalty-last-n "${DRY_PENALTY_LAST_N}"
    )
fi

DRAFT_ARGS=()
if [[ -n "${DRAFT_MODEL_PATH}" ]]; then
    if [[ ! -f "${DRAFT_MODEL_PATH}" ]]; then
        echo "DRAFT_MODEL_PATH does not exist: ${DRAFT_MODEL_PATH}" >&2
        echo "Fetch the Qwen3 draft with:" >&2
        echo "  hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf \\" >&2
        echo "    --local-dir ./models/qwen3" >&2
        exit 1
    fi
    DRAFT_ARGS=(
        --spec-draft-model "${DRAFT_MODEL_PATH}"
        --spec-draft-ngl "${DRAFT_GPU_LAYERS}"
        --spec-draft-n-max "${DRAFT_N_MAX}"
    )
fi

exec /app/llama-server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --ctx-size "${CTX_SIZE}" \
    --repeat-penalty "${REPEAT_PENALTY}" \
    --top-k "${TOP_K}" \
    "${GPU_LAYERS_ARGS[@]}" \
    "${KV_CACHE_ARGS[@]}" \
    "${MEMLOCK_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${CACHE_REUSE_ARGS[@]}" \
    "${DRAFT_ARGS[@]}" \
    "${DRY_ARGS[@]}" \
    --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
    --reasoning-budget "${REASONING_BUDGET}" \
    --parallel "${PARALLEL}" \
    --jinja \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    ${EXTRA_FLAGS}
