#!/usr/bin/env bash
# Fetch private-mode model weights into ./models/ before `make up-private`.
#
# - Gemma 3 27B IT, Q4_K_M GGUF → models/gemma/
# - Faster-Whisper Large-v3 (CTranslate2 conversion) → models/whisper/
# - Coqui XTTS-v2 → models/xtts/ (downloaded automatically at first
#   container start; this script just creates the cache dir so the
#   volume mount has somewhere to land)
#
# Tools:
# - uv (with `uv tool run` / `uvx`) on the host — we don't require a
#   global huggingface-cli; uv pulls it into an ephemeral tool venv.
# - Optionally HF_TOKEN exported in the environment for gated repos.
#
# Re-running is safe; huggingface-cli skips files already on disk.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

mkdir -p models/gemma models/whisper models/xtts

# Prefer uv tool run (no global install needed). The modern HF client
# is `hf`; the older `huggingface-cli` is deprecated upstream and the
# `[cli]` extra is gone. Fall back to a system-installed `hf` if uv
# isn't on PATH.
if command -v uv >/dev/null; then
    HF=(uv tool run --from huggingface_hub hf)
elif command -v hf >/dev/null; then
    HF=(hf)
else
    echo "Need either uv or hf (huggingface_hub) on PATH. Install uv from https://docs.astral.sh/uv/" >&2
    echo "(uv is already used by the rest of the AIDM toolchain.)" >&2
    exit 1
fi

# Gemma 4 26B-A4B IT, **QAT** (quantization-aware-trained) +
# unsloth Dynamic Q4_K_XL. Google QAT-pretrains the model with Q4
# simulation in the loop, so the 4-bit weights carry near-fp16
# quality instead of the usual PTQ degradation. unsloth's dynamic
# per-tensor bit-width tuning sits on top. The file is 14.25 GB —
# smaller AND higher quality than the non-QAT UD-Q4_K_M (16.95 GB).
: "${GEMMA_REPO:=unsloth/gemma-4-26B-A4B-it-qat-GGUF}"
: "${GEMMA_FILE:=gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
echo ">> Fetching ${GEMMA_REPO}/${GEMMA_FILE} -> models/gemma/"
"${HF[@]}" download "${GEMMA_REPO}" "${GEMMA_FILE}" --local-dir models/gemma

# Whisper Large-v3 Turbo: ~5x faster decoding than Large-v3, same
# Hungarian quality on our test set. CT2-converted by deepdml for
# faster-whisper.
: "${WHISPER_REPO:=deepdml/faster-whisper-large-v3-turbo-ct2}"
echo ">> Fetching ${WHISPER_REPO} -> models/whisper/"
"${HF[@]}" download "${WHISPER_REPO}" --local-dir models/whisper

# MOSS-TTS-Nano + the shared audio tokenizer (~120MB combined). The
# TTS container loads these at startup; no lazy download path.
: "${MOSS_TTS_REPO:=OpenMOSS-Team/MOSS-TTS-Nano-100M}"
: "${MOSS_TOK_REPO:=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano}"
echo ">> Fetching ${MOSS_TTS_REPO} -> models/moss-tts/"
"${HF[@]}" download "${MOSS_TTS_REPO}" --local-dir models/moss-tts
echo ">> Fetching ${MOSS_TOK_REPO} -> models/moss-audio-tokenizer/"
"${HF[@]}" download "${MOSS_TOK_REPO}" --local-dir models/moss-audio-tokenizer

echo
echo "Done. Total size:"
du -sh models/* 2>/dev/null || true
