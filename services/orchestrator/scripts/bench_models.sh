#!/usr/bin/env bash
# Rotate through GGUF models, run the regress bench against each, archive
# results to docs/bench/<date>/models/<model-tag>/. Use to compare RP/tool
# discipline across candidate models on the same hardware.
#
# Usage:
#   bench_models.sh <model-tag>=<gguf-path>[,<env=val>,...] [...]
#
# Each entry:
#   TAG is the folder name under docs/bench/<date>/models/.
#   GGUF-PATH is the .gguf to mount into llm-gateway (path INSIDE the
#     container — i.e. starts with /models/).
#   Optional comma-separated env=val pairs override per-model defaults
#     (e.g. N_CPU_MOE=15 for MoE candidates).
#
# Example — comparing dense 14B baseline against an MoE 30B candidate:
#   bench_models.sh \
#     baseline-14b=/models/qwen3/Qwen3-14B-Q5_K_M.gguf \
#     qwen3-30b-a3b=/models/candidates/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf,N_CPU_MOE=15
#
# Simplification: no temp files, no parallel runs (the gateway is a singleton), one
# loop. Outer script does the rotation; bench_regress.py does the scoring.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <tag>=<gguf-path>[,<env=val>...] [<tag>=<gguf-path>[,<env=val>...] ...]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATE="$(date +%Y-%m-%d)"
OUT_ROOT="docs/bench/${DATE}/models"
mkdir -p "${OUT_ROOT}"

declare -a SCENARIOS
SCENARIOS=(06-lyra-inline-discipline 07-long-run-discipline 08-locale-fidelity-hu 10-status-tracking)

for arg in "$@"; do
    # Split tag=path,env=val,env=val
    TAG="${arg%%=*}"
    REST="${arg#*=}"
    GGUF="${REST%%,*}"
    EXTRA_ENV=""
    if [[ "${REST}" == *,* ]]; then
        EXTRA_ENV="${REST#*,}"
    fi

    # The host-side check: convert /models/... container path to the
    # host-side ./models/... for existence verification.
    HOST_GGUF="${GGUF/#\/models\//./models/}"
    if [[ ! -f "${HOST_GGUF}" ]]; then
        echo "skip ${TAG}: ${HOST_GGUF} not found" >&2
        continue
    fi

    echo "==> ${TAG} (${GGUF}) ${EXTRA_ENV:+[env: ${EXTRA_ENV}]}"

    # Build the env-var prefix for the docker compose call.
    COMPOSE_ENV=("AIDM_LLM_MODEL_PATH=${GGUF}")
    if [[ -n "${EXTRA_ENV}" ]]; then
        # Split comma-separated env=val entries.
        IFS=',' read -ra PAIRS <<< "${EXTRA_ENV}"
        for pair in "${PAIRS[@]}"; do
            # Prefix bare env names with AIDM_LLM_ so they pass through
            # the docker-compose env mapping consistently.
            key="${pair%%=*}"
            val="${pair#*=}"
            COMPOSE_ENV+=("AIDM_LLM_${key}=${val}")
        done
    fi

    env "${COMPOSE_ENV[@]}" \
        docker compose --profile private up -d --force-recreate llm-gateway >/dev/null 2>&1

    # Wait for ready, with a 5-minute ceiling (cold MoE loads can be slow).
    for _ in $(seq 1 60); do
        curl -sf http://localhost:8002/v1/models >/dev/null 2>&1 && break
        sleep 5
    done
    if ! curl -sf http://localhost:8002/v1/models >/dev/null 2>&1; then
        echo "gateway never came ready for ${TAG}" >&2
        continue
    fi

    # bench_regress exits non-zero when any scenario fails — that's a
    # real bench signal, not a script error. Suspend errexit just here
    # so the rotation continues through all candidates.
    set +e
    AIDM_MODE=private \
        AIDM_LOCAL_LLM_MODEL="$(basename "${GGUF}")" \
        uv run python services/orchestrator/scripts/bench_regress.py \
            --scenarios "${SCENARIOS[@]}" \
            --replays 5
    set -e

    # bench_regress drops output under docs/bench/<date>/regress/ —
    # move into the model's slot so the next iteration doesn't clobber.
    if [[ -d "docs/bench/${DATE}/regress" ]]; then
        rm -rf "${OUT_ROOT}/${TAG}"
        mv "docs/bench/${DATE}/regress" "${OUT_ROOT}/${TAG}"
        echo "==> ${TAG} done, results at ${OUT_ROOT}/${TAG}/summary.md"
    else
        echo "==> ${TAG} produced no output dir — skipping archive" >&2
    fi
done

echo
echo "All runs complete. Compare with:"
for arg in "$@"; do
    TAG="${arg%%=*}"
    echo "  ${OUT_ROOT}/${TAG}/summary.md"
done
