#!/bin/bash
set -e

# Clean startup for AIDM dev modes.
# Usage: dev-start.sh <mode> <transport> [debug]
#   mode: public|private
#   transport: webrtc|text
#   debug: optional, set to 1 to enable DEBUG logging
#
# Binding: the orchestrator and the vite dev server listen on 127.0.0.1 only.
# Dev mode usually runs with CF_ACCESS_DISABLED=1, so the API has no auth;
# putting it on the LAN or tailnet is an explicit opt-in:
#   AIDM_SERVER_HOST=0.0.0.0 make start-private-text   (any start-* target)

MODE=${1:-public}
TRANSPORT=${2:-webrtc}
DEBUG=${3:-0}

if [[ "$MODE" != "public" && "$MODE" != "private" ]]; then
    echo "Error: mode must be 'public' or 'private', got '$MODE'"
    exit 1
fi

if [[ "$TRANSPORT" != "webrtc" && "$TRANSPORT" != "text" ]]; then
    echo "Error: transport must be 'webrtc' or 'text', got '$TRANSPORT'"
    exit 1
fi

# Private text mode needs docker compose up
if [[ "$MODE" == "private" ]]; then
    echo "==> starting llm-gateway (private mode)"
    docker compose --profile private up -d --build --no-deps llm-gateway
fi

# Prepare env vars
export AIDM_MODE="$MODE"
export AIDM_TRANSPORT="$TRANSPORT"
export AIDM_SERVER_HOST="${AIDM_SERVER_HOST:-127.0.0.1}"

if [[ "$DEBUG" == "1" ]]; then
    # Respect a caller-set level (e.g. AIDM_LOG_LEVEL=TRACE for the firehose)
    export AIDM_LOG_LEVEL="${AIDM_LOG_LEVEL:-DEBUG}"
fi

# Private mode needs local service URLs (Docker publishes to localhost)
if [[ "$MODE" == "private" ]]; then
    export AIDM_LOCAL_LLM_URL="http://localhost:8002/v1"
    export AIDM_LOCAL_STT_URL="http://localhost:8001"
    export AIDM_LOCAL_TTS_URL="ws://localhost:8003/v1/tts/stream"
fi

# Detect IPs
LOCALHOST_URL="http://127.0.0.1:5173"
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")

echo ""
echo "╭─────────────────────────────────────────╮"
echo "│ AIDM Dev Server"
echo "├─────────────────────────────────────────┤"
echo "│ Mode:      $MODE"
echo "│ Transport: $TRANSPORT"
if [[ "$DEBUG" == "1" ]]; then
    echo "│ Log Level: ${AIDM_LOG_LEVEL}"
fi
echo "├─────────────────────────────────────────┤"
echo "│ Localhost: $LOCALHOST_URL"
if [[ "$AIDM_SERVER_HOST" == "127.0.0.1" || "$AIDM_SERVER_HOST" == "localhost" ]]; then
    echo "│ LAN/Tailscale: off. To expose (no auth in dev mode):"
    echo "│   AIDM_SERVER_HOST=0.0.0.0 make <the same start target>"
else
    echo "│ Listening on $AIDM_SERVER_HOST (no auth in dev mode if CF_ACCESS_DISABLED=1)"
    if [[ -n "$TAILSCALE_IP" ]]; then
        echo "│ Tailscale: http://$TAILSCALE_IP:5173"
    fi
fi
echo "│"
echo "│ Press Ctrl+C to stop everything"
echo "╰─────────────────────────────────────────╯"
echo ""

# Kill on exit (orchestrator + web dev server + compose services)
cleanup() {
    echo ""
    echo "==> shutting down..."
    # Kill any remaining background jobs
    jobs -p | xargs -r kill -9 2>/dev/null || true
    # Stop docker services
    if [[ "$MODE" == "private" ]]; then
        docker compose --profile private down 2>/dev/null || true
    fi
    echo "==> done"
}
trap cleanup EXIT INT TERM

# Start orchestrator and web dev server in parallel
uv run aidm &
npm --prefix web run dev -- --host "$AIDM_SERVER_HOST" &

wait
