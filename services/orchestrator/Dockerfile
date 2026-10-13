# syntax=docker/dockerfile:1
#
# Orchestrator container.
#
# Builds cleanly and can run aidm helper subcommands (make audio-devices,
# make mic-test). The voice loop itself requires audio device access from
# the host (ALSA / PipeWire socket mount), which is fiddly and varies per
# platform. For local development keep using `uv run aidm` directly on the
# host; this image is here so the deployment story is unblocked when a
# WebRTC transport replaces local audio in a later phase, and for CI builds
# that just verify the package installs.

FROM python:3.13-slim AS base

# System deps:
#   - portaudio19-dev / libportaudio2: pyaudio runtime + build
#   - libsndfile1: audio file IO some pipecat extras use
#   - gcc / build-essential: pyaudio wheel compile
#   - ca-certificates: TLS to ElevenLabs / Google
#   - libxcb1 / libgl1 / libglib2.0-0: opencv-python's shared deps. pipecat's
#     [webrtc] extra pulls opencv (audio+video share the aiortc extra); we
#     don't use video but `import cv2` runs at import time of
#     pipecat.transports.smallwebrtc.transport, so the libs must be present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1 \
    gcc \
    build-essential \
    ca-certificates \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv from the official image (just copy the binary, no separate install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy workspace metadata first so dependency layers cache well
COPY pyproject.toml uv.lock ./
COPY services/orchestrator/pyproject.toml services/orchestrator/pyproject.toml

# Pre-resolve the dependency graph without source — fastest build cache hit
RUN uv sync --frozen --no-install-workspace --all-extras --all-packages

# Now the actual source
COPY services/orchestrator services/orchestrator

# Install workspace packages
RUN uv sync --frozen --all-extras --all-packages

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["aidm"]
