# MOSS-TTS-Nano streaming TTS server (private mode, Phase 7).
#
# CPU-only by design: at 0.1B params the model is comfortably
# realtime on a 4-core CPU, and keeping it off the GPU leaves the
# entire 16GB for Gemma (LLM) and Whisper (STT) co-residency.
#
# Pins torch 2.7.0 and transformers 4.57.1 because MOSS-TTS-Nano's
# upstream pyproject pins both. Diverging produces obscure runtime
# errors in the audio tokenizer path.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates curl libsndfile1 ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch + torchaudio. MOSS-TTS-Nano pins the patch versions;
# the +cpu suffix routes to pytorch.org's CPU wheel index so no CUDA
# libraries land in the image (saves ~3GB).
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.7.0 \
        torchaudio==2.7.0

# Application dependencies. MOSS-TTS-Nano installs from git directly:
# no PyPI release exists, and the package layout ships the runtime
# module (`moss_tts_nano_runtime.py`) at the repo root, which our
# server imports.
RUN pip install \
        transformers==4.57.1 \
        sentencepiece>=0.1.99 \
        numpy>=1.24 \
        onnxruntime>=1.20.0 \
        fastapi==0.115.6 \
        'uvicorn[standard]==0.32.1' \
        websockets==13.1 \
        soundfile \
        einops \
    && pip install --no-deps git+https://github.com/OpenMOSS/MOSS-TTS-Nano.git@main

COPY services/tts/server.py /app/server.py

EXPOSE 8003
HEALTHCHECK --interval=5s --timeout=3s --start-period=180s --retries=18 \
    CMD curl -fsS http://localhost:8003/health | grep -q '"ready": *true' || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
