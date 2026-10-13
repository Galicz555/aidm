# Faster-Whisper Large-v3 STT service (private mode, Phase 7).
#
# Requires the NVIDIA Container Toolkit on the host. Compose grants the
# GPU via deploy.resources.reservations.devices.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1

# Ubuntu 22.04 (jammy) ships python3.10; we need 3.12 to match the rest
# of the stack. deadsnakes is the canonical PPA for newer Python on
# Ubuntu LTS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app
COPY services/stt/pyproject.toml /app/pyproject.toml
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip \
    && pip install \
        faster-whisper==1.0.3 \
        ctranslate2>=4.4 \
        fastapi==0.115.6 \
        uvicorn[standard]==0.32.1 \
        python-multipart==0.0.20 \
        numpy>=1.26 \
        requests>=2.31 \
        huggingface_hub>=0.24

COPY services/stt/server.py /app/server.py

EXPOSE 8001
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=12 \
    CMD curl -fsS http://localhost:8001/health | grep -q '"ready": *true' || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
