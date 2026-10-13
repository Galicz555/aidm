# llama.cpp server hosting Gemma 4 26B-A4B QAT (private mode, Phase 7).
#
# The upstream image already ships llama-server compiled against CUDA;
# we only add curl (for healthcheck) and our entrypoint script. The
# chat template is read from the GGUF metadata via --jinja, so no
# /templates/ payload is needed.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/llm-gateway/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8002
HEALTHCHECK --interval=5s --timeout=3s --start-period=300s --retries=24 \
    CMD curl -fsS http://localhost:8002/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
