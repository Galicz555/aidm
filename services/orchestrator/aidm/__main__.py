"""CLI entry: `python -m aidm` or `aidm` (after install).

Dispatches on ``AIDM_TRANSPORT``:

* ``local`` (default) — direct PyAudio mic + speaker loop. One process,
  one session, ``Ctrl+C`` to stop. Today's path.
* ``webrtc`` — uvicorn server hosting the FastAPI app from
  ``aidm.server``. Browsers connect, each peer-connection becomes a
  session. See ``docs/session-lifecycle.md``.
* ``text`` — uvicorn server hosting the FastAPI app from
  ``aidm.text_server``. Same session model, but the client speaks JSON
  over HTTP (POST turn, SSE for streamed DM tokens) instead of audio
  over WebRTC. No STT/TTS in the pipeline.
"""

from __future__ import annotations

import asyncio
import os
import sys

from loguru import logger

from aidm.log import configure as configure_logging
from aidm.pipeline import run_voice_loop


def main() -> int:
    configure_logging(level=os.environ.get("AIDM_LOG_LEVEL", "INFO"))
    transport = os.environ.get("AIDM_TRANSPORT", "local")
    try:
        if transport == "webrtc":
            return _run_webrtc_server()
        if transport == "text":
            return _run_text_server()
        if transport == "local":
            return _run_local_loop()
        logger.error(
            "unsupported AIDM_TRANSPORT: {} (expected 'local', 'webrtc' or 'text')",
            transport,
        )
        return 2
    except KeyboardInterrupt:
        logger.info("interrupted, bye")
        return 0


def _run_local_loop() -> int:
    try:
        asyncio.run(run_voice_loop())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.exception("voice loop crashed: {}", exc)
        return 1
    return 0


def _run_webrtc_server() -> int:
    # Imported lazily so the local-mode path does not pay the uvicorn +
    # fastapi + aiortc import cost. Browser-mode users opt in by setting
    # AIDM_TRANSPORT=webrtc.
    import uvicorn  # noqa: PLC0415 (lazy import — local mode skips this cost)

    from aidm.server import create_app  # noqa: PLC0415

    host = os.environ.get("AIDM_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("AIDM_SERVER_PORT", "8000"))
    logger.info("starting aidm webrtc server on http://{}:{}", host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0


def _run_text_server() -> int:
    # Lazy-imported for the same reason as the WebRTC server: local
    # mode shouldn't pay the FastAPI / uvicorn import cost.
    import uvicorn  # noqa: PLC0415

    from aidm.text_server import create_app  # noqa: PLC0415

    host = os.environ.get("AIDM_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("AIDM_SERVER_PORT", "8000"))
    logger.info("starting aidm text server on http://{}:{}", host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
