"""Text-debug channel — debug.bot must come from the post-LLM tracer.

Regression for the 2026-07-11 loop-harness finding: TextDebugLogger sits
upstream of the LLM and never sees reply frames, so its old reply-flush
branch was dead code and ``debug.bot`` never reached the SSE stream.
The reply assembly now lives on LLMOutputTracer (post-LLM position).
"""

from __future__ import annotations

import asyncio
from typing import Any

from aidm.text_debug import LLMOutputTracer


def test_tracer_flushes_assembled_reply_as_debug_bot() -> None:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    tracer = LLMOutputTracer(queue)
    tracer._text = ["The gate ", "yard is ", "quiet."]
    tracer._flush_reply()
    payload = queue.get_nowait()
    assert payload == {"type": "debug.bot", "text": "The gate yard is quiet."}
    assert tracer._text == []  # buffer cleared for the next dispatch


def test_tracer_skips_empty_reply_and_survives_no_queue() -> None:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    tracer = LLMOutputTracer(queue)
    tracer._flush_reply()
    assert queue.empty()
    # Log-only construction (no queue) must not crash on flush.
    bare = LLMOutputTracer()
    bare._text = ["hello"]
    bare._flush_reply()
