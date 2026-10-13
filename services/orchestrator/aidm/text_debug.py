"""Text-mode debug logging.

Emits a per-turn trace to BOTH the server log (loguru) and the SSE
stream (so the browser can render a debug panel):

* ``debug.context`` — the full LLM message history at the moment the
  user's turn is about to fire (system prompt, prior turns, tools).
* ``debug.bot`` — the assistant's reply once the LLM has finished
  streaming.
* ``debug.tool`` — tool-call invocations with name + arguments + result.

Only wired in the text pipeline today. Voice mode has its own
turn-tracking observer and the audio surface is a poor debug channel.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_CONTEXT_PREVIEW_CHARS = 200
_BOT_LOG_PREVIEW_CHARS = 500


class TextDebugLogger(FrameProcessor):
    """Capture the outbound LLM context, log + push ``debug.context``.

    Sits UPSTREAM of the LLM, so it only ever sees frames flowing into
    it — the reply frames (``LLMTextFrame`` etc.) are emitted by the LLM
    downstream and never pass through here. ``debug.bot`` is therefore
    emitted by ``LLMOutputTracer`` below, which sits after the LLM.
    (Until 2026-07-11 this class also carried a reply-flush branch that
    could never fire from this position — the loop harness exposed it.)
    """

    def __init__(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        super().__init__()
        self._queue = queue

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            self._log_context(frame)

    def _log_context(self, frame: LLMContextFrame) -> None:
        messages = list(getattr(frame.context, "messages", None) or [])
        serialized = [_serialize_message(m) for m in messages]
        last_user = next(
            (m["text"] for m in reversed(serialized) if m["role"] == "user"),
            "",
        )
        logger.info(
            "USER -> LLM ({} msgs in context): {!r}",
            len(serialized),
            _truncate(last_user, _CONTEXT_PREVIEW_CHARS),
        )
        self._queue.put_nowait(
            {
                "type": "debug.context",
                "messages": serialized,
                "count": len(serialized),
            }
        )


class LLMOutputTracer(FrameProcessor):
    """Log everything the LLM service emits downstream, with timing.

    Sits between the LLM and CharacterStreamSplitter to settle "is the
    LLM service stuck or did it complete with an empty response?" Each
    LLM dispatch logs Start → text-chunk count → End with wall-time.

    When constructed with a queue, also assembles the reply text and
    pushes it as one ``debug.bot`` SSE payload on response end — this
    is the post-LLM position where the reply frames actually flow.
    """

    def __init__(self, queue: asyncio.Queue[dict[str, Any] | None] | None = None) -> None:
        super().__init__()
        self._queue = queue
        self._dispatch_started: float | None = None
        self._chunks: int = 0
        self._chars: int = 0
        self._text: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._dispatch_started = time.perf_counter()
                self._chunks = 0
                self._chars = 0
                self._text = []
                logger.debug("llm_trace: LLMFullResponseStartFrame")
            elif isinstance(frame, LLMTextFrame):
                self._chunks += 1
                self._chars += len(frame.text or "")
                if frame.text:
                    self._text.append(frame.text)
                if self._chunks <= 3 or self._chunks % 25 == 0:
                    logger.trace(
                        "llm_trace: text chunk #{} ({} chars, head={!r})",
                        self._chunks,
                        len(frame.text or ""),
                        (frame.text or "")[:80],
                    )
            elif isinstance(frame, LLMFullResponseEndFrame):
                elapsed = (
                    time.perf_counter() - self._dispatch_started
                    if self._dispatch_started is not None
                    else 0.0
                )
                logger.info(
                    "llm_trace: LLMFullResponseEndFrame after {:.1f}s "
                    "({} text chunks, {} chars total)",
                    elapsed,
                    self._chunks,
                    self._chars,
                )
                self._dispatch_started = None
                self._flush_reply()
        await self.push_frame(frame, direction)

    def _flush_reply(self) -> None:
        """Push the assembled reply as one ``debug.bot`` payload."""
        full = "".join(self._text)
        self._text = []
        if not full:
            return
        logger.info(
            "LLM -> USER ({} chars): {!r}",
            len(full),
            _truncate(full, _BOT_LOG_PREVIEW_CHARS),
        )
        if self._queue is not None:
            self._queue.put_nowait({"type": "debug.bot", "text": full})


def _serialize_message(message: object) -> dict[str, str]:
    """Best-effort flatten of OpenAI/Anthropic-style context messages.

    The aggregator stores raw dicts coming from whichever LLM adapter
    populates them; shapes differ by adapter, so we extract the parts
    a debug panel actually wants (role + text) and stringify the rest.
    """
    if isinstance(message, dict):
        role = str(message.get("role", "?"))
        content = message.get("content")
    else:
        role = getattr(message, "role", "?")
        content = getattr(message, "content", None)

    text = _stringify_content(content)
    return {"role": str(role), "text": text}


def _stringify_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "tool_use":
                    parts.append(f"[tool_use {item.get('name', '?')}({item.get('input', {})})]")
                elif item.get("type") == "tool_result":
                    parts.append(f"[tool_result {item.get('content', '')}]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} chars)"
