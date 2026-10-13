"""Text-mode sink: forwards data-channel payloads onto an asyncio.Queue.

In WebRTC mode, ``DataChannelForwarder`` emits ``OutputTransportMessageFrame``s
which the ``SmallWebRTCTransport`` ships over the browser's data channel.
In text mode there is no transport at the tail of the pipeline — we
intercept those same frames here and push their payloads onto a queue
that the FastAPI SSE response drains for the browser.

Transparent: every frame is also forwarded downstream so the LLM-context
aggregator still sees ``LLMFullResponseEndFrame`` and commits the bot
turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputTransportMessageFrame,
    TextFrame,
    TranscriptionFrame,
    TranslationFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

IDLE_PAYLOAD: dict[str, Any] = {"type": "turn.status", "phase": "idle"}


class TurnActivity:
    """Whether a text turn is still in flight: the server's gate and the client's idle.

    A turn can be several LLM responses (narration + tool call, then the
    narration the tool result re-runs), and the sheet writer emits its own
    ``turn.status: idle`` after each one that carried text — so the first
    idle is not the end (2026-09-28 review: a regenerate restored while
    leg 2 was streaming). The turn has settled when all hold:

    - no response is streaming;
    - no tool call is pending and no final tool result is waiting on the
      LLM re-run it triggers;
    - every response that carried text has reached the end of the
      post-turn tail (its writer and summarizer processors passed).

    ``TextSink`` feeds the LLM-side frames (in order, before the assistant
    aggregator), ``TurnSettled`` the tail. On settling ``busy`` clears and
    ONE ``turn.status: idle`` is emitted — the sink drops the writer's
    per-response idles, so the client ungates on the same signal. A
    response with no text and no tool call (an empty completion) settles
    at its end: no writer runs for it, and it used to leave both stuck.
    """

    def __init__(self, emit: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.busy = False
        self._emit = emit
        self._streaming = False
        self._has_text = False
        self._pending_calls: set[str] = set()
        self._awaiting_rerun = False
        self._text_responses = 0
        self._landed = 0
        self._settled = asyncio.Event()
        self._settled.set()

    def begin(self) -> None:
        """A player turn (or regenerate, or /end) was queued."""
        self.busy = True
        self._settled.clear()

    def release(self) -> None:
        """A queued-nothing request gave up (a refused regenerate): not busy, no idle."""
        self.busy = False
        self._settled.set()

    async def wait_settled(self, timeout_s: float) -> bool:
        """True once the current turn settled; False on timeout."""
        try:
            await asyncio.wait_for(self._settled.wait(), timeout=timeout_s)
        except TimeoutError:
            return False
        return True

    def observe(self, frame: Frame) -> None:
        """An LLM-side downstream frame at the sink (before the assistant aggregator)."""
        if isinstance(frame, InterruptionFrame):
            # A new turn cut this one off: its queued frames are dropped,
            # so its counts would never balance. The new turn re-arms busy.
            self._streaming = self._awaiting_rerun = False
            self._pending_calls.clear()
            self._text_responses = self._landed = 0
        elif isinstance(frame, LLMFullResponseStartFrame):
            self.begin()
            self._streaming, self._has_text, self._awaiting_rerun = True, False, False
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._pending_calls.update(fc.tool_call_id for fc in frame.function_calls)
        elif isinstance(frame, FunctionCallResultFrame):
            if frame.properties is not None and frame.properties.is_final is False:
                return
            self._finish_call(frame.tool_call_id, rerun=_reruns_llm(frame))
        elif isinstance(frame, FunctionCallCancelFrame):
            self._finish_call(frame.tool_call_id, rerun=False)
        elif _is_context_text(frame):
            self._has_text = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._streaming = False
            if self._has_text:
                self._text_responses += 1  # settles when its timestamp reaches the tail
            else:
                self._maybe_settle()

    def response_landed(self) -> None:
        """An end-of-turn frame passed the whole post-turn tail."""
        self._landed += 1
        self._maybe_settle()

    def _finish_call(self, tool_call_id: str, *, rerun: bool) -> None:
        self._pending_calls.discard(tool_call_id)
        if not self._pending_calls and rerun:
            self._awaiting_rerun = True
        self._maybe_settle()

    def _maybe_settle(self) -> None:
        if not self.busy or self._streaming or self._pending_calls or self._awaiting_rerun:
            return
        if self._landed < self._text_responses:
            return
        self.busy = False
        self._text_responses = self._landed = 0
        self._settled.set()
        logger.debug("turn settled")
        if self._emit is not None:
            self._emit(dict(IDLE_PAYLOAD))


def _reruns_llm(frame: FunctionCallResultFrame) -> bool:
    """Mirrors the assistant aggregator: an explicit ``run_llm=False`` is the only no."""
    if frame.properties is not None and frame.properties.run_llm is not None:
        return bool(frame.properties.run_llm)
    return frame.run_llm is not False


def _is_context_text(frame: Frame) -> bool:
    """Text the assistant aggregator folds into the reply (same test it applies)."""
    return (
        isinstance(frame, TextFrame)
        and not isinstance(frame, TranscriptionFrame | TranslationFrame | InterimTranscriptionFrame)
        and frame.append_to_context
        and len(frame.text) > 0
    )


class TextSink(FrameProcessor):
    """Capture data-channel-bound JSON payloads into a per-session queue.

    The queue is supplied at construction time by the text-server's
    session spawn. SSE consumers ``await queue.get()``; a ``None`` value
    is the close sentinel.
    """

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any] | None],
        *,
        activity: TurnActivity | None = None,
    ) -> None:
        super().__init__()
        self._queue = queue
        self._activity = activity

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        # Bookkeeping first: the aggregator's timestamp for this response
        # must never reach the tail before the sink counted the response.
        if self._activity is not None and direction == FrameDirection.DOWNSTREAM:
            self._activity.observe(frame)
        # Pass everything through unchanged so the assistant aggregator
        # downstream still sees LLMFullResponseEndFrame and friends.
        await self.push_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            logger.trace("text_sink: LLMFullResponseStartFrame reached sink")
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.trace(
                "text_sink: LLMFullResponseEndFrame reached sink, qsize={}",
                self._queue.qsize(),
            )
        elif isinstance(frame, OutputTransportMessageFrame):
            payload = frame.message
            if isinstance(payload, dict):
                ptype = str(payload.get("type"))
                # Per-token transcript payloads are pure volume; state
                # payloads (scene.sheets, npc.state, turn.status, error)
                # are the turn narrative.
                log = logger.trace if ptype.startswith("transcript.") else logger.debug
                if self._activity is not None and payload == IDLE_PAYLOAD:
                    # The writer's per-response idle; the turn's own idle
                    # comes from TurnActivity once the turn settled.
                    return
                log("text_sink: queued payload type={}", ptype)
                # put_nowait — the queue is unbounded by intent: the SSE
                # consumer reads on its own loop, and dropping a token
                # mid-turn would silently corrupt the transcript.
                self._queue.put_nowait(payload)
