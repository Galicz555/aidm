"""Browser data-channel forwarder for the player-transcript leg.

Bridges a final ``TranscriptionFrame`` from the STT to a
``transcript.user`` event on the WebRTC data channel (text mode reuses
the same envelope via ``TextSink``). The forwarder is transparent: the
original frame still flows downstream to the LLM context aggregator.

The assistant-side events (``transcript.segment`` per speaker, plus
``transcript.bot_end``) are emitted by ``CharacterStreamSplitter`` —
not by this forwarder. We only sit on the user leg.
"""

from __future__ import annotations

from typing import Any

from pipecat.frames.frames import (
    Frame,
    OutputTransportMessageFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class DataChannelForwarder(FrameProcessor):
    """Forward final player transcripts to the WebRTC data channel.

    Transparent: every frame still flows downstream unchanged. The
    forwarder only *additionally* emits an ``OutputTransportMessageFrame``
    for a non-empty ``TranscriptionFrame``.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            return
        payload = self._payload_for(frame)
        if payload is not None:
            await self.push_frame(OutputTransportMessageFrame(message=payload), direction)

    def _payload_for(self, frame: Frame) -> dict[str, Any] | None:
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return None
            return {"type": "transcript.user", "text": text}
        return None
