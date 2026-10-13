"""Split DM output into narrator / NPC segments and attribute them.

The DM is instructed (see ``prompts/dm.hu.md``) to wrap each NPC's
spoken dialogue in ``<npc id="slug">...</npc>`` tags. Narrator prose
and scene description stay untagged. This frame processor parses the
streaming ``LLMTextFrame`` chunks, attributes each segment to the
right speaker, and forwards everything downstream.

Two outputs per segment:

1. A ``transcript.segment`` data-channel event with ``speaker`` set to
   ``"dm"`` or ``"npc:<slug>"``. This drives both the web client's
   character-attributed transcript panel and the text-mode SSE stream.
2. The segment text — with the surrounding ``<npc ...>`` tags stripped
   — pushed downstream as a fresh ``LLMTextFrame`` so the TTS speaks
   the dialogue cleanly. ElevenLabs v3 emotion tags (``[whispers]``,
   ``[laughs]``) inside the segment are preserved.

Per-NPC TTS voice switching is a follow-up. The Pipecat TTS service
aggregates text into sentences before synthesis; cleanly mutating
``tts_service._settings.voice`` mid-stream needs a flush handshake
that is risky to land in the same change as the parser. The current
processor lays the structural foundation (tag parsing, attribution,
event emission) so the voice-routing layer can build on top of it
without re-shaping the pipeline.

The processor REPLACES the legacy ``transcript.bot_delta`` /
``transcript.bot_end`` events from ``DataChannelForwarder(name=
"forwarder.bot")`` for the LLM → TTS leg. ``forwarder.bot`` is removed
from the pipeline once this processor is in place.

Positioning: between the LLM service and the TTS service in the voice
pipeline; between the LLM service and ``TextSink`` in text mode. Sees
``LLMTextFrame`` chunks as they stream from the LLM, ``LLMFullResponseEndFrame``
as the turn closes.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_OPEN_TAG = re.compile(r'<npc\s+id="([^"]+)"\s*>')
_CLOSE_TAG = re.compile(r"</npc\s*>")

# Bound on how much un-emitted text we keep around waiting for a tag
# to close. Long enough that a verbose NPC monologue (~1 KB) doesn't
# trip it; short enough that a runaway broken `<npc id="...">` without
# a closing tag does not balloon memory.
_MAX_PENDING_CHARS = 4096

_DM_SPEAKER = "dm"


class CharacterStreamSplitter(FrameProcessor):
    """Tag-aware splitter for the LLM → TTS leg.

    Maintains a small buffer across ``LLMTextFrame`` chunks because tag
    delimiters can span chunk boundaries (``<npc id="no`` arrives, then
    ``va-tessen">…`` arrives in the next chunk). On ``LLMFullResponseEndFrame``
    the buffer is flushed and the speaker resets to the DM for the next
    turn.
    """

    def __init__(self) -> None:
        super().__init__()
        self._buffer: str = ""
        self._current_speaker: str = _DM_SPEAKER
        self._open_segment: bool = False  # True while inside an <npc> block

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            logger.trace("character_splitter: Start reached, reset state")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            await self._consume_text(frame.text)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            logger.debug(
                "character_splitter: End reached, buffer={} chars, open_segment={}",
                len(self._buffer),
                self._open_segment,
            )
            await self._flush_remaining()
            await self.push_frame(
                OutputTransportMessageFrame(message={"type": "transcript.bot_end"}),
                direction,
            )
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    def _reset(self) -> None:
        self._buffer = ""
        self._current_speaker = _DM_SPEAKER
        self._open_segment = False

    async def _consume_text(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer += chunk
        await self._drain()

    async def _drain(self) -> None:
        while True:
            if self._open_segment:
                close_match = _CLOSE_TAG.search(self._buffer)
                if close_match is None:
                    await self._emit_safe_prefix_as_current()
                    return
                segment_text = self._buffer[: close_match.start()]
                self._buffer = self._buffer[close_match.end() :]
                await self._emit_segment(self._current_speaker, segment_text)
                self._current_speaker = _DM_SPEAKER
                self._open_segment = False
                continue

            open_match = _OPEN_TAG.search(self._buffer)
            if open_match is None:
                await self._emit_safe_prefix_as_current()
                return

            narrator_text = self._buffer[: open_match.start()]
            if narrator_text:
                await self._emit_segment(_DM_SPEAKER, narrator_text)
            slug = open_match.group(1).strip()
            self._buffer = self._buffer[open_match.end() :]
            self._current_speaker = _npc_speaker(slug)
            self._open_segment = True

    async def _emit_safe_prefix_as_current(self) -> None:
        # Emit the part of the buffer that cannot possibly be part of a
        # straddling tag delimiter (``<`` or ``</``). Keep the trailing
        # candidate fragment in the buffer until the next chunk arrives
        # so we never split a tag across emit boundaries.
        if not self._buffer:
            return

        safe_end = _safe_split_index(self._buffer, opening=not self._open_segment)
        if safe_end <= 0:
            if len(self._buffer) > _MAX_PENDING_CHARS:
                # Pathological: the buffer is full of would-be tag
                # candidates with no resolution. Flush as plain text
                # and reset so the pipeline keeps moving.
                logger.warning(
                    "character splitter: oversized pending buffer ({} chars); flushing as plain text",
                    len(self._buffer),
                )
                await self._emit_segment(self._current_speaker, self._buffer)
                self._buffer = ""
            return

        prefix = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]
        await self._emit_segment(self._current_speaker, prefix)

    async def _flush_remaining(self) -> None:
        if not self._buffer:
            self._reset()
            return
        # On end-of-response we no longer wait for a closing tag — the
        # DM produced what it produced. Strip any dangling open tag
        # bytes so the TTS does not speak literal angle brackets.
        leftover = _strip_dangling_tag_fragments(self._buffer)
        if leftover:
            await self._emit_segment(self._current_speaker, leftover)
        self._reset()

    async def _emit_segment(self, speaker: str, text: str) -> None:
        if not text:
            return
        logger.trace(
            "character_splitter: emit speaker={} chars={} head={!r}",
            speaker,
            len(text),
            text[:60],
        )
        await self._push_transcript_event(speaker, text)
        await self.push_frame(LLMTextFrame(text=text), FrameDirection.DOWNSTREAM)

    async def _push_transcript_event(self, speaker: str, text: str) -> None:
        payload: dict[str, Any] = {
            "type": "transcript.segment",
            "speaker": speaker,
            "text": text,
        }
        await self.push_frame(
            OutputTransportMessageFrame(message=payload),
            FrameDirection.DOWNSTREAM,
        )


def _npc_speaker(slug: str) -> str:
    cleaned = slug.strip().lower()
    return f"npc:{cleaned}" if cleaned else _DM_SPEAKER


def _safe_split_index(buffer: str, *, opening: bool) -> int:
    """Find the largest prefix length that cannot contain a partial tag.

    When ``opening`` is True we are watching for ``<npc`` openings; we
    must keep a trailing candidate that could grow into one. When False
    we are inside a segment watching for ``</npc>``; same idea.
    """
    last_lt = buffer.rfind("<")
    if last_lt == -1:
        return len(buffer)
    candidate = buffer[last_lt:]
    if opening:
        if _OPEN_TAG.fullmatch(candidate) is not None:
            return len(buffer)
        if _is_open_tag_prefix(candidate):
            return last_lt
    else:
        if _CLOSE_TAG.fullmatch(candidate) is not None:
            return len(buffer)
        if _is_close_tag_prefix(candidate):
            return last_lt
    return len(buffer)


def _is_open_tag_prefix(text: str) -> bool:
    # Could this grow into ``<npc id="..."> ``?
    target = '<npc id="'
    if text == target[: len(text)]:
        return True
    if not text.startswith("<npc"):
        return text in {"<", "<n", "<np"}
    rest = text[len("<npc") :]
    # Past ``<npc`` we expect whitespace then ``id="…"`` then ``>``.
    return ('"' not in rest) or (not rest.endswith(">"))


def _is_close_tag_prefix(text: str) -> bool:
    target = "</npc>"
    return text == target[: len(text)]


def _strip_dangling_tag_fragments(text: str) -> str:
    # Remove any partial ``<npc id="…">`` or ``</npc>`` fragments left
    # over at end-of-response. A complete tag would already have been
    # consumed by the regex match in ``_drain``; what's left here is
    # the broken fragment.
    last_lt = text.rfind("<")
    if last_lt == -1:
        return text
    tail = text[last_lt:]
    if _is_open_tag_prefix(tail) or _is_close_tag_prefix(tail):
        return text[:last_lt]
    return text
