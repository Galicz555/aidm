"""Show a story-clock advance (and any Spell line) at the start of the next DM reply.

A pending ``DirectorState.spell_line`` (aidm.first_nightmare: the trial
greeting or the appraisal) is shown the same way, once, ahead of the time
line.

The local 14B does not reliably narrate time passing (2026-09-26
marathon: 1 of ~20 transitions narrated), so code shows it: when
``DirectorState.clock_line`` has moved since the last reply, one short
line ("*Day 2, morning.*") is pushed as an ``LLMTextFrame`` right before
the reply's first text. Downstream it is ordinary DM text: the
``CharacterStreamSplitter`` attributes it to the narrator (text: SSE
``transcript.segment``; voice: TTS), and the assistant aggregator folds
it into the same assistant message, so the DM sees time moved too.

Positioning: directly after the LLM service in both pipelines. The
private-mode guarded retry buffers drafts *inside* the LLM service, so
only the released reply's frames reach this processor — the line is
never part of a draft and never duplicated across a regenerate.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from aidm.story_events import time_shown_line
from aidm.turn_director import DirectorState


class TimeShownProcessor(FrameProcessor):
    """Prefix the reply after a clock advance with the shown-time line."""

    def __init__(self, director: DirectorState, locale: str, *, initial_clock_line: str) -> None:
        super().__init__()
        self._director = director
        self._locale = locale
        # Last clock line the player has been shown (or started from).
        # Empty means no baseline yet: the first line seen is adopted
        # silently, since it is the clock's position, not an advance.
        self._shown = initial_clock_line
        # Set on response start, cleared at its first text: the line only
        # ever opens a reply. A tool-only leg has no text, so the line
        # rides the leg that actually speaks.
        self._armed = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._armed = True
            elif isinstance(frame, LLMTextFrame) and frame.text and self._armed:
                self._armed = False
                spell = self._take_spell_line()
                if spell:
                    logger.info("spell: shown {} chars", len(spell))
                    await self.push_frame(LLMTextFrame(text=f"{spell}\n\n"), direction)
                line = self._take_advance()
                if line:
                    logger.info('time: shown "{}"', line)
                    await self.push_frame(LLMTextFrame(text=f"{line}\n\n"), direction)
        await self.push_frame(frame, direction)

    def snapshot(self) -> object:
        return self._shown

    def restore(self, state: object) -> None:
        self._shown = str(state)
        self._armed = False

    def _take_spell_line(self) -> str:
        line, self._director.spell_line = self._director.spell_line, ""
        return line

    def _take_advance(self) -> str:
        current = self._director.clock_line
        if not current or current == self._shown:
            return ""
        previous, self._shown = self._shown, current
        return time_shown_line(current, self._locale) if previous else ""
