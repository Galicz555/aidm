"""Per-turn tool-subset selector.

Pipecat's LLM adapter reads the tool schema off ``LLMContext`` every
invocation. We exploit that to swap a smaller subset in for most
turns and only widen to the full catalog when the player's message
looks like it warrants a heavier tool (session-end signals, mainly).

This is context-engineering, not prompt-engineering: the prompt is
identical, the *menu of tools* the model sees in the request body
shrinks. On a 4096 ctx that saves the ~280-token
``vault_write_session_log`` description on every non-end turn — a
meaningful chunk of the budget when the persona + recent history
already crowd the window.

Positioning: insert between ``user_agg`` and the LLM service.
``user_agg`` emits an ``LLMContextFrame`` after appending the new
user message; we read that frame, look at the latest user content,
and call ``context.set_tools`` with the chosen subset. The frame
continues downstream unchanged, and the LLM service picks up the
narrowed tool list on its very next chat-completion call.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from aidm.vault import Vault
from aidm.vault_tools import (
    default_tools_schema,
    end_session_tools_schema,
    has_character_sheet,
    session_zero_tools_schema,
)


class ToolSubsetSelector(FrameProcessor):
    """Set ``LLMContext.tools`` per turn based on the current session phase.

    Transparent: every frame still flows downstream. The selector only
    inspects ``LLMContextFrame``s (emitted by the user aggregator) and
    mutates the carried context's tools list in place.

    Phases:
    - ``session-zero`` — no character sheet yet on the active run.
      Interviewer-only tools.
    - ``end-session`` — the HTTP /end endpoint flipped ``end_session``
      to ``True``. Only ``vault_write_session_log`` is on the menu.
    - ``play`` — default. Every tool except ``vault_write_session_log``.

    ``end_session`` is a public mutable attribute: the End-session HTTP
    handler sets it before injecting the synthetic closing utterance,
    so the very next ``LLMContextFrame`` carries the end-session
    schema.
    """

    def __init__(self, vault: Vault | None = None) -> None:
        super().__init__()
        self._vault = vault
        self.end_session: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            return
        if not isinstance(frame, LLMContextFrame):
            return

        if self.end_session:
            subset = end_session_tools_schema()
            phase = "end-session"
        elif self._vault is not None and not has_character_sheet(self._vault):
            subset = session_zero_tools_schema()
            phase = "session-zero"
        else:
            subset = default_tools_schema()
            phase = "play"
        frame.context.set_tools(subset)
        if self.end_session:
            # Force the session-log call: offered but optional, the model
            # narrated on and never wrote the log (2026-09-27 playtest 3).
            # llm_local resets tool_choice after one dispatch, so the
            # farewell after the tool result streams normally.
            frame.context.set_tool_choice("required")
        logger.debug(
            "tool subset ({}): {} tools",
            phase,
            len(subset.standard_tools),
        )
