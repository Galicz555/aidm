"""End-of-turn pruning of stale conversation history.

The scene-sheet writer keeps NPC/PC continuity bounded on disk. But the
raw turn-by-turn dialogue still grows unbounded in
``LLMContext.messages``; long sessions begin to recycle their own past
prose ("the shelter hums around you, warm, steady, *alive*" appears N
times across M turns — verified empirically on the previous stack).

This processor drops old user→assistant(→tool*) clusters when the
session exceeds ``max_recent_turns`` user turns, replacing the dropped
span with one synthetic system note. Leading system messages (persona,
resume preamble, scene-sheets injection slot) are always preserved.

Pipeline position: after ``aggregator.assistant()`` so the
just-finalized assistant turn is in ``context.messages``. Triggered by
``LLMContextAssistantTimestampFrame``.

Extracted from the retired ``NPCStateHistoryStabilizer`` — only the
``_maybe_trim_old_turns`` half survived the move to sheet-driven state.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextAssistantTimestampFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_TRIM_TAG = "<!--aidm:history-trim-->"
# Chars-per-token for the prompt estimate: measured 3.8-4.7 on real DM
# contexts and 3.8 on the tool schema (Qwen3.8 tokenizer, 2026-09-27); 3.5
# over-counted the ~11k fixed tokens enough to keep only 2-3 long turns.
_CHARS_PER_TOKEN = 4.0
_MIN_KEPT_TURNS = 2


class HistoryTrimmer(FrameProcessor):
    """Trim ``context.messages`` once user-turn count exceeds the threshold."""

    def __init__(
        self,
        context: Any,
        *,
        max_recent_turns: int = 12,
        token_budget: int = 0,
        fixed_chars: int = 0,
    ) -> None:
        """``token_budget`` > 0 also trims by estimated prompt size.

        The turn count alone let a long-reply model overflow the 20k window
        (2026-09-27: 20534 tokens at T12). ``fixed_chars`` is what the
        request carries outside ``context.messages`` — the persona
        (system_instruction, ~9k tokens) and the tool schema (~2k) — which
        a messages-only estimate misses.
        """
        super().__init__()
        self._context = context
        self._max_recent_turns = max_recent_turns
        self._token_budget = token_budget
        self._fixed_chars = fixed_chars

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            return
        if not isinstance(frame, LLMContextAssistantTimestampFrame):
            return

        self._maybe_trim(self._max_recent_turns)
        keep = self._max_recent_turns
        while self._over_budget() and keep > _MIN_KEPT_TURNS:
            keep = min(keep, self._user_turns()) - 1
            self._maybe_trim(keep)

    def _user_turns(self) -> int:
        messages = getattr(self._context, "messages", None) or []
        return sum(1 for m in messages if _role_of(m) == "user")

    def _over_budget(self) -> bool:
        if self._token_budget <= 0:
            return False
        messages = getattr(self._context, "messages", None) or []
        chars = self._fixed_chars + sum(len(_content_text(m)) for m in messages)
        return chars / _CHARS_PER_TOKEN > self._token_budget

    def _maybe_trim(self, keep: int) -> None:
        messages = getattr(self._context, "messages", None)
        if not isinstance(messages, list):
            return
        user_indices = [i for i, m in enumerate(messages) if _role_of(m) == "user"]
        if len(user_indices) <= keep:
            return
        first_keep_idx = user_indices[-keep]
        # Preserve the leading system block. Scan from the start until
        # the first non-system message; everything from there up to
        # first_keep_idx is the trim target.
        system_block_end = 0
        for i, m in enumerate(messages):
            if _role_of(m) != "system":
                system_block_end = i
                break
        else:
            return  # all-system context, nothing to trim
        if first_keep_idx <= system_block_end:
            return

        # If a prior trim already inserted a stub at the system-block edge,
        # absorb its count and overwrite it — one rolling stub, not many.
        prior_dropped = 0
        if system_block_end > 0:
            existing = messages[system_block_end - 1]
            existing_text = _content_text(existing)
            if existing_text.startswith(_TRIM_TAG):
                prior_dropped = _parse_dropped(existing_text)
                # Drop the prior stub and shift the window back one slot.
                del messages[system_block_end - 1]
                system_block_end -= 1
                first_keep_idx -= 1

        dropped_now = first_keep_idx - system_block_end
        if dropped_now <= 0 and prior_dropped == 0:
            return
        total_dropped = prior_dropped + dropped_now
        placeholder = {
            "role": "system",
            "content": (
                f"{_TRIM_TAG}\n"
                f"[Earlier session — {total_dropped} message(s) trimmed to keep "
                "context focused on the recent scene. The scene-sheet block "
                "above carries the up-to-date NPC and PC state; the older "
                "turns are absorbed into each sheet's ## Session so far.]"
            ),
        }
        if dropped_now > 0:
            del messages[system_block_end:first_keep_idx]
        messages.insert(system_block_end, placeholder)
        logger.info(
            "history_trimmer: trimmed {} stale message(s) (cumulative {}), kept {} recent user turns",
            dropped_now,
            total_dropped,
            keep,
        )


def _role_of(message: Any) -> str:
    raw = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
    return str(raw or "").lower()


def _content_text(message: Any) -> str:
    raw = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


_DROPPED_RE = re.compile(r"Earlier session — (\d+) message")


def _parse_dropped(text: str) -> int:
    """Pull the integer count out of a prior trim stub. Tolerates missing/garbled."""
    match = _DROPPED_RE.search(text)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0
