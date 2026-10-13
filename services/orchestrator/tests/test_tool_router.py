"""Tests for the per-turn tool-subset selection.

The runtime ``ToolSubsetSelector`` frame processor pushes the chosen
subset onto ``LLMContext.set_tools`` whenever an ``LLMContextFrame``
flows downstream. The phase is decided by:

- ``end_session`` flag (flipped by the HTTP /end handler) → end-session
  schema (only ``vault_write_session_log``).
- character-sheet presence on the active vault run → session-zero or
  play schema.
"""

from __future__ import annotations

import pytest
from aidm.tool_router import ToolSubsetSelector
from aidm.vault_tools import (
    _build_tools_schema,
    default_tools_schema,
    end_session_tools_schema,
)

# --- schema helpers ---


def test_default_subset_drops_session_log() -> None:
    full = _build_tools_schema()
    default = default_tools_schema()
    full_names = {f.name for f in full.standard_tools}
    default_names = {f.name for f in default.standard_tools}
    assert "vault_write_session_log" in full_names
    assert "vault_write_session_log" not in default_names
    assert default_names == full_names - {"vault_write_session_log"}


def test_end_session_subset_is_only_session_log() -> None:
    schema = end_session_tools_schema()
    names = {f.name for f in schema.standard_tools}
    assert names == {"vault_write_session_log"}


# --- ToolSubsetSelector frame processor ---


class _Ctx:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = messages
        self._tools: object = None
        self.tool_choice: object = None

    def set_tools(self, tools: object) -> None:
        self._tools = tools

    def set_tool_choice(self, choice: object) -> None:
        self.tool_choice = choice


@pytest.mark.asyncio
async def test_selector_default_play_subset_on_normal_turn() -> None:
    from pipecat.frames.frames import LLMContextFrame  # noqa: PLC0415
    from pipecat.processors.frame_processor import FrameDirection  # noqa: PLC0415

    selector = ToolSubsetSelector()

    async def _capture(_f: object, _d: object = None) -> None:
        return None

    selector.push_frame = _capture  # type: ignore[method-assign]
    ctx = _Ctx([{"role": "user", "content": "Belépek a kocsmába."}])
    await selector.process_frame(
        LLMContextFrame(context=ctx),  # type: ignore[arg-type]
        FrameDirection.DOWNSTREAM,
    )
    assert ctx._tools is not None
    names = {f.name for f in ctx._tools.standard_tools}  # type: ignore[attr-defined]
    assert "vault_write_session_log" not in names
    assert len(names) > 1


@pytest.mark.asyncio
async def test_selector_end_session_flag_shrinks_to_session_log() -> None:
    from pipecat.frames.frames import LLMContextFrame  # noqa: PLC0415
    from pipecat.processors.frame_processor import FrameDirection  # noqa: PLC0415

    selector = ToolSubsetSelector()
    selector.end_session = True

    async def _capture(_f: object, _d: object = None) -> None:
        return None

    selector.push_frame = _capture  # type: ignore[method-assign]
    ctx = _Ctx([{"role": "user", "content": "anything"}])
    await selector.process_frame(
        LLMContextFrame(context=ctx),  # type: ignore[arg-type]
        FrameDirection.DOWNSTREAM,
    )
    names = {f.name for f in ctx._tools.standard_tools}  # type: ignore[attr-defined]
    assert names == {"vault_write_session_log"}
    # The log call is forced: offered but optional, the model narrated on
    # instead (2026-09-27 playtest 3 wrote no session log).
    assert ctx.tool_choice == "required"
