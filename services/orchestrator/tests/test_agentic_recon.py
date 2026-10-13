"""Agentic-recon gate: forced tool_choice when the player names unknown canon.

The injector arms ``tool_choice="required"`` only when the latest user
message carries a mid-sentence proper noun whose slug isn't one of
THIS turn's actually-injected identity sheets; the local provider
consumes the flag after one dispatch so the post-tool narration
completion runs on auto.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from aidm.providers.llm_local import _LoggingLocalLLMService
from aidm.scene_sheet_injector import _set_recon_tool_choice
from openai import NOT_GIVEN


class _FakeContext:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.tool_choice: Any = NOT_GIVEN

    def set_tool_choice(self, choice: Any) -> None:
        self.tool_choice = choice


def _ctx(user_text: str) -> _FakeContext:
    messages: list[Any] = [{"role": "system", "content": "persona"}]
    messages.append({"role": "user", "content": user_text})
    return _FakeContext(messages)


_LYRA_SHEET: list[tuple[str, str, str]] = [("npc", "lyra", "Lyra, wary.")]


def test_unfamiliar_proper_noun_forces_required() -> None:
    ctx = _ctx("I ask her about the Crimson Hand and wait.")
    _set_recon_tool_choice(ctx, _LYRA_SHEET)
    assert ctx.tool_choice == "required"


def test_sheet_covered_slug_stays_auto() -> None:
    ctx = _ctx("I look straight at Lyra and nod.")
    _set_recon_tool_choice(ctx, _LYRA_SHEET)
    assert ctx.tool_choice is NOT_GIVEN


def test_name_only_mentioned_in_another_sheets_prose_still_forces_required() -> None:
    # Lyra's own sheet text namedrops Dora in a Bonds section — Dora has
    # no sheet of her own in scene, so the gate must still fire even
    # though "dora" appears inside the rendered sheets block.
    sheets = [("npc", "lyra", "Lyra, wary.\n\n## Bonds\n\n- [[dora]] — an old debt.")]
    ctx = _ctx("I ask Lyra whether Dora already knows.")
    _set_recon_tool_choice(ctx, sheets)
    assert ctx.tool_choice == "required"


def test_stale_required_is_reset_on_quiet_turn() -> None:
    ctx = _ctx("I sit down and wait quietly.")
    ctx.tool_choice = "required"  # leaked from a previous turn
    _set_recon_tool_choice(ctx, _LYRA_SHEET)
    assert ctx.tool_choice is NOT_GIVEN


@pytest.mark.asyncio
async def test_provider_consumes_required_after_one_dispatch() -> None:
    with patch(
        "aidm.providers.llm_local._LoggingLocalLLMService.create_client",
        return_value=None,
    ):
        svc = _LoggingLocalLLMService.__new__(_LoggingLocalLLMService)
    ctx = _FakeContext([{"role": "user", "content": "hi"}])
    ctx.tool_choice = "required"

    with patch(
        "pipecat.services.openai.llm.OpenAILLMService.get_chat_completions",
        return_value="stream-handle",
    ):
        stream = await svc._dispatch_logged(ctx)  # type: ignore[arg-type]

    assert stream == "stream-handle"
    assert ctx.tool_choice is NOT_GIVEN
