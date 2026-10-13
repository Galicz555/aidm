"""judge_scene_ends: the logprob cut, and fail-open."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aidm.advance_judge import judge_scene_ends


def _client(top: dict[str, float] | Exception) -> Any:
    create = AsyncMock()
    if isinstance(top, Exception):
        create.side_effect = top
    else:
        tokens = [SimpleNamespace(token=t, logprob=math.log(p)) for t, p in top.items()]
        choice = SimpleNamespace(
            logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=tokens)])
        )
        create.return_value = SimpleNamespace(choices=[choice])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def _judge(client: Any, current: str = "You cross the bridge.") -> bool | None:
    return await judge_scene_ends(client, "m", player="Let's go.", current=current)


@pytest.mark.asyncio
async def test_yes_share_above_the_cut_is_a_scene_end() -> None:
    # The model rarely says YES outright: 25% YES mass already counts.
    assert await _judge(_client({"NO": 0.7, "YES": 0.2, "Yes": 0.05})) is True


@pytest.mark.asyncio
async def test_yes_share_below_the_cut_is_not() -> None:
    assert await _judge(_client({"NO": 0.9, "YES": 0.1})) is False


@pytest.mark.asyncio
async def test_no_verdict_on_error_empty_reply_or_foreign_tokens() -> None:
    assert await _judge(_client(RuntimeError("down"))) is None
    assert await _judge(_client({"NO": 1.0}), current="  ") is None
    assert await _judge(_client({"Maybe": 1.0})) is None
