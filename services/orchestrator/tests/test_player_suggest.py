"""Tests for the sheet-driven player-action suggestion endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aidm.player_suggest import SuggestionError, suggest_player_action

_PC_SHEET = """---
type: character
lang: en
name: Rowan
class_role: software engineer
hexaco_honesty_humility: 4
hexaco_emotionality: 3
hexaco_extraversion: 2
hexaco_agreeableness: 5
hexaco_conscientiousness: 4
hexaco_openness: 3
hp_current: [8]
inventory: [knife, curved blade]
---

# Rowan

Rowan is a former military engineer.

## Recent (last 5 events)
- T1: arrived at shelter.

## Session so far

Rowan stands at the shelter door.
"""


class _FakeVault:
    def __init__(self, files: dict[str, str]) -> None:
        self._files = files

    def read_file(self, path: str) -> str:
        return self._files[path]


class _FakeRegistry:
    def __init__(self, pc: str, npcs: list[str]) -> None:
        self.current_pc_slug = pc
        self._npcs = npcs

    def active(self) -> list[str]:
        return list(self._npcs)


def _make_config(mode: str = "private") -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        local_llm_base_url="http://localhost:8080/v1",
        local_llm_model="qwen3-14b",
        openrouter_api_key="x",
        openrouter_model="x",
        npc_failsafe_timeout_s=30.0,
    )


def _completion(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.mark.asyncio
async def test_suggest_includes_sheet_and_history_and_returns_text() -> None:
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _completion('  "I draw my knife and step back."  ')

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    with patch("aidm.player_suggest.AsyncOpenAI", return_value=client):
        vault = _FakeVault({"characters/rowan.md": _PC_SHEET})
        registry = _FakeRegistry(pc="rowan", npcs=[])
        context = SimpleNamespace(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "The door creaks open."},
            ]
        )
        text = await suggest_player_action(_make_config(), vault, registry, context, "en")

    assert text == "I draw my knife and step back."
    msg = captured["messages"][0]["content"]  # type: ignore[index]
    assert "Rowan" in msg
    assert "knife, curved blade" in msg  # PC sheet inventory included
    assert "The door creaks open." in msg  # last DM beat included
    assert "Player: hi" in msg


@pytest.mark.asyncio
async def test_suggest_rejects_when_no_pc() -> None:
    registry = _FakeRegistry(pc="", npcs=[])
    with pytest.raises(SuggestionError) as exc:
        await suggest_player_action(
            _make_config(), _FakeVault({}), registry, SimpleNamespace(messages=[]), "en"
        )
    assert exc.value.status == 409


@pytest.mark.asyncio
async def test_suggest_translates_llm_error_to_504() -> None:
    create = AsyncMock(side_effect=RuntimeError("boom"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch("aidm.player_suggest.AsyncOpenAI", return_value=client):
        registry = _FakeRegistry(pc="rowan", npcs=[])
        vault = _FakeVault({"characters/rowan.md": _PC_SHEET})
        with pytest.raises(SuggestionError) as exc:
            await suggest_player_action(
                _make_config(), vault, registry, SimpleNamespace(messages=[]), "en"
            )
    assert exc.value.status == 504
