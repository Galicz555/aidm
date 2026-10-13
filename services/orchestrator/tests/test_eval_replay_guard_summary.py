"""``eval_replay_guard.py --summary-json``: one comparable record per (model, profile)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_replay_guard.py"
_BASE_DISPATCH = "pipecat.services.openai.base_llm.BaseOpenAILLMService.get_chat_completions"

_DM1 = (
    "Rain drums on the depot roof. Lyra keeps one hand on the rusted rail "
    "and watches the tunnel mouth. Water drips somewhere below."
)
_FRESH2 = "The tunnel exhales cold air. A lantern sways far down the track, then goes dark."
_DM3 = "Marta limps out toward the pharmacy district, wounded, and does not look back."
_DIRECTIVE = "Marta leaves the shelter heading for the pharmacy district wounded."


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_replay_guard", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chunk(content: str | None = None, reasoning: str | None = None) -> ChatCompletionChunk:
    fields: dict[str, Any] = {"content": content}
    if reasoning is not None:
        fields["reasoning_content"] = reasoning  # llama.cpp extra field
    delta = ChoiceDelta.model_validate(fields)
    return ChatCompletionChunk(
        id="t",
        object="chat.completion.chunk",
        created=0,
        model="test",
        choices=[Choice(index=0, delta=delta, finish_reason=None)],
    )


def _replay_dir(tmp_path: Path) -> Path:
    sheets = f"<!--aidm:scene-sheets-->\n[World events this turn]\n- {_DIRECTIVE}\n"
    turns: list[dict[str, Any]] = [
        {"turn": 1, "player": "I wait.", "dm": _DM1},
        {"turn": 2, "player": "I listen.", "dm": _DM1},
        {"turn": 3, "player": "I look for Marta.", "dm": _DM3},
    ]
    history: list[dict[str, str]] = [{"role": "system", "text": "persona"}]
    events = []
    for t in turns:
        extra = [{"role": "system", "text": sheets}] if t["turn"] == 3 else []
        messages = [*history, *extra, {"role": "user", "text": t["player"]}]
        events.append({"type": "debug.context", "messages": messages})
        history += [{"role": "user", "text": t["player"]}, {"role": "assistant", "text": t["dm"]}]
    replay = tmp_path / "replay-001"
    replay.mkdir()
    (replay / "transcript.json").write_text(json.dumps(turns), encoding="utf-8")
    (replay / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return replay


def _scripted(replies: list[list[ChatCompletionChunk]]) -> Any:
    calls: list[int] = []

    async def _dispatch(self: object, context: Any) -> AsyncIterator[ChatCompletionChunk]:
        calls.append(1)
        chunks = replies[len(calls) - 1]

        async def _gen() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in chunks:
                yield chunk

        return _gen()

    return _dispatch


@pytest.mark.asyncio
async def test_summary_json_records_guard_activity_latency_and_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AIDM_MODE", "private")
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "c"))
    monkeypatch.setenv("AIDM_PROFILE", "qwen3-14b-thinking")
    monkeypatch.setenv("AIDM_DM_THINKING", "1")
    for key in ("AIDM_GUARD_TICS", "AIDM_GUARD_PARAGRAPH", "AIDM_GUARD_REPLAY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AIDM_GUARD_DIRECTIVES", "0")  # measured, not enforced
    replay = _replay_dir(tmp_path)
    out = tmp_path / "summary.json"
    replies = [
        [_chunk(reasoning="a"), _chunk(reasoning="b"), _chunk(content=_DM1)],  # turn 1
        [_chunk(reasoning="c"), _chunk(content=_DM1)],  # turn 2 draft: replays turn 1
        [_chunk(content=_FRESH2)],  # turn 2 retry clears it
        [_chunk(content="<think>x</think> The rain stops.")],  # turn 3: miss + leak
    ]
    argv = ["x", str(replay), "--all", "--live-guards", "--summary-json", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with patch(_BASE_DISPATCH, _scripted(replies)):
        await _module().main()

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["profile"] == "qwen3-14b-thinking"
    assert summary["dm_thinking"] is True
    assert summary["guards"]["directives"] is False
    assert summary["turns"] == 3
    assert (summary["replay_drafts"], summary["replay_guard_fired"]) == (1, 1)
    assert summary["replay_guard_cleared"] == 1
    assert summary["retries"] == 1
    assert summary["directive_misses"] == 1  # guard off: counted, not retried
    assert summary["reasoning_leaks"] == 1
    assert summary["reasoning_tokens_mean"] == pytest.approx(3 / 4)
    for key in ("gen_seconds_mean", "gen_seconds_median", "gen_seconds_p90"):
        assert summary[key] >= 0.0
    assert {"residual_banned", "residual_paragraph", "residual_8gram_turns", "excisions"} <= set(
        summary
    )
