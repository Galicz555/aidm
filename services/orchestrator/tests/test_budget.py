"""Smoke tests for the cost guard.

Verifies:
* SessionBudgetGuard accumulates LLM + TTS metrics and trips at the cap.
* The trip pushes a TTSSpeakFrame followed by an EndTaskFrame.
* DailyBudgetLedger persists across instances and accumulates per UTC day.
* enforce_daily_budget raises 503 once the cap is reached.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.budget import (
    BudgetExceededError,
    SessionBudgetGuard,
    make_ledger,
)
from aidm.web_security import enforce_daily_budget
from fastapi import HTTPException
from pipecat.frames.frames import EndTaskFrame, MetricsFrame, TTSSpeakFrame
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection


class _Capture:
    """Replacement for the parent processor — captures pushed frames."""

    def __init__(self) -> None:
        self.pushed: list[tuple[object, FrameDirection]] = []

    async def queue_frame(self, frame: object, direction: FrameDirection) -> None:
        self.pushed.append((frame, direction))


@pytest.mark.asyncio
async def test_session_guard_trips_on_token_cap(tmp_path: Path) -> None:
    guard = SessionBudgetGuard(
        token_cap=100,
        tts_char_cap=10_000,
        locale="hu",
        ledger=None,
        session_id="test-session",
    )
    capture = _Capture()
    # The base FrameProcessor wants a parent set before push_frame works;
    # we monkey-patch push_frame to bypass the worker plumbing.
    guard.push_frame = capture.queue_frame  # type: ignore[method-assign]

    # 50 prompt + 60 completion = 110 tokens, over the 100 cap.
    metrics = MetricsFrame(
        data=[
            LLMUsageMetricsData(
                processor="llm",
                value=LLMTokenUsage(prompt_tokens=50, completion_tokens=60, total_tokens=110),
            )
        ]
    )
    await guard.process_frame(metrics, FrameDirection.DOWNSTREAM)

    kinds = [type(f).__name__ for f, _ in capture.pushed]
    assert "MetricsFrame" in kinds
    assert "TTSSpeakFrame" in kinds
    assert "EndTaskFrame" in kinds
    # The goodbye is in Hungarian.
    speak = next(f for f, _ in capture.pushed if isinstance(f, TTSSpeakFrame))
    assert "kapitány" in speak.text.lower()
    # End frame comes after the goodbye.
    end_idx = next(i for i, (f, _) in enumerate(capture.pushed) if isinstance(f, EndTaskFrame))
    speak_idx = next(i for i, (f, _) in enumerate(capture.pushed) if isinstance(f, TTSSpeakFrame))
    assert speak_idx < end_idx


@pytest.mark.asyncio
async def test_session_guard_trips_on_tts_cap() -> None:
    guard = SessionBudgetGuard(
        token_cap=1_000_000,
        tts_char_cap=100,
        locale="en",
        ledger=None,
        session_id="t",
    )
    capture = _Capture()
    guard.push_frame = capture.queue_frame  # type: ignore[method-assign]

    metrics = MetricsFrame(data=[TTSUsageMetricsData(processor="tts", value=150)])
    await guard.process_frame(metrics, FrameDirection.DOWNSTREAM)
    assert any(isinstance(f, EndTaskFrame) for f, _ in capture.pushed)
    speak = next(f for f, _ in capture.pushed if isinstance(f, TTSSpeakFrame))
    assert "chronicle" in speak.text.lower()


@pytest.mark.asyncio
async def test_session_guard_trips_only_once() -> None:
    guard = SessionBudgetGuard(
        token_cap=10, tts_char_cap=10_000, locale="hu", ledger=None, session_id="t"
    )
    capture = _Capture()
    guard.push_frame = capture.queue_frame  # type: ignore[method-assign]
    metrics = MetricsFrame(
        data=[
            LLMUsageMetricsData(
                processor="llm",
                value=LLMTokenUsage(prompt_tokens=20, completion_tokens=0, total_tokens=20),
            )
        ]
    )
    await guard.process_frame(metrics, FrameDirection.DOWNSTREAM)
    await guard.process_frame(metrics, FrameDirection.DOWNSTREAM)
    # First call: MetricsFrame + TTSSpeakFrame + EndTaskFrame = 3
    # Second call: only MetricsFrame (already tripped) = 1
    end_frames = [f for f, _ in capture.pushed if isinstance(f, EndTaskFrame)]
    assert len(end_frames) == 1


def test_daily_ledger_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "budget.db"
    a = make_ledger(db)
    a.add_spend(0.10)
    a.add_spend(0.05)
    assert a.current_spend_usd() == pytest.approx(0.15)

    # New ledger pointing at the same file: cumulative total survives.
    b = make_ledger(db)
    assert b.current_spend_usd() == pytest.approx(0.15)
    b.add_spend(0.02)
    assert b.current_spend_usd() == pytest.approx(0.17)


def test_daily_ledger_check_cap_raises(tmp_path: Path) -> None:
    db = tmp_path / "budget.db"
    ledger = make_ledger(db)
    ledger.add_spend(0.30)
    with pytest.raises(BudgetExceededError):
        ledger.check_cap(0.30)
    # Sub-cap is fine.
    ledger.check_cap(0.50)


def test_enforce_daily_budget_translates_to_503(tmp_path: Path) -> None:
    db = tmp_path / "budget.db"
    ledger = make_ledger(db)
    ledger.add_spend(1.0)
    # None / zero cap is a no-op.
    enforce_daily_budget(None, 0.5)
    enforce_daily_budget(ledger, 0.0)
    # Over cap → HTTPException(503).
    with pytest.raises(HTTPException) as exc_info:
        enforce_daily_budget(ledger, 0.5)
    assert exc_info.value.status_code == 503
    assert "költségkerete" in exc_info.value.detail
