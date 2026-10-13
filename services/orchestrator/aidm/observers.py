"""Pipecat observer wiring.

Two observers carry their weight in Phase 1:

- TurnTrackingObserver (auto-created by PipelineWorker when
  enable_turn_tracking=True) — we subscribe to its events to bump the
  per-turn logging context.
- UserBotLatencyObserver — measures end-to-end mic→speaker latency and
  emits per-turn breakdowns. We log them in a structured form so the
  Phase 1 latency-poc check fills itself in.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    UserBotLatencyObserver,
)

from aidm.log import current_turn, set_turn


def build_latency_observer() -> UserBotLatencyObserver:
    observer = UserBotLatencyObserver()  # type: ignore[no-untyped-call]

    @observer.event_handler("on_latency_measured")  # type: ignore[untyped-decorator]
    async def _on_latency(_: UserBotLatencyObserver, latency: float) -> None:
        logger.info("turn latency: {:.0f}ms (mic→speaker)", latency * 1000)

    @observer.event_handler("on_latency_breakdown")  # type: ignore[untyped-decorator]
    async def _on_breakdown(_: UserBotLatencyObserver, breakdown: LatencyBreakdown) -> None:
        events = breakdown.chronological_events()
        if not events:
            return
        logger.info("latency breakdown: {}", " | ".join(events))

    return observer


def attach_turn_logging(turn_observer: Any) -> None:
    """Wire the auto-created turn observer to our log context."""

    @turn_observer.event_handler("on_turn_started")  # type: ignore[untyped-decorator]
    async def _on_start(_: Any, turn_count: int) -> None:
        set_turn(turn_count)
        logger.info("turn started")

    @turn_observer.event_handler("on_turn_ended")  # type: ignore[untyped-decorator]
    async def _on_end(_: Any, turn_count: int, duration: float, was_interrupted: bool) -> None:
        suffix = " (interrupted)" if was_interrupted else ""
        logger.info("turn ended: {:.1f}s{}", duration, suffix)
        # Leave the contextvar at the ended turn so post-turn logs still
        # carry the right id; the next on_turn_started will bump it.
        _ = current_turn()
