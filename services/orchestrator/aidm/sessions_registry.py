"""Shared session-registry primitives for browser-mode transports.

Both ``aidm.server`` (WebRTC) and ``aidm.text_server`` (text) hold a
short grace window after a client disconnects so a fast reconnect (Wi-Fi
handoff, tab refresh, reopened browser) can resume the same ``LLMContext``
instead of starting a brand new session. The mechanism and the grace
budget are identical for both transports; this module owns the shared
state structures and the sweeper task so the two server modules don't
copy-paste it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from aidm.session import SessionContext

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext


# Phrases that match the persona prompt's "the player ends the session"
# triggers in both locales. The end-session endpoints push one of these
# as a synthetic TranscriptionFrame so the persona's existing
# end-of-session logic fires (write the session log, say a brief
# farewell) — no parallel "end of session" code path in the persona to
# maintain.
END_PHRASES: dict[str, str] = {
    "hu": "Köszi, vége mára.",
    "en": "Thanks, we're done for today.",
}


# How long the LLMContext lingers after client disconnect before we
# consider the session lost. Picked per docs/session-lifecycle.md:
# Wi-Fi handoffs typically resolve under 3s; 5s covers the common case
# without holding context indefinitely under bad network. The sweeper
# checks ~once per second.
RECONNECT_GRACE_SECONDS: float = 5.0
SWEEPER_INTERVAL: float = 1.0


@dataclass
class PendingContext:
    """LLMContext stashed after client disconnect, eligible for resume."""

    context: LLMContext
    session: SessionContext
    deadline: float  # monotonic clock


async def sweep_pending(pending: dict[str, PendingContext]) -> None:
    """Periodically purge expired PendingContext entries.

    Keeps the dict bounded under a steady stream of failed reconnects
    and prevents memory creep across days. Cancelled on shutdown.
    """
    while True:
        try:
            await asyncio.sleep(SWEEPER_INTERVAL)
        except asyncio.CancelledError:
            break
        now = time.monotonic()
        expired = [sid for sid, p in pending.items() if p.deadline <= now]
        for sid in expired:
            pending.pop(sid, None)
            logger.debug("reconnect grace expired: session_id={}", sid)


def is_pending(session_id: str | None, pending: dict[str, PendingContext]) -> bool:
    """True when ``session_id`` has an unexpired stash (non-consuming peek)."""
    entry = pending.get(session_id) if isinstance(session_id, str) else None
    return entry is not None and entry.deadline > time.monotonic()


def claim_pending(
    session_id: str | None, pending: dict[str, PendingContext]
) -> PendingContext | None:
    """Pop and return the PendingContext matching ``session_id``.

    Returns None when no session_id is supplied, the entry is missing,
    or the deadline has passed (stale entries are purged here too —
    saves the sweeper one tick).
    """
    if not isinstance(session_id, str) or not session_id:
        return None
    entry = pending.pop(session_id, None)
    if entry is None:
        return None
    if entry.deadline <= time.monotonic():
        return None
    return entry


def stash_pending(
    pending: dict[str, PendingContext],
    session: SessionContext,
    context: LLMContext,
) -> None:
    """Park the context under its session id with a fresh deadline."""
    pending[session.session_id] = PendingContext(
        context=context,
        session=session,
        deadline=time.monotonic() + RECONNECT_GRACE_SECONDS,
    )
