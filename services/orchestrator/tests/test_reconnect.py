"""Reconnect-grace registry behaviour.

The grace window stashes a session's LLMContext after client disconnect
and lets a quick reconnect (same session_id, within 5s) resume the
conversation. See ``docs/session-lifecycle.md`` for the design.

We unit-test the helpers in ``aidm.sessions_registry`` directly rather
than spinning up a full WebRTC peer. The plumbing around them
(FastAPI lifespan, SmallWebRTCRequestHandler) is exercised by the smoke
tests; here we verify the registry rules in isolation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest
from aidm import sessions_registry
from aidm.session import session_for_locale
from aidm.sessions_registry import (
    PendingContext,
    claim_pending,
    stash_pending,
    sweep_pending,
)


def _pending(*, session_id: str = "abc12345", grace: float = 5.0) -> PendingContext:
    session = session_for_locale("hu", session_id=session_id)
    return PendingContext(
        # The actual LLMContext type does not affect the registry logic —
        # we only need an opaque object that round-trips through pop().
        context=cast(Any, object()),
        session=session,
        deadline=time.monotonic() + grace,
    )


def test_claim_pending_returns_entry_for_known_session() -> None:
    pending = {"abc12345": _pending(session_id="abc12345")}
    claimed = claim_pending("abc12345", pending)
    assert claimed is not None
    assert claimed.session.session_id == "abc12345"
    # The entry is consumed — a second reconnect attempt for the same
    # session_id finds nothing.
    assert "abc12345" not in pending


def test_claim_pending_returns_none_for_unknown_session() -> None:
    pending: dict[str, PendingContext] = {}
    assert claim_pending("no-such", pending) is None


def test_claim_pending_returns_none_without_session_id() -> None:
    pending = {"abc12345": _pending(session_id="abc12345")}
    assert claim_pending(None, pending) is None
    # The entry stayed put — we only consume on a real claim.
    assert "abc12345" in pending


def test_claim_pending_returns_none_for_empty_string() -> None:
    assert claim_pending("", {"abc12345": _pending()}) is None


def test_claim_pending_returns_none_for_expired_entry() -> None:
    expired = PendingContext(
        context=cast(Any, object()),
        session=session_for_locale("hu", session_id="abc12345"),
        deadline=time.monotonic() - 1.0,
    )
    pending = {"abc12345": expired}
    assert claim_pending("abc12345", pending) is None


def test_stash_pending_writes_entry_with_grace_deadline() -> None:
    pending: dict[str, PendingContext] = {}
    session = session_for_locale("hu", session_id="ghi98765")
    context = cast(Any, object())
    before = time.monotonic()
    stash_pending(pending, session, context)
    after = time.monotonic()
    entry = pending["ghi98765"]
    # Deadline lives in [now + grace, after + grace] — bracket-checking
    # avoids brittleness around the monotonic clock resolution.
    grace = sessions_registry.RECONNECT_GRACE_SECONDS
    assert before + grace <= entry.deadline <= after + grace + 1e-3
    assert entry.context is context


@pytest.mark.asyncio
async def test_sweep_pending_purges_expired_entries() -> None:
    # Two entries: one already expired, one fresh enough to survive a
    # single sweeper tick. The sweeper sleeps roughly 1s between
    # passes, so we monkey-patch the interval down for the test.
    pending: dict[str, PendingContext] = {
        "stale": PendingContext(
            context=cast(Any, object()),
            session=session_for_locale("hu", session_id="stale"),
            deadline=time.monotonic() - 1.0,
        ),
        "fresh": _pending(session_id="fresh", grace=10.0),
    }

    original_interval = sessions_registry.SWEEPER_INTERVAL
    sessions_registry.SWEEPER_INTERVAL = 0.01
    sweeper = asyncio.create_task(sweep_pending(pending))
    try:
        # Give the sweeper a couple of ticks to find and remove the
        # stale entry without racing it.
        await asyncio.sleep(0.05)
    finally:
        sweeper.cancel()
        await asyncio.gather(sweeper, return_exceptions=True)
        sessions_registry.SWEEPER_INTERVAL = original_interval

    assert "stale" not in pending
    assert "fresh" in pending


def test_is_pending_peeks_without_consuming() -> None:
    pending = {"abc12345": _pending(session_id="abc12345")}
    assert sessions_registry.is_pending("abc12345", pending)
    assert "abc12345" in pending
    assert not sessions_registry.is_pending("no-such", pending)
    assert not sessions_registry.is_pending(None, pending)
    expired = {"old": _pending(session_id="old", grace=-1.0)}
    assert not sessions_registry.is_pending("old", expired)


class _FakeTransport:
    handlers: dict[str, Any]

    def __init__(self, **_kwargs: object) -> None:
        _FakeTransport.handlers = {}

    def event_handler(self, name: str) -> Any:
        def register(fn: Any) -> Any:
            _FakeTransport.handlers[name] = fn
            return fn

        return register


class _FakeWorker:
    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.ended_by = ""

    async def cancel(self, *, reason: str | None = None) -> None:
        self.ended_by = f"cancel:{reason}"
        self.done.set()

    async def stop_when_done(self) -> None:
        self.ended_by = "stop_when_done"
        self.done.set()


async def _spawn_fake_webrtc_session(
    monkeypatch: pytest.MonkeyPatch, *, ending: bool
) -> tuple[Any, _FakeWorker, dict[str, PendingContext]]:
    from pathlib import Path  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from aidm import server  # noqa: PLC0415
    from aidm.tool_router import ToolSubsetSelector  # noqa: PLC0415

    selector = ToolSubsetSelector()
    selector.end_session = ending
    worker = _FakeWorker()
    handle = SimpleNamespace(
        worker=worker,
        user_agg=None,
        context=object(),
        tool_selector=selector,
        budget_guard=None,
    )

    async def fake_run_worker(h: Any, _session: object) -> None:
        await h.worker.done.wait()

    monkeypatch.setattr(server, "SmallWebRTCTransport", _FakeTransport)
    monkeypatch.setattr(server, "build_pipeline", lambda *_a, **_k: handle)
    monkeypatch.setattr(server, "run_worker", fake_run_worker)
    monkeypatch.setattr(server, "build_session_vault", lambda *_a: None)
    monkeypatch.setattr(server, "read_run_title", lambda _v: "")
    monkeypatch.setattr(server, "read_campaign_title", lambda _v: "")
    pending: dict[str, PendingContext] = {}
    app = SimpleNamespace(
        state=SimpleNamespace(
            config=SimpleNamespace(locale="en", vault_path=Path("/v/camp"), run_slug="default"),
            http_session=None,
            pending_contexts=pending,
            budget_ledger=None,
            session_endpoints={},
            pipeline_tasks=set(),
        )
    )
    await server._spawn_pipeline(
        cast(Any, app), cast(Any, SimpleNamespace(pc_id="pc1")), {"campaign_id": "camp"}
    )
    return app, worker, pending


@pytest.mark.asyncio
async def test_peer_disconnect_cancels_pipeline_and_stashes_for_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, worker, pending = await _spawn_fake_webrtc_session(monkeypatch, ending=False)
    await _FakeTransport.handlers["on_client_disconnected"](None, None)
    await asyncio.gather(*app.state.pipeline_tasks)

    assert worker.ended_by == "cancel:peer disconnected"
    assert len(pending) == 1  # reconnect within the grace resumes this context


@pytest.mark.asyncio
async def test_peer_disconnect_after_end_finishes_farewell_and_is_not_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, worker, pending = await _spawn_fake_webrtc_session(monkeypatch, ending=True)
    await _FakeTransport.handlers["on_client_disconnected"](None, None)
    await asyncio.gather(*app.state.pipeline_tasks)

    assert worker.ended_by == "stop_when_done"
    assert pending == {}
