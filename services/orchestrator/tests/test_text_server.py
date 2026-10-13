"""Text-server smoke tests.

We don't spin up a real pipeline (the LLM provider would try to phone
home). Instead the test installs a fake ``build_text_pipeline`` that
returns a handle with a recording ``user_agg`` and an injectable sink
queue, then drives the HTTP routes end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

fastapi = pytest.importorskip("fastapi")

from aidm.session import session_for_locale  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@dataclass
class _FakeWorker:
    """Stand-in for PipelineWorker: the fake run_worker blocks until it ends."""

    cancelled: asyncio.Event
    ended_by: str = ""

    async def cancel(self, *, reason: str | None = None) -> None:
        self.ended_by = f"cancel:{reason}"
        self.cancelled.set()

    async def stop_when_done(self) -> None:
        self.ended_by = "stop_when_done"
        self.cancelled.set()


class _RecordingAgg:
    """Captures every queued frame so the test can assert on it."""

    def __init__(self) -> None:
        self.frames: list[tuple[str, str]] = []

    async def queue_frame(self, frame: Any, direction: Any) -> None:
        self.frames.append((frame.text, getattr(direction, "name", str(direction))))


@dataclass
class _FakeHandle:
    user_agg: _RecordingAgg
    sink_queue: asyncio.Queue[dict[str, Any] | None]
    context: object
    worker: _FakeWorker
    tool_selector: Any = None
    scene_registry: Any = None
    transcript: list[dict[str, str]] = field(default_factory=list)
    rollback: Any = None
    activity: Any = None


@pytest.fixture
def text_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    # Seed a minimal vault layout under tmp_path = vault_root.
    # Campaign vault: vault_root/test-campaign/. Run: runs/default/.
    campaign_dir = tmp_path / "test-campaign"
    (campaign_dir / "runs" / "default" / "characters").mkdir(parents=True)
    (campaign_dir / "runs" / "default" / "sessions").mkdir(exist_ok=True)
    (campaign_dir / "runs" / "default" / "notes").mkdir(exist_ok=True)
    (campaign_dir / "campaign.md").write_text(
        "---\ntype: campaign\ntitle: Test Campaign\nstatus: active\nlang: hu\n---\n",
        encoding="utf-8",
    )
    (campaign_dir / "runs" / "default" / "characters" / "elara.md").write_text(
        "# Elara\n", encoding="utf-8"
    )
    (campaign_dir / "runs" / "default" / "run.md").write_text(
        "---\ntype: run\ntitle: Default\nstatus: active\ncreated: 2026-06-17\n---\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AIDM_MODE", "public")
    monkeypatch.setenv("AIDM_TRANSPORT", "text")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("VAULT_PATH", str(campaign_dir))
    # Set the fallback run so existing tests don't have to thread run_id
    # through every POST payload. Tests that exercise the picker flow
    # explicitly should pass run_id and override this.
    monkeypatch.setenv("RUN_SLUG", "default")

    from aidm import text_server  # noqa: PLC0415

    async def _noop_check(_config: object) -> None:
        return None

    monkeypatch.setattr(text_server, "check_providers", _noop_check)

    handles: list[_FakeHandle] = []

    def _fake_build(_config: object, session: object, vault: Any, **_kwargs: object) -> _FakeHandle:
        from aidm.scene_state import SceneRegistry  # noqa: PLC0415
        from aidm.text_sink import TurnActivity  # noqa: PLC0415
        from aidm.tool_router import ToolSubsetSelector  # noqa: PLC0415
        from aidm.turn_director import DirectorState  # noqa: PLC0415
        from aidm.turn_rollback import TurnRollback  # noqa: PLC0415
        from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: PLC0415

        handle = _FakeHandle(
            user_agg=_RecordingAgg(),
            sink_queue=asyncio.Queue(),
            context=LLMContext(),
            worker=_FakeWorker(cancelled=asyncio.Event()),
            tool_selector=ToolSubsetSelector(),
            scene_registry=SceneRegistry(),
            activity=TurnActivity(),
        )
        handle.rollback = TurnRollback(
            context=handle.context,  # type: ignore[arg-type]
            transcript=handle.transcript,
            registry=handle.scene_registry,
            director=DirectorState(),
            vault=vault,
            processors=[],
            summarizer=None,
            inlined_sheets=set(),
        )
        handles.append(handle)
        return handle

    async def _fake_run_worker(handle: _FakeHandle, _session: object) -> None:
        # Block until the lifespan teardown cancels us — mirrors the
        # real worker's "runs forever until the pipeline closes" shape.
        try:
            await handle.worker.cancelled.wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(text_server, "build_text_pipeline", _fake_build)
    monkeypatch.setattr(text_server, "run_worker", _fake_run_worker)

    with TestClient(text_server.create_app()) as client:
        client.handles = handles  # type: ignore[attr-defined]
        yield client


def test_health_responds_ok(text_client: TestClient) -> None:
    response = text_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_registers_text_routes(text_client: TestClient) -> None:
    routes = {route.path for route in text_client.app.routes}  # type: ignore[attr-defined]
    assert "/api/text/session" in routes
    assert "/api/text/session/{session_id}/turn" in routes
    assert "/api/text/session/{session_id}/stream" in routes
    assert "/api/text/session/{session_id}/end" in routes
    assert "/api/vault/characters/{slug}" in routes


def test_start_session_returns_session_id(text_client: TestClient) -> None:
    response = text_client.post("/api/text/session", json={"locale": "hu"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("session_id"), str)
    assert body["locale"] == "hu"


def test_turn_queues_transcription_frame(text_client: TestClient) -> None:
    started = text_client.post("/api/text/session", json={"locale": "hu"}).json()
    sid = started["session_id"]

    response = text_client.post(
        f"/api/text/session/{sid}/turn", json={"text": "Belépek a kocsmába"}
    )
    assert response.status_code == 202
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    assert handle.user_agg.frames, "turn endpoint should have queued a frame"
    text, direction = handle.user_agg.frames[0]
    assert text == "Belépek a kocsmába"
    assert direction == "DOWNSTREAM"


def test_turn_empty_text_returns_400(text_client: TestClient) -> None:
    started = text_client.post("/api/text/session", json={"locale": "hu"}).json()
    sid = started["session_id"]
    response = text_client.post(f"/api/text/session/{sid}/turn", json={"text": "   "})
    assert response.status_code == 400


def test_turn_unknown_session_returns_404(text_client: TestClient) -> None:
    response = text_client.post("/api/text/session/no-such/turn", json={"text": "hello"})
    assert response.status_code == 404


def test_end_session_queues_locale_phrase(text_client: TestClient) -> None:
    started = text_client.post("/api/text/session", json={"locale": "hu"}).json()
    sid = started["session_id"]
    response = text_client.post(f"/api/text/session/{sid}/end")
    assert response.status_code == 202
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    assert handle.user_agg.frames
    text, _direction = handle.user_agg.frames[0]
    assert "vége" in text.lower()


def test_end_session_uses_english_when_locale_en(text_client: TestClient) -> None:
    started = text_client.post("/api/text/session", json={"locale": "en"}).json()
    sid = started["session_id"]
    response = text_client.post(f"/api/text/session/{sid}/end")
    assert response.status_code == 202
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    text, _ = handle.user_agg.frames[0]
    assert "done" in text.lower()


def test_stream_delivers_queue_payloads_as_sse(text_client: TestClient) -> None:
    started = text_client.post("/api/text/session", json={"locale": "hu"}).json()
    sid = started["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]

    # Pre-load the queue so the SSE generator drains immediately and
    # the close sentinel terminates the stream. The TestClient's stream
    # support is synchronous; we read all bytes and parse SSE lines.
    handle.sink_queue.put_nowait({"type": "transcript.bot_delta", "text": "Helló"})
    handle.sink_queue.put_nowait({"type": "transcript.bot_end"})
    handle.sink_queue.put_nowait(None)

    with text_client.stream("GET", f"/api/text/session/{sid}/stream") as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes())

    # SSE frames are "data: <json>\n\n" — split on the double newline.
    chunks = [c for c in body.decode().split("\n\n") if c.strip()]
    payloads = [json.loads(c.removeprefix("data: ")) for c in chunks]
    assert payloads == [
        {"type": "transcript.bot_delta", "text": "Helló"},
        {"type": "transcript.bot_end"},
    ]


def test_character_sheet_route(text_client: TestClient) -> None:
    response = text_client.get("/api/vault/characters/elara?campaign=test-campaign&run=default")
    assert response.status_code == 200
    assert "Elara" in response.text
    # Last-Modified anchors optimistic concurrency on PUT.
    assert "last-modified" in {k.lower() for k in response.headers}


def test_character_sheet_put_writes_through_vault(text_client: TestClient, tmp_path: Path) -> None:
    # Use a character we know exists (planted by the fixture).
    get_resp = text_client.get("/api/vault/characters/elara?campaign=test-campaign&run=default")
    last_modified = get_resp.headers["last-modified"]
    put_resp = text_client.put(
        "/api/vault/characters/elara?campaign=test-campaign&run=default",
        content="# Edited\n\nNew body.\n",
        headers={"If-Unmodified-Since": last_modified},
    )
    assert put_resp.status_code == 204
    written = (
        tmp_path / "test-campaign" / "runs" / "default" / "characters" / "elara.md"
    ).read_text(encoding="utf-8")
    assert "New body." in written


def test_character_sheet_put_conflict_when_stale(text_client: TestClient, tmp_path: Path) -> None:
    import time as _time  # noqa: PLC0415

    target = tmp_path / "test-campaign" / "runs" / "default" / "characters" / "elara.md"
    # Backdate the supplied If-Unmodified-Since to one year ago; the
    # current mtime is now, so the server must reject as stale.
    _time.sleep(0)  # no-op; mtime already established
    response = text_client.put(
        "/api/vault/characters/elara?campaign=test-campaign&run=default",
        content="anything\n",
        headers={"If-Unmodified-Since": "Sun, 06 Jun 2020 12:00:00 GMT"},
    )
    assert response.status_code == 409
    assert target.read_text(encoding="utf-8") == "# Elara\n"


def test_character_sheet_put_missing_returns_404(text_client: TestClient) -> None:
    response = text_client.put(
        "/api/vault/characters/no-such?campaign=test-campaign&run=default",
        content="x\n",
    )
    assert response.status_code == 404


def test_character_roleplay_prompt_route(text_client: TestClient, tmp_path: Path) -> None:
    # Plant a richer sheet so the builder has fields to surface.
    (tmp_path / "test-campaign" / "runs" / "default" / "characters" / "richard.md").write_text(
        "---\n"
        "type: character\n"
        "lang: hu\n"
        "name: Richard\n"
        "status: active\n"
        "location: Vasfátyol\n"
        "last_updated: 2026-06-08\n"
        "hexaco_honesty_humility: 4\n"
        "hexaco_emotionality: 3\n"
        "hexaco_extraversion: 2\n"
        "hexaco_agreeableness: 3\n"
        "hexaco_conscientiousness: 5\n"
        "hexaco_openness: 4\n"
        "---\n\n"
        "# Richard\n\n"
        "## Psychology (HEXACO)\n"
        "- **Honesty-Humility (4):** őszinte, becsületes.\n"
        "- **Emotionality (3):** kiegyensúlyozott.\n"
        "- **eXtraversion (2):** zárkózott.\n"
        "- **Agreeableness (3):** együttműködő.\n"
        "- **Conscientiousness (5):** pedáns.\n"
        "- **Openness (4):** kíváncsi.\n",
        encoding="utf-8",
    )
    response = text_client.get(
        "/api/vault/characters/richard/roleplay-prompt?campaign=test-campaign&run=default"
    )
    assert response.status_code == 200
    text = response.text
    assert "You are Richard" in text
    assert "Honesty-Humility (4)" in text
    assert "őszinte, becsületes" in text
    assert "first person" in text
    assert "Respond in hu" in text


def test_unknown_session_id_resume_starts_fresh(text_client: TestClient) -> None:
    # Passing a session_id that doesn't match any pending entry should
    # be tolerated — the server starts a fresh session with a new id.
    response = text_client.post("/api/text/session", json={"locale": "hu", "session_id": "no-such"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] != "no-such"


def test_campaigns_route_returns_grouping(text_client: TestClient, tmp_path: Path) -> None:
    # Sessions live under vault_root/<campaign>/runs/<run>/sessions/. The
    # grouping endpoint takes ?campaign=&run= and walks only that scope.
    vault_slug = "test-campaign"
    sessions_dir = tmp_path / vault_slug / "runs" / "default" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "2026-06-07-100000.md").write_text(
        f"---\ncampaign: {vault_slug}\ndate: 2026-06-07\n---\n# Belépés\n\nbody\n",
        encoding="utf-8",
    )
    (sessions_dir / "2026-06-06-100000.md").write_text(
        "---\ncampaign: some-other-vault\ndate: 2026-06-06\n---\n# Idegen kampány\n\nbody\n",
        encoding="utf-8",
    )
    (sessions_dir / "2026-06-05-100000.md").write_text(
        "---\ndate: 2026-06-05\n---\n# No-field default\n\nbody\n",
        encoding="utf-8",
    )

    response = text_client.get(f"/api/sessions/grouping?campaign={vault_slug}&run=default")
    assert response.status_code == 200
    body = response.json()
    assert [c["slug"] for c in body["campaigns"]] == [vault_slug]
    session_ids = {s["id"] for s in body["campaigns"][0]["sessions"]}
    assert session_ids == {"2026-06-07-100000", "2026-06-05-100000"}
    assert [e["id"] for e in body["orphans"]] == ["2026-06-06-100000"]


def test_campaigns_route_picker_shape(text_client: TestClient) -> None:
    # /api/campaigns enumerates every vault_root/* with a campaign.md.
    response = text_client.get("/api/campaigns")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert any(c["slug"] == "test-campaign" for c in body)


def test_sessions_route_still_returns_flat_list(text_client: TestClient, tmp_path: Path) -> None:
    # Backward-compat: /api/sessions returns the active campaign's runs
    # as a flat list when ?run= is passed; defaults to RUN_SLUG env.
    sessions_dir = tmp_path / "test-campaign" / "runs" / "default" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "2026-06-07-100000.md").write_text(
        "---\ncampaign: vasfatyol\ndate: 2026-06-07\n---\n# x\n",
        encoding="utf-8",
    )
    response = text_client.get("/api/sessions?run=default")
    assert response.status_code == 200
    body = response.json()
    assert "sessions" in body
    assert any(s["id"] == "2026-06-07-100000" for s in body["sessions"])


def test_hot_reattach_returns_live_session(text_client: TestClient) -> None:
    # Start a session; pass the returned session_id back to the
    # start route. The server should detect the live registry hit and
    # respond with resumed_from="live" *without* spawning a second
    # pipeline.
    started = text_client.post("/api/text/session", json={"locale": "hu"}).json()
    sid = started["session_id"]
    handles_before = len(text_client.handles)  # type: ignore[attr-defined]

    reattach = text_client.post("/api/text/session", json={"session_id": sid, "locale": "hu"})
    assert reattach.status_code == 200
    body = reattach.json()
    assert body["session_id"] == sid
    assert body["resumed_from"] == "live"
    handles_after = len(text_client.handles)  # type: ignore[attr-defined]
    assert handles_after == handles_before  # no new pipeline


def test_history_serves_full_transcript_after_summarizer_folded_context(
    text_client: TestClient,
) -> None:
    started = text_client.post("/api/text/session", json={"locale": "hu"}).json()
    sid = started["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]

    # Player lines are recorded at the HTTP boundary; DM replies by the
    # pipeline's transcript recorder (simulated here).
    text_client.post(f"/api/text/session/{sid}/turn", json={"text": "Belépek a kocsmába"})
    handle.transcript.append({"role": "assistant", "text": "A pultos felnéz."})
    handle.activity.response_landed()  # the reply passed the post-turn tail
    text_client.post(f"/api/text/session/{sid}/turn", json={"text": "Mit lát?"})
    handle.transcript.append({"role": "assistant", "text": "Egy alakot"})

    # The summarizer has folded the context down to its tail; /history
    # must not shrink with it.
    class _Ctx:
        messages: ClassVar[list[dict[str, str]]] = [
            {"role": "system", "content": "<!--aidm:session-summary-->\n[...]\n\nfolded"},
            {"role": "user", "content": "Mit lát?"},
            {"role": "assistant", "content": "Egy alakot"},
        ]

    endpoints = text_client.app.state.session_endpoints  # type: ignore[attr-defined]
    endpoints[sid].context = _Ctx()

    response = text_client.get(f"/api/text/session/{sid}/history")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == sid
    assert body["turns"] == [
        {"role": "user", "text": "Belépek a kocsmába"},
        {"role": "assistant", "text": "A pultos felnéz."},
        {"role": "user", "text": "Mit lát?"},
        {"role": "assistant", "text": "Egy alakot"},
    ]


def test_hot_reattach_is_not_blocked_by_session_cap(
    text_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses  # noqa: PLC0415

    app_state = text_client.app.state  # type: ignore[attr-defined]
    app_state.config = dataclasses.replace(app_state.config, max_concurrent_sessions=1)
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]

    # A page refresh at the cap rejoins the live session...
    reattach = text_client.post("/api/text/session", json={"session_id": sid})
    assert reattach.status_code == 200
    assert reattach.json()["resumed_from"] == "live"
    # ...while a genuinely new session is still refused.
    assert text_client.post("/api/text/session", json={"locale": "en"}).status_code == 503


def _wait_until_gone(client: TestClient, sid: str, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get(f"/api/text/session/{sid}/history").status_code == 404:
            return True
        time.sleep(0.02)
    return False


def test_session_without_stream_ends_after_reconnect_grace(text_client: TestClient) -> None:
    state = text_client.app.state  # type: ignore[attr-defined]
    state.config = replace(state.config, text_reconnect_grace_s=0.1)
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]

    # The SSE stream attaches and then drops (browser gone).
    handle.sink_queue.put_nowait(None)
    with text_client.stream("GET", f"/api/text/session/{sid}/stream") as response:
        b"".join(response.iter_bytes())

    assert _wait_until_gone(text_client, sid)
    assert handle.worker.ended_by == "cancel:client disconnected"
    # The slot is released: the pipeline task is gone.
    assert not any(not t.done() for t in text_client.app.state.pipeline_tasks)  # type: ignore[attr-defined]


def test_reattach_inside_grace_keeps_the_live_session(text_client: TestClient) -> None:
    state = text_client.app.state  # type: ignore[attr-defined]
    state.config = replace(state.config, text_reconnect_grace_s=0.4)
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    time.sleep(0.25)
    # Reattach restarts the window, giving the client time to reopen SSE.
    assert (
        text_client.post("/api/text/session", json={"session_id": sid}).json()["resumed_from"]
        == "live"
    )
    time.sleep(0.25)  # past the original deadline, inside the new one
    assert text_client.get(f"/api/text/session/{sid}/history").status_code == 200
    assert _wait_until_gone(text_client, sid)


def test_detached_ending_session_finishes_farewell_instead_of_cancelling(
    text_client: TestClient,
) -> None:
    state = text_client.app.state  # type: ignore[attr-defined]
    state.config = replace(state.config, text_reconnect_grace_s=0.1)
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    text_client.post(f"/api/text/session/{sid}/end")

    assert _wait_until_gone(text_client, sid)
    assert handle.worker.ended_by == "stop_when_done"


def test_history_unknown_session_returns_404(text_client: TestClient) -> None:
    response = text_client.get("/api/text/session/no-such/history")
    assert response.status_code == 404


def test_session_for_locale_is_locale_aware() -> None:
    # Sanity: the session factory the server delegates to recognizes
    # both supported locales. If this regresses, the start-session route
    # will 400 silently.
    assert session_for_locale("hu").locale == "hu"
    assert session_for_locale("en").locale == "en"


# --- regenerate the last DM reply --------------------------------------------


def _land_reply(handle: _FakeHandle, reply: str) -> None:
    """What the real pipeline does for a turn: reply in context + transcript,
    a state file on disk, then the writer's idle through TextSink."""
    handle.context.add_message({"role": "user", "content": handle.transcript[-1]["text"]})  # type: ignore[attr-defined]
    handle.context.add_message({"role": "assistant", "content": reply})  # type: ignore[attr-defined]
    handle.transcript.append({"role": "assistant", "text": reply})
    handle.rollback._vault.write_npc_state("stranger", "---\ntype: npc_state\n---\n")
    handle.activity.response_landed()


def test_regenerate_is_409_while_a_turn_is_in_flight(text_client: TestClient) -> None:
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    text_client.post(f"/api/text/session/{sid}/turn", json={"text": "I open the door"})
    # No turn.status idle has passed the sink yet.
    response = text_client.post(f"/api/text/session/{sid}/regenerate")
    assert response.status_code == 409


def test_regenerate_without_a_turn_is_409(text_client: TestClient) -> None:
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    assert text_client.post(f"/api/text/session/{sid}/regenerate").status_code == 409


def test_regenerate_archives_the_reply_rolls_back_and_replays_the_player_line(
    text_client: TestClient, tmp_path: Path
) -> None:
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    run = tmp_path / "test-campaign" / "runs" / "default"
    text_client.post(f"/api/text/session/{sid}/turn", json={"text": "I open the door"})
    _land_reply(handle, "A stranger waits in the hall.")
    assert (run / "npcs" / "stranger.state.md").exists()

    response = text_client.post(f"/api/text/session/{sid}/regenerate")

    assert response.status_code == 202
    archived = list((run / "discarded").glob("*.md"))
    assert len(archived) == 1
    body = archived[0].read_text(encoding="utf-8")
    assert "A stranger waits in the hall." in body and "I open the door" in body
    assert "turn: 1" in body
    assert not (run / "npcs" / "stranger.state.md").exists()  # the turn's write undone
    assert handle.context.messages == []
    assert [text for text, _ in handle.user_agg.frames] == ["I open the door"] * 2
    turns = text_client.get(f"/api/text/session/{sid}/history").json()["turns"]
    assert turns == [{"role": "user", "text": "I open the door"}]
    assert handle.activity.busy  # the re-run is in flight

    # The new reply lands; a second regenerate reuses the same snapshot.
    _land_reply(handle, "The hall is empty but for a lamp.")
    assert text_client.post(f"/api/text/session/{sid}/regenerate").status_code == 202
    assert len(list((run / "discarded").glob("*.md"))) == 2
    turns = text_client.get(f"/api/text/session/{sid}/history").json()["turns"]
    assert turns == [{"role": "user", "text": "I open the door"}]


@pytest.mark.asyncio
async def test_the_archived_reply_is_invisible_to_the_dm_vault_tools(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.vault import Vault  # noqa: PLC0415
    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    (tmp_path / "runs" / "default").mkdir(parents=True)
    vault = Vault(tmp_path, run_slug="default")
    archived = vault.write_discarded_reply("## Discarded reply\n\nThe dragon laughs.\n")
    name = Path(archived).name
    handlers = _handlers_for(vault)

    async def call(tool: str, **arguments: str) -> str:
        params = MagicMock()
        params.arguments = arguments
        params.result_callback = AsyncMock()
        await handlers[tool](params)
        return str(params.result_callback.call_args.args[0])

    assert "dragon" not in await call("vault_read_file", path=f"discarded/{name}")
    assert "dragon" not in await call("vault_grep", pattern="dragon")
    assert "discarded" not in await call("vault_glob", pattern="*")
    assert "discarded" not in await call("vault_list_dir", path=".")


def test_regenerate_is_not_offered_on_the_farewell_turn(text_client: TestClient) -> None:
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    text_client.post(f"/api/text/session/{sid}/turn", json={"text": "I open the door"})
    handle.activity.response_landed()
    text_client.post(f"/api/text/session/{sid}/end")
    handle.activity.response_landed()  # the farewell landed
    assert handle.rollback.snapshot is None
    assert text_client.post(f"/api/text/session/{sid}/regenerate").status_code == 409


# --- review fixes (2026-09-28) ------------------------------------------------


def _played_turn(text_client: TestClient, reply: str = "A stranger waits.") -> tuple[str, Any]:
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    text_client.post(f"/api/text/session/{sid}/turn", json={"text": "I open the door"})
    _land_reply(handle, reply)
    return sid, handle


def test_turn_is_409_while_a_turn_is_in_flight(text_client: TestClient) -> None:
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    assert (
        text_client.post(f"/api/text/session/{sid}/turn", json={"text": "one"}).status_code == 202
    )
    assert (
        text_client.post(f"/api/text/session/{sid}/turn", json={"text": "two"}).status_code == 409
    )
    assert [t for t, _ in handle.user_agg.frames] == ["one"]  # no mid-turn snapshot or frame


def test_a_browser_edit_drops_the_snapshot_so_regenerate_cannot_revert_it(
    text_client: TestClient, tmp_path: Path
) -> None:
    sid, handle = _played_turn(text_client)
    handle.rollback._vault.write_character_sheet("lyra", "turn-written\n")  # the turn made it
    put = text_client.put(
        "/api/vault/characters/lyra?campaign=test-campaign&run=default", content="PLAYER EDIT\n"
    )
    assert put.status_code == 204
    assert handle.rollback.snapshot is None
    assert text_client.post(f"/api/text/session/{sid}/regenerate").status_code == 409
    sheet = tmp_path / "test-campaign" / "runs" / "default" / "characters" / "lyra.md"
    assert sheet.read_text(encoding="utf-8") == "PLAYER EDIT\n"


def test_regenerate_is_409_when_another_writer_changed_a_journaled_file(
    text_client: TestClient, tmp_path: Path
) -> None:
    from aidm.vault import Vault  # noqa: PLC0415

    sid, handle = _played_turn(text_client)
    other_session = Vault(tmp_path / "test-campaign", run_slug="default")
    other_session.write_npc_state("stranger", "---\ntype: npc_state\n---\nother session\n")

    assert text_client.post(f"/api/text/session/{sid}/regenerate").status_code == 409
    state = tmp_path / "test-campaign" / "runs" / "default" / "npcs" / "stranger.state.md"
    assert "other session" in state.read_text(encoding="utf-8")
    assert not handle.activity.busy  # released: the player can go on
    assert len(handle.transcript) == 2  # nothing rolled back in memory


def test_a_failed_restore_replays_nothing_archives_nothing_and_a_retry_finishes(
    text_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os  # noqa: PLC0415

    sid, handle = _played_turn(text_client)
    run = tmp_path / "test-campaign" / "runs" / "default"
    real_replace = os.replace
    handle.rollback._vault.write_beat("turn beat")  # a created file (unlink) ...
    handle.rollback._vault._journal[run / "npcs" / "stranger.state.md"].pre_image = b"old\n"

    def flaky(src: str | Path, dst: str | Path) -> None:  # ... and a staged one that fails
        raise OSError("device busy")

    monkeypatch.setattr(os, "replace", flaky)
    response = text_client.post(f"/api/text/session/{sid}/regenerate")
    assert response.status_code == 500
    assert not handle.activity.busy
    assert len(handle.transcript) == 2  # memory untouched
    assert [t for t, _ in handle.user_agg.frames] == ["I open the door"]  # no replay
    assert not (run / "discarded").exists()  # archived only after a successful restore

    monkeypatch.setattr(os, "replace", real_replace)
    assert text_client.post(f"/api/text/session/{sid}/regenerate").status_code == 202
    assert len(list((run / "discarded").glob("*.md"))) == 1
    assert (run / "npcs" / "stranger.state.md").read_bytes() == b"old\n"
    assert not (run / "beat.md").exists()


def test_an_unexpected_regenerate_error_never_leaves_busy_stuck(text_client: TestClient) -> None:
    sid, handle = _played_turn(text_client)

    def boom() -> None:
        raise RuntimeError("bug")

    handle.rollback.restore = boom
    with pytest.raises(RuntimeError):
        text_client.post(f"/api/text/session/{sid}/regenerate")
    assert not handle.activity.busy


def test_end_waits_for_the_in_flight_turn_or_refuses(
    text_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aidm import text_server  # noqa: PLC0415

    monkeypatch.setattr(text_server, "_END_WAIT_S", 0.05)
    sid = text_client.post("/api/text/session", json={"locale": "en"}).json()["session_id"]
    handle = text_client.handles[-1]  # type: ignore[attr-defined]
    text_client.post(f"/api/text/session/{sid}/turn", json={"text": "I open the door"})

    assert text_client.post(f"/api/text/session/{sid}/end").status_code == 409
    assert handle.tool_selector.end_session is False  # the tool re-run keeps its guards

    handle.activity.response_landed()
    assert text_client.post(f"/api/text/session/{sid}/end").status_code == 202
    assert handle.tool_selector.end_session is True


# --- campaign / run ids must stay inside the vault root ---


def _outside_dir(tmp_path: Path) -> Path:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    return outside


def test_tone_put_refuses_an_absolute_campaign_path(
    text_client: TestClient, tmp_path: Path
) -> None:
    outside = _outside_dir(tmp_path)
    response = text_client.put(f"/api/vault/tone?campaign={outside}", content="pwned")
    assert response.status_code == 404
    assert not (outside / "tone.md").exists()


def test_tone_put_refuses_a_dotdot_campaign(text_client: TestClient, tmp_path: Path) -> None:
    outside = _outside_dir(tmp_path)
    response = text_client.put(f"/api/vault/tone?campaign=../{outside.name}", content="pwned")
    assert response.status_code == 404
    assert not (outside / "tone.md").exists()


def test_tone_get_refuses_an_absolute_campaign_path(
    text_client: TestClient, tmp_path: Path
) -> None:
    outside = _outside_dir(tmp_path)
    (outside / "tone.md").write_text("secret", encoding="utf-8")
    response = text_client.get(f"/api/vault/tone?campaign={outside}")
    assert response.status_code == 404
    assert "secret" not in response.text


def test_tone_put_still_writes_for_a_real_campaign(text_client: TestClient, tmp_path: Path) -> None:
    response = text_client.put("/api/vault/tone?campaign=test-campaign", content="grim")
    assert response.status_code == 204
    assert (tmp_path / "test-campaign" / "tone.md").read_text(encoding="utf-8") == "grim"


def test_beat_put_refuses_a_traversing_run(text_client: TestClient, tmp_path: Path) -> None:
    outside = _outside_dir(tmp_path)
    response = text_client.put(
        f"/api/vault/beat?campaign=test-campaign&run=../../../{outside.name}", content="pwned"
    )
    assert response.status_code == 404
    assert not (outside / "beat.md").exists()


def test_start_session_refuses_a_traversing_run_and_campaign(
    text_client: TestClient, tmp_path: Path
) -> None:
    outside = _outside_dir(tmp_path)
    bad_run = text_client.post(
        "/api/text/session",
        json={"campaign_id": "test-campaign", "run_id": f"../../../{outside.name}"},
    )
    assert bad_run.status_code == 404
    bad_campaign = text_client.post(
        "/api/text/session", json={"campaign_id": str(outside), "run_id": "default"}
    )
    assert bad_campaign.status_code == 404


def test_sessions_grouping_refuses_a_traversing_campaign(
    text_client: TestClient, tmp_path: Path
) -> None:
    outside = _outside_dir(tmp_path)
    response = text_client.get(f"/api/sessions/grouping?campaign={outside}&run=default")
    assert response.status_code == 404


def test_sessions_with_a_traversing_run_lists_nothing(
    text_client: TestClient, tmp_path: Path
) -> None:
    outside = _outside_dir(tmp_path)
    (outside / "sessions").mkdir()
    (outside / "sessions" / "2026-01-01-000000.md").write_text("# secret\n", encoding="utf-8")
    response = text_client.get(f"/api/sessions?run=../../../{outside.name}")
    assert response.json() == {"sessions": []}
