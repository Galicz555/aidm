"""Server module smoke tests.

We do not spin up uvicorn or open a WebRTC peer connection here — those
need a real browser and live STT/LLM/TTS providers. The goal is to
verify:

* The FastAPI app builds (imports + route registration work).
* The locale-from-request fallback chain is correct for the offer
  endpoint (this is the only piece of session bootstrap that runs
  before the pipeline starts and is easy to get wrong).
* The health endpoint answers without a running lifespan.

Anything more requires the integration setup that's coming with the
SvelteKit client commit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

# Importing aidm.server pulls fastapi + pipecat.smallwebrtc + aiortc. If
# the webrtc extra is not installed the whole module is unimportable;
# skip rather than crash the suite.
fastapi = pytest.importorskip("fastapi")
aiortc = pytest.importorskip("aiortc")

from aidm.server import _locale_from_request, create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def test_app_builds_without_lifespan() -> None:
    # No lifespan run — we only want to confirm the routes register and
    # the middleware stack assembles. Endpoints that touch app.state
    # would 500 in this mode; /health does not.
    app = create_app()
    routes = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/health" in routes
    assert "/api/offer" in routes
    assert "/api/offer/{pc_id}/candidate" in routes
    assert "/api/vault/characters/{slug}" in routes


def test_health_endpoint_responds_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    # TestClient invokes the lifespan. load_config needs valid env, and
    # check_providers would hit the network — both are dodged by setting
    # mode/transport, blanking the keys, and replacing check_providers
    # with a no-op.
    monkeypatch.setenv("AIDM_MODE", "public")
    monkeypatch.setenv("AIDM_TRANSPORT", "webrtc")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")

    import aidm.server as server_module  # noqa: PLC0415 (monkeypatched after env setup)

    async def _noop_check(_config: object) -> None:
        return None

    monkeypatch.setattr(server_module, "check_providers", _noop_check)

    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_locale_from_request_takes_explicit_value() -> None:
    assert _locale_from_request({"locale": "en"}, default="hu") == "en"


def test_locale_from_request_falls_back_to_default() -> None:
    assert _locale_from_request(None, default="hu") == "hu"
    assert _locale_from_request({}, default="hu") == "hu"
    assert _locale_from_request("not-a-dict", default="hu") == "hu"
    assert _locale_from_request({"locale": 42}, default="hu") == "hu"


# --- /api/vault/characters/{slug} ---


@pytest.fixture
def vault_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Build a TestClient with a temp vault containing one character sheet.

    Most server tests skip the lifespan by hitting endpoints that don't
    touch app.state; the vault endpoint cannot, so we set up the
    minimum env needed for load_config and shim out check_providers.
    """
    # Campaign vault lives one level under the vault_root (tmp_path); the
    # campaign slug becomes the requestData campaign_id everywhere.
    campaign_dir = tmp_path / "test-campaign"
    (campaign_dir / "runs" / "default" / "characters").mkdir(parents=True)
    (campaign_dir / "campaign.md").write_text(
        "---\ntype: campaign\ntitle: Test Campaign\nstatus: active\nlang: hu\n---\n",
        encoding="utf-8",
    )
    (campaign_dir / "runs" / "default" / "characters" / "elara.md").write_text(
        "# Elara\n\nA wandering druid.\n", encoding="utf-8"
    )
    (campaign_dir / "runs" / "default" / "run.md").write_text(
        "---\ntype: run\ntitle: Test Run\nstatus: active\ncreated: 2026-06-17\n---\n",
        encoding="utf-8",
    )
    # Files outside characters/ are not exposed by the endpoint, but we
    # seed one anyway so the path-traversal test has a concrete target.
    (tmp_path / "secret.md").write_text("not for browser eyes", encoding="utf-8")

    monkeypatch.setenv("AIDM_MODE", "public")
    monkeypatch.setenv("AIDM_TRANSPORT", "webrtc")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("VAULT_PATH", str(campaign_dir))
    monkeypatch.setenv("RUN_SLUG", "default")

    import aidm.server as server_module  # noqa: PLC0415

    async def _noop_check(_config: object) -> None:
        return None

    monkeypatch.setattr(server_module, "check_providers", _noop_check)

    with TestClient(create_app()) as client:
        yield client


def test_character_sheet_happy_path(vault_client: TestClient) -> None:
    response = vault_client.get("/api/vault/characters/elara?campaign=test-campaign&run=default")
    assert response.status_code == 200
    assert "Elara" in response.text
    assert "wandering druid" in response.text
    # PlainTextResponse picks text/plain; the browser parses the markdown
    # itself (or renders it as-is). We don't care about a markdown MIME.
    assert response.headers["content-type"].startswith("text/plain")


def test_missing_character_returns_404(vault_client: TestClient) -> None:
    response = vault_client.get("/api/vault/characters/nope?campaign=test-campaign&run=default")
    assert response.status_code == 404


def test_path_traversal_returns_404(vault_client: TestClient) -> None:
    # Vault._safe should refuse to resolve above the root. The endpoint
    # builds the lookup path as f"characters/{slug}.md"; even with a
    # crafted slug we should never read the seeded ``secret.md`` above
    # the characters/ directory.
    response = vault_client.get(
        "/api/vault/characters/..%2Fsecret?campaign=test-campaign&run=default"
    )
    assert response.status_code == 404
    assert "not for browser eyes" not in response.text


def test_character_sheet_missing_run_query_returns_422(vault_client: TestClient) -> None:
    # The endpoint now requires ?run=<slug> to disambiguate which run's
    # overlay to read from. FastAPI's Query validation returns 422.
    response = vault_client.get("/api/vault/characters/elara")
    assert response.status_code == 422


# --- /api/session/{pc_id}/end ---


def test_end_session_unknown_pc_id_returns_404(vault_client: TestClient) -> None:
    response = vault_client.post("/api/session/no-such-pc/end")
    assert response.status_code == 404


def test_end_session_queues_locale_phrase(vault_client: TestClient) -> None:
    # Inject a fake SessionEndpoint with a user_agg that records every
    # queue_frame call. The endpoint handler does not care that the
    # processor isn't wired into a real pipeline — it just calls
    # queue_frame, which we intercept here.
    from aidm.server import SessionEndpoint  # noqa: PLC0415

    app = vault_client.app  # type: ignore[attr-defined]

    captured: list[tuple[str, str]] = []

    class _RecordingAgg:
        async def queue_frame(self, frame, direction):  # type: ignore[no-untyped-def]
            captured.append((frame.text, getattr(direction, "name", str(direction))))

    from aidm.tool_router import ToolSubsetSelector  # noqa: PLC0415

    selector = ToolSubsetSelector()
    app.state.session_endpoints["fake-pc"] = SessionEndpoint(
        user_agg=_RecordingAgg(),  # type: ignore[arg-type]
        locale="hu",
        session_id="abcd1234",
        tool_selector=selector,
    )

    response = vault_client.post("/api/session/fake-pc/end")
    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}

    assert captured, "expected the end phrase to be queued on the user_agg"
    text, direction = captured[0]
    assert "vége" in text.lower(), f"expected Hungarian end phrase, got {text!r}"
    assert direction == "DOWNSTREAM"


def test_end_session_uses_english_when_locale_en(vault_client: TestClient) -> None:
    from aidm.server import SessionEndpoint  # noqa: PLC0415

    app = vault_client.app  # type: ignore[attr-defined]
    captured: list[str] = []

    class _RecordingAgg:
        async def queue_frame(self, frame, _direction):  # type: ignore[no-untyped-def]
            captured.append(frame.text)

    from aidm.tool_router import ToolSubsetSelector  # noqa: PLC0415

    selector = ToolSubsetSelector()
    app.state.session_endpoints["en-pc"] = SessionEndpoint(
        user_agg=_RecordingAgg(),  # type: ignore[arg-type]
        locale="en",
        session_id="abcd5678",
        tool_selector=selector,
    )

    response = vault_client.post("/api/session/en-pc/end")
    assert response.status_code == 202
    assert captured and "done" in captured[0].lower()
    assert selector.end_session is True
