"""Server-side tests for the campaign + run discovery and create endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

CAMPAIGN_A = "vasfatyol"
CAMPAIGN_B = "doskvol"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    # Two-campaign layout to verify the picker enumerates multiple.
    # ``tmp_path`` IS the vault root; each subdir is a campaign vault.
    for slug, title in ((CAMPAIGN_A, "Vasfátyol"), (CAMPAIGN_B, "Doskvol")):
        cdir = tmp_path / slug
        cdir.mkdir()
        (cdir / "campaign.md").write_text(
            f"---\ntype: campaign\ntitle: {title}\nstatus: active\nlang: hu\n---\n",
            encoding="utf-8",
        )

    # Pre-existing run under campaign A so /runs has content to surface.
    run_root = tmp_path / CAMPAIGN_A / "runs" / "default"
    run_root.mkdir(parents=True)
    (run_root / "run.md").write_text(
        "---\ntype: run\ntitle: Default\nstatus: active\ncreated: 2026-01-01\n---\n",
        encoding="utf-8",
    )
    (run_root / "sessions").mkdir()
    (run_root / "sessions" / "2026-06-10-100000.md").write_text(
        "---\ntype: session_log\ndate: 2026-06-10\n---\n# Session\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AIDM_MODE", "public")
    monkeypatch.setenv("AIDM_TRANSPORT", "webrtc")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / CAMPAIGN_A))

    import aidm.server as server_module  # noqa: PLC0415

    async def _noop_check(_config: object) -> None:
        return None

    monkeypatch.setattr(server_module, "check_providers", _noop_check)

    from aidm.server import create_app  # noqa: PLC0415

    with TestClient(create_app()) as c:
        yield c


def test_list_campaigns_enumerates_vault_root(client: TestClient) -> None:
    response = client.get("/api/campaigns")
    assert response.status_code == 200
    body = response.json()
    slugs = {c["slug"] for c in body}
    assert slugs == {CAMPAIGN_A, CAMPAIGN_B}
    titles = {c["title"] for c in body}
    assert titles == {"Vasfátyol", "Doskvol"}


def test_list_runs_returns_existing_default(client: TestClient) -> None:
    response = client.get(f"/api/campaigns/{CAMPAIGN_A}/runs")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    run = body[0]
    assert run["slug"] == "default"
    assert run["title"] == "Default"
    assert run["session_count"] == 1
    assert run["last_session_date"] == "2026-06-10"


def test_list_runs_other_campaign_is_empty(client: TestClient) -> None:
    # Second campaign has no runs yet — the inventory is per-campaign.
    response = client.get(f"/api/campaigns/{CAMPAIGN_B}/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_runs_rejects_unknown_campaign(client: TestClient) -> None:
    response = client.get("/api/campaigns/no-such/runs")
    assert response.status_code == 404


def test_create_run_writes_new_subtree(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_A}/runs",
        json={"title": "Replay 2026"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["slug"] == "replay-2026"
    assert body["title"] == "Replay 2026"
    new_run = tmp_path / CAMPAIGN_A / "runs" / "replay-2026"
    assert (new_run / "run.md").is_file()
    assert (new_run / "characters").is_dir()
    assert (new_run / "sessions").is_dir()


def test_create_run_lands_under_correct_campaign(client: TestClient, tmp_path: Path) -> None:
    # A run created against campaign B must NOT pollute campaign A's tree.
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_B}/runs",
        json={"title": "First Party"},
    )
    assert response.status_code == 201
    assert (tmp_path / CAMPAIGN_B / "runs" / "first-party" / "run.md").is_file()
    assert not (tmp_path / CAMPAIGN_A / "runs" / "first-party").exists()


def test_create_run_409_on_duplicate(client: TestClient) -> None:
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_A}/runs",
        json={"title": "Default"},
    )
    assert response.status_code == 409


def test_create_run_400_on_empty_title(client: TestClient) -> None:
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_A}/runs",
        json={"title": "   "},
    )
    assert response.status_code == 400


def test_create_run_400_when_title_slugifies_to_empty(client: TestClient) -> None:
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_A}/runs",
        json={"title": "!!!"},
    )
    assert response.status_code == 400


def test_offer_with_expired_session_id_still_validates_campaign(client: TestClient) -> None:
    # An expired / unknown session_id starts a fresh session, so a bad
    # (campaign, run) must be refused before any SDP is accepted.
    response = client.post(
        "/api/offer",
        json={
            "sdp": "v=0",
            "type": "offer",
            "requestData": {"session_id": "expired1", "campaign_id": "nope", "run_id": "default"},
        },
    )
    assert response.status_code == 404
    assert "campaign not found" in response.json()["detail"]


def test_offer_refuses_a_traversing_run_id(client: TestClient, tmp_path: Path) -> None:
    # The escape target is a real directory, so a bare ``is_dir`` check passes.
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    response = client.post(
        "/api/offer",
        json={
            "sdp": "v=0",
            "type": "offer",
            "requestData": {"campaign_id": CAMPAIGN_A, "run_id": f"../../../{outside.name}"},
        },
    )
    assert response.status_code == 404
    assert "invalid run id" in response.json()["detail"]


def test_offer_refuses_an_absolute_campaign_id(client: TestClient, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    (outside / "runs" / "default").mkdir(parents=True)
    response = client.post(
        "/api/offer",
        json={
            "sdp": "v=0",
            "type": "offer",
            "requestData": {"campaign_id": str(outside), "run_id": "default"},
        },
    )
    assert response.status_code == 404
    assert "campaign not found" in response.json()["detail"]


def test_character_routes_refuse_a_traversing_run(client: TestClient, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    (outside / "characters").mkdir(parents=True)
    (outside / "characters" / "x.md").write_text("# x\n", encoding="utf-8")
    run = f"../../../{outside.name}"
    got = client.get(f"/api/vault/characters/x?campaign={CAMPAIGN_A}&run={run}")
    put = client.put(f"/api/vault/characters/x?campaign={CAMPAIGN_A}&run={run}", content="pwned")
    assert got.status_code == 404
    assert put.status_code == 404
    assert (outside / "characters" / "x.md").read_text(encoding="utf-8") == "# x\n"
