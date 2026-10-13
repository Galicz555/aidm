"""Production-posture safety nets for the HTTP layer.

The web-security module refuses to assemble in two misconfiguration
shapes that would silently expose a prod deploy:

1. ``CF_ACCESS_DISABLED`` not set, ``CF_ACCESS_TEAM_DOMAIN``/``AUD``
   empty — open-by-default auth.
2. ``CF_ACCESS_DISABLED`` not set, ``AIDM_ALLOWED_ORIGINS`` empty —
   dev-origin CORS fallback in prod.

Both should raise at app-build time, not at first request.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("aiortc")

from aidm.server import create_app
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient


def _common_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIDM_MODE", "public")
    monkeypatch.setenv("AIDM_TRANSPORT", "webrtc")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")


def test_refuses_to_start_when_cf_access_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Override the conftest autouse default that disables CF Access.
    # setenv to "" rather than delenv: load_dotenv(override=False) would
    # re-populate the var from .env after a delenv (the local-dev .env
    # carries CF_ACCESS_DISABLED=1 so the picker-driven dev flow works),
    # which leaks the dev posture into prod-posture tests.
    monkeypatch.setenv("CF_ACCESS_DISABLED", "")
    monkeypatch.setenv("AIDM_ALLOWED_ORIGINS", "https://aidm.example")
    _common_env(monkeypatch)

    with pytest.raises(RuntimeError, match="CF Access not configured"):
        create_app()


def test_refuses_to_start_when_allowed_origins_empty_in_prod_posture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # setenv to "" rather than delenv: load_dotenv(override=False) would
    # re-populate the var from .env after a delenv (the local-dev .env
    # carries CF_ACCESS_DISABLED=1 so the picker-driven dev flow works),
    # which leaks the dev posture into prod-posture tests.
    monkeypatch.setenv("CF_ACCESS_DISABLED", "")
    monkeypatch.setenv("CF_ACCESS_TEAM_DOMAIN", "team.cloudflareaccess.com")
    monkeypatch.setenv("CF_ACCESS_AUD", "aud-tag")
    # ALLOWED_ORIGINS deliberately unset.
    _common_env(monkeypatch)

    with pytest.raises(RuntimeError, match="AIDM_ALLOWED_ORIGINS is empty"):
        create_app()


# Liveness and mode probes carry no data; every other route is behind Access.
_PUBLIC_ROUTES = {"/health", "/api/info"}


@pytest.mark.parametrize("transport", ["webrtc", "text"])
def test_every_route_but_the_probes_demands_an_access_token(
    monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    from aidm.text_server import create_app as create_text_app  # noqa: PLC0415

    monkeypatch.setenv("CF_ACCESS_DISABLED", "")
    monkeypatch.setenv("CF_ACCESS_TEAM_DOMAIN", "team.cloudflareaccess.com")
    monkeypatch.setenv("CF_ACCESS_AUD", "aud-tag")
    monkeypatch.setenv("AIDM_ALLOWED_ORIGINS", "https://aidm.example")
    _common_env(monkeypatch)
    monkeypatch.setenv("AIDM_TRANSPORT", transport)

    app = (create_app if transport == "webrtc" else create_text_app)()
    # No lifespan: an unauthenticated request must be refused before any
    # handler touches app.state, so nothing here needs a running server.
    client = TestClient(app, raise_server_exceptions=False)
    checked = 0
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path in _PUBLIC_ROUTES:
            continue
        path = re.sub(r"\{[^}]+\}", "x", route.path)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            response = client.request(method, path)
            assert response.status_code == 401, f"{method} {route.path} is not behind Access"
            checked += 1
    assert checked > 5  # the walk really visited the API surface
