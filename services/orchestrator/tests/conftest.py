"""Test-suite-wide fixtures.

Tests construct the FastAPI app in a permissive dev posture by default.
``configure_cors`` and ``make_cf_access_verifier`` refuse to start
without explicit production envs — that's the safety net for prod, not
something every test should opt out of. Set ``CF_ACCESS_DISABLED=1``
once here so individual tests don't have to.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _dev_security_posture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CF_ACCESS_DISABLED", "1")
    # Don't leak a real ALLOWED_ORIGINS from the developer's .env into
    # tests; the dev-origin fallback is what we want here.
    if "AIDM_ALLOWED_ORIGINS" in os.environ:
        monkeypatch.delenv("AIDM_ALLOWED_ORIGINS", raising=False)
