"""Default-locale resolution in ``load_config``.

The runtime default was switched from ``hu`` to ``en`` in commit dfd3963
(landed alongside the DM-session analysis fixes). A regression here would
silently flip every fresh deploy back to Hungarian responses — easy to miss
in a code review because the only diff is the literal in ``config.py:140``.

These tests pin the env-var contract: omitting ``AIDM_LOCALE`` gives ``en``;
setting it to ``hu`` still works; setting it to garbage raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.config import load_config


def _set_min_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Minimum env for ``load_config`` in private mode (no API-key gates)."""
    monkeypatch.setenv("AIDM_MODE", "private")
    monkeypatch.setenv("AIDM_TRANSPORT", "text")
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "test-campaign"))
    # Pre-create the campaign dir so VAULT_PATH resolution succeeds.
    (tmp_path / "test-campaign").mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("AIDM_LOCALE", raising=False)


def test_locale_defaults_to_en_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_min_env(monkeypatch, tmp_path)
    config = load_config()
    assert config.locale == "en"


def test_locale_honours_explicit_hu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_min_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AIDM_LOCALE", "hu")
    config = load_config()
    assert config.locale == "hu"


def test_locale_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_min_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AIDM_LOCALE", "de")
    with pytest.raises(ValueError, match="Unsupported AIDM_LOCALE"):
        load_config()


def test_npc_failsafe_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The failsafe defaults to on with 3 attempts. A regression here would
    silently disable the DM-retry side-car on every deploy.
    """
    _set_min_env(monkeypatch, tmp_path)
    for key in (
        "AIDM_NPC_FAILSAFE_ENABLED",
        "AIDM_NPC_FAILSAFE_TIMEOUT_S",
    ):
        monkeypatch.delenv(key, raising=False)
    config = load_config()
    assert config.npc_failsafe_enabled is True
    assert config.npc_failsafe_timeout_s == 60.0


def test_npc_failsafe_disable_via_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`AIDM_NPC_FAILSAFE_ENABLED=false` flips to DM-only mode — used for
    the no-failsafe baseline regress scenario and emergency rollback.
    """
    _set_min_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AIDM_NPC_FAILSAFE_ENABLED", "false")
    config = load_config()
    assert config.npc_failsafe_enabled is False


def test_npc_failsafe_bool_parses_yes_no_on_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bool parser is permissive — accepts the common shorthands so a
    Docker-compose env-block can use either ``on``/``off`` or ``true``/``false``.
    """
    for raw, expected in (("0", False), ("on", True), ("no", False), ("yes", True)):
        _set_min_env(monkeypatch, tmp_path)
        monkeypatch.setenv("AIDM_NPC_FAILSAFE_ENABLED", raw)
        assert load_config().npc_failsafe_enabled is expected


def test_npc_failsafe_timeout_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_min_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AIDM_NPC_FAILSAFE_TIMEOUT_S", "8.5")
    config = load_config()
    assert config.npc_failsafe_timeout_s == 8.5
