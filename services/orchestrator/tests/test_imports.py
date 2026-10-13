"""Sanity tests: the package imports cleanly and the persona prompts are valid."""

from __future__ import annotations

import aidm
import aidm.persona as persona_module
import pytest
from aidm.persona import persona_for


def test_package_imports() -> None:
    assert aidm.__version__


def test_hungarian_persona_has_content() -> None:
    prompt = persona_for("hu")
    assert "mesélő" in prompt
    assert len(prompt) > 200


def test_english_persona_has_content() -> None:
    prompt = persona_for("en")
    assert "Dungeon Master" in prompt
    assert len(prompt) > 200


def test_unknown_locale_raises() -> None:
    with pytest.raises(ValueError, match="No persona prompt"):
        persona_for("de")


def test_private_mode_uses_private_variant_when_present() -> None:
    public = persona_for("hu", mode="public")
    private = persona_for("hu", mode="private")
    # Private prompt is a separate file with deliberately different
    # guidance (shorter turns, stricter tool-use exhortation). It MUST
    # diverge from the public one in private mode.
    assert public != private
    assert "vault_grep" in private  # explicit tool-use directive
    assert len(private) > 200


def test_private_mode_falls_back_to_public_when_variant_missing(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If a future locale has no .private prompt yet, private mode must
    # still boot rather than blowing up at startup — fallback to the
    # public prompt is the documented behavior in persona.py.
    fake_prompts = tmp_path  # type: ignore[assignment]
    (fake_prompts / "dm.zz.md").write_text("zz public prompt", encoding="utf-8")
    monkeypatch.setattr(persona_module, "_PROMPTS_DIR", fake_prompts)
    assert persona_for("zz", mode="private") == "zz public prompt"


def test_session_log_schema_describes_per_campaign_vault() -> None:
    # The vault_write_session_log description is the load-bearing prompt
    # that tells the DM what the session-log frontmatter should look
    # like. After the per-campaign vault overhaul, the date field is
    # server-set and the campaign field is optional (the vault IS a
    # campaign). Guard the relevant phrases so the description doesn't
    # regress into pre-overhaul wording that demands a `date` or asks
    # the DM to list a non-existent campaigns/ directory.
    from aidm.vault_tools import _build_tools_schema  # noqa: PLC0415

    schema = _build_tools_schema()
    write_log = next(f for f in schema.standard_tools if f.name == "vault_write_session_log")
    desc = write_log.description.lower()
    assert "session_log" in desc
    # date is server-filled, not LLM-supplied.
    assert "auto-filled" in desc or "server" in desc
    # The vault root carries campaign.md; the LLM shouldn't be told to
    # list a campaigns/ directory.
    assert "vault_list_dir('campaigns')" not in desc
    assert "campaign.md" in desc
