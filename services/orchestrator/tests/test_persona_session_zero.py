"""Session-zero persona selection and vault-summary embedding."""

from __future__ import annotations

from aidm.persona import persona_for


def test_session_zero_phase_loads_interviewer_prompt() -> None:
    text = persona_for("en", phase="session-zero")
    # Marker only present in the session-zero prompt body.
    assert "session zero" in text.lower()
    assert "vault_update_character" in text


def test_play_phase_loads_dm_prompt() -> None:
    text = persona_for("en", phase="play")
    # The play-mode prompt opens with the DM intro and references the
    # NPC dialogue contract — both absent from the session-zero file.
    assert "Dungeon Master" in text or "dungeon master" in text.lower()


def test_vault_summary_appended_to_persona() -> None:
    text = persona_for("en", vault_summary="## Vault map\n\ncampaign.md\nrun.md")
    assert "Vault map" in text
    assert "campaign.md" in text


def test_session_zero_falls_back_to_english_for_unknown_locale() -> None:
    # No `dm.xx.session-zero.md` exists; the resolver should fall back to
    # the English session-zero prompt rather than raising.
    text = persona_for("xx", phase="session-zero")
    assert "session zero" in text.lower()
