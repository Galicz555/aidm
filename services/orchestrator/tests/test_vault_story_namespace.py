"""Story-engine vault namespace — hidden from the LLM, open to the engine.

``runs/<slug>/events/`` and ``clock.md`` belong to aidm.story_events:
the DM reading future events is a plot spoiler, so the LLM-facing
surface (read_file / list_dir / grep / glob / tree) must not see them,
while the dedicated accessors must.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.vault import Vault, VaultPathError


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    (tmp_path / "campaign.md").write_text(
        "---\ntype: campaign\ntitle: T\nstatus: active\nlang: en\n---\n", encoding="utf-8"
    )
    run = tmp_path / "runs" / "test"
    (run / "events").mkdir(parents=True)
    (run / "run.md").write_text(
        "---\ntype: run\ntitle: T\nstatus: active\ncreated: 2026-07-11\n---\n", encoding="utf-8"
    )
    (run / "clock.md").write_text(
        "---\ntype: clock\nday: 2\nsegment: night\nturn: 5\nlast_updated: 2026-07-11\n---\n"
        "\n## Time log\n\n_(no advances yet)_\n",
        encoding="utf-8",
    )
    (run / "events" / "secret-raid.md").write_text(
        "---\ntype: event\nlang: en\ntitle: Secret raid\nstatus: pending\ntrigger_day: 5\n"
        "last_updated: 2026-07-11\n---\n\n## Directive\n\nThe raid comes.\n",
        encoding="utf-8",
    )
    return Vault(tmp_path, run_slug="test")


def test_read_file_refuses_clock_and_events(vault: Vault) -> None:
    with pytest.raises(VaultPathError):
        vault.read_file("clock.md")
    with pytest.raises(VaultPathError):
        vault.read_file("events/secret-raid.md")


def test_hidden_guard_survives_path_games(vault: Vault) -> None:
    # Un-normalized segments must not slip past the prefix guards
    # (reproduced bypass shapes from review 2026-07-11).
    for sneaky in (
        "./clock.md",
        ".//clock.md",
        "notes/../clock.md",
        "./events/secret-raid.md",
        "events/./secret-raid.md",
        "runs/test/clock.md",
        "./runs/test/events/secret-raid.md",
    ):
        with pytest.raises(VaultPathError):
            vault.read_file(sneaky)
    with pytest.raises(VaultPathError):
        vault.list_dir(".//./events")
    root_games = vault.list_dir(".//.")
    assert "clock.md" not in root_games
    assert "events/" not in root_games


def test_list_dir_hides_the_story_namespace(vault: Vault) -> None:
    entries = vault.list_dir(".")
    assert "clock.md" not in entries
    assert "events/" not in entries
    with pytest.raises(VaultPathError):
        vault.list_dir("events")


def test_grep_and_glob_do_not_leak_future_events(vault: Vault) -> None:
    assert vault.grep("raid") == []
    assert vault.glob("*.md") == ["campaign.md", "run.md"]
    assert "events" not in vault.tree()


def test_engine_accessors_bypass_the_guard(vault: Vault) -> None:
    clock = vault.read_clock()
    assert clock is not None and "segment: night" in clock
    events = vault.list_events()
    assert [slug for slug, _ in events] == ["secret-raid"]
    vault.write_clock(clock.replace("night", "morning"))
    reread = vault.read_clock()
    assert reread is not None and "segment: morning" in reread
    assert "last_updated:" in reread


def test_write_event_stamps_and_round_trips(vault: Vault) -> None:
    vault.write_event(
        "secret-raid",
        "---\ntype: event\nlang: en\ntitle: Secret raid\nstatus: fired\ntrigger_day: 5\n---\n"
        "\n## Directive\n\nThe raid comes.\n",
    )
    _, raw = vault.list_events()[0]
    assert "status: fired" in raw
    assert "last_updated:" in raw


def test_canon_only_vault_has_no_clock_or_events(tmp_path: Path) -> None:
    (tmp_path / "campaign.md").write_text(
        "---\ntype: campaign\ntitle: T\nstatus: active\nlang: en\n---\n", encoding="utf-8"
    )
    vault = Vault(tmp_path)
    assert vault.read_clock() is None
    assert vault.list_events() == []
    with pytest.raises(VaultPathError):
        vault.write_clock("---\ntype: clock\n---\n")
