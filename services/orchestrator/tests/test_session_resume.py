"""Tests for the session-resume picker helpers.

Per-campaign vault layout: each Vault is a single campaign, with
``campaign.md`` at the root and session logs under ``runs/<slug>/sessions/``.
Tests use the ``default`` run slug for consistency with the migration.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from aidm.session_resume import (
    build_campaign_grouping,
    list_resumable,
    load_resume_message,
    read_campaigns,
)
from aidm.vault import Vault, VaultPathError


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    run_root = tmp_path / "runs" / "default"
    (run_root / "sessions").mkdir(parents=True)
    (run_root / "notes").mkdir(parents=True)
    (run_root / "run.md").write_text(
        "---\ntype: run\ntitle: Test\nstatus: active\ncreated: 2026-01-01\n---\n",
        encoding="utf-8",
    )
    return Vault(tmp_path, run_slug="default")


def _write(path: Path, content: str, *, mtime: float | None = None) -> None:
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        path.touch()
        import os  # noqa: PLC0415

        os.utime(path, (mtime, mtime))


def test_list_resumable_returns_both_sources_newest_first(vault: Vault) -> None:
    base = time.time()
    _write(
        vault.run_root / "sessions" / "2026-01-01-100000.md",
        "---\ntype: session_log\n---\n# Vasfátyol bukása\n\nElara meghalt.\n",
        mtime=base - 7200,
    )
    _write(
        vault.run_root / "sessions" / "2026-01-15-200000.md",
        "# Új város\n\nBalor csatlakozott.\n",
        mtime=base - 300,
    )
    _write(
        vault.run_root / "notes" / "session-abc12345.md",
        "# session-abc12345\n\n---\n*2026-01-20 19:00*\n\nA banditákkal tárgyaltunk.\n",
        mtime=base - 60,
    )

    entries = list_resumable(vault)
    assert [e.id for e in entries] == ["abc12345", "2026-01-15-200000", "2026-01-01-100000"]
    assert [e.source for e in entries] == ["running", "log", "log"]
    assert entries[1].title == "Új város"
    assert entries[2].title == "Vasfátyol bukása"
    assert "type: session_log" not in entries[2].snippet
    assert "Elara meghalt" in entries[2].snippet


def test_list_resumable_ignores_unrelated_note_files(vault: Vault) -> None:
    _write(vault.run_root / "notes" / "elara.md", "# Elara\n\nNotes about Elara.\n")
    _write(vault.run_root / "notes" / "session-aaa.md", "# session-aaa\n\nSummary.\n")
    entries = list_resumable(vault)
    assert [e.id for e in entries] == ["aaa"]


def test_list_resumable_limit(vault: Vault) -> None:
    for i in range(60):
        _write(vault.run_root / "sessions" / f"2026-02-{i:02d}-000000.md", "# x\n\nbody\n")
    entries = list_resumable(vault, limit=10)
    assert len(entries) == 10


def test_load_resume_message_log(vault: Vault) -> None:
    _write(
        vault.run_root / "sessions" / "2026-01-01-100000.md",
        "---\ntype: session_log\n---\nA temető szélén álltunk.\n",
    )
    msg = load_resume_message(vault, "log", "2026-01-01-100000", "hu")
    assert "korábbi session emléke" in msg
    assert "A temető szélén álltunk." in msg
    assert "type: session_log" not in msg


def test_load_resume_message_running(vault: Vault) -> None:
    _write(
        vault.run_root / "notes" / "session-abc12345.md",
        "# session-abc12345\n\nTartalom.\n",
    )
    msg = load_resume_message(vault, "running", "abc12345", "en")
    assert "Memory of a previous session" in msg
    assert "Tartalom." in msg


def test_load_resume_message_running_uses_only_latest_rolling_summary(vault: Vault) -> None:
    vault.append_note("session-abc12345", "Old summary: the bridge still stands.")
    vault.append_note("session-abc12345", "New summary: the bridge burned.")
    msg = load_resume_message(vault, "running", "abc12345", "en")
    assert "New summary: the bridge burned." in msg
    assert "Old summary" not in msg


def test_load_resume_message_unknown_source_raises(vault: Vault) -> None:
    with pytest.raises(VaultPathError, match="unknown resume source"):
        load_resume_message(vault, "bogus", "x", "hu")  # type: ignore[arg-type]


def test_load_resume_message_unknown_id_raises(vault: Vault) -> None:
    with pytest.raises(VaultPathError):
        load_resume_message(vault, "log", "does-not-exist", "hu")


def test_load_resume_message_empty_file_raises(vault: Vault) -> None:
    _write(vault.run_root / "sessions" / "empty.md", "---\nx: y\n---\n\n")
    with pytest.raises(VaultPathError, match="empty"):
        load_resume_message(vault, "log", "empty", "hu")


def test_entry_from_file_populates_metadata(vault: Vault) -> None:
    _write(
        vault.run_root / "sessions" / "2026-06-07-100000.md",
        "---\n"
        "type: session_log\n"
        "date: 2026-06-07\n"
        "campaign: vasfatyol-birodalom\n"
        "characters: [richard]\n"
        "locations: [Fellegvár]\n"
        "---\n"
        "# Session\n\nThe knight died.\n",
    )
    entries = list_resumable(vault)
    assert len(entries) == 1
    e = entries[0]
    assert e.campaign == "vasfatyol-birodalom"
    assert e.date == "2026-06-07"
    assert e.characters == ("richard",)
    assert e.locations == ("Fellegvár",)


# --- read_campaigns + grouping (per-campaign vault model) ---


def test_read_campaigns_returns_active_campaign(vault: Vault) -> None:
    _write(
        vault.root / "campaign.md",
        "---\ntype: campaign\ntitle: Vasfátyol Birodalom\nstatus: active\n"
        "tone: sötét fantasy\nlang: hu\n---\n# Vasfátyol\n",
    )
    campaigns = read_campaigns(vault)
    assert len(campaigns) == 1
    c = campaigns[0]
    # Slug is the vault directory name, not the filename.
    assert c.slug == vault.root.name
    assert c.title == "Vasfátyol Birodalom"
    assert c.status == "active"
    assert c.lang == "hu"


def test_read_campaigns_returns_empty_when_no_campaign_file(vault: Vault) -> None:
    assert read_campaigns(vault) == []


def test_group_by_campaign_routes_sessions_to_active_and_orphans(vault: Vault) -> None:
    vault_slug = vault.root.name
    _write(
        vault.root / "campaign.md",
        "---\ntype: campaign\ntitle: Active\nstatus: active\nlang: hu\n---\n",
    )
    _write(
        vault.run_root / "sessions" / "2026-06-07-100000.md",
        f"---\ncampaign: {vault_slug}\ndate: 2026-06-07\n---\n# A\n",
    )
    _write(
        vault.run_root / "sessions" / "2026-06-06-100000.md",
        "---\ncampaign: some-other\ndate: 2026-06-06\n---\n# B\n",
    )
    _write(
        vault.run_root / "sessions" / "2026-06-05-100000.md",
        "---\ndate: 2026-06-05\n---\n# C (no campaign field)\n",
    )

    grouping = build_campaign_grouping(vault)
    assert [c.slug for c in grouping.campaigns] == [vault_slug]
    active = grouping.campaigns[0]
    # Explicit match (06-07) and no-field-default (06-05) both land in
    # the active campaign; only the explicit cross-vault entry (06-06)
    # orphans.
    assert {s.id for s in active.sessions} == {
        "2026-06-07-100000",
        "2026-06-05-100000",
    }
    assert {e.id for e in grouping.orphans} == {"2026-06-06-100000"}


def test_group_by_campaign_no_campaign_field_defaults_to_active(vault: Vault) -> None:
    # A session log written without a campaign: field (the new tool
    # default) lands in the active campaign, not orphans.
    _write(
        vault.root / "campaign.md",
        "---\ntype: campaign\ntitle: Active\nstatus: active\nlang: hu\n---\n",
    )
    _write(
        vault.run_root / "sessions" / "2026-06-07-100000.md",
        "---\ntype: session_log\ndate: 2026-06-07\n---\n# Recent session\n",
    )

    grouping = build_campaign_grouping(vault)
    assert grouping.orphans == []
    active = grouping.campaigns[0]
    assert [s.id for s in active.sessions] == ["2026-06-07-100000"]


def test_group_by_campaign_with_no_campaign_file(vault: Vault) -> None:
    _write(
        vault.run_root / "sessions" / "2026-06-07-100000.md",
        "---\ncampaign: vasfatyol\n---\n# x\n",
    )
    grouping = build_campaign_grouping(vault)
    assert grouping.campaigns == []
    assert len(grouping.orphans) == 1
    assert grouping.orphans[0].campaign == "vasfatyol"
