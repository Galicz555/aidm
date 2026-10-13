"""Vault turn write journal (text regenerate) and the discarded-reply archive."""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.vault import Vault, VaultConflictError, VaultPathError, VaultRestoreIncomplete


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    (tmp_path / "npcs").mkdir()
    (tmp_path / "npcs" / "mara.md").write_text("---\ntype: npc\n---\n# Mara\n", encoding="utf-8")
    run = tmp_path / "runs" / "r1"
    (run / "characters").mkdir(parents=True)
    (run / "characters" / "lyra.state.md").write_text(
        "---\ntype: character_state\nstatus: active\n---\nbefore\n", encoding="utf-8"
    )
    (run / "clock.md").write_text("---\ntype: clock\nday: 1\n---\n", encoding="utf-8")
    return Vault(tmp_path, run_slug="r1")


def test_restore_puts_back_the_pre_image_of_every_written_file(vault: Vault) -> None:
    run = vault.run_root
    assert run is not None
    state_before = (run / "characters" / "lyra.state.md").read_bytes()
    npc_before = (vault.root / "npcs" / "mara.md").read_bytes()
    clock_before = (run / "clock.md").read_bytes()
    vault.begin_journal()

    vault.write_character_state("lyra", "---\ntype: character_state\n---\nafter one\n")
    vault.write_character_state("lyra", "---\ntype: character_state\n---\nafter two\n")
    vault.write_npc_sheet("mara", "---\ntype: npc\n---\n# Mara, rewritten\n")
    vault.write_clock("---\ntype: clock\nday: 2\n---\n")
    vault.append_note("session-x", "a summary")  # new file
    vault.append_note("session-x", "a second block")

    vault.restore_journal()
    assert (run / "characters" / "lyra.state.md").read_bytes() == state_before
    assert (vault.root / "npcs" / "mara.md").read_bytes() == npc_before
    assert (run / "clock.md").read_bytes() == clock_before


def test_restore_deletes_files_the_turn_created(vault: Vault) -> None:
    run = vault.run_root
    assert run is not None
    vault.begin_journal()
    log = vault.write_session_log("---\ntype: session_log\n---\nthe log\n")
    vault.write_npc_state("stranger", "---\ntype: npc_state\n---\n")
    vault.append_note("clue", "found a key")

    vault.restore_journal()
    assert not (run / log).exists()
    assert not (run / "npcs" / "stranger.state.md").exists()
    assert not (run / "notes" / "clue.md").exists()
    assert not list(run.rglob(".*.rollback"))  # no stage left behind


def test_writes_outside_a_journal_are_not_rolled_back(vault: Vault) -> None:
    run = vault.run_root
    assert run is not None
    vault.write_beat("slow and quiet")  # before begin_journal
    vault.begin_journal()
    vault.end_journal()
    vault.write_beat("fast")  # after end_journal
    vault.restore_journal()
    assert (run / "beat.md").read_text(encoding="utf-8") == "fast"


def test_a_journal_restores_once_and_then_starts_empty(vault: Vault) -> None:
    run = vault.run_root
    assert run is not None
    vault.begin_journal()
    vault.write_beat("tense")
    vault.restore_journal()
    vault.write_beat("tense again")  # journaled afresh: pre-image = absent
    vault.restore_journal()
    assert not (run / "beat.md").exists()


def test_a_staging_failure_restores_nothing(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    run = vault.run_root
    assert run is not None
    vault.begin_journal()
    vault.write_character_state("lyra", "---\ntype: character_state\n---\nafter\n")
    vault.write_beat("new beat")

    def refuse(_self: Path, _data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", refuse)
    with pytest.raises(VaultPathError, match="nothing restored"):
        vault.restore_journal()
    monkeypatch.undo()
    assert "after" in (run / "characters" / "lyra.state.md").read_text(encoding="utf-8")
    assert (run / "beat.md").exists()  # the deletion phase never ran
    vault.restore_journal()  # the journal survived for a retry
    assert "before" in (run / "characters" / "lyra.state.md").read_text(encoding="utf-8")


def test_the_discarded_archive_is_never_journaled_and_hidden_from_the_dm(vault: Vault) -> None:
    vault.begin_journal()
    archived = vault.write_discarded_reply("## Discarded reply\n\nThe dragon laughs.\n")
    vault.restore_journal()

    path = vault.root / archived
    assert archived.startswith("runs/r1/discarded/") and path.is_file()
    name = path.name
    with pytest.raises(VaultPathError):
        vault.read_file(f"discarded/{name}")
    with pytest.raises(VaultPathError):
        vault.list_dir("discarded")
    assert "discarded/" not in vault.list_dir(".")
    assert vault.grep("dragon laughs") == []
    assert all("discarded" not in hit for hit in vault.glob("*"))


def test_a_file_another_writer_changed_blocks_the_whole_restore(vault: Vault) -> None:
    # Review #1 repro: session zero creates lyra.md, the player edits it in the
    # browser (another Vault instance), then regenerates: lyra.md was unlinked.
    run = vault.run_root
    assert run is not None
    editor = Vault(vault.root, run_slug="r1")  # what build_session_vault() hands the PUT routes
    vault.begin_journal()
    vault.write_character_sheet("lyra", "turn-written sheet\n")
    vault.write_beat("turn beat")
    editor.write_character_sheet("lyra", "PLAYER EDIT\n")

    with pytest.raises(VaultConflictError, match=r"characters/lyra\.md"):
        vault.restore_journal()

    assert (run / "characters" / "lyra.md").read_text(encoding="utf-8") == "PLAYER EDIT\n"
    assert (run / "beat.md").exists()  # all-or-nothing: the clean file was not restored either
    with pytest.raises(VaultConflictError):
        vault.restore_journal()  # still journaled, still refused


def test_a_file_deleted_by_another_writer_is_a_conflict(vault: Vault) -> None:
    run = vault.run_root
    assert run is not None
    vault.begin_journal()
    vault.write_beat("turn beat")
    (run / "beat.md").unlink()
    with pytest.raises(VaultConflictError):
        vault.restore_journal()


def test_an_incomplete_commit_keeps_the_rest_journaled_for_a_retry(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os  # noqa: PLC0415

    run = vault.run_root
    assert run is not None
    state = run / "characters" / "lyra.state.md"
    before = state.read_bytes()
    vault.begin_journal()
    vault.write_character_state("lyra", "---\ntype: character_state\n---\nafter\n")
    vault.write_beat("turn beat")
    real_replace = os.replace

    def flaky_replace(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == state:
            raise OSError("device busy")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(VaultRestoreIncomplete, match=r"lyra\.state\.md"):
        vault.restore_journal()
    assert not (run / "beat.md").exists()  # the file that could come back, did
    assert not list(run.rglob(".*.rollback"))  # the failed stage was cleaned up
    monkeypatch.undo()

    vault.restore_journal()  # the retry finishes only what stayed journaled
    assert state.read_bytes() == before
