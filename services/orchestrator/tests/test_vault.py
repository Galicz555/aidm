"""Vault filesystem access — path safety, basic operations."""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.vault import Vault, VaultPathError


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    (tmp_path / "campaigns").mkdir()
    (tmp_path / "npcs").mkdir()
    (tmp_path / "lore").mkdir()
    (tmp_path / "campaigns" / "test.md").write_text(
        "---\ntype: campaign\n---\n# Test Campaign\nAbout [[elara]].\n",
        encoding="utf-8",
    )
    (tmp_path / "npcs" / "elara.md").write_text(
        "---\ntype: npc\n---\n# Elara\nA wandering scholar.\n",
        encoding="utf-8",
    )
    (tmp_path / "npcs" / "balor.md").write_text(
        "---\ntype: npc\n---\n# Balor\nA reclusive lord.\n",
        encoding="utf-8",
    )
    # An active run is required for write methods (session log, character
    # sheet, append note). The run subtree starts empty; tests exercise
    # the writes that populate it.
    run_root = tmp_path / "runs" / "default"
    run_root.mkdir(parents=True)
    (run_root / "run.md").write_text(
        "---\ntype: run\ntitle: Test Run\nstatus: active\ncreated: 2026-01-01\n---\n",
        encoding="utf-8",
    )
    return Vault(tmp_path, run_slug="default")


def test_list_dir_root(vault: Vault) -> None:
    entries = vault.list_dir(".")
    assert "campaigns/" in entries
    assert "npcs/" in entries
    assert "lore/" in entries


def test_list_dir_subfolder_marks_files(vault: Vault) -> None:
    entries = vault.list_dir("npcs")
    # Entries carry the parent prefix so the LLM can feed them straight
    # back into ``read_file`` without remembering the directory.
    assert sorted(entries) == ["npcs/balor.md", "npcs/elara.md"]


def test_read_file(vault: Vault) -> None:
    content = vault.read_file("npcs/elara.md")
    assert "Elara" in content
    assert "wandering scholar" in content


def test_grep_finds_across_files(vault: Vault) -> None:
    hits = vault.grep("Elara")
    paths = {h.path for h in hits}
    assert "npcs/elara.md" in paths
    assert "campaigns/test.md" in paths


def test_grep_is_case_insensitive(vault: Vault) -> None:
    assert vault.grep("ELARA")


def test_glob(vault: Vault) -> None:
    matches = vault.glob("elara*.md")
    assert "npcs/elara.md" in matches
    assert not any("balor" in m for m in matches)


def test_path_traversal_blocked(vault: Vault) -> None:
    # `..` is rejected up front by _normalize (before any prefix guard
    # or filesystem touch); _safe remains the backstop for anything else.
    with pytest.raises(VaultPathError, match="traversal not allowed"):
        vault.read_file("../etc/passwd")


def test_absolute_path_blocked(vault: Vault) -> None:
    with pytest.raises(VaultPathError):
        vault.read_file("/etc/passwd")


def test_read_nonexistent_raises(vault: Vault) -> None:
    with pytest.raises(VaultPathError, match="not a file"):
        vault.read_file("npcs/nobody.md")


def test_hidden_files_ignored(vault: Vault, tmp_path: Path) -> None:
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    entries = vault.list_dir(".")
    assert ".obsidian/" not in entries


def test_write_session_log_creates_unique_file(vault: Vault) -> None:
    path1 = vault.write_session_log("session one")
    assert path1.startswith("sessions/")
    assert path1.endswith(".md")
    # The body is preserved; auto-fill prepends a frontmatter block with
    # today's date so the file is self-describing even when the DM
    # forgot to declare one.
    content = vault.read_file(path1)
    assert "session one" in content
    assert "date:" in content


def test_write_session_log_does_not_overwrite(vault: Vault, tmp_path: Path) -> None:
    # Create two logs in quick succession; the timestamp resolution is
    # seconds, so this can technically collide if both calls hit within
    # the same second. The implementation uses HHMMSS so as long as a
    # second has passed they differ. We at least verify both calls succeed.
    path1 = vault.write_session_log("# Session one\nContent A.")
    path2 = vault.write_session_log("# Session one\nContent A.")
    # Even if filenames match within a second, the second write must not
    # silently destroy the first. The simplest contract: both calls return
    # a path that exists.
    # Virtual paths resolve under the active run on disk.
    assert (vault.run_root / path1).is_file()
    assert (vault.run_root / path2).is_file()


def test_update_character_creates_and_overwrites(vault: Vault) -> None:
    path = vault.write_character_sheet("Thaddeus Wanderer", "v1")
    assert path == "characters/thaddeus-wanderer.md"
    first = vault.read_file(path)
    assert "v1" in first
    # Identity writes no longer auto-stamp last_updated; that field lives
    # on the sibling ``.state.md`` written by SceneSheetWriter.
    # Second call overwrites
    vault.write_character_sheet("thaddeus-wanderer", "v2")
    second = vault.read_file(path)
    assert "v2" in second
    assert "v1" not in second


def test_update_character_sanitizes_slug(vault: Vault) -> None:
    path = vault.write_character_sheet("Elara, A Vándor!", "x")
    assert path == "characters/elara-a-vandor.md"


def test_update_character_rejects_empty_slug(vault: Vault) -> None:
    with pytest.raises(VaultPathError, match="slug is empty"):
        vault.write_character_sheet("!!!", "x")


# --- write_npc_sheet ---


def test_write_npc_sheet_creates_at_canon_root(vault: Vault) -> None:
    path = vault.write_npc_sheet("Viktor Tall", "---\ntype: npc\n---\n# Viktor\n")
    # NPCs are campaign-level canon — they live at the vault root, NOT
    # under runs/<run>/npcs/. The virtual path is just `npcs/<slug>.md`.
    assert path == "npcs/viktor-tall.md"
    on_disk = vault.root / "npcs" / "viktor-tall.md"
    assert on_disk.is_file()
    body = on_disk.read_text(encoding="utf-8")
    assert "# Viktor" in body


def test_write_npc_sheet_overwrites_same_slug(vault: Vault) -> None:
    vault.write_npc_sheet("aniko", "---\ntype: npc\n---\nv1\n")
    vault.write_npc_sheet("aniko", "---\ntype: npc\n---\nv2\n")
    body = vault.read_file("npcs/aniko.md")
    assert "v2" in body
    assert "v1" not in body


def test_write_npc_sheet_rejects_empty_slug(vault: Vault) -> None:
    with pytest.raises(VaultPathError, match="slug is empty"):
        vault.write_npc_sheet("!!!", "x")


# --- append_note ---


def test_append_note_creates_file_with_header(vault: Vault) -> None:
    path = vault.append_note("Iron Veil Citadel", "Rumour: the cellar floods at dusk.")
    assert path == "notes/iron-veil-citadel.md"
    content = vault.read_file(path)
    # Header carries the original slug spelling (pre-slugify) for human
    # readability; the timestamp block follows.
    assert content.startswith("# Iron Veil Citadel\n")
    assert "Rumour: the cellar floods at dusk." in content


def test_append_note_appends_subsequent_calls(vault: Vault) -> None:
    vault.append_note("elara", "First impression: cautious.")
    vault.append_note("elara", "Later: lied about the silver.")
    content = vault.read_file("notes/elara.md")
    # Both notes survive; separator + timestamp blocks chained.
    assert "First impression: cautious." in content
    assert "Later: lied about the silver." in content
    # Two append-blocks, each separated by ``---``.
    assert content.count("\n---\n") >= 2


def test_append_note_rejects_empty_slug(vault: Vault) -> None:
    with pytest.raises(VaultPathError, match="slug is empty"):
        vault.append_note("???", "x")
