"""Vault overlay tests — campaign canon + run-scoped subtree.

The campaign vault holds world-canonical content at the root (npcs,
locations, lore, ...). A run lives under ``runs/<slug>/`` and carries
run-specific state (characters, sessions, arcs, notes). The Vault
class exposes a flat virtual view to the LLM — reads of run-scoped
paths route to the active run; canon paths fall through. Writes only
land under the active run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.vault import Vault, VaultPathError, slugify


def _build_layout(root: Path) -> None:
    (root / "npcs").mkdir()
    (root / "npcs" / "elara.md").write_text(
        "---\ntype: npc\n---\n# Elara\nA wandering scholar.\n", encoding="utf-8"
    )
    (root / "lore").mkdir()
    (root / "lore" / "world.md").write_text("# World\n", encoding="utf-8")
    (root / "campaign.md").write_text(
        "---\ntype: campaign\ntitle: Test\nstatus: active\nlang: hu\n---\n",
        encoding="utf-8",
    )
    run_a = root / "runs" / "first-party"
    run_a.mkdir(parents=True)
    (run_a / "run.md").write_text(
        "---\ntype: run\ntitle: First Party\nstatus: active\ncreated: 2026-01-01\n---\n",
        encoding="utf-8",
    )
    (run_a / "characters").mkdir()
    (run_a / "characters" / "lyra.md").write_text(
        "---\ntype: character\nname: Lyra\n---\n# Lyra\n", encoding="utf-8"
    )
    (run_a / "sessions").mkdir()
    run_b = root / "runs" / "replay"
    (run_b / "characters").mkdir(parents=True)
    (run_b / "run.md").write_text(
        "---\ntype: run\ntitle: Replay\nstatus: active\ncreated: 2026-02-01\n---\n",
        encoding="utf-8",
    )
    (run_b / "characters" / "lyra.md").write_text(
        "---\ntype: character\nname: Lyra Replay\n---\n# Lyra Replay\n",
        encoding="utf-8",
    )


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    _build_layout(tmp_path)
    return Vault(tmp_path, run_slug="first-party")


def test_constructor_rejects_unknown_run(tmp_path: Path) -> None:
    _build_layout(tmp_path)
    with pytest.raises(VaultPathError, match="run not found"):
        Vault(tmp_path, run_slug="no-such")


def test_canon_only_mode_blocks_writes(tmp_path: Path) -> None:
    _build_layout(tmp_path)
    v = Vault(tmp_path)
    with pytest.raises(VaultPathError, match="no active run"):
        v.write_session_log("# log\n")
    with pytest.raises(VaultPathError, match="no active run"):
        v.write_character_sheet("a", "# x")


def test_read_run_scoped_path_routes_to_run(vault: Vault) -> None:
    text = vault.read_file("characters/lyra.md")
    assert "Lyra" in text
    assert "Replay" not in text  # other run's lyra.md must not leak in


def test_read_canon_path_succeeds(vault: Vault) -> None:
    text = vault.read_file("npcs/elara.md")
    assert "Elara" in text


def test_read_run_scoped_without_active_run_raises(tmp_path: Path) -> None:
    _build_layout(tmp_path)
    v = Vault(tmp_path)
    with pytest.raises(VaultPathError, match="no active run"):
        v.read_file("characters/lyra.md")


def test_runs_namespace_rejected_on_read(vault: Vault) -> None:
    with pytest.raises(VaultPathError, match="reserved namespace"):
        vault.read_file("runs/replay/characters/lyra.md")


def test_list_dir_root_unions_canon_and_run(vault: Vault) -> None:
    entries = vault.list_dir(".")
    # Canon dirs visible
    assert "npcs/" in entries
    assert "lore/" in entries
    assert "campaign.md" in entries
    # Run-scoped subdirs surface as if at the root
    assert "characters/" in entries
    assert "sessions/" in entries
    assert "run.md" in entries
    # Implementation namespace is hidden
    assert "runs/" not in entries


def test_glob_excludes_inactive_run(vault: Vault) -> None:
    hits = vault.glob("lyra.md")
    # Only one match — the active run's lyra.md, not the other run's
    assert hits == ["characters/lyra.md"]


def test_grep_uses_virtual_path(vault: Vault) -> None:
    hits = vault.grep("Lyra")
    paths = {h.path for h in hits}
    # No "runs/first-party/..." paths bleed into output
    for p in paths:
        assert not p.startswith("runs/")


def test_write_session_log_lands_in_run(vault: Vault) -> None:
    virtual = vault.write_session_log("# Test\nbody\n")
    assert virtual.startswith("sessions/")
    on_disk = vault.run_root / virtual
    assert on_disk.is_file()
    assert (vault.root / virtual).is_file() is False  # not at campaign root


def test_write_character_uses_run_overlay(vault: Vault) -> None:
    virtual = vault.write_character_sheet("ezra", "# Ezra\n")
    assert virtual == "characters/ezra.md"
    assert (vault.run_root / "characters" / "ezra.md").is_file()
    assert not (vault.root / "characters" / "ezra.md").exists()


def test_append_note_lands_in_run(vault: Vault) -> None:
    virtual = vault.append_note("rumour", "first")
    assert virtual.startswith("notes/")
    assert (vault.run_root / "notes" / "rumour.md").is_file()


def test_slugify_rejects_reserved_runs() -> None:
    assert slugify("runs") == ""
    assert slugify("Runs") == ""


def test_slugify_handles_hungarian() -> None:
    assert slugify("Első parti") == "elso-parti"
