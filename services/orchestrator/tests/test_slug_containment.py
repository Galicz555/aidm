"""Campaign and run ids reach the filesystem; they must never leave the vault.

One validator (``aidm.vault.validate_slug``) plus resolved-path containment
guards every route that takes a campaign or run id, and every Vault write
that does not go through a slugified filename.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.server_utils import resolve_campaign_dir
from aidm.vault import Vault, VaultPathError, validate_slug
from fastapi import HTTPException

# Campaign and run directory names in the shapes vaults use, plus the
# demo campaign.
REAL_SLUGS = [
    "ember-hollow",
    "default",
    "eval",
    "eval-5t",
    "eval-10t",
    "eval-30t",
    "first-party",
    "playtest",
    "playtest2",
    "playtest3",
]
HOSTILE = [
    "",
    ".",
    "..",
    "../x",
    "a/b",
    "a\\b",
    "/etc",
    "-lead",
    "_template",
    ".hidden",
    "Upper",
    "with space",
    "nul\x00",
    "x\n",
]


@pytest.mark.parametrize("slug", REAL_SLUGS)
def test_real_slugs_pass_the_validator(slug: str) -> None:
    assert validate_slug("campaign", slug) == slug


@pytest.mark.parametrize("slug", HOSTILE)
def test_hostile_slugs_fail_the_validator(slug: str) -> None:
    with pytest.raises(VaultPathError):
        validate_slug("campaign", slug)


def _campaign(root: Path, name: str = "camp") -> Path:
    camp = root / name
    (camp / "runs" / "default").mkdir(parents=True)
    (camp / "campaign.md").write_text("---\ntype: campaign\n---\n", encoding="utf-8")
    return camp


def test_resolve_campaign_dir_accepts_a_real_campaign(tmp_path: Path) -> None:
    camp = _campaign(tmp_path)
    assert resolve_campaign_dir(tmp_path, "camp") == camp


@pytest.mark.parametrize("slug", ["ABS", "../outside", ".."])
def test_resolve_campaign_dir_refuses_paths_outside_the_vault_root(
    tmp_path: Path, slug: str
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = str(outside) if slug == "ABS" else slug
    with pytest.raises(HTTPException) as err:
        resolve_campaign_dir(root, candidate)
    assert err.value.status_code == 404


def test_resolve_campaign_dir_refuses_a_symlink_out_of_the_vault_root(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HTTPException) as err:
        resolve_campaign_dir(root, "link")
    assert err.value.status_code == 404


@pytest.mark.parametrize("run", ["../../outside", "/abs", "..", "a/b"])
def test_vault_refuses_a_run_slug_that_leaves_the_runs_dir(tmp_path: Path, run: str) -> None:
    camp = _campaign(tmp_path / "vault")
    (tmp_path / "outside").mkdir()
    with pytest.raises(VaultPathError):
        Vault(camp, run_slug=run)


def test_vault_refuses_a_run_dir_that_is_a_symlink_out_of_the_vault(tmp_path: Path) -> None:
    camp = _campaign(tmp_path / "vault")
    outside = tmp_path / "outside"
    outside.mkdir()
    (camp / "runs" / "evil").symlink_to(outside, target_is_directory=True)
    with pytest.raises(VaultPathError):
        Vault(camp, run_slug="evil")


# --- run-scoped writes and reads that never went through ``_safe`` ---------


def _vault_with_escaping_links(tmp_path: Path) -> tuple[Vault, Path]:
    camp = _campaign(tmp_path / "vault")
    outside = tmp_path / "outside"
    outside.mkdir()
    run = camp / "runs" / "default"
    (run / "beat.md").symlink_to(outside / "beat-target.md")
    (run / "clock.md").symlink_to(outside / "clock-target.md")
    (run / "sessions").symlink_to(outside, target_is_directory=True)
    return Vault(camp, run_slug="default"), outside


def test_write_beat_cannot_follow_a_link_out_of_the_vault(tmp_path: Path) -> None:
    vault, outside = _vault_with_escaping_links(tmp_path)
    with pytest.raises(VaultPathError):
        vault.write_beat("pwned")
    assert not (outside / "beat-target.md").exists()


def test_write_clock_cannot_follow_a_link_out_of_the_vault(tmp_path: Path) -> None:
    vault, outside = _vault_with_escaping_links(tmp_path)
    with pytest.raises(VaultPathError):
        vault.write_clock("pwned")
    assert not (outside / "clock-target.md").exists()


def test_read_clock_cannot_follow_a_link_out_of_the_vault(tmp_path: Path) -> None:
    vault, outside = _vault_with_escaping_links(tmp_path)
    (outside / "clock-target.md").write_text("secret", encoding="utf-8")
    with pytest.raises(VaultPathError):
        vault.read_clock()


def test_write_session_log_cannot_follow_a_link_out_of_the_vault(tmp_path: Path) -> None:
    vault, outside = _vault_with_escaping_links(tmp_path)
    with pytest.raises(VaultPathError):
        vault.write_session_log("pwned")
    assert list(outside.glob("*.md")) == []


def test_write_tone_stays_inside_the_campaign(tmp_path: Path) -> None:
    camp = _campaign(tmp_path / "vault")
    Vault(camp).write_tone("grim")
    assert (camp / "tone.md").read_text(encoding="utf-8") == "grim"


def test_write_tone_cannot_follow_a_link_out_of_the_vault(tmp_path: Path) -> None:
    camp = _campaign(tmp_path / "vault")
    outside = tmp_path / "outside"
    outside.mkdir()
    (camp / "tone.md").symlink_to(outside / "tone-target.md")
    with pytest.raises(VaultPathError):
        Vault(camp).write_tone("pwned")
    assert not (outside / "tone-target.md").exists()
