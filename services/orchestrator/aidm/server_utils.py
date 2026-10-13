"""Shared utilities used by both aidm.server (WebRTC) and aidm.text_server (HTTP/SSE).

Vault + campaign helpers that are identical across transports: resolving
campaign directories, building per-session Vaults, listing campaigns and
runs on disk, reading title frontmatter. Both servers import these under
their original private aliases so no call sites need updating.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException

from aidm import frontmatter
from aidm.config import Config
from aidm.vault import Vault, VaultPathError, validate_slug


def from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def resolve_campaign_dir(vault_root: Path, slug: str) -> Path:
    """Resolve ``vault_root/<slug>``; raises 404 if absent, malformed or outside the root.

    ``slug`` comes straight from a query string, a path or a request body.
    The slug shape rules out ``..`` and absolute paths; the resolved-path
    check also catches a campaign directory that is a symlink out of the root.
    """
    not_found = HTTPException(status_code=404, detail=f"campaign not found: {slug}")
    try:
        validate_slug("campaign", slug)
    except VaultPathError as exc:
        raise not_found from exc
    candidate = vault_root / slug
    if not candidate.is_dir() or not candidate.resolve().is_relative_to(vault_root.resolve()):
        raise not_found
    return candidate


def build_session_vault(config: Config, campaign_slug: str, run_slug: str) -> Vault:
    """Build a per-session Vault rooted at ``vault_root/<campaign>`` with the active run."""
    campaign_dir = resolve_campaign_dir(config.vault_root, campaign_slug)
    try:
        return Vault(campaign_dir, run_slug=run_slug)
    except VaultPathError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def list_campaigns_on_disk(vault_root: Path) -> list[dict[str, str]]:
    """Inventory ``vault_root/*`` for directories that contain a ``campaign.md``."""
    if not vault_root.is_dir():
        return []
    out: list[dict[str, str]] = []
    for entry in sorted(vault_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        campaign_md = entry / "campaign.md"
        if not campaign_md.is_file():
            continue
        title = entry.name
        try:
            fm = frontmatter.parse(campaign_md.read_text(encoding="utf-8"))
            fm_title = str(fm.get("title") or "").strip()
            if fm_title:
                title = fm_title
        except OSError:
            pass
        out.append({"slug": entry.name, "title": title})
    return out


# ``runs/_template/`` — story-engine seed (``clock.md`` + ``events/``) copied
# into every run created from the UI. Under ``runs/`` it is invisible to the
# DM's vault tools (plot spoilers); it is not a run and is never listed.
RUN_TEMPLATE_DIR = "_template"


def seed_run_from_template(run_dir: Path) -> None:
    """Copy the campaign's run template into a freshly created ``run_dir``.

    A UI-created run had no clock and no events, so the story engine was
    off for the whole 2026-09-27 playtest. No template = nothing copied.
    """
    template = run_dir.parent / RUN_TEMPLATE_DIR
    if template.is_dir():
        shutil.copytree(template, run_dir, dirs_exist_ok=True)


def list_runs_on_disk(campaign_dir: Path) -> list[dict[str, object]]:
    """Inventory ``runs/*/run.md`` for ``/api/campaigns/{slug}/runs``."""
    runs_root = campaign_dir / "runs"
    if not runs_root.is_dir():
        return []
    out: list[dict[str, object]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith(".") or run_dir.name == RUN_TEMPLATE_DIR:
            continue
        out.append(run_summary(run_dir))
    return out


def run_summary(run_dir: Path) -> dict[str, object]:
    """Parse ``run.md`` frontmatter and count sessions for one run."""
    run_md = run_dir / "run.md"
    fm: dict[str, object] = {}
    if run_md.is_file():
        fm = frontmatter.parse(run_md.read_text(encoding="utf-8"))
    sessions_dir = run_dir / "sessions"
    sessions = sorted(sessions_dir.glob("*.md")) if sessions_dir.is_dir() else []
    last_session_date: str | None = None
    if sessions:
        # Filenames are ISO timestamps (YYYY-MM-DD-HHMMSS.md); last sorts newest.
        last_session_date = "-".join(sessions[-1].stem.split("-")[:3])
    return {
        "slug": run_dir.name,
        "title": str(fm.get("title") or run_dir.name),
        "status": str(fm.get("status") or "active"),
        "created": str(fm.get("created") or ""),
        "session_count": len(sessions),
        "last_session_date": last_session_date,
    }


def read_run_title(vault: Vault) -> str:
    """Return the ``run.md`` title for the active run, or empty string if missing."""
    run_root = vault.run_root
    if run_root is None:
        return ""
    run_md = run_root / "run.md"
    if not run_md.is_file():
        return ""
    fm = frontmatter.parse(run_md.read_text(encoding="utf-8"))
    return str(fm.get("title") or "").strip()


def read_campaign_title(vault: Vault) -> str:
    """Return the ``campaign.md`` title at the vault root, or empty string if missing."""
    campaign_md = vault.root / "campaign.md"
    if not campaign_md.is_file():
        return ""
    fm = frontmatter.parse(campaign_md.read_text(encoding="utf-8"))
    return str(fm.get("title") or "").strip()
