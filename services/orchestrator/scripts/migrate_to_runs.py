"""Migrate flat-layout vaults to the campaign → run hierarchy.

Before: ``sessions/``, ``characters/``, ``arcs/``, ``notes/`` lived at the
campaign vault root, implicitly tied to a single playthrough.

After: those subdirectories move under ``runs/default/<subdir>/``, freeing
the vault to host multiple playthroughs of the same world. World-canonical
content (``npcs/``, ``locations/``, ``lore/``, ``factions/``, ``items/``,
``rules/``, ``campaign.md``) stays at the root.

Idempotent: re-running against an already-migrated vault is a no-op.

Usage::

    uv run python -m services.orchestrator.scripts.migrate_to_runs [VAULT_DIR ...]

With no arguments, migrates every campaign vault under ``<repo>/vault/``.
"""

from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

from aidm import frontmatter

DEFAULT_RUN_SLUG = "default"
DEFAULT_RUN_TITLE = "Default Run"
RUN_SUBDIRS = ("characters", "arcs", "sessions", "notes")
RUNS_DIR = "runs"


def migrate(vault_dir: Path) -> str:
    """Migrate one campaign vault. Returns a one-line status string."""
    if not vault_dir.is_dir():
        return f"skip {vault_dir}: not a directory"
    runs_root = vault_dir / RUNS_DIR
    if runs_root.exists():
        return f"skip {vault_dir.name}: runs/ already present"
    movable = [vault_dir / sub for sub in RUN_SUBDIRS if (vault_dir / sub).is_dir()]
    if not movable:
        return f"skip {vault_dir.name}: nothing to migrate (flat dirs absent)"
    target_run = runs_root / DEFAULT_RUN_SLUG
    target_run.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for src in movable:
        dst = target_run / src.name
        shutil.move(str(src), str(dst))
        moved.append(src.name)
    # Ensure every run-scoped subdir exists even if empty post-move so
    # the runtime can write without checking.
    for sub in RUN_SUBDIRS:
        (target_run / sub).mkdir(exist_ok=True)
    _write_run_md(target_run, parent_campaign=vault_dir.name)
    return f"migrated {vault_dir.name}: moved {', '.join(moved)} → runs/{DEFAULT_RUN_SLUG}/"


def _write_run_md(run_root: Path, *, parent_campaign: str) -> None:
    run_md = run_root / "run.md"
    if run_md.exists():
        return
    fm = {
        "type": "run",
        "title": DEFAULT_RUN_TITLE,
        "status": "active",
        "created": date.today().isoformat(),
        "parent_campaign": parent_campaign,
    }
    body = (
        "# Default Run\n\n"
        "This run was created automatically by the migration from the legacy "
        "flat layout. Rename or replace it as the campaign grows new playthroughs.\n"
    )
    run_md.write_text(frontmatter.serialize(fm, body), encoding="utf-8")


def _resolve_targets(argv: list[str]) -> list[Path]:
    if len(argv) > 1:
        return [Path(arg).expanduser().resolve() for arg in argv[1:]]
    vault_root = Path(__file__).resolve().parents[3] / "vault"
    if not vault_root.is_dir():
        return []
    return sorted(p for p in vault_root.iterdir() if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    targets = _resolve_targets(argv)
    if not targets:
        print("migrate_to_runs: no vault directories found", file=sys.stderr)
        return 1
    for vault_dir in targets:
        print(migrate(vault_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
