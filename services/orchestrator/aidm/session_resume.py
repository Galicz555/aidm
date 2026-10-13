"""Persistent session resume — per-campaign vault redesign.

The orchestrator binds to one campaign vault at a time
(``VAULT_PATH``). A campaign IS the vault directory; its top-level
state lives in ``campaign.md`` at the vault root. Session logs live in
``sessions/`` and may carry a ``campaign:`` frontmatter field for
cross-vault provenance (kept for backward compatibility with logs
written before the restructure). Sessions inside the active vault that
don't match the active campaign slug are still surfaced as orphans so
the UI can flag them rather than silently drop them.

Two file sources feed the session listing:

* ``vault/sessions/YYYY-MM-DD-HHMMSS.md`` — end-of-session logs the DM
  writes via ``vault_write_session_log``.
* ``vault/notes/session-<id>.md`` — running summaries the auto
  ``SessionSummarizer`` appends every N turns.

On selection, the chosen file's body is loaded as a system message
ahead of the empty conversation history so the LLM starts the new
session already remembering where the campaign was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from aidm import frontmatter as _fm
from aidm.vault import Vault, VaultPathError

ResumeSource = Literal["log", "running"]

_SNIPPET_CHARS = 240
# Leads the resume system message so the summarizer can fold the earlier
# session into the new rolling summary (and thus the new notes file).
RESUME_TAG = "<!--aidm:resume-memory-->"
# Vault.append_note's block separator: "\n---\n*YYYY-MM-DD HH:MM*\n".
_NOTE_BLOCK_RE = re.compile(r"\n---\n\*\d{4}-\d{2}-\d{2} \d{2}:\d{2}\*\n")


@dataclass(frozen=True)
class ResumeEntry:
    """One row in the session picker."""

    id: str
    source: ResumeSource
    path: str  # vault-relative
    title: str
    mtime: float
    snippet: str
    # Frontmatter-derived metadata. Empty / None when the source file
    # has no frontmatter (running summaries, malformed logs).
    campaign: str | None = None
    date: str | None = None
    characters: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()


@dataclass(frozen=True)
class CampaignSummary:
    """One row in the campaign-list pane."""

    slug: str
    title: str
    status: str  # "active" | "archived" | "" (unset)
    tone: str
    lang: str
    path: str  # vault-relative
    sessions: list[ResumeEntry] = field(default_factory=list)


@dataclass(frozen=True)
class CampaignGrouping:
    """What the ``/api/campaigns`` endpoint returns to the UI."""

    campaigns: list[CampaignSummary]
    orphans: list[ResumeEntry]


# --- public API ----------------------------------------------------------


def list_resumable(vault: Vault, *, limit: int = 40) -> list[ResumeEntry]:
    """Return up to ``limit`` past-session entries, newest first.

    Backward-compatible flat list; the campaign-aware UI uses
    ``build_campaign_grouping`` instead, but the legacy ``/api/sessions``
    endpoint still serves this shape.
    """
    entries: list[ResumeEntry] = []
    entries.extend(_collect_sessions_dir(vault))
    entries.extend(_collect_running_notes(vault))
    entries.sort(key=lambda e: e.mtime, reverse=True)
    return entries[:limit]


def read_campaigns(vault: Vault) -> list[CampaignSummary]:
    """Return a one-element list describing the active campaign.

    Reads ``campaign.md`` at the vault root. The campaign slug is the
    vault directory name. Returns ``[]`` if no ``campaign.md`` exists —
    callers (the picker UI) treat that as "no campaign metadata", and
    every session log becomes an orphan.

    Multi-campaign vaults are out of scope; the orchestrator binds to
    one campaign at a time via ``VAULT_PATH``.
    """
    path = vault.root / "campaign.md"
    if not path.is_file():
        return []
    summary = _campaign_from_file(path, vault.root, slug=vault.root.name)
    return [summary] if summary is not None else []


def build_campaign_grouping(vault: Vault) -> CampaignGrouping:
    """Pull campaigns + sessions and group them.

    Sessions whose frontmatter ``campaign`` slug matches a known
    campaign join that bucket. Sessions with no ``campaign`` field at
    all default to the active campaign when the vault holds exactly
    one (the per-campaign-vault model: each vault IS a campaign, and
    the session-log tool no longer demands the field). Sessions whose
    ``campaign`` is set but doesn't match any known campaign still
    land in ``orphans`` — they're cross-vault provenance and we don't
    want to silently re-bucket them.

    Campaign ordering: ``status: active`` first, then unset, then
    ``archived``. Sessions newest-first within each bucket.
    """
    campaigns = read_campaigns(vault)
    entries = list_resumable(vault, limit=200)
    # Index campaigns by slug for O(1) bucketing.
    by_slug: dict[str, list[ResumeEntry]] = {c.slug: [] for c in campaigns}
    orphans: list[ResumeEntry] = []
    # When the vault carries exactly one campaign, sessions without an
    # explicit campaign: field belong to it by default.
    default_slug = campaigns[0].slug if len(campaigns) == 1 else None
    for entry in entries:
        target = entry.campaign or default_slug or ""
        if target and target in by_slug:
            by_slug[target].append(entry)
        else:
            orphans.append(entry)
    # Attach sessions to summaries; ResumeEntry is frozen so we rebuild
    # the CampaignSummary with its sessions populated.
    filled: list[CampaignSummary] = []
    for c in campaigns:
        filled.append(
            CampaignSummary(
                slug=c.slug,
                title=c.title,
                status=c.status,
                tone=c.tone,
                lang=c.lang,
                path=c.path,
                sessions=by_slug[c.slug],
            )
        )
    filled.sort(key=_campaign_sort_key)
    return CampaignGrouping(campaigns=filled, orphans=orphans)


def load_resume_message(vault: Vault, source: ResumeSource, entry_id: str, locale: str) -> str:
    """Return the system-message body that injects the chosen session.

    Raises ``VaultPathError`` for unknown id / source so the HTTP layer
    can 404 cleanly.
    """
    if source == "log":
        path = f"sessions/{entry_id}.md"
    elif source == "running":
        path = f"notes/session-{entry_id}.md"
    else:
        raise VaultPathError(f"unknown resume source: {source!r}")
    body = vault.read_file(path)
    cleaned = _fm.strip(body).strip()
    if source == "running":
        # Rolling summaries: each appended block supersedes all earlier ones.
        cleaned = _NOTE_BLOCK_RE.split(cleaned)[-1].strip()
    if not cleaned:
        raise VaultPathError(f"resume file is empty: {path}")
    return _wrap_for_context(cleaned, source, entry_id, locale)


# --- internals -----------------------------------------------------------


def _collect_sessions_dir(vault: Vault) -> list[ResumeEntry]:
    base = vault.run_root
    if base is None:
        return []
    sessions_root = base / "sessions"
    if not sessions_root.is_dir():
        return []
    out: list[ResumeEntry] = []
    for path in sessions_root.glob("*.md"):
        if path.name.startswith("."):
            continue
        entry = _entry_from_file(path, "log", base)
        if entry is not None:
            out.append(entry)
    return out


def _collect_running_notes(vault: Vault) -> list[ResumeEntry]:
    base = vault.run_root
    if base is None:
        return []
    notes_root = base / "notes"
    if not notes_root.is_dir():
        return []
    out: list[ResumeEntry] = []
    for path in notes_root.glob("session-*.md"):
        if path.name.startswith("."):
            continue
        entry = _entry_from_file(path, "running", base)
        if entry is not None:
            out.append(entry)
    return out


def _entry_from_file(path: Path, source: ResumeSource, vault_root: Path) -> ResumeEntry | None:
    try:
        text = path.read_text(encoding="utf-8")
        stat = path.stat()
    except OSError:
        return None
    rel = str(path.relative_to(vault_root))
    raw_id = path.stem
    if source == "running" and raw_id.startswith("session-"):
        entry_id = raw_id[len("session-") :]
    else:
        entry_id = raw_id
    fm = _fm.parse(text)
    title = _derive_title(text, fallback=path.stem)
    return ResumeEntry(
        id=entry_id,
        source=source,
        path=rel,
        title=title,
        mtime=stat.st_mtime,
        snippet=_snippet(text),
        campaign=_str_or_none(fm.get("campaign")),
        date=_str_or_none(fm.get("date")),
        characters=_list_of_str(fm.get("characters")),
        locations=_list_of_str(fm.get("locations")),
    )


def _campaign_from_file(
    path: Path, vault_root: Path, *, slug: str | None = None
) -> CampaignSummary | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fm = _fm.parse(text)
    return CampaignSummary(
        slug=slug or path.stem,
        title=str(fm.get("title") or path.stem),
        status=str(fm.get("status") or ""),
        tone=str(fm.get("tone") or ""),
        lang=str(fm.get("lang") or ""),
        path=str(path.relative_to(vault_root)),
        sessions=[],
    )


def _campaign_sort_key(c: CampaignSummary) -> tuple[int, str]:
    # active (0) → unset (1) → archived (2) → other (3); within a
    # bucket, alphabetical by title for stability.
    rank = {"active": 0, "": 1, "archived": 2}.get(c.status.lower(), 3)
    return (rank, c.title.lower())


def _derive_title(text: str, *, fallback: str) -> str:
    body = _fm.strip(text).lstrip()
    # Prefer the first markdown heading; fall back to the first
    # non-empty line; fall back to the filename stem.
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip() or fallback
        return stripped[:80]
    return fallback


def _snippet(text: str) -> str:
    body = _fm.strip(text).strip()
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    snippet = " ".join(line.strip() for line in lines if line.strip())
    if len(snippet) > _SNIPPET_CHARS:
        snippet = snippet[:_SNIPPET_CHARS].rstrip() + "…"
    return snippet


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return None  # frontmatter said "campaign: [a, b]"; treat as malformed
    text = str(value).strip()
    return text or None


def _list_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    # Single-string value typed in the frontmatter — wrap it so the UI
    # gets one chip instead of an empty list.
    text = str(value).strip()
    return (text,) if text else ()


def _wrap_for_context(body: str, source: ResumeSource, entry_id: str, locale: str) -> str:
    label = "session-log" if source == "log" else "running-summary"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if locale == "hu":
        return (
            f"{RESUME_TAG}\n"
            f"[Egy korábbi session emléke ({label}, id={entry_id}, betöltve {timestamp}). "
            "Kezeld úgy mintha a saját emléked lenne. NE idézd szó szerint a játékosnak, "
            "de használd a folytonossághoz — a karakterek, helyszínek, és nyitott szálak "
            "innen jönnek. Ez a folytonosság már nálad van — NE listázd újra a "
            "`sessions/` mappát és NE olvasd be újra ezt a logot a nyitó körben; "
            "a nyitó mondatod erre a szövegre épüljön.]\n\n"
            f"{body}"
        )
    return (
        f"{RESUME_TAG}\n"
        f"[Memory of a previous session ({label}, id={entry_id}, loaded {timestamp}). "
        "Treat it as your own memory. Do not quote it verbatim to the player, but use it "
        "for continuity — characters, locations, and open threads come from here. "
        "This continuity is already loaded — do NOT call vault_list_dir('sessions') "
        "or re-read this log in your opening turn; anchor your first reply to the "
        "body below.]\n\n"
        f"{body}"
    )
