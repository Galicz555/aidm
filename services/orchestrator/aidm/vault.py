"""Safe filesystem access to the Obsidian vault.

Karpathy-style "RAG": no embeddings, no index, no vector DB. The DM-LLM
navigates a folder of markdown files via the methods below. The Vault
class is the only abstraction — it gives the LLM tools just rich enough
to find what it needs, and locks them inside the vault root so a hostile
prompt cannot escape into the host filesystem.

Two augmentations layered on top of the original read/write surface:

1. ``read_file`` appends a ``# Referenced by`` footer to markdown files
   listing other vault files that wikilink to the one being read. This
   gives the LLM cheap relational context without a separate tool call.
2. Write paths (``write_character_sheet``, ``write_session_log``) auto-
   fill timestamp frontmatter fields so LLM-supplied (and inevitably
   wrong) dates can't rot the vault.

Overlay (campaign → run): the vault hosts a campaign world plus N runs
under ``runs/<run-slug>/``. World-canonical content (npcs, locations,
lore, factions, items, rules, campaign.md) lives at the vault root; run-
specific content (characters, arcs, sessions, notes, run.md) lives under
the active run subtree. The LLM sees a *flattened* view — it reads and
writes ``characters/lyra.md`` and the overlay routes the call to
``runs/<run-slug>/characters/lyra.md``. The ``runs/`` directory itself
is hidden from list_dir/glob/grep so inactive runs don't bleed in.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import frontmatter


@dataclass(frozen=True)
class GrepHit:
    path: str
    line_number: int
    text: str


class VaultPathError(ValueError):
    """Raised when a requested path escapes the vault root or doesn't resolve."""


class VaultConflictError(VaultPathError):
    """A journaled file was changed by another writer since this turn wrote it."""


class VaultRestoreIncomplete(VaultPathError):
    """A rollback restored some files but not all; the rest stay journaled for a retry."""


@dataclass
class _JournalEntry:
    """One file in the turn journal: its bytes before the turn (None = absent) and
    the digest of what this journal last wrote to it."""

    pre_image: bytes | None
    written: bytes | None = None


def _digest(path: Path) -> bytes | None:
    return hashlib.sha256(path.read_bytes()).digest() if path.is_file() else None


# Wikilinks: ``[[target|alias]]`` and ``[[target#anchor]]``. We only
# capture the target portion, since that's what resolves to a file.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\|#]+)")

# Subdirectories whose content is specific to a single playthrough. Reads
# and writes through these prefixes resolve under the active run subtree;
# requesting them with no active run raises VaultPathError.
RUN_SUBDIRS: frozenset[str] = frozenset({"characters", "arcs", "sessions", "notes"})

# Top-level files that are run-scoped. ``run.md`` carries the run's
# frontmatter (title, status, created, parent_campaign). ``beat.md`` is
# the live, per-run narrative beat — short prose the DM consults on
# every turn to modulate tempo / closeness / cruelty / humour /
# tension. Players (or the DM via tools) can edit it mid-session.
RUN_FILES: frozenset[str] = frozenset({"run.md", "beat.md"})

# Reserved directory name at the campaign vault root. Holds the per-run
# subtrees. Hidden from the LLM-facing surface.
RUNS_DIR = "runs"

# Run-scoped paths owned by the deterministic story engine
# (aidm.story_events) and hidden from the LLM-facing surface entirely:
# the DM reading future events is a plot spoiler, and the clock reaches
# the model only through the turn-contract note. list_dir/read_file/
# grep/glob/tree all skip them; the engine goes through the dedicated
# read_clock/write_clock/list_events/write_event accessors below.
# ``discarded/`` holds DM replies the player regenerated away
# (``write_discarded_reply``): archived for humans, never story input.
HIDDEN_RUN_DIRS: frozenset[str] = frozenset({"events", "discarded"})
HIDDEN_RUN_FILES: frozenset[str] = frozenset({"clock.md"})


class Vault:
    """Read/write navigator over a campaign vault with run overlay."""

    def __init__(self, root: Path, run_slug: str | None = None):
        self._root = root.expanduser().resolve(strict=True)
        if not self._root.is_dir():
            raise VaultPathError(f"vault root is not a directory: {self._root}")
        self._run_slug: str | None = None
        if run_slug:
            validate_slug("run", run_slug)
            run_root = self._root / RUNS_DIR / run_slug
            if not run_root.is_dir():
                raise VaultPathError(f"run not found: {run_slug!r} under {self._root}")
            self._safe(f"{RUNS_DIR}/{run_slug}")  # a symlinked run dir must not leave the root
            self._run_slug = run_slug
        # Backlink index: virtual_rel_path -> set of virtual_rel_paths that
        # link to it. Built lazily on first read; invalidated by every write.
        self._backlinks: dict[str, set[str]] | None = None
        # Turn write journal (text regenerate): on-disk path -> its bytes
        # before the turn's first write to it, None when it did not exist.
        # None = not journaling.
        self._journal: dict[Path, _JournalEntry] | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def run_slug(self) -> str | None:
        return self._run_slug

    @property
    def run_root(self) -> Path | None:
        """On-disk path of the active run, or ``None`` in canon-only mode."""
        if self._run_slug is None:
            return None
        return self._root / RUNS_DIR / self._run_slug

    def list_dir(self, path: str = ".") -> list[str]:
        """List vault entries, returning full vault-relative paths.

        Returning ``rules/firearms.md`` rather than ``firearms.md`` lets
        the LLM feed the result straight back into ``read_file`` without
        having to remember the parent directory. Directories carry a
        trailing slash.
        """
        path = self._normalize(path)
        self._reject_runs_prefix(path)
        self._reject_hidden(path)
        normalized = path.strip("/").strip()
        prefix = "" if normalized in ("", ".") else normalized.rstrip("/") + "/"
        entries: dict[str, str] = {}
        found_any = False
        for source in self._list_sources(path):
            if not source.is_dir():
                continue
            found_any = True
            for entry in source.iterdir():
                if entry.name.startswith("."):
                    continue
                # Hide the ``runs/`` implementation directory at the root.
                if source == self._root and entry.name == RUNS_DIR:
                    continue
                # Hide the story-engine namespace (events/, clock.md).
                if normalized in ("", ".") and (
                    entry.name in HIDDEN_RUN_DIRS or entry.name in HIDDEN_RUN_FILES
                ):
                    continue
                display = prefix + entry.name + ("/" if entry.is_dir() else "")
                entries.setdefault(entry.name, display)
        if not found_any:
            raise VaultPathError(f"not a directory: {path}")
        return sorted(entries.values())

    def read_file(self, path: str, *, max_bytes: int = 200_000) -> str:
        """Return file contents as text. Truncates above ``max_bytes``.

        For ``.md`` files, appends a backlink footer listing other vault
        files that wikilink to this one. Cheap relational expansion at
        zero exploration cost — the LLM gets the graph context for free.
        """
        path = self._normalize(path)
        self._reject_runs_prefix(path)
        self._reject_hidden(path)
        target = self._resolve_read(path)
        if not target.is_file():
            raise VaultPathError(f"not a file: {path}")
        data = target.read_bytes()
        text = data[:max_bytes].decode("utf-8", errors="replace")
        if len(data) > max_bytes:
            text += f"\n\n[... truncated, {len(data) - max_bytes} more bytes]"
        if target.suffix == ".md":
            virtual = self._virtual_path(target)
            backlinks = self._link_index().get(virtual)
            if backlinks:
                text = text.rstrip() + "\n\n# Referenced by\n"
                for link in sorted(backlinks):
                    text += f"- {link}\n"
        return text

    def grep(
        self, pattern: str, *, max_hits: int = 30, include_rules_subdirs: bool = False
    ) -> list[GrepHit]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise VaultPathError(f"invalid grep pattern: {exc}") from exc
        hits: list[GrepHit] = []
        for md in self._iter_markdown():
            if not include_rules_subdirs and self._is_rules_subdir(md):
                # Base-system reference (e.g. rules/srd/) is huge and noisy
                # for narrative grep. Campaign-layer rules at rules/*.md
                # stay in. Caller can opt back in via include_rules_subdirs.
                continue
            try:
                for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), start=1):
                    if regex.search(line):
                        hits.append(
                            GrepHit(
                                path=self._virtual_path(md),
                                line_number=lineno,
                                text=line.strip()[:200],
                            )
                        )
                        if len(hits) >= max_hits:
                            return hits
            except OSError:
                continue
        return hits

    def _is_rules_subdir(self, p: Path) -> bool:
        """True if ``p`` lives under a subdirectory of ``rules/`` (e.g. ``rules/srd/``)."""
        try:
            rel = p.relative_to(self._root)
        except ValueError:
            return False
        parts = rel.parts
        return len(parts) >= 3 and parts[0] == "rules"

    def tree(self, *, max_depth: int = 2, max_entries_per_dir: int = 40) -> str:
        """Compact map of the vault for prompt seeding.

        Breadth-first walk to ``max_depth`` levels. Files appear as full
        vault-relative paths; directories carry trailing slashes. Caps
        the entry count per directory so a runaway ``sessions/`` doesn't
        blow up the system prompt.
        """
        lines: list[str] = []
        # ``queue`` items are (path, depth). Depth 0 is the root.
        queue: list[tuple[str, int]] = [(".", 0)]
        while queue:
            path, depth = queue.pop(0)
            try:
                entries = self.list_dir(path)
            except VaultPathError:
                continue
            shown = entries[:max_entries_per_dir]
            for entry in shown:
                lines.append(entry)
                if entry.endswith("/") and depth + 1 < max_depth:
                    queue.append((entry.rstrip("/"), depth + 1))
            if len(entries) > max_entries_per_dir:
                lines.append(f"  ... {len(entries) - max_entries_per_dir} more entries in {path}")
        return "\n".join(lines)

    def glob(self, pattern: str, *, max_hits: int = 100) -> list[str]:
        hits: list[str] = []
        for md in self._iter_markdown():
            if fnmatch.fnmatch(md.name, pattern):
                hits.append(self._virtual_path(md))
                if len(hits) >= max_hits:
                    break
        return sorted(hits)

    def write_session_log(self, content: str) -> str:
        """Create a new session log under the active run's ``sessions/``.

        Auto-fills ``date:`` in the frontmatter to match the filename
        timestamp — the LLM cannot be trusted to know what day it is.
        Filename is ``YYYY-MM-DD-HHMMSS.md`` so multiple sessions per day
        don't collide.
        """
        sessions_dir = self._require_run_dir("sessions")
        self._safe(self._rel(sessions_dir))
        sessions_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%d-%H%M%S")
        target = sessions_dir / f"{stamp}.md"
        self._safe(self._rel(target))
        self._write(target, _stamped(content, "date", now.strftime("%Y-%m-%d")))
        self._invalidate_index()
        return self._virtual_path(target)

    def append_note(self, slug: str, content: str) -> str:
        clean = _slugify(slug)
        if not clean:
            raise VaultPathError("note slug is empty after sanitization")
        notes_dir = self._require_run_dir("notes")
        notes_dir.mkdir(parents=True, exist_ok=True)
        target = notes_dir / f"{clean}.md"
        # _safe still applies — the path must stay inside the vault root.
        self._safe(target.relative_to(self._root).as_posix())
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"\n\n---\n*{stamp}*\n\n{content.strip()}\n"
        if target.exists():
            self._write(target, block, append=True)
        else:
            self._write(target, f"# {slug.strip()}\n{block}")
        self._invalidate_index()
        return self._virtual_path(target)

    def write_beat(self, content: str) -> str:
        """Overwrite the active run's ``beat.md``.

        The beat is short prose the DM reads every turn to modulate
        tempo / closeness / cruelty / humour / tension. Replacing it is
        the player's (or DM's) live "scene-direct" control. Empty
        ``content`` is allowed — it disables the beat injection until a
        new one is written.
        """
        if self.run_root is None:
            raise VaultPathError("no active run; cannot write beat.md")
        target = self.run_root / "beat.md"
        self._safe(self._rel(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write(target, content)
        self._invalidate_index()
        return self._virtual_path(target)

    def write_tone(self, content: str) -> None:
        """Overwrite the campaign-canon ``tone.md`` at the vault root."""
        target = self._safe("tone.md")
        self._write(target, content)
        self._invalidate_index()

    def read_clock(self) -> str | None:
        """Raw text of the run's ``clock.md``, or None (no run / no clock).

        Story-engine accessor — bypasses the LLM-facing hidden-namespace
        guard on purpose. A run without a clock file has the event
        engine disabled.
        """
        if self.run_root is None:
            return None
        target = self.run_root / "clock.md"
        self._safe(self._rel(target))
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

    def write_clock(self, content: str) -> str:
        """Overwrite the run's ``clock.md``; stamps ``last_updated``."""
        if self.run_root is None:
            raise VaultPathError("no active run; cannot write clock.md")
        target = self.run_root / "clock.md"
        self._safe(self._rel(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        self._write(target, _stamped(content, "last_updated", today))
        return f"{RUNS_DIR}/{self._run_slug}/clock.md"

    def list_events(self) -> list[tuple[str, str]]:
        """``(slug, raw text)`` for every run event file, sorted by slug.

        Story-engine accessor; empty when the run has no ``events/``.
        """
        if self.run_root is None:
            return []
        events_dir = self.run_root / "events"
        if not events_dir.is_dir():
            return []
        out: list[tuple[str, str]] = []
        for path in sorted(events_dir.glob("*.md")):
            if path.name.startswith("."):
                continue
            try:
                out.append((path.stem, path.read_text(encoding="utf-8")))
            except OSError:
                continue
        return out

    def write_event(self, slug: str, content: str) -> str:
        """Overwrite one run event file; stamps ``last_updated``."""
        if self.run_root is None:
            raise VaultPathError("no active run; cannot write event file")
        clean = _slugify(slug)
        if not clean:
            raise VaultPathError("event slug is empty after sanitization")
        target = self.run_root / "events" / f"{clean}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._safe(target.relative_to(self._root).as_posix())
        today = datetime.now().strftime("%Y-%m-%d")
        self._write(target, _stamped(content, "last_updated", today))
        return f"{RUNS_DIR}/{self._run_slug}/events/{clean}.md"

    def write_character_sheet(self, slug: str, content: str) -> str:
        """Create or overwrite a character IDENTITY sheet under the active run.

        Identity sheets hold persona + HEXACO + fixed mechanical ceiling.
        Per-turn volatile state lives on the sibling ``.state.md`` (see
        ``write_character_state``); no auto-stamp here.
        """
        clean = _slugify(slug)
        if not clean:
            raise VaultPathError("character slug is empty after sanitization")
        characters_dir = self._require_run_dir("characters")
        characters_dir.mkdir(parents=True, exist_ok=True)
        target = characters_dir / f"{clean}.md"
        self._safe(target.relative_to(self._root).as_posix())
        self._write(target, content)
        self._invalidate_index()
        return self._virtual_path(target)

    def write_arc(self, slug: str, content: str) -> str:
        """Create or overwrite a run-scoped ``arcs/<slug>.md`` (code-owned arcs, e.g. the
        First Nightmare plan). No auto-stamp; journaled like every turn write."""
        clean = _slugify(slug)
        if not clean:
            raise VaultPathError("arc slug is empty after sanitization")
        arcs_dir = self._require_run_dir("arcs")
        arcs_dir.mkdir(parents=True, exist_ok=True)
        target = arcs_dir / f"{clean}.md"
        self._safe(target.relative_to(self._root).as_posix())
        self._write(target, content)
        self._invalidate_index()
        return self._virtual_path(target)

    def write_npc_sheet(self, slug: str, content: str) -> str:
        """Create or overwrite an NPC IDENTITY sheet under campaign-canon ``npcs/``.

        NPCs are campaign-level canon (recurring across runs). Volatile
        per-scene state lives on the sibling ``.state.md`` written under
        the active run by ``write_npc_state``; no auto-stamp here.
        """
        clean = _slugify(slug)
        if not clean:
            raise VaultPathError("npc slug is empty after sanitization")
        target = self._safe(f"npcs/{clean}.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write(target, content)
        self._invalidate_index()
        return self._virtual_path(target)

    def write_character_state(self, slug: str, content: str) -> str:
        """Create or overwrite the PC's per-turn state file under the active run.

        Auto-stamps ``last_updated`` so the LLM-supplied (and inevitably
        wrong) date cannot rot the file.
        """
        clean = _slugify(slug)
        if not clean:
            raise VaultPathError("character slug is empty after sanitization")
        characters_dir = self._require_run_dir("characters")
        characters_dir.mkdir(parents=True, exist_ok=True)
        target = characters_dir / f"{clean}.state.md"
        self._safe(target.relative_to(self._root).as_posix())
        today = datetime.now().strftime("%Y-%m-%d")
        self._write(target, _stamped(content, "last_updated", today))
        self._invalidate_index()
        return self._virtual_path(target)

    def write_npc_state(self, slug: str, content: str) -> str:
        """Create or overwrite an NPC's per-turn state under the active run.

        NPC identity is canon (recurring across runs); NPC state is per-
        run (each playthrough has its own scenes). State files therefore
        live under ``runs/<run>/npcs/<slug>.state.md``; the canon-path
        overlay surfaces them at the virtual ``npcs/<slug>.state.md``.
        """
        if self.run_root is None:
            raise VaultPathError("no active run; cannot write npc state file")
        clean = _slugify(slug)
        if not clean:
            raise VaultPathError("npc slug is empty after sanitization")
        target = self.run_root / "npcs" / f"{clean}.state.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._safe(target.relative_to(self._root).as_posix())
        today = datetime.now().strftime("%Y-%m-%d")
        self._write(target, _stamped(content, "last_updated", today))
        self._invalidate_index()
        return self._virtual_path(target)

    def read_state(self, kind: str, slug: str) -> str | None:
        """Read a per-turn state file (``npc`` or ``pc``), or ``None`` if missing.

        Strips the backlink footer ``read_file`` appends so callers get
        the raw on-disk text. ``kind`` is ``"npc"`` or ``"pc"``.
        """
        subdir = "characters" if kind == "pc" else "npcs"
        path = f"{subdir}/{slug}.state.md"
        try:
            text = self.read_file(path)
        except VaultPathError:
            return None
        marker = "\n\n# Referenced by\n"
        if marker in text:
            text = text[: text.index(marker)]
        return text

    def write_discarded_reply(self, content: str) -> str:
        """Archive a regenerated-away DM reply under ``runs/<run>/discarded/``.

        Never journaled (a rollback must not take the archive with it) and
        hidden from the LLM-facing surface (``HIDDEN_RUN_DIRS``): a
        discarded reply is for humans, never story input.
        """
        if self.run_root is None:
            raise VaultPathError("no active run; cannot archive a discarded reply")
        target = self.run_root / "discarded" / f"{datetime.now():%Y-%m-%d-%H%M%S-%f}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._safe(target.relative_to(self._root).as_posix())
        target.write_text(content, encoding="utf-8")
        return target.relative_to(self._root).as_posix()

    # --- turn write journal ---------------------------------------------

    def begin_journal(self) -> None:
        """Start a fresh journal; the previous one (if any) is dropped."""
        self._journal = {}

    def end_journal(self) -> None:
        self._journal = None

    def restore_journal(self) -> None:
        """Put every journaled file back to its pre-image; the journal empties.

        All-or-nothing where the filesystem allows it, and never silent:

        1. **Conflict check.** A file whose bytes differ from what this
           journal last wrote was changed by someone else since (the
           browser sheet/beat editor, another session on the run, a canon
           ``npcs/`` write from another campaign session). Restoring would
           revert or delete their write: ``VaultConflictError``, nothing
           touched.
        2. **Stage.** Every pre-image goes to a hidden sibling temp file; a
           failure removes the stages and raises ``VaultPathError``,
           nothing touched.
        3. **Commit.** Stages are renamed over their targets, files the
           turn created are deleted. An entry leaves the journal only once
           it is back; a failure here raises ``VaultRestoreIncomplete``
           naming the paths still journaled, so a retry can finish them.
        """
        journal = self._journal or {}
        changed = [
            self._rel(path) for path, entry in journal.items() if _digest(path) != entry.written
        ]
        if changed:
            raise VaultConflictError(f"changed since this turn wrote them: {', '.join(changed)}")
        staged: dict[Path, Path] = {}
        try:
            for path, entry in journal.items():
                if entry.pre_image is not None:
                    stage = path.with_name(f".{path.name}.rollback")
                    stage.write_bytes(entry.pre_image)
                    staged[path] = stage
        except OSError as exc:
            for stage in staged.values():
                stage.unlink(missing_ok=True)
            raise VaultPathError(f"rollback aborted, nothing restored: {exc}") from exc
        failed: list[str] = []
        for path in list(journal):
            try:
                if path in staged:
                    os.replace(staged[path], path)
                    del staged[path]
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                failed.append(self._rel(path))
                continue
            del journal[path]
        for stage in staged.values():
            stage.unlink(missing_ok=True)
        self._invalidate_index()
        if failed:
            raise VaultRestoreIncomplete(f"not restored (still journaled): {', '.join(failed)}")

    def _write(self, target: Path, text: str, *, append: bool = False) -> None:
        """Every turn-reachable disk write: journal the pre-image, write, note what we wrote."""
        journal = self._journal
        if journal is not None and target not in journal:
            journal[target] = _JournalEntry(target.read_bytes() if target.is_file() else None)
        if append:
            with target.open("a", encoding="utf-8") as fh:
                fh.write(text)
        else:
            target.write_text(text, encoding="utf-8")
        if journal is not None:
            journal[target].written = _digest(target)

    def _rel(self, path: Path) -> str:
        return path.relative_to(self._root).as_posix()

    # --- backlink index -------------------------------------------------

    def _link_index(self) -> dict[str, set[str]]:
        if self._backlinks is None:
            self._backlinks = self._build_link_index()
        return self._backlinks

    def _invalidate_index(self) -> None:
        self._backlinks = None

    def _build_link_index(self) -> dict[str, set[str]]:
        """Walk every markdown file, resolve wikilinks, return backlink map.

        Wikilink ``[[slug]]`` resolves to the unique ``.md`` file whose
        stem matches ``slug``. Ambiguous slugs (matching multiple files)
        are skipped — there's no way to disambiguate without context, so
        we under-report rather than mislead. Keys and values are virtual
        paths (the run prefix is stripped) so backlinks match what the
        LLM sees through ``read_file``.
        """
        by_stem: dict[str, list[str]] = {}
        all_files: list[Path] = []
        for md in self._iter_markdown():
            virtual = self._virtual_path(md)
            all_files.append(md)
            by_stem.setdefault(md.stem, []).append(virtual)
        index: dict[str, set[str]] = {}
        for md in all_files:
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            source_virtual = self._virtual_path(md)
            for match in _WIKILINK_RE.finditer(text):
                target_slug = match.group(1).strip()
                candidates = by_stem.get(target_slug)
                if not candidates or len(candidates) > 1:
                    continue
                target_virtual = candidates[0]
                if target_virtual == source_virtual:
                    continue
                index.setdefault(target_virtual, set()).add(source_virtual)
        return index

    # --- path safety ----------------------------------------------------

    def _normalize(self, path: str) -> str:
        """Collapse ``.``/empty segments and reject ``..`` before any guard.

        The prefix guards below (``runs/``, hidden story-engine paths)
        inspect the *first path segment* — un-normalized input like
        ``./clock.md`` or ``notes/../clock.md`` would slip past them and
        only be collapsed later by ``_safe()``'s resolve, after the
        guard already approved it. Normalizing once up front closes that
        gap; ``..`` is rejected outright because no legitimate virtual
        path needs it.
        """
        parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise VaultPathError(f"path traversal not allowed: {path}")
        return "/".join(parts) or "."

    def _safe(self, path: str) -> Path:
        candidate = (self._root / path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise VaultPathError(f"path escapes vault root: {path}") from exc
        return candidate

    def _reject_runs_prefix(self, path: str) -> None:
        """Block direct access to the implementation ``runs/`` namespace.

        The LLM-facing surface is the flat virtual view; reaching into
        ``runs/<other-run>/...`` would let one playthrough peek at another.
        """
        first = (path.split("/", 1)[0] or "").strip()
        if first == RUNS_DIR:
            raise VaultPathError(f"path uses reserved namespace {RUNS_DIR!r}: {path}")

    def _reject_hidden(self, path: str) -> None:
        """Block LLM-facing access to the story-engine namespace.

        Reads mask as plain not-found so the model self-corrects instead
        of learning that a forbidden pocket of the vault exists.
        """
        normalized = path.strip("/").strip()
        first = normalized.split("/", 1)[0]
        if first in HIDDEN_RUN_DIRS or normalized in HIDDEN_RUN_FILES:
            raise VaultPathError(f"not a file: {path}")

    def _resolve_read(self, path: str) -> Path:
        """Map a virtual path to the on-disk path with overlay rules.

        Run-scoped paths (``characters/...``, ``sessions/...``, etc.) and
        run-scoped files (``run.md``) always route to the active run; if
        no run is active, raise. Canon paths route to the campaign root
        unless the active run has an override file at the same path —
        then the override wins.
        """
        normalized = path.strip("/")
        first = normalized.split("/", 1)[0]
        is_run_scoped = first in RUN_SUBDIRS or normalized in RUN_FILES
        if is_run_scoped:
            if self.run_root is None:
                raise VaultPathError(f"no active run; cannot access run-scoped path: {path}")
            return self._safe(f"{RUNS_DIR}/{self._run_slug}/{normalized}")
        # Canon path: check run override first.
        if self.run_root is not None:
            override = self._safe(f"{RUNS_DIR}/{self._run_slug}/{normalized}")
            if override.exists():
                return override
        return self._safe(normalized)

    def _list_sources(self, path: str) -> list[Path]:
        """Sources to union for ``list_dir``.

        At the root ``.``, show canon root + active run root so the LLM
        sees the flat virtual layout. Inside a run-scoped subdir, only
        the run side. Inside a canon subdir, both sides (run can add).
        """
        normalized = path.strip("/")
        if normalized in ("", "."):
            sources: list[Path] = [self._root]
            if self.run_root is not None:
                sources.append(self.run_root)
            return sources
        first = normalized.split("/", 1)[0]
        if first in RUN_SUBDIRS:
            if self.run_root is None:
                raise VaultPathError(f"no active run; cannot list run-scoped path: {path}")
            return [self.run_root / normalized]
        # Canon: canon dir + optional run-side overlay at the same path.
        sources = [self._safe(normalized)]
        if self.run_root is not None:
            sources.append(self.run_root / normalized)
        return sources

    def _require_run_dir(self, subdir: str) -> Path:
        if self.run_root is None:
            raise VaultPathError(f"no active run; cannot write to run-scoped subdir: {subdir}")
        return self.run_root / subdir

    def _iter_markdown(self) -> list[Path]:
        """All markdown files visible to the active run.

        Canon walk skips every ``runs/`` subtree (so inactive runs don't
        leak into grep/glob/backlink); the active run is then walked
        separately. Hidden files and directories are pruned.
        """
        files: list[Path] = []
        for p in self._root.rglob("*.md"):
            rel = p.relative_to(self._root)
            parts = rel.parts
            if any(part.startswith(".") for part in parts):
                continue
            if parts and parts[0] == RUNS_DIR:
                continue
            files.append(p)
        if self.run_root is not None and self.run_root.is_dir():
            for p in self.run_root.rglob("*.md"):
                rel = p.relative_to(self.run_root)
                parts = rel.parts
                if any(part.startswith(".") for part in parts):
                    continue
                # Story-engine namespace stays out of grep/glob/backlinks.
                if parts and (parts[0] in HIDDEN_RUN_DIRS or parts[0] in HIDDEN_RUN_FILES):
                    continue
                files.append(p)
        return files

    def _virtual_path(self, p: Path) -> str:
        """Path the LLM sees: the run prefix is stripped from on-disk paths."""
        rel = p.relative_to(self._root)
        parts = rel.parts
        if (
            len(parts) >= 2
            and parts[0] == RUNS_DIR
            and self._run_slug is not None
            and parts[1] == self._run_slug
        ):
            return str(Path(*parts[2:])) if len(parts) > 2 else ""
        return str(rel)


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """ASCII-folded lowercase-dash slug. Empty string for unsalvageable input.

    Hungarian accents are NFKD-normalised away so vault filenames remain
    portable. The reserved ``runs`` namespace is rejected (the campaign-
    canon root must never collide with the per-run subtree directory).
    """
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", errors="ignore").decode("ascii")
    slug = _SLUG_PATTERN.sub("-", ascii_only.lower()).strip("-")
    if slug == RUNS_DIR:
        return ""
    return slug


# Backwards-compatible alias for the previous private name; existing call
# sites inside this module keep working.
_slugify = slugify

# The shape ``slugify`` produces, and the only shape a campaign or run id
# may have when it arrives from outside (HTTP query, request body, env).
# Anything else — ``..``, ``/abs``, ``a/b``, a leading ``_`` or ``.`` — is
# refused before it is ever joined onto a path.
_SLUG_SHAPE = re.compile(r"[a-z0-9][a-z0-9-]*")


def validate_slug(kind: str, value: str) -> str:
    """Return ``value`` if it is a well-formed campaign/run slug, else raise."""
    if not _SLUG_SHAPE.fullmatch(value):
        raise VaultPathError(f"invalid {kind} id: {value!r}")
    return value


def _stamped(content: str, field: str, value: str) -> str:
    """Set ``field: value`` in the frontmatter, preserving the rest.

    If the document has no frontmatter, prepends a minimal block. The
    field is appended to the existing frontmatter when missing, or
    overwritten when present. Other fields and field order are kept
    intact so a human reading the file later sees a stable layout.
    """
    fm, body = frontmatter.split(content)
    if not fm:
        fm = {field: value}
        return frontmatter.serialize(fm, content.lstrip("\n"))
    fm[field] = value
    return frontmatter.serialize(fm, body)
