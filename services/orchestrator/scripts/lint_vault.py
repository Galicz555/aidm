"""Static lint over a vault directory.

Validates every ``.md`` file against the per-type schema declared in
``aidm.vault_schema``. Run from ``make check``; nonzero exit on errors.

Why not strict write-time validation? The voice loop is latency-bound;
forcing the LLM to retry on a schema violation costs 1-3s of dead air
mid-turn. Instead, write tools log warnings, and this lint surfaces
violations to humans at commit time.

Usage::

    uv run python -m services.orchestrator.scripts.lint_vault [VAULT_PATH]

``VAULT_PATH`` defaults to ``$VAULT_PATH`` or the repo's
``vault/ember-hollow/`` (i.e. the same resolution the orchestrator uses).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from aidm import frontmatter
from aidm.story_events import DIRECTIVE_HEADER, ON_FIRE_HEADER, parse_mutations
from aidm.vault_schema import HEXACO_DIMENSIONS, HEXACO_KEYS, hexaco_body_score, schema_for


@dataclass(frozen=True)
class Issue:
    path: Path
    message: str


def lint(vault_root: Path) -> list[Issue]:
    """Walk the vault, validate every markdown file, return a flat issue list.

    The walk covers both the canon root and every ``runs/<slug>/`` subtree
    — run-scoped files use the same per-type schemas as the canon, only
    the on-disk prefix differs.
    """

    issues: list[Issue] = []
    for path in sorted(vault_root.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(vault_root).parts):
            continue
        # README.md files are scaffolding for empty folders; they have
        # no frontmatter and exist purely to keep the directory in git.
        if path.name == "README.md":
            continue
        # tone.md (campaign canon) and beat.md (run-scoped) are
        # free-form prose modulators the DM reads every turn — no
        # frontmatter contract, no lint coverage. tone.private.md is
        # the private-mode override of tone.md and carries the same
        # free-form contract, so it's exempt too.
        if path.name in {"tone.md", "tone.private.md", "beat.md"}:
            continue
        # notes/<slug>.md files are written by ``vault.append_note`` as
        # a free-form running log (h1 + timestamped blocks). No
        # frontmatter contract — exempt the whole notes/ subtree.
        if "notes" in path.relative_to(vault_root).parts:
            continue
        # runs/<run>/discarded/ archives regenerated-away DM replies
        # (Vault.write_discarded_reply) — a human log, not a schema type.
        if "discarded" in path.relative_to(vault_root).parts:
            continue
        issues.extend(_lint_file(path))
    return issues


def _lint_file(path: Path) -> list[Issue]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Issue(path, f"unreadable: {exc}")]
    fm = frontmatter.parse(text)
    if not fm:
        return [Issue(path, "no frontmatter block found")]
    type_name = str(fm.get("type") or "").strip()
    if not type_name:
        return [Issue(path, "missing required field: type")]
    schema = schema_for(type_name)
    if schema is None:
        return [Issue(path, f"unknown type: {type_name!r}")]

    issues: list[Issue] = []
    for required in schema.required:
        if required not in fm:
            issues.append(Issue(path, f"missing required field: {required}"))
    for key, allowed in schema.enums.items():
        if key in fm and str(fm[key]) not in allowed:
            issues.append(
                Issue(
                    path,
                    f"{key} enum mismatch: {fm[key]!r} "
                    f"(expected one of: {', '.join(sorted(allowed))})",
                )
            )
    if schema.hexaco_required:
        issues.extend(_lint_hexaco(path, fm, text))
    if type_name == "event":
        issues.extend(_lint_event_body(path, text))
    if "lang" in schema.required and fm.get("lang") not in {"hu", "en"}:
        issues.append(Issue(path, f"lang must be 'hu' or 'en', got {fm.get('lang')!r}"))
    return issues


def _lint_event_body(path: Path, text: str) -> list[Issue]:
    """Event body contract: a non-empty ``## Directive`` section plus an
    optional ``## On fire`` block whose every bullet parses against the
    story-engine mutation grammar (see ``aidm.story_events``).
    """
    issues: list[Issue] = []
    body = frontmatter.strip(text)
    directive = _event_section(body, DIRECTIVE_HEADER)
    if not directive:
        issues.append(Issue(path, f"event body missing a non-empty '{DIRECTIVE_HEADER}' section"))
    on_fire = _event_section(body, ON_FIRE_HEADER)
    if on_fire:
        bullets = [line for line in on_fire.splitlines() if line.strip().startswith("- ")]
        parsed = parse_mutations(on_fire)
        if len(parsed) != len(bullets):
            issues.append(
                Issue(
                    path,
                    f"'{ON_FIRE_HEADER}' has {len(bullets) - len(parsed)} invalid mutation "
                    "line(s) — expected '- <npc>.<state-field>: <value>', '- beat: <text>' "
                    "or '- note: <text>'",
                )
            )
    return issues


def _event_section(body: str, header: str) -> str:
    if header not in body:
        return ""
    after = body.split(header, 1)[1]
    for next_header in ("\n## ", "\n# "):
        if next_header in after:
            after = after.split(next_header, 1)[0]
    return after.strip()


def _lint_hexaco(path: Path, fm: dict[str, object], body: str) -> list[Issue]:
    issues: list[Issue] = []
    for key in HEXACO_KEYS:
        value = fm.get(key)
        if value is None:
            issues.append(Issue(path, f"missing required HEXACO field: {key}"))
            continue
        try:
            score = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            issues.append(Issue(path, f"{key}: expected integer, got {value!r}"))
            continue
        if not 1 <= score <= 5:
            issues.append(Issue(path, f"{key}: {score} out of range 1-5"))
    # Body cross-check: each HEXACO dimension should appear in the
    # "Psychology (HEXACO)" block with a parenthesized score that
    # matches the frontmatter. We're permissive about formatting —
    # the goal is to catch drift, not enforce a specific layout.
    for dim in HEXACO_DIMENSIONS:
        fm_score = fm.get(f"hexaco_{dim}")
        if fm_score is None:
            continue
        body_score = hexaco_body_score(body, dim)
        if body_score is None:
            issues.append(Issue(path, f"HEXACO body block missing entry for {dim}"))
            continue
        if str(fm_score) != body_score:
            issues.append(
                Issue(
                    path,
                    f"HEXACO body score for {dim} ({body_score}) "
                    f"disagrees with frontmatter ({fm_score})",
                )
            )
    return issues


def _resolve_vault(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).expanduser().resolve()
    env = os.environ.get("VAULT_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "vault" / "ember-hollow"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    root = _resolve_vault(argv)
    if not root.is_dir():
        print(f"lint_vault: not a directory: {root}", file=sys.stderr)
        return 2
    issues = lint(root)
    if not issues:
        print(f"lint_vault: {root} clean")
        return 0
    # Group by file for human-friendly output.
    current: Path | None = None
    for issue in issues:
        if issue.path != current:
            print(f"\n{issue.path.relative_to(root)}:")
            current = issue.path
        print(f"  - {issue.message}")
    print(f"\n{len(issues)} error(s) in vault {root}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
