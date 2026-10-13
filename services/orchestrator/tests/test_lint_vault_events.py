"""Vault lint — event/clock schema coverage and event-body contract."""

from __future__ import annotations

from pathlib import Path

from scripts.lint_vault import lint

_VALID_EVENT = """---
type: event
lang: en
title: Marta slips out
status: pending
trigger_day: 3
trigger_segment: morning
subject_npc: marta
defer_policy: expire
last_updated: 2026-07-11
---

## Directive

Marta is gone before first light.

## On fire

- marta.location: pharmacy district
- beat: Marta is missing.
"""

_VALID_CLOCK = """---
type: clock
day: 1
segment: morning
turn: 0
last_updated: 2026-07-11
---

## Time log

_(no advances yet)_
"""


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "campaign.md").write_text(
        "---\ntype: campaign\ntitle: T\nstatus: active\nlang: en\n---\n", encoding="utf-8"
    )
    events = tmp_path / "runs" / "test" / "events"
    events.mkdir(parents=True)
    return tmp_path


def test_valid_event_and_clock_pass(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "runs/test/events/marta.md").write_text(_VALID_EVENT, encoding="utf-8")
    (root / "runs/test/clock.md").write_text(_VALID_CLOCK, encoding="utf-8")
    assert lint(root) == []


def test_event_missing_directive_section_flags(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    broken = _VALID_EVENT.replace("## Directive", "## Something else")
    (root / "runs/test/events/marta.md").write_text(broken, encoding="utf-8")
    issues = lint(root)
    assert any("Directive" in i.message for i in issues)


def test_event_invalid_mutation_line_flags(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    broken = _VALID_EVENT.replace(
        "- marta.location: pharmacy district", "- marta.not_a_field: nope"
    )
    (root / "runs/test/events/marta.md").write_text(broken, encoding="utf-8")
    issues = lint(root)
    assert any("invalid mutation" in i.message for i in issues)


def test_event_bad_enum_flags(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    broken = _VALID_EVENT.replace("defer_policy: expire", "defer_policy: whenever")
    (root / "runs/test/events/marta.md").write_text(broken, encoding="utf-8")
    issues = lint(root)
    assert any("defer_policy" in i.message for i in issues)


def test_clock_bad_segment_flags(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "runs/test/clock.md").write_text(
        _VALID_CLOCK.replace("segment: morning", "segment: brunch"), encoding="utf-8"
    )
    issues = lint(root)
    assert any("segment" in i.message for i in issues)
