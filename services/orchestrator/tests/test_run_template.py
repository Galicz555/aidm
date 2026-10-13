"""A UI-created run is seeded from the campaign's runs/_template/."""

from __future__ import annotations

from pathlib import Path

from aidm.server_utils import list_runs_on_disk, seed_run_from_template


def test_new_run_gets_the_template_clock_and_events_and_the_template_is_not_a_run(
    tmp_path: Path,
) -> None:
    # 2026-09-27 playtest: a UI-created run had no clock.md / events/, so the
    # story engine never ran.
    template = tmp_path / "runs" / "_template"
    (template / "events").mkdir(parents=True)
    (template / "clock.md").write_text("---\ntype: clock\nday: 1\n---\n", encoding="utf-8")
    (template / "events" / "raid.md").write_text("---\ntype: event\n---\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "new-run"
    run_dir.mkdir()
    (run_dir / "run.md").write_text("---\ntype: run\ntitle: New\n---\n", encoding="utf-8")

    seed_run_from_template(run_dir)

    assert (run_dir / "clock.md").is_file()
    assert (run_dir / "events" / "raid.md").is_file()
    assert (run_dir / "run.md").read_text(encoding="utf-8").startswith("---\ntype: run")
    assert [r["slug"] for r in list_runs_on_disk(tmp_path)] == ["new-run"]


def test_campaign_without_template_seeds_nothing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "plain"
    run_dir.mkdir(parents=True)
    seed_run_from_template(run_dir)
    assert list(run_dir.iterdir()) == []
