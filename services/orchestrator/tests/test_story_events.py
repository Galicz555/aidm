"""Story clock + event engine — deterministic plot rules.

Covers the advancement matrix (baseline tick, scene tick, clamped model
hint) and the trigger/defer/cancel/branch rules against the three
authored chains from the design plan: Marta's antibiotics run, the Vas
raid branches, and Lyra at the gate.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from aidm import frontmatter, story_events
from aidm.scene_state import SceneRegistry
from aidm.vault import Vault


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_straight_and_curly_possessives_stem_alike() -> None:
    # "Marta's note" (sim/player) must match a directive with a typographic apostrophe.
    typographic = "Marta’s note"  # noqa: RUF001 — the model emits typographic apostrophes
    assert story_events._word_stems("Marta's note") == story_events._word_stems(typographic)
    assert "marta" in story_events._word_stems("Marta's note")


def _npc_state(location: str, status: str = "active") -> str:
    return (
        f"---\ntype: npc_state\nstatus: {status}\nlocation: {location}\n"
        "in_scene: false\nlast_updated: 2026-07-11\n---\n\n"
        "## Recent (last 5 events)\n\n_(no events yet)_\n\n"
        "## Session so far\n\n_(no summary yet)_\n"
    )


def _clock(day: int, segment: str, *, turn: int = 0, last_advance: int = 0) -> str:
    return (
        f"---\ntype: clock\nday: {day}\nsegment: {segment}\nturn: {turn}\n"
        f"last_advance_turn: {last_advance}\nlast_updated: 2026-07-11\n---\n\n"
        "## Time log\n\n_(no advances yet)_\n"
    )


def _event(  # noqa: PLR0913 — event frontmatter is naturally wide
    slug: str,
    *,
    day: int,
    segment: str = "",
    subject: str = "",
    npc_status: str = "",
    npc_location: str = "",
    chain: str = "",
    step: int | None = None,
    policy: str = "",
    expires: int | None = None,
    priority: int = 0,
    mutations: str = "",
) -> str:
    fm = ["---", "type: event", "lang: en", f"title: {slug} title", "status: pending"]
    fm.append(f"trigger_day: {day}")
    if segment:
        fm.append(f"trigger_segment: {segment}")
    if subject:
        fm.append(f"subject_npc: {subject}")
    if npc_status:
        fm.append(f"trigger_npc_status: {npc_status}")
    if npc_location:
        fm.append(f"trigger_npc_location: {npc_location}")
    if chain:
        fm.append(f"chain: {chain}")
    if step is not None:
        fm.append(f"chain_step: {step}")
    if policy:
        fm.append(f"defer_policy: {policy}")
    if expires is not None:
        fm.append(f"expires_after_days: {expires}")
    if priority:
        fm.append(f"priority: {priority}")
    fm.extend(["last_updated: 2026-07-11", "---"])
    body = f"\n## Directive\n\n{slug} happened.\n"
    if mutations:
        body += f"\n## On fire\n\n{mutations}\n"
    return "\n".join(fm) + "\n" + body


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(
        tmp_path / "campaign.md", "---\ntype: campaign\ntitle: T\nstatus: active\nlang: en\n---\n"
    )
    for slug in ("marta", "vas", "lyra"):
        _write(tmp_path / "npcs" / f"{slug}.md", f"---\ntype: npc\nlang: en\nname: {slug}\n---\n")
    _write(tmp_path / "npcs" / "marta.state.md", _npc_state("shelter"))
    _write(tmp_path / "npcs" / "vas.state.md", _npc_state("ridge"))
    _write(tmp_path / "npcs" / "lyra.state.md", _npc_state("the gate"))
    run = tmp_path / "runs" / "test"
    _write(run / "run.md", "---\ntype: run\ntitle: T\nstatus: active\ncreated: 2026-07-11\n---\n")
    (run / "events").mkdir(parents=True)
    return tmp_path


def _engine(
    root: Path, *, hint: str | None = None, scene_changed: bool = False
) -> tuple[story_events.EngineOutcome | None, Vault]:
    vault = Vault(root, run_slug="test")
    registry = SceneRegistry()
    if scene_changed:
        registry.note_scene_change()
    return story_events.run_end_of_turn(vault, registry, time_advance_hint=hint), vault


def _clock_state(vault: Vault) -> dict[str, object]:
    raw = vault.read_clock()
    assert raw is not None
    fm, _ = frontmatter.split(raw)
    return fm


def _event_status(vault: Vault, slug: str) -> str:
    for name, raw in vault.list_events():
        if name == slug:
            fm, _ = frontmatter.split(raw)
            return str(fm.get("status"))
    raise AssertionError(f"event {slug} not found")


# --- clock ---------------------------------------------------------------


def test_engine_is_noop_without_clock_file(root: Path) -> None:
    outcome, _ = _engine(root)
    assert outcome is None


def test_clock_quiet_turn_does_not_advance(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=1, last_advance=1))
    outcome, vault = _engine(root)
    assert outcome is not None
    assert outcome.clock_line == "Day 1, morning."
    assert _clock_state(vault)["segment"] == "morning"


def test_clock_scene_change_advances_one_segment(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=1, last_advance=0))
    _, vault = _engine(root, scene_changed=True)
    assert _clock_state(vault)["segment"] == "afternoon"


def test_clock_scene_change_soon_after_an_advance_is_suppressed(root: Path) -> None:
    # Gate -> hall -> yard in three turns is one evening, not three.
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=2, last_advance=1))
    _, vault = _engine(root, scene_changed=True)
    assert _clock_state(vault)["segment"] == "morning"
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=4, last_advance=1))
    _, vault = _engine(root, scene_changed=True)
    assert _clock_state(vault)["segment"] == "afternoon"


def test_clock_hint_next_day_lands_on_next_morning(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(1, "afternoon", turn=1, last_advance=1))
    _, vault = _engine(root, hint="next_day")
    fm = _clock_state(vault)
    assert (fm["day"], fm["segment"]) == (2, "morning")


def test_clock_night_rolls_into_next_day(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(1, "night", turn=1, last_advance=1))
    _, vault = _engine(root, hint="next_segment")
    fm = _clock_state(vault)
    assert (fm["day"], fm["segment"]) == (2, "morning")


def test_clock_applies_at_most_one_advance_per_turn(root: Path) -> None:
    # Hint + scene change + baseline all due at once → exactly one day.
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=9, last_advance=0))
    _, vault = _engine(root, hint="next_day", scene_changed=True)
    fm = _clock_state(vault)
    assert (fm["day"], fm["segment"]) == (2, "morning")


# --- firing + mutations ----------------------------------------------------


def test_event_fires_with_mutations_and_directive(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(3, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-goes-alone.md",
        _event(
            "marta-goes-alone",
            day=3,
            segment="morning",
            subject="marta",
            npc_location="shelter",
            chain="marta-antibiotics",
            step=2,
            policy="expire",
            mutations="- marta.location: pharmacy district\n- beat: Marta is missing.",
        ),
    )
    outcome, vault = _engine(root)
    assert outcome is not None
    assert outcome.directives == ["(Day 3, morning) marta-goes-alone happened."]
    assert _event_status(vault, "marta-goes-alone") == "fired"
    state = vault.read_state("npc", "marta")
    assert state is not None
    fm, _ = frontmatter.split(state)
    assert fm["location"] == "pharmacy district"
    beat = vault.read_file("beat.md")
    assert "Marta is missing." in beat


def test_expire_policy_kills_player_preempted_event(root: Path) -> None:
    # Lyra was let inside — her location no longer matches the trigger.
    _write(root / "npcs" / "lyra.state.md", _npc_state("inside the shelter"))
    _write(root / "runs/test/clock.md", _clock(1, "night", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/lyra-leaves.md",
        _event(
            "lyra-leaves",
            day=1,
            segment="night",
            subject="lyra",
            npc_location="the gate",
            policy="expire",
        ),
    )
    outcome, vault = _engine(root)
    assert outcome is not None
    assert outcome.directives == []
    assert _event_status(vault, "lyra-leaves") == "expired"


def test_dead_subject_cancels_pending_chain_events(root: Path) -> None:
    _write(root / "npcs" / "marta.state.md", _npc_state("shelter", status="dead"))
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    for step, day in ((2, 3), (3, 4)):
        _write(
            root / f"runs/test/events/marta-step{step}.md",
            _event(
                f"marta-step{step}", day=day, subject="marta", chain="marta-antibiotics", step=step
            ),
        )
    _, vault = _engine(root)
    assert _event_status(vault, "marta-step2") == "cancelled"
    assert _event_status(vault, "marta-step3") == "cancelled"


def test_dead_targeting_branch_survives_and_fires(root: Path) -> None:
    _write(root / "npcs" / "vas.state.md", _npc_state("ridge", status="dead"))
    _write(root / "runs/test/clock.md", _clock(5, "night", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/vas-raid.md",
        _event("vas-raid", day=5, segment="night", subject="vas", chain="vas-raid", step=2),
    )
    _write(
        root / "runs/test/events/vas-raid-leaderless.md",
        _event(
            "vas-raid-leaderless",
            day=5,
            segment="night",
            subject="vas",
            npc_status="dead",
            chain="vas-raid",
            step=2,
        ),
    )
    outcome, vault = _engine(root)
    assert outcome is not None
    assert _event_status(vault, "vas-raid") == "cancelled"
    assert _event_status(vault, "vas-raid-leaderless") == "fired"
    assert outcome.directives == ["(Day 5, night) vas-raid-leaderless happened."]


def test_fired_branch_cancels_its_pending_sibling(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(5, "night", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/vas-raid.md",
        _event("vas-raid", day=5, segment="night", subject="vas", chain="vas-raid", step=2),
    )
    _write(
        root / "runs/test/events/vas-raid-leaderless.md",
        _event(
            "vas-raid-leaderless",
            day=5,
            segment="night",
            subject="vas",
            npc_status="dead",
            chain="vas-raid",
            step=2,
        ),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "vas-raid") == "fired"
    assert _event_status(vault, "vas-raid-leaderless") == "cancelled"


def test_missing_subject_holds_the_event(root: Path) -> None:
    _write(root / "npcs" / "marta.state.md", _npc_state("shelter", status="missing"))
    _write(root / "runs/test/clock.md", _clock(3, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-step2.md",
        _event(
            "marta-step2",
            day=3,
            subject="marta",
            chain="marta-antibiotics",
            step=2,
            policy="expire",
        ),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "marta-step2") == "pending"


def test_event_that_resolves_a_missing_subject_still_fires(root: Path) -> None:
    # 2026-09-25 marathon: step 2's beat made the writer mark Marta missing,
    # which then held "Marta staggers back" (it sets her status) until expiry.
    _write(root / "npcs" / "marta.state.md", _npc_state("pharmacy district", status="missing"))
    _write(root / "runs/test/clock.md", _clock(4, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-returns.md",
        _event(
            "marta-returns",
            day=4,
            subject="marta",
            chain="marta-antibiotics",
            step=3,
            mutations="- marta.status: wounded\n- marta.location: the gate",
        ),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "marta-returns") == "fired"
    assert "status: wounded" in (vault.read_state("npc", "marta") or "")


def test_held_event_expires_after_its_window(root: Path) -> None:
    _write(root / "npcs" / "marta.state.md", _npc_state("elsewhere"))
    _write(root / "runs/test/clock.md", _clock(5, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-step1.md",
        _event("marta-step1", day=2, subject="marta", npc_location="shelter", expires=1),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "marta-step1") == "expired"


def test_directive_cap_overflow_carries_once_narration_is_confirmed(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    for i in range(3):
        _write(
            root / f"runs/test/events/ev{i}.md",
            _event(f"ev{i}", day=1, priority=10 - i),
        )
    outcome, vault = _engine(root)
    assert outcome is not None
    assert outcome.directives == [
        "(Day 2, morning) ev0 happened.",
        "(Day 2, morning) ev1 happened.",
    ]
    # The DM narrated both rendered directives → confirmed and retired;
    # the overflow directive gets its first render.
    registry = SceneRegistry()
    second = story_events.run_end_of_turn(
        vault,
        registry,
        time_advance_hint=None,
        assistant_text="Indeed, ev0 happened and then ev1 happened, plain as day.",
    )
    assert second is not None
    assert second.directives == ["(Day 2, morning) ev2 happened."]


def test_ignored_directive_is_rerendered_until_confirmed(root: Path) -> None:
    # The 30-turn run's core failure: the DM ignored every directive and
    # the old mark-on-render policy retired them anyway. Now an ignored
    # directive keeps riding the prompt.
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    _write(root / "runs/test/events/ev0.md", _event("ev0", day=1))
    first, vault = _engine(root)
    assert first is not None and first.directives == ["(Day 2, morning) ev0 happened."]
    # DM reply shares nothing with the directive → re-rendered.
    second = story_events.run_end_of_turn(
        vault,
        SceneRegistry(),
        time_advance_hint=None,
        assistant_text="Rain drums on the corrugated roof while nothing moves outside.",
    )
    assert second is not None and second.directives == ["(Day 2, morning) ev0 happened."]
    # Narrated at last → confirmed, retired.
    third = story_events.run_end_of_turn(
        vault,
        SceneRegistry(),
        time_advance_hint=None,
        assistant_text="And so ev0 happened, exactly as feared.",
    )
    assert third is not None and third.directives == []
    _, raw = next((s, r) for s, r in vault.list_events() if s == "ev0")
    assert "narrated: true" in raw


def test_ignored_directive_is_abandoned_loudly_after_render_cap(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    _write(root / "runs/test/events/ev0.md", _event("ev0", day=1))
    outcome, vault = _engine(root)
    assert outcome is not None and outcome.directives == ["(Day 2, morning) ev0 happened."]
    ignoring_reply = "The wind keeps its own counsel; the shelter sleeps uneasily tonight."
    for _ in range(story_events.MAX_DIRECTIVE_RENDERS):
        result = story_events.run_end_of_turn(
            vault, SceneRegistry(), time_advance_hint=None, assistant_text=ignoring_reply
        )
        assert result is not None
    # After the cap the directive stops riding and the failure is on disk.
    final = story_events.run_end_of_turn(
        vault, SceneRegistry(), time_advance_hint=None, assistant_text=ignoring_reply
    )
    # (The baseline tick lands meanwhile.)
    assert final is not None and not any("ev0" in d for d in final.directives)
    _, raw = next((s, r) for s, r in vault.list_events() if s == "ev0")
    assert "narrated: abandoned" in raw


def test_clock_jump_fires_at_most_one_event_per_chain_per_pass(root: Path) -> None:
    # A next_day skip makes both chain steps time-eligible at once; step
    # 1's mutation would satisfy step 2's condition in the same pass —
    # the DM must not get "she arrives" and "she leaves" in one contract.
    _write(root / "npcs" / "lyra.state.md", _npc_state("perimeter ruins"))
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/lyra-arrives.md",
        _event(
            "lyra-arrives",
            day=1,
            segment="evening",
            subject="lyra",
            chain="lyra-gate",
            step=1,
            mutations="- lyra.location: the gate",
        ),
    )
    _write(
        root / "runs/test/events/lyra-leaves.md",
        _event(
            "lyra-leaves",
            day=1,
            segment="night",
            subject="lyra",
            npc_location="the gate",
            chain="lyra-gate",
            step=2,
        ),
    )
    outcome, vault = _engine(root)
    assert outcome is not None
    assert outcome.directives == ["(Day 2, morning) lyra-arrives happened."]
    assert _event_status(vault, "lyra-arrives") == "fired"
    assert _event_status(vault, "lyra-leaves") == "pending"  # re-evaluates next turn


def test_status_flip_preserves_authored_body_sections(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    raw = _event("keeper", day=1) + "\n## DM notes\n\nAuthored prose that must survive.\n"
    _write(root / "runs/test/events/keeper.md", raw)
    _, vault = _engine(root)
    _, reread = next((s, r) for s, r in vault.list_events() if s == "keeper")
    assert "status: fired" in reread
    assert "## DM notes" in reread
    assert "Authored prose that must survive." in reread


def test_in_scene_condition_holds_offscreen_event_during_conversation(root: Path) -> None:
    # First live smoke run: "Lyra gives up and leaves" fired while the
    # player was mid-negotiation with her. trigger_npc_in_scene: false
    # holds the event while she's engaged; it fires once she isn't.
    state = _npc_state("the gate").replace("in_scene: false", "in_scene: true")
    _write(root / "npcs" / "lyra.state.md", state)
    _write(root / "runs/test/clock.md", _clock(1, "night", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/lyra-leaves.md",
        _event(
            "lyra-leaves", day=1, segment="night", subject="lyra", npc_location="the gate"
        ).replace("subject_npc: lyra", "subject_npc: lyra\ntrigger_npc_in_scene: false"),
    )
    outcome, vault = _engine(root)
    assert outcome is not None
    assert outcome.directives == []
    assert _event_status(vault, "lyra-leaves") == "pending"  # held while engaged
    # Player walks away — writer stamps in_scene: false; event now fires.
    _write(root / "runs" / "test" / "npcs" / "lyra.state.md", _npc_state("the gate"))
    second, vault = _engine(root)
    assert second is not None
    assert _event_status(vault, "lyra-leaves") == "fired"


def test_confirmation_survives_paraphrase_via_anchor_and_stems(root: Path) -> None:
    # replay-003 ground truth: the DM narrated "Lyra at the fence" in
    # its own words and exact-word overlap scored 0.04 — confirmation
    # must accept a proper-name anchor plus a few stemmed echoes.
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    directive_body = (
        "\n## Directive\n\nThe sentry platform calls it down as the light "
        "fails: a lone woman on the approach road, hands empty and raised, "
        "asking to be let in before dark. She gives her name as Lyra.\n"
    )
    raw = _event("lyra-arrives", day=1).split("\n## Directive")[0] + directive_body
    _write(root / "runs/test/events/lyra-arrives.md", raw)
    first, vault = _engine(root)
    assert first is not None and len(first.directives) == 1
    paraphrase = (
        "You approach the fence slowly, hands where she can see them. The "
        "woman at the wire watches you come — Lyra, she says her name is, "
        "asking again to be let inside before the light is gone."
    )
    second = story_events.run_end_of_turn(
        vault, SceneRegistry(), time_advance_hint=None, assistant_text=paraphrase
    )
    assert second is not None and second.directives == []
    _, reread = next((s, r) for s, r in vault.list_events() if s == "lyra-arrives")
    assert "narrated: true" in reread


def test_abandoned_directive_is_not_rewarned_every_turn(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    _write(root / "runs/test/events/ev0.md", _event("ev0", day=1))
    outcome, vault = _engine(root)
    assert outcome is not None
    ignoring = "The wind keeps its own counsel tonight; nothing else stirs."
    for _ in range(story_events.MAX_DIRECTIVE_RENDERS + 2):
        story_events.run_end_of_turn(
            vault, SceneRegistry(), time_advance_hint=None, assistant_text=ignoring
        )
    _, raw = next((s, r) for s, r in vault.list_events() if s == "ev0")
    assert "narrated: abandoned" in raw
    # Settled: renders must not keep climbing past the cap.
    fm, _ = frontmatter.split(raw)
    assert fm["renders"] == story_events.MAX_DIRECTIVE_RENDERS


def test_tell_line_is_rendered_instead_of_directive_prose(root: Path) -> None:
    # The 60-90-word Directive paragraphs read as skippable color to the
    # 14B; when a terse ## Tell line is authored, THAT is what rides the
    # prompt — and confirmation matches against the same text.
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    raw = (
        _event("marta-gone", day=1)
        + "\n## Tell\n\nNarrate now: Marta is gone; her note sits in the sentry log.\n"
    )
    _write(root / "runs/test/events/marta-gone.md", raw)
    outcome, vault = _engine(root)
    assert outcome is not None
    assert outcome.directives == [
        "(Day 2, morning) Narrate now: Marta is gone; her note sits in the sentry log."
    ]
    confirmed = story_events.run_end_of_turn(
        vault,
        SceneRegistry(),
        time_advance_hint=None,
        assistant_text="Word spreads fast: Marta left before dawn — gone, her note found in the sentry log.",
    )
    assert confirmed is not None and confirmed.directives == []
    _, reread = next((s, r) for s, r in vault.list_events() if s == "marta-gone")
    assert "narrated: true" in reread


def test_invalid_mutation_lines_are_dropped() -> None:
    parsed = story_events.parse_mutations(
        "- marta.location: shelter\n"
        "- marta.not_a_field: x\n"
        "- marta.status: not-an-enum-value\n"
        "- garbage without separator\n"
        "- note: a real note"
    )
    assert parsed == [
        ("npc", "marta", "location", "shelter"),
        ("note", "a real note"),
    ]


def test_directive_covered_matches_paraphrase() -> None:
    # Anchored match: proper name + supporting content words.
    directive = "Marta leaves the shelter, heading for the pharmacy district wounded."
    reply = "Marta limped past the gate, clutching her side, bound for the pharmacy."
    assert story_events.directive_covered(reply, directive)


def test_directive_covered_rejects_unrelated_reply() -> None:
    directive = "Marta leaves the shelter, heading for the pharmacy district wounded."
    reply = "The sun rose quietly over the empty streets."
    assert not story_events.directive_covered(reply, directive)


def test_directive_covered_empty_inputs_return_false() -> None:
    assert not story_events.directive_covered("", "anything")
    assert not story_events.directive_covered("anything", "")


def test_directive_covered_matches_accented_hungarian_words() -> None:
    # Hungarian play is first-class at runtime; accented content words
    # must survive tokenization or the guarded retry burns an extra
    # completion on every Hungarian event turn.
    directive = "Márta elhagyja a menedéket, és sebesülten a gyógyszertár felé indul."
    reply = "Márta sebesülten sántikál ki a kapun, egyenesen a gyógyszertár irányába indul."
    assert story_events.directive_covered(reply, directive)


def test_directive_anchor_words_extracts_proper_nouns() -> None:
    # A name as the very first word of the directive doesn't match —
    # same sentence-initial anti-noise tradeoff _ANCHOR_RE already makes
    # for the narration-confirmation check.
    directive = "The plan falls apart when Marta leaves the shelter wounded."
    assert story_events.directive_anchor_words(directive) == ["Marta"]


def test_directive_anchor_words_dedupes_preserving_order() -> None:
    directive = "By dusk, Marta warns Kerek, then Marta leaves before Kerek can answer."
    assert story_events.directive_anchor_words(directive) == ["Marta", "Kerek"]


def test_directive_anchor_words_empty_without_anchors() -> None:
    assert story_events.directive_anchor_words("the water purifier fails at dawn") == []


# --- regressions from the 50-turn marathon review ---------------------------


def test_npc_location_trigger_matches_free_text_writer_location(root: Path) -> None:
    # 07-12 replay-003: the writer wrote 'hall of ridgeway fallout shelter' and
    # marta-goes-alone (requires 'ridgeway fallout shelter') expired on equality.
    _write(root / "npcs" / "marta.state.md", _npc_state("hall of ridgeway fallout shelter"))
    _write(root / "runs/test/clock.md", _clock(3, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-goes-alone.md",
        _event(
            "marta-goes-alone",
            day=3,
            subject="marta",
            npc_location="ridgeway fallout shelter",
            policy="expire",
        ),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "marta-goes-alone") == "fired"


def test_pc_location_trigger_matches_free_text_writer_location(root: Path) -> None:
    _write(
        root / "runs/test/characters/hero.state.md",
        "---\ntype: pc_state\nstatus: active\nlocation: infirmary of ridgeway fallout shelter\n"
        "last_updated: 2026-07-11\n---\n",
    )
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/inspection.md",
        _event("inspection", day=2, policy="expire").replace(
            "trigger_day: 2", "trigger_day: 2\ntrigger_pc_location: ridgeway fallout shelter"
        ),
    )
    vault = Vault(root, run_slug="test")
    registry = SceneRegistry()
    registry.set_pc("hero")
    story_events.run_end_of_turn(vault, registry, time_advance_hint=None)
    assert _event_status(vault, "inspection") == "fired"


def test_time_hint_is_rate_limited_to_one_per_segment_pace(root: Path) -> None:
    # 07-12 replay-003: a hint drove 12 of the last 12 advances, including
    # two next_day skips one turn apart — the run ended on day 12 vs 7.
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=1, last_advance=1))
    _, vault = _engine(root, hint="next_day")
    assert (_clock_state(vault)["day"], _clock_state(vault)["segment"]) == (2, "morning")
    shown = "Hours later, the shelter stirs."
    _turn(vault, shown, hint="next_day")  # one turn later: suppressed
    assert (_clock_state(vault)["day"], _clock_state(vault)["segment"]) == (2, "morning")
    for _ in range(2):
        _turn(vault, shown, hint="next_segment")  # still inside the window
    assert _clock_state(vault)["segment"] == "morning"
    # turns_per_segment (4) turns after the last honored hint, hints count again.
    _turn(vault, shown, hint="next_segment")
    assert (_clock_state(vault)["day"], _clock_state(vault)["segment"]) == (2, "afternoon")


def test_rerendered_directive_is_not_reported_as_a_world_change(root: Path) -> None:
    # 07-12 replay-004: vas-sighting was re-emitted as `world.event fired`
    # on turns 12-15 and each re-render reset the stall counter.
    _write(root / "runs/test/clock.md", _clock(2, "morning", turn=1, last_advance=1))
    _write(root / "runs/test/events/ev0.md", _event("ev0", day=1))
    first, vault = _engine(root)
    assert first is not None and first.changed_events == [("ev0", "ev0 title", "fired")]
    second = story_events.run_end_of_turn(
        vault,
        SceneRegistry(),
        time_advance_hint=None,
        assistant_text="Rain drums on the corrugated roof while nothing moves outside.",
    )
    assert second is not None and second.directives == ["(Day 2, morning) ev0 happened."]
    assert second.changed_events == []
    _, raw = next((s, r) for s, r in vault.list_events() if s == "ev0")
    assert frontmatter.split(raw)[0]["renders"] == 2  # render count still persisted


def test_deceased_subject_fires_the_dead_targeting_branch(root: Path) -> None:
    _write(root / "npcs" / "vas.state.md", _npc_state("ridge", status="deceased"))
    _write(root / "runs/test/clock.md", _clock(5, "night", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/vas-raid.md",
        _event("vas-raid", day=5, segment="night", subject="vas", chain="vas-raid", step=2),
    )
    _write(
        root / "runs/test/events/vas-raid-leaderless.md",
        _event(
            "vas-raid-leaderless",
            day=5,
            segment="night",
            subject="vas",
            npc_status="dead",
            chain="vas-raid",
            step=2,
        ),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "vas-raid") == "cancelled"
    assert _event_status(vault, "vas-raid-leaderless") == "fired"


def test_wounded_subject_does_not_block_a_default_event(root: Path) -> None:
    _write(root / "npcs" / "marta.state.md", _npc_state("shelter", status="wounded"))
    _write(root / "runs/test/clock.md", _clock(3, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-step2.md",
        _event("marta-step2", day=3, subject="marta", policy="expire"),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "marta-step2") == "fired"


@pytest.mark.parametrize("failing", ["write_clock", "write_event"])
def test_failed_write_never_refires_an_event(
    root: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    # A write failure after the mutations landed left the event pending on
    # disk → it fired again next turn, appending the beat twice.
    _write(root / "runs/test/clock.md", _clock(3, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-gone.md",
        _event("marta-gone", day=3, mutations="- beat: Marta is missing."),
    )
    vault = Vault(root, run_slug="test")
    real = getattr(vault, failing)
    calls = {"n": 0}

    def fail_once(*args: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return str(real(*args))

    monkeypatch.setattr(vault, failing, fail_once)
    with contextlib.suppress(OSError):
        story_events.run_end_of_turn(vault, SceneRegistry(), time_advance_hint=None)
    story_events.run_end_of_turn(vault, SceneRegistry(), time_advance_hint=None)
    assert vault.read_file("beat.md").count("Marta is missing.") == 1
    assert _event_status(vault, "marta-gone") == "fired"


def test_missing_trigger_day_orders_like_day_one(root: Path) -> None:
    # Ordering must agree with eligibility (_trigger_abs defaults to day 1):
    # at equal effective trigger time, priority decides which step fires.
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/a-undated.md",
        _event("a-undated", day=1, chain="c", step=1).replace("trigger_day: 1\n", ""),
    )
    _write(
        root / "runs/test/events/b-urgent.md",
        _event("b-urgent", day=1, chain="c", step=2, priority=5),
    )
    _, vault = _engine(root)
    assert _event_status(vault, "b-urgent") == "fired"
    assert _event_status(vault, "a-undated") == "pending"


def test_narrate_now_prefix_does_not_dilute_confirmation() -> None:
    # Four content words, one echoed = exactly the 0.25 bar; the Tell's
    # imperative prefix must not add "narrate" to the denominator.
    directive = "the water purifier fails overnight"
    reply = "Someone mutters about the purifier."
    assert story_events.directive_covered(reply, directive)
    assert story_events.directive_covered(reply, f"Narrate now: {directive}")


# --- time: code-shown, never held ------------------------------------------
# 2026-09-26 marathon-50t replay-002: the narration hold made the clock ~1.5x
# slower (day 5 vs day 7) because the 14B almost never narrated time, so
# authored events never came due. The clock no longer waits on the DM; the
# advance is shown by code (aidm.time_shown).

_IGNORING_REPLY = "Rain drums on the corrugated roof while nothing moves outside."
_TRANSITION_D3_MORNING = (
    "Time has passed: it is now day 3, morning (was day 2, night). "
    "Show the passage of time before continuing."
)


def _turn(vault: Vault, reply: str, **kw: object) -> story_events.EngineOutcome:
    registry = SceneRegistry()
    if kw.get("scene_changed"):
        registry.note_scene_change()
    hint = kw.get("hint")
    outcome = story_events.run_end_of_turn(
        vault,
        registry,
        time_advance_hint=hint if isinstance(hint, str) else None,
        assistant_text=reply,
    )
    assert outcome is not None
    return outcome


def test_clock_advance_queues_no_directive(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(2, "night", turn=1, last_advance=0))
    outcome, _ = _engine(root, scene_changed=True)
    assert outcome is not None
    assert outcome.directives == []
    assert outcome.clock_line == "Day 3, morning."


def test_quiet_turns_never_advance_the_clock_mid_scene(root: Path) -> None:
    # No baseline tick (playtest, 2026-09-27): however many quiet turns pass,
    # only a narrated hint or a scene change moves time.
    _write(root / "runs/test/clock.md", _clock(1, "morning", turn=1, last_advance=0))
    vault = Vault(root, run_slug="test")
    for _ in range(12):
        _turn(vault, _IGNORING_REPLY)
    assert _clock_state(vault)["segment"] == "morning"
    _turn(vault, _IGNORING_REPLY, hint="next_day")
    fm = _clock_state(vault)
    assert (fm["day"], fm["segment"]) == (2, "morning")
    assert fm["last_hint_turn"] == fm["turn"]  # the hint was honored


def test_legacy_transition_keys_in_clock_md_load_harmlessly(root: Path) -> None:
    clock = _clock(3, "morning", turn=5, last_advance=5).replace(
        "last_advance_turn: 5",
        "last_advance_turn: 5\ntransition_from_day: 2\n"
        "transition_from_segment: night\ntransition_renders: 2",
    )
    _write(root / "runs/test/clock.md", clock)
    outcome = _turn(Vault(root, run_slug="test"), _IGNORING_REPLY)
    assert outcome.directives == []
    fm = _clock_state(Vault(root, run_slug="test"))
    assert (fm["day"], fm["segment"], fm["turn"]) == (3, "morning", 6)
    assert not any(key.startswith("transition_") for key in fm)  # dropped on write


def test_time_shown_line_renders_the_clock_line_per_locale() -> None:
    assert story_events.time_shown_line("Day 2, morning.", "en") == "*Day 2, morning.*"
    # Locales without their own template fall back to English.
    assert story_events.time_shown_line("Day 2, morning.", "xx") == "*Day 2, morning.*"
    assert story_events.time_shown_line("", "en") == ""


def test_fired_event_directive_carries_its_in_fiction_time(root: Path) -> None:
    _write(root / "runs/test/clock.md", _clock(5, "night", turn=1, last_advance=1))
    _write(root / "runs/test/events/ev0.md", _event("ev0", day=5))
    first, vault = _engine(root)
    assert first is not None and first.directives == ["(Day 5, night) ev0 happened."]
    # The prefix is staging, not content: it never blocks confirmation.
    done = _turn(vault, "And so ev0 happened, exactly as feared.")
    assert done.directives == []


def test_is_time_transition_tells_transitions_from_event_directives() -> None:
    assert story_events.is_time_transition(_TRANSITION_D3_MORNING)
    assert not story_events.is_time_transition("(Day 5, night) ev0 happened.")
    assert story_events.directive_covered("And so ev0 happened.", "(Day 5, night) ev0 happened.")


# --- subject named this turn → no off-screen move ----------------------------


def _moving_event(root: Path, *, policy: str, mutations: str) -> None:
    _write(root / "runs/test/clock.md", _clock(3, "morning", turn=1, last_advance=1))
    _write(
        root / "runs/test/events/marta-goes-alone.md",
        _event(
            "marta-goes-alone",
            day=3,
            segment="morning",
            subject="marta",
            npc_location="shelter",
            policy=policy,
            mutations=mutations,
        ),
    )


def _named_turn(root: Path, *, user: str = "", reply: str = "") -> Vault:
    vault = Vault(root, run_slug="test")
    story_events.run_end_of_turn(
        vault, SceneRegistry(), time_advance_hint=None, assistant_text=reply, user_text=user
    )
    return vault


def test_location_moving_event_held_while_player_names_subject(root: Path) -> None:
    # 2026-09-25 replay-003: "Marta left alone" fired against her stale
    # shelter location while she walked beside the player.
    _moving_event(root, policy="hold", mutations="- marta.location: pharmacy district")
    vault = _named_turn(root, user="I fall into step beside Marta.", reply="The bridge groans.")
    assert _event_status(vault, "marta-goes-alone") == "pending"
    assert "pharmacy" not in (vault.read_state("npc", "marta") or "")
    # Next turn nobody names her: the same event fires.
    vault = _named_turn(root, user="I check the wire.", reply="Rain on the roof.")
    assert _event_status(vault, "marta-goes-alone") == "fired"
    assert "location: pharmacy district" in (vault.read_state("npc", "marta") or "")


def test_status_moving_expire_event_expires_when_dm_names_subject(root: Path) -> None:
    _moving_event(root, policy="expire", mutations="- marta.status: wounded")
    vault = _named_turn(root, user="I wait.", reply="Marta tightens the strap on her pack.")
    assert _event_status(vault, "marta-goes-alone") == "expired"
    assert "status: wounded" not in (vault.read_state("npc", "marta") or "")


def test_event_not_moving_its_subject_fires_even_when_subject_named(root: Path) -> None:
    _moving_event(root, policy="hold", mutations="- marta.mood: tense\n- beat: Tension rises.")
    vault = _named_turn(root, user="Marta, wait.", reply="Marta looks up.")
    assert _event_status(vault, "marta-goes-alone") == "fired"


def test_location_moving_event_held_while_subject_is_with_pc(root: Path) -> None:
    # Presence the narration never names ("the two of you cross"): the PC
    # writer's with_pc still keeps her off-screen events on hold.
    _moving_event(root, policy="hold", mutations="- marta.location: pharmacy district")
    vault = Vault(root, run_slug="test")
    story_events.run_end_of_turn(
        vault,
        SceneRegistry(),
        time_advance_hint=None,
        assistant_text="The bridge groans as the two of you cross.",
        user_text="I keep walking.",
        with_pc={"marta"},
    )
    assert _event_status(vault, "marta-goes-alone") == "pending"
    assert "pharmacy" not in (vault.read_state("npc", "marta") or "")


def test_arrival_already_lived_through_is_satisfied_not_held(root: Path) -> None:
    # 2026-09-27 playtest 3: lyra-at-the-gate was held 8 turns while Lyra stood
    # beside the player — an arrival the player had already played out.
    _moving_event(root, policy="hold", mutations="- marta.location: the gate")
    path = root / "runs/test/events/marta-goes-alone.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "defer_policy: hold", "defer_policy: hold\nif_subject_present: satisfied"
        ),
        encoding="utf-8",
    )
    vault = Vault(root, run_slug="test")
    outcome = story_events.run_end_of_turn(
        vault,
        SceneRegistry(),
        time_advance_hint=None,
        assistant_text="Marta walks beside you.",
        user_text="I keep walking.",
        with_pc={"marta"},
    )
    assert _event_status(vault, "marta-goes-alone") == "fired"
    assert outcome.directives == []  # no re-staging
    assert "the gate" not in (vault.read_state("npc", "marta") or "")  # no mutation
