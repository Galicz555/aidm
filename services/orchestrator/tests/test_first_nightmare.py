"""First Nightmare — campaign feature ``first-nightmare`` (aidm.first_nightmare)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from aidm import first_nightmare, story_events
from aidm.advance_judge import judge_falls_asleep, judge_trial_resolved
from aidm.scene_sheet_writer import SceneSheetWriter
from aidm.scene_state import SceneRegistry
from aidm.time_shown import TimeShownProcessor
from aidm.turn_director import DirectorState
from aidm.turn_rollback import TurnRollback
from aidm.vault import Vault
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test
from scripts.lint_vault import _lint_file

_VOICE = (
    Path(__file__).resolve().parent / "fixtures" / "spell-voice.md"
).read_text(encoding="utf-8")
_SHEET = """---
type: character
lang: en
name: Rowan
class_role: Aspirant — guarded district
hexaco_honesty_humility: 4
hexaco_emotionality: 3
hexaco_extraversion: 2
hexaco_agreeableness: 4
hexaco_conscientiousness: 3
hexaco_openness: 4
---

# Rowan

Wants to be free; fears never being accepted.

## Psychology (HEXACO)

- Honesty-Humility (4): plain.
- Emotionality (3): steady.
- Extraversion (2): shy.
- Agreeableness (4): goes along.
- Conscientiousness (3): drifts.
- Openness (4): curious.
"""
_PLAN = {
    "title": "The Drowned Bell",
    "setting": "A flooded bell tower of an ancient river city, the night the dam broke.",
    "role": "The bell-ringer's mute apprentice, chained to the rope.",
    "conflict": "The Lamprey King climbs the tower toward the sound; it is blind to anything silent.",
    "stages": ["The water rises.", "The Lamprey King enters the tower.", "The bell must ring."],
    "win": "The apprentice rings the bell three times from the roof and the flood recedes.",
    "lose": "The Lamprey King drags the apprentice into the water.",
    "aspect": "Bell Warden",
    "aspect_description": "Loud enough to be found, which is the problem.",
    "aspect_ability": "Toll: one struck note stuns everything within ten paces; costs essence.",
    "flaw": "Cannot keep silent when someone calls their name.",
    "attributes": ["[Keen Hearing]", "Stubborn"],
}


def _vault(tmp_path: Path, *, feature: bool = True, pc_sheet: bool = True) -> Vault:
    features = "features: [first-nightmare]\n" if feature else ""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "campaign.md").write_text(
        f"---\ntype: campaign\nlang: en\ntitle: T\nstatus: active\n{features}---\n",
        encoding="utf-8",
    )
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "spell-voice.md").write_text(_VOICE, encoding="utf-8")
    run = tmp_path / "runs" / "r1"
    (run / "characters").mkdir(parents=True)
    if pc_sheet:
        (run / "characters" / "rowan.md").write_text(_SHEET, encoding="utf-8")
    (run / "clock.md").write_text(
        "---\ntype: clock\nday: 1\nsegment: morning\nturn: 3\nlast_updated: 2026-09-28\n---\n",
        encoding="utf-8",
    )
    return Vault(tmp_path, run_slug="r1")


def _logprobs(p_yes: float) -> Any:
    top = [
        SimpleNamespace(token="YES", logprob=math.log(p_yes)),
        SimpleNamespace(token="NO", logprob=math.log(1 - p_yes)),
    ]
    content = [SimpleNamespace(top_logprobs=top)]
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])


def _text(content: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _Client:
    """Answers judge calls (``logprobs=True``) and JSON calls from two queues."""

    def __init__(self, *, p_yes: list[float] = (), texts: list[str] = ()) -> None:  # type: ignore[assignment]
        self.p_yes, self.texts, self.calls = list(p_yes), list(texts), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls += 1
        if kwargs.get("logprobs"):
            return _logprobs(self.p_yes.pop(0))
        return _text(self.texts.pop(0))


def _writer(vault: Vault, client: Any, director: DirectorState) -> SceneSheetWriter:
    return SceneSheetWriter(
        context=LLMContext(),
        registry=SceneRegistry(current_pc_slug="rowan"),
        vault=vault,
        client=client,
        model="m",
        locale="en",
        timeout_s=5,
        emitter=MagicMock(),
        director=director,
        nightmare=True,
    )


async def _turn(writer: SceneSheetWriter, player: str = "I sleep.", reply: str = "Dark.") -> bool:
    trial, asleep, resolved = writer._start_nightmare_judges("rowan", player, reply)
    return await writer._advance_nightmare(trial, asleep, resolved, "rowan", player, reply)


# --- judges --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_asleep_judge_cuts_at_its_conservative_threshold() -> None:
    # 0.712 is what the live model gave the real falling-asleep turn (probe, 2026-09-28).
    assert await judge_falls_asleep(_Client(p_yes=[0.712]), "m", player="p", current="c") is True
    assert await judge_falls_asleep(_Client(p_yes=[0.3]), "m", player="p", current="c") is False
    assert await judge_falls_asleep(_Client(), "m", player="p", current="") is None


@pytest.mark.asyncio
async def test_the_judges_fail_open() -> None:
    broken = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_raise)))
    assert await judge_falls_asleep(broken, "m", player="p", current="c") is None
    verdict = await judge_trial_resolved(broken, "m", win="w", lose="l", player="p", current="c")
    assert verdict is None
    resolved = await judge_trial_resolved(
        _Client(p_yes=[0.45]), "m", win="w", lose="l", player="p", current="c"
    )
    assert resolved is False


async def _raise(**_kwargs: Any) -> Any:
    raise TimeoutError("gateway down")


# --- plan ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_valid_plan_is_written_as_a_lint_clean_active_arc(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    first_nightmare.start(vault)
    client = _Client(texts=["no json here", json.dumps(_PLAN)])  # one bad attempt, then good

    plan = await first_nightmare.write_plan(vault, client, "m", pc_slug="rowan", timeout_s=5)

    assert plan is not None and plan.attributes == ("[Keen Hearing]", "[Stubborn]")
    arc = tmp_path / "runs" / "r1" / "arcs" / "first-nightmare.md"
    assert [i.message for i in _lint_file(arc)] == []
    trial = first_nightmare.read_trial(vault)
    assert trial is not None and trial.status == "active" and trial.plan == plan
    block = first_nightmare.dm_block(vault)
    assert "The Lamprey King climbs" in block and "reveal ONLY when the player checks" in block


@pytest.mark.asyncio
async def test_a_plan_that_reuses_a_listed_name_is_rejected(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    first_nightmare.start(vault)
    stolen = {**_PLAN, "conflict": "The Glass Tyrant hunts the caravan down the slope."}
    client = _Client(texts=[json.dumps(stolen), json.dumps({**_PLAN, "stages": ["one"]})])

    assert (
        await first_nightmare.write_plan(vault, client, "m", pc_slug="rowan", timeout_s=5) is None
    )
    trial = first_nightmare.read_trial(vault)
    assert trial is not None and trial.status == "dormant"  # retried next turn


# --- transitions -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_falling_asleep_starts_the_trial_and_queues_the_greeting(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    director = DirectorState()
    writer = _writer(vault, _Client(p_yes=[0.9], texts=[json.dumps(_PLAN)]), director)

    assert await _turn(writer) is True  # clock frozen from this turn on
    trial = first_nightmare.read_trial(vault)
    assert trial is not None and trial.status == "active"
    assert director.spell_line.startswith("*Aspirant Rowan. The Nightmare Spell has chosen you.")
    assert "The Drowned Bell" in director.spell_line


@pytest.mark.asyncio
async def test_no_start_while_awake_in_session_zero_or_without_the_feature(tmp_path: Path) -> None:
    awake = _Client(p_yes=[0.2])
    assert await _turn(_writer(_vault(tmp_path / "a"), awake, DirectorState())) is False
    assert first_nightmare.read_trial(Vault(tmp_path / "a", run_slug="r1")) is None
    for vault in (_vault(tmp_path / "b", pc_sheet=False), _vault(tmp_path / "c", feature=False)):
        client = _Client()
        assert await _turn(_writer(vault, client, DirectorState())) is False
        assert client.calls == 0  # no judge at all


async def _active(tmp_path: Path, turns: int) -> Vault:
    vault = _vault(tmp_path)
    first_nightmare.start(vault)
    await first_nightmare.write_plan(
        vault, _Client(texts=[json.dumps(_PLAN)]), "m", pc_slug="rowan", timeout_s=5
    )
    trial = first_nightmare.read_trial(vault)
    assert trial is not None
    for _ in range(turns):
        trial = first_nightmare.count_turn(vault, trial)
    return vault


@pytest.mark.asyncio
async def test_the_end_judge_waits_for_the_minimum_trial_turns(tmp_path: Path) -> None:
    vault = await _active(tmp_path, turns=0)
    client = _Client()
    assert await _turn(_writer(vault, client, DirectorState())) is True
    assert client.calls == 0  # turn 1 of the trial: no end judge yet
    trial = first_nightmare.read_trial(vault)
    assert trial is not None and trial.turns == 1 and trial.status == "active"


@pytest.mark.asyncio
async def test_a_survived_trial_appraises_writes_the_runes_and_ends(tmp_path: Path) -> None:
    vault = await _active(tmp_path, turns=2)
    director = DirectorState()
    appraisal = json.dumps({"outcome": "survived", "appraisal": "glorious"})
    writer = _writer(vault, _Client(p_yes=[0.8], texts=[appraisal]), director)

    assert await _turn(writer, "I ring it.", "The flood recedes.") is True

    trial = first_nightmare.read_trial(vault)
    assert trial is not None and trial.status == "resolved"
    assert "Performance rated: Glorious." in director.spell_line
    assert "**Flaw:** Cannot keep silent" in director.spell_line
    sheet_path = tmp_path / "runs" / "r1" / "characters" / "rowan.md"
    sheet = sheet_path.read_text(encoding="utf-8")
    assert "## Runes\n\n- Rank: Sleeper\n- Aspect: [Bell Warden]" in sheet
    assert [i.message for i in _lint_file(sheet_path)] == []
    # Resolved for good: the next turn neither judges nor freezes the clock.
    client = _Client()
    assert await _turn(_writer(vault, client, DirectorState())) is False
    assert client.calls == 0


@pytest.mark.asyncio
async def test_a_trial_lost_is_logged_not_enforced(tmp_path: Path) -> None:
    vault = await _active(tmp_path, turns=2)
    director = DirectorState()
    died = json.dumps({"outcome": "died", "appraisal": "Poor"})
    await _turn(_writer(vault, _Client(p_yes=[0.9], texts=[died]), director))

    assert director.spell_line == ""  # no appraisal, no runes
    assert "## Runes" not in (tmp_path / "runs/r1/characters/rowan.md").read_text(encoding="utf-8")
    arc = (tmp_path / "runs/r1/arcs/first-nightmare.md").read_text(encoding="utf-8")
    assert "status: resolved" in arc and "outcome: died" in arc


# --- clock, prefix, injector, rollback ------------------------------------------


def test_the_clock_is_frozen_inside_the_trial(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    registry = SceneRegistry()
    registry.note_scene_change()  # the scene-end judge read "falls asleep" as leaving
    before = (tmp_path / "runs/r1/clock.md").read_text(encoding="utf-8")

    outcome = story_events.run_end_of_turn(
        vault, registry, time_advance_hint="next_segment", frozen=True
    )

    assert (
        outcome is not None and outcome.clock_line == "Day 1, morning." and outcome.directives == []
    )
    assert registry.scene_changes == 0
    assert (tmp_path / "runs/r1/clock.md").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_the_spell_line_opens_the_next_reply_once() -> None:
    director = DirectorState(spell_line="*Welcome, Aspirant!*")
    proc = TimeShownProcessor(director, "en", initial_clock_line="Day 1, morning.")

    def reply(text: str) -> list[Any]:
        return [LLMFullResponseStartFrame(), LLMTextFrame(text=text), LLMFullResponseEndFrame()]

    down, _ = await run_test(proc, frames_to_send=[*reply("Water."), *reply("Later.")])
    texts = [f.text for f in down if isinstance(f, LLMTextFrame)]
    assert texts == ["*Welcome, Aspirant!*\n\n", "Water.", "Later."]


@pytest.mark.asyncio
async def test_the_dm_sees_the_plan_and_no_waking_time_line(tmp_path: Path) -> None:
    vault = await _active(tmp_path, turns=1)
    from aidm.scene_sheet_injector import SceneSheetInjector  # noqa: PLC0415
    from pipecat.frames.frames import LLMContextFrame  # noqa: PLC0415
    from pipecat.processors.frame_processor import FrameDirection  # noqa: PLC0415

    ctx = LLMContext(messages=[{"role": "user", "content": "I look around."}])
    director = DirectorState(clock_line="Day 1, morning.")
    injector = SceneSheetInjector(SceneRegistry(), "en", vault, director=director)
    injector.push_frame = MagicMock(side_effect=_anoop)  # type: ignore[method-assign]
    await injector.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)

    block = next(m["content"] for m in ctx.messages if m["role"] == "system")
    assert "First Nightmare — the player is INSIDE their trial (turn 2)" in block
    assert "[Time]" not in block


async def _anoop(*_args: Any, **_kwargs: Any) -> None:
    return None


@pytest.mark.asyncio
async def test_regenerate_rolls_back_a_nightmare_start(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    director = DirectorState()
    transcript: list[dict[str, str]] = []
    rollback = TurnRollback(
        context=LLMContext(),
        transcript=transcript,
        registry=SceneRegistry(current_pc_slug="rowan"),
        director=director,
        vault=vault,
        processors=[],
        summarizer=None,
        inlined_sheets=set(),
    )
    await rollback.capture("I let sleep take me.")
    writer = _writer(vault, _Client(p_yes=[0.9], texts=[json.dumps(_PLAN)]), director)
    await _turn(writer, "I let sleep take me.")
    assert first_nightmare.read_trial(vault) is not None and director.spell_line

    rollback.restore()

    assert first_nightmare.read_trial(vault) is None
    assert not (tmp_path / "runs/r1/arcs/first-nightmare.md").exists()
    assert director.spell_line == ""
