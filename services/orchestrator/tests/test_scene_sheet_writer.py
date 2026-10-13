"""Tests for SceneSheetWriter: JSON-patch parse → apply → write state file."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from aidm import frontmatter
from aidm.scene_sheet_writer import (
    _STUB_GLOSS_MARKER,
    CONDENSED_REPLY_TAG,
    SceneSheetWriter,
    _apply_patch,
    _clean_patch,
    _extract_recent_exchanges,
    _parse_patch,
    _stub_identity_sheet,
    _validate_consolidation,
    detect_addressed_names,
    detect_introduced_names,
    detect_unnamed_referents,
    detect_voiced_names,
    rewrite_sheet,
)
from aidm.scene_state import SceneRegistry
from aidm.turn_director import DirectorState
from aidm.vault import VaultPathError
from loguru import logger
from pipecat.frames.frames import LLMContextAssistantTimestampFrame
from pipecat.processors.frame_processor import FrameDirection

_OLD_STATE = """---
type: npc_state
status: active
location: shelter
in_scene: true
mood: guarded
scene_goal: test trust
social_mask: calm front
attitude: wary
last_updated: 2026-06-26
---

## Recent (last 5 events)

- T1: arrived at shelter.

## Session so far

Lyra has been wary since arrival.
"""


@dataclass
class _FakeVault:
    sheets: dict[str, str] = field(default_factory=dict)
    written_npc_state: list[tuple[str, str]] = field(default_factory=list)
    written_pc_state: list[tuple[str, str]] = field(default_factory=list)

    def read_file(self, path: str, *, max_bytes: int = 200_000) -> str:
        if path not in self.sheets:
            raise VaultPathError(f"not a file: {path}")
        return self.sheets[path]

    def read_state(self, kind: str, slug: str) -> str | None:
        subdir = "characters" if kind == "pc" else "npcs"
        return self.sheets.get(f"{subdir}/{slug}.state.md")

    def list_dir(self, path: str) -> list[str]:
        # Used by _collect_blocked_npc_names; return identity files only.
        return [p for p in self.sheets if p.startswith(f"{path}/") and not p.endswith(".state.md")]

    def write_npc_state(self, slug: str, content: str) -> str:
        self.written_npc_state.append((slug, content))
        self.sheets[f"npcs/{slug}.state.md"] = content
        return f"npcs/{slug}.state.md"

    def write_npc_sheet(self, slug: str, content: str) -> str:
        self.sheets[f"npcs/{slug}.md"] = content
        return f"npcs/{slug}.md"

    def write_character_state(self, slug: str, content: str) -> str:
        self.written_pc_state.append((slug, content))
        self.sheets[f"characters/{slug}.state.md"] = content
        return f"characters/{slug}.state.md"


def _client_returning(content: str) -> Any:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    create = AsyncMock(return_value=completion)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _client_seq(*contents: str) -> Any:
    """One reply per patch call, in target order (NPCs sorted, then PC); then '{}'."""
    queue = list(contents)

    async def create(**_: Any) -> Any:
        content = queue.pop(0) if queue else "{}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create)))
    )


def _turn_writer(
    vault: _FakeVault, reg: SceneRegistry, client: Any, reply: str, user: str = "I act."
) -> SceneSheetWriter:
    """Writer whose context holds one finished exchange (``user`` → ``reply``)."""
    return SceneSheetWriter(
        context=SimpleNamespace(
            messages=[{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
        ),
        registry=reg,
        vault=vault,
        client=client,
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
    )


def _phases(writer: SceneSheetWriter) -> list[str]:
    calls = writer._emitter.push_frame.call_args_list  # type: ignore[attr-defined]
    return [c.args[0].message["phase"] for c in calls if c.args[0].message["type"] == "turn.status"]


# ---- _parse_patch ------------------------------------------------------


def test_parse_patch_plain_json() -> None:
    assert _parse_patch('{"mood": "soft"}') == {"mood": "soft"}


def test_parse_patch_strips_code_fences() -> None:
    assert _parse_patch('```json\n{"mood": "soft"}\n```') == {"mood": "soft"}


def test_parse_patch_recovers_from_surrounding_prose() -> None:
    assert _parse_patch('Sure: {"mood": "soft"} done.') == {"mood": "soft"}


def test_parse_patch_returns_none_on_garbage() -> None:
    assert _parse_patch("totally not json") is None


def test_parse_patch_empty_object_is_valid() -> None:
    assert _parse_patch("{}") == {}


# ---- _clean_patch ------------------------------------------------------


def test_clean_patch_drops_unknown_keys() -> None:
    out = _clean_patch({"mood": "soft", "wat": "nope"}, kind="npc")
    assert out == {"mood": "soft"}


def test_clean_patch_validates_status_enum() -> None:
    out = _clean_patch({"status": "wounded"}, kind="npc")
    assert out == {"status": "wounded"}


def test_clean_patch_drops_bad_enum() -> None:
    out = _clean_patch({"status": "totally-fine"}, kind="npc")
    assert out == {}


def test_clean_patch_handles_inventory_list() -> None:
    out = _clean_patch({"inventory": ["knife", "rope"]}, kind="npc")
    assert out == {"inventory": ["knife", "rope"]}


def test_clean_patch_drops_inventory_if_not_list() -> None:
    out = _clean_patch({"inventory": "knife, rope"}, kind="npc")
    assert out == {}


def test_clean_patch_pc_subset_differs_from_npc() -> None:
    # NPC-only keys (mood/social_mask) get dropped on PC patches.
    out = _clean_patch({"mood": "soft", "hp_current": 8}, kind="pc")
    assert out == {"hp_current": 8}


# ---- _apply_patch ------------------------------------------------------


def test_apply_patch_appends_recent_bullet_with_turn_prefix() -> None:
    fm, body = frontmatter.split(_OLD_STATE)
    _, new_body, bullet = _apply_patch(fm, body, {"recent_bullet": "softened over food"}, turn=2)
    assert bullet == "softened over food"
    assert "- T2: softened over food" in new_body
    assert "- T1: arrived at shelter." in new_body  # prior bullet preserved


def test_apply_patch_rotates_window_at_five() -> None:
    # Pre-seed five bullets, add a sixth — the first should drop.
    body = (
        "## Recent (last 5 events)\n\n"
        "- T1: a\n- T2: b\n- T3: c\n- T4: d\n- T5: e\n\n"
        "## Session so far\n\nx\n"
    )
    _, new_body, _ = _apply_patch({}, body, {"recent_bullet": "f"}, turn=6)
    assert "T1: a" not in new_body
    assert "T6: f" in new_body
    assert "T2: b" in new_body


def test_apply_patch_replaces_session_so_far() -> None:
    fm, body = frontmatter.split(_OLD_STATE)
    _, new_body, _ = _apply_patch(fm, body, {"session_so_far": "completely new"}, turn=2)
    assert "completely new" in new_body
    assert "wary since arrival" not in new_body


def test_apply_patch_overwrites_scalar_fields() -> None:
    fm, body = frontmatter.split(_OLD_STATE)
    new_fm, _, _ = _apply_patch(fm, body, {"mood": "softening", "location": "indoors"}, turn=2)
    assert new_fm["mood"] == "softening"
    assert new_fm["location"] == "indoors"
    # Untouched scalars survive
    assert new_fm["scene_goal"] == "test trust"


# ---- rewrite_sheet end-to-end -----------------------------------------


@pytest.mark.asyncio
async def test_rewrite_writes_state_on_valid_patch() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning(json.dumps({"mood": "softening", "recent_bullet": "ate the food"}))

    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="here, eat",
        assistant_text="Lyra tries the omelette.",
        turn=2,
        timeout_s=10.0,
    )
    assert out.written is True
    assert out.skipped_reason is None
    assert len(vault.written_npc_state) == 1
    written_slug, written_content = vault.written_npc_state[0]
    assert written_slug == "lyra"
    assert "mood: softening" in written_content
    assert "T2: ate the food" in written_content


@pytest.mark.asyncio
async def test_rewrite_empty_patch_is_noop_skip() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning("{}")

    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=2,
        timeout_s=10.0,
    )
    assert out.written is False
    assert out.skipped_reason == "empty_patch"
    assert vault.written_npc_state == []


@pytest.mark.asyncio
async def test_rewrite_bad_json_skips_safely() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning("not even close to json")

    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=2,
        timeout_s=10.0,
    )
    assert out.written is False
    assert out.skipped_reason == "bad_json"
    assert vault.written_npc_state == []


@pytest.mark.asyncio
async def test_rewrite_seeds_state_when_missing() -> None:
    # No state file on disk — writer should still patch (first-turn entry).
    vault = _FakeVault(sheets={})
    client = _client_returning(json.dumps({"mood": "tense", "recent_bullet": "appeared"}))

    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="newbie",
        locale="en",
        user_text="who's that",
        assistant_text="A figure steps in.",
        turn=1,
        timeout_s=10.0,
    )
    assert out.written is True
    assert len(vault.written_npc_state) == 1
    _, content = vault.written_npc_state[0]
    assert "mood: tense" in content


@pytest.mark.asyncio
async def test_rewrite_handles_llm_error() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    create = AsyncMock(side_effect=RuntimeError("boom"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=1,
        timeout_s=10.0,
    )
    assert out.written is False
    assert out.skipped_reason is not None
    assert out.skipped_reason.startswith("llm_error:")
    assert vault.written_npc_state == []


@pytest.mark.asyncio
async def test_rewrite_salvages_valid_keys_from_malformed_json() -> None:
    # Observed live: an unquoted `session_so_far:` line cost the whole
    # patch, including a well-formed recent_bullet.
    raw = (
        '{\n  "recent_bullet": "Gave her name at the gate.",\n  "physical_state": "",\n'
        "  session_so_far: Rowan stands at the door as Lyra hesitates.\n}"
    )
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    out = await rewrite_sheet(
        client=_client_returning(raw),
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="who are you",
        assistant_text="Lyra.",
        turn=2,
        timeout_s=10.0,
    )
    assert out.written is True
    content = vault.sheets["npcs/lyra.state.md"]
    assert "T2: Gave her name at the gate." in content
    assert "hesitates" not in content  # the malformed key is not guessed at


@pytest.mark.asyncio
async def test_rewrite_logs_why_a_completion_was_empty() -> None:
    message = SimpleNamespace(content=None, reasoning_content="Let me think about Lyra...")
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="length")])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=completion)))
    )
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}")
    try:
        out = await rewrite_sheet(
            client=client,
            model="test",
            vault=_FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE}),
            kind="npc",
            slug="lyra",
            locale="en",
            user_text="x",
            assistant_text="y",
            turn=2,
            timeout_s=10.0,
        )
    finally:
        logger.remove(sink)
    assert out.skipped_reason == "bad_json"
    assert any(
        "finish_reason='length'" in line and "reasoning_content=present" in line for line in lines
    )


# ---- _auto_exit_diverged uses the state file --------------------------


def _npc_state(location: str, *, in_scene: str = "true") -> str:
    return (
        f"---\ntype: npc_state\nstatus: active\nlocation: {location}\n"
        f"in_scene: {in_scene}\nlast_updated: 2026-06-29\n---\n\n"
        f"## Recent (last 5 events)\n\n_(none)_\n\n## Session so far\n\n_(none)_\n"
    )


def _pc_state(location: str) -> str:
    return (
        f"---\ntype: character_state\nstatus: active\nlocation: {location}\n"
        f"last_updated: 2026-06-29\n---\n\n"
        f"## Recent (last 5 events)\n\n_(none)_\n\n## Session so far\n\n_(none)_\n"
    )


def _writer_with(vault: _FakeVault, registry: SceneRegistry) -> SceneSheetWriter:
    return SceneSheetWriter(
        context=SimpleNamespace(messages=[]),  # type: ignore[arg-type]
        registry=registry,
        vault=vault,  # type: ignore[arg-type]
        client=_client_returning(""),
        model="test",
        locale="en",
        timeout_s=1.0,
        emitter=SimpleNamespace(),  # type: ignore[arg-type]
    )


def test_auto_exit_drops_npc_when_location_diverges() -> None:
    vault = _FakeVault(
        sheets={
            "npcs/vera.state.md": _npc_state("perimeter ridge"),
            "characters/rowan.state.md": _pc_state("inside shelter"),
        }
    )
    registry = SceneRegistry(in_scene={"vera"}, current_pc_slug="rowan")
    _writer_with(vault, registry)._auto_exit_diverged("rowan", {("pc", "rowan")}, "")
    assert "vera" not in registry.in_scene
    # ``in_scene: false`` stamped on the state file so the next read is honest.
    assert vault.written_npc_state and "in_scene: false" in vault.written_npc_state[-1][1]


def test_auto_exit_keeps_npc_the_reply_names_despite_stale_pc_location() -> None:
    # 2026-09-25 marathon: Marta escorted the player across the river; her
    # sheet moved to 'river gate', the PC's stayed 'ridgeway fallout shelter'.
    vault = _FakeVault(
        sheets={
            "npcs/marta.state.md": _npc_state("river gate"),
            "characters/rowan.state.md": _pc_state("ridgeway fallout shelter"),
        }
    )
    registry = SceneRegistry(in_scene={"marta"}, current_pc_slug="rowan")
    _writer_with(vault, registry)._auto_exit_diverged(
        "rowan", {("npc", "marta")}, '"With me, Marta," you murmur, crossing the open ground.'
    )
    assert "marta" in registry.in_scene
    assert vault.written_npc_state == []


def test_auto_exit_keeps_npc_when_location_matches() -> None:
    vault = _FakeVault(
        sheets={
            "npcs/vera.state.md": _npc_state("inside shelter"),
            "characters/rowan.state.md": _pc_state("inside shelter"),
        }
    )
    registry = SceneRegistry(in_scene={"vera"}, current_pc_slug="rowan")
    _writer_with(vault, registry)._auto_exit_diverged("rowan", {("pc", "rowan")}, "")
    assert "vera" in registry.in_scene
    assert vault.written_npc_state == []


def test_auto_exit_noop_when_pc_state_missing() -> None:
    vault = _FakeVault(sheets={"npcs/vera.state.md": _npc_state("anywhere")})
    registry = SceneRegistry(in_scene={"vera"}, current_pc_slug="rowan")
    _writer_with(vault, registry)._auto_exit_diverged("rowan", {("pc", "rowan")}, "")
    assert "vera" in registry.in_scene


@pytest.mark.asyncio
async def test_auto_exit_keeps_npc_in_another_room_of_the_same_place() -> None:
    # Marathon logs: 'hall of ridgeway fallout shelter' vs 'infirmary of
    # ridgeway fallout shelter' evicted on exact string compare.
    vault = _FakeVault(
        sheets={
            "npcs/marta.state.md": _npc_state("infirmary of ridgeway fallout shelter"),
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry(in_scene={"marta"}, current_pc_slug="rowan")
    client = _client_seq("{}", json.dumps({"location": "hall of ridgeway fallout shelter"}))
    await _turn_writer(vault, reg, client, "You walk down into the hall.")._run()
    assert "marta" in reg.in_scene


@pytest.mark.asyncio
async def test_auto_exit_waits_until_someone_moves() -> None:
    # A stale mismatch nobody acted on this turn is not a departure.
    vault = _FakeVault(
        sheets={
            "npcs/marta.state.md": _npc_state("perimeter ridge"),
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry(in_scene={"marta"}, current_pc_slug="rowan")
    await _turn_writer(vault, reg, _client_seq(), "The wind picks up.")._run()
    assert "marta" in reg.in_scene


@pytest.mark.asyncio
async def test_auto_exit_evicts_npc_left_behind_when_pc_moves() -> None:
    vault = _FakeVault(
        sheets={
            "npcs/marta.state.md": _npc_state("gatehouse"),
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry(in_scene={"marta"}, current_pc_slug="rowan")
    client = _client_seq("{}", json.dumps({"location": "perimeter ridge"}))
    await _turn_writer(vault, reg, client, "You climb to the ridge alone.")._run()
    assert "marta" not in reg.in_scene
    assert "in_scene: false" in vault.sheets["npcs/marta.state.md"]


# ---- DM-introduced NPC backstop -----------------------------------------


def _writer(vault: _FakeVault, registry: SceneRegistry) -> SceneSheetWriter:
    return SceneSheetWriter(
        context=SimpleNamespace(messages=[]),
        registry=registry,
        vault=vault,
        client=_client_returning("{}"),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
    )


def test_detect_introduced_names_patterns() -> None:
    assert detect_introduced_names('"Name’s Lyra," they say, voice steady.') == ["Lyra"]  # noqa: RUF001 — the model emits typographic apostrophes
    assert detect_introduced_names("My name is Talwen.") == ["Talwen"]
    assert detect_introduced_names('"Call me Vex."') == ["Vex"]
    assert detect_introduced_names('"I’m Kael," he grunts.') == ["Kael"]  # noqa: RUF001 — the model emits typographic apostrophes
    assert detect_introduced_names("A nevem Vika.") == ["Vika"]


def test_detect_introduced_names_rejects_stopwords_and_dupes() -> None:
    assert detect_introduced_names("I'm Sorry. I'm Not doing that. I am Sure.") == []
    assert detect_introduced_names('"Name’s Lyra." "My name is Lyra."') == ["Lyra"]  # noqa: RUF001 — the model emits typographic apostrophes
    assert detect_introduced_names("i'm lowercase throughout") == []


def test_seed_introduced_npcs_creates_stub_and_enters() -> None:
    vault = _FakeVault(
        sheets={
            "characters/rowan.md": "---\ntype: character\nname: Rowan\n---\n\nbody\n",
            "characters/rowan.state.md": (
                "---\ntype: character_state\nstatus: active\nlocation: shelter entrance\n---\n"
            ),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    writer = _writer(vault, reg)

    writer._seed_introduced_npcs('"Name’s Lyra," they say, voice steady.')  # noqa: RUF001 — the model emits typographic apostrophes

    assert "lyra" in reg.in_scene
    id_fm, id_body = frontmatter.split(vault.sheets["npcs/lyra.md"])
    assert id_fm is not None
    assert id_fm["name"] == "Lyra"
    assert id_fm["type"] == "npc"
    assert int(id_fm["hexaco_openness"]) == 3
    assert "## Psychology (HEXACO)" in id_body
    st_fm, _ = frontmatter.split(vault.sheets["npcs/lyra.state.md"])
    assert st_fm is not None
    assert str(st_fm["in_scene"]).lower() == "true"
    # Co-located with the PC so _auto_exit_diverged doesn't evict her.
    assert st_fm["location"] == "shelter entrance"


def test_seed_skips_pc_echo_and_enters_existing_npc() -> None:
    vault = _FakeVault(
        sheets={
            "characters/rowan.md": "---\ntype: character\nname: Rowan\n---\n\nbody\n",
            "npcs/vika.md": "---\ntype: npc\nname: Vika\n---\n\noriginal body\n",
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    writer = _writer(vault, reg)

    writer._seed_introduced_npcs('"I am Rowan," you say. "Call me Vika," she answers.')

    # PC's own introduction never becomes an NPC.
    assert "rowan" not in reg.in_scene
    assert "npcs/rowan.md" not in vault.sheets
    # Existing NPC voiced without npc_scene_enter: entered, sheet untouched.
    assert "vika" in reg.in_scene
    assert "original body" in vault.sheets["npcs/vika.md"]


@pytest.mark.asyncio
async def test_introduced_existing_npc_is_co_located_and_stays() -> None:
    # Marathon logs: 4/4 NPCs entered via the introduction path were
    # evicted the same turn — never co-located with the PC.
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Vika", "perimeter ridge"),
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    await _turn_writer(vault, reg, _client_seq(), '"Name\'s Vika." The rifle lowers.')._run()
    assert "vika" in reg.in_scene
    assert "location: gatehouse" in vault.sheets["npcs/vika.state.md"]


def test_introduced_dead_npc_is_not_entered() -> None:
    vault = _FakeVault(sheets=_npc_fixture("Vas", "perimeter ridge", status="dead"))
    reg = SceneRegistry()
    _writer(vault, reg)._seed_introduced_npcs('"Name\'s Vas." A voice from the dark.')
    assert reg.in_scene == set()


def test_introduced_first_name_resolves_to_multiword_vault_npc() -> None:
    lyra = "---\ntype: npc\nname: Lyra Moonwhisper\n---\n\nbody\n"
    vault = _FakeVault(sheets={"npcs/lyra-moonwhisper.md": lyra})
    reg = SceneRegistry()
    _writer(vault, reg)._seed_introduced_npcs('"I\'m Lyra," she says.')
    assert "npcs/lyra.md" not in vault.sheets  # no duplicate stub in canon
    assert reg.in_scene == {"lyra-moonwhisper"}


def test_ambiguous_first_name_neither_enters_nor_seeds() -> None:
    vault = _FakeVault(
        sheets={
            "npcs/lyra-moonwhisper.md": "---\ntype: npc\nname: Lyra Moonwhisper\n---\n",
            "npcs/lyra-stone.md": "---\ntype: npc\nname: Lyra Stone\n---\n",
        }
    )
    reg = SceneRegistry()
    _writer(vault, reg)._seed_introduced_npcs('"I\'m Lyra," she says.')
    assert "npcs/lyra.md" not in vault.sheets
    assert reg.in_scene == set()


def test_pc_first_name_echo_never_seeds_an_npc() -> None:
    vault = _FakeVault(
        sheets={
            "characters/rowan-blackthorn.md": "---\ntype: character\nname: Rowan Blackthorn\n---\n"
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan-blackthorn")
    _writer(vault, reg)._seed_introduced_npcs('"I am Rowan," you say.')
    assert "npcs/rowan.md" not in vault.sheets
    assert reg.in_scene == set()


def test_detect_unnamed_referents_positives_and_of_construction() -> None:
    text = "The figure doesn’t move. You watch the stranger’s hands; the Creature hisses."  # noqa: RUF001 — the model emits typographic apostrophes
    assert detect_unnamed_referents(text) == ["The figure", "The stranger", "The creature"]
    assert detect_unnamed_referents("It takes the shape of the hill, a silhouette of pines.") == []
    # Indefinite, plural, and off-whitelist nouns never count.
    assert detect_unnamed_referents("A figure waits. The figures scatter. The guard nods.") == []


def _two_reply_writer(
    vault: _FakeVault, reg: SceneRegistry, prev_reply: str, reply: str
) -> SceneSheetWriter:
    writer = _turn_writer(vault, reg, _client_seq(), reply)
    writer._context.messages[:0] = [
        {"role": "user", "content": "I look."},
        {"role": "assistant", "content": prev_reply},
    ]
    return writer


def _pc_fixture() -> dict[str, str]:
    return {
        "characters/rowan.md": "---\ntype: character\nname: Rowan\n---\n\nbody\n",
        "characters/rowan.state.md": _pc_state("depot passage"),
    }


@pytest.mark.asyncio
async def test_recurring_unnamed_referent_is_seeded_and_entered_on_second_mention() -> None:
    # 2026-09-25 marathon replay-002: "the figure" recurred T5-T9 with no
    # sheet and was re-imagined from "too tall, too still" into a man in a coat.
    vault = _FakeVault(sheets=_pc_fixture())
    reg = SceneRegistry()
    reg.set_pc("rowan")
    await _turn_writer(vault, reg, _client_seq(), "The figure stands too still.")._run()
    assert "npcs/the-figure.md" not in vault.sheets  # one mention is not recurrence

    prev, reply = "The figure stands too still.", "The figure tilts its elongated head."
    await _two_reply_writer(vault, reg, prev, reply)._run()

    assert "the-figure" in reg.in_scene
    id_fm, _ = frontmatter.split(vault.sheets["npcs/the-figure.md"])
    assert id_fm is not None and id_fm["name"] == "The figure"
    assert _STUB_GLOSS_MARKER in vault.sheets["npcs/the-figure.md"]
    st_fm, _ = frontmatter.split(vault.sheets["npcs/the-figure.state.md"])
    assert st_fm is not None
    assert str(st_fm["in_scene"]).lower() == "true"
    assert st_fm["location"] == "depot passage"


@pytest.mark.asyncio
async def test_of_construction_in_both_replies_seeds_nothing() -> None:
    vault = _FakeVault(sheets=_pc_fixture())
    reg = SceneRegistry()
    reg.set_pc("rowan")
    prev, reply = "The shape of the hill looms.", "You climb the shape of the hill."
    await _two_reply_writer(vault, reg, prev, reply)._run()
    assert reg.in_scene == set()
    assert "npcs/the-shape.md" not in vault.sheets


def test_existing_unnamed_referent_sheet_is_entered_not_reseeded() -> None:
    vault = _FakeVault(
        sheets={
            "npcs/the-figure.md": "---\ntype: npc\nname: The figure\n---\n\ntoo tall\n",
            "npcs/the-figure.state.md": _npc_state("corridor", in_scene="false"),
        }
    )
    reg = SceneRegistry()
    _writer(vault, reg)._seed_introduced_npcs("The figure waits.", "The figure is there.")
    assert reg.in_scene == {"the-figure"}
    assert "too tall" in vault.sheets["npcs/the-figure.md"]


def test_unnamed_referent_matching_known_npc_never_seeds() -> None:
    vault = _FakeVault(sheets={"npcs/ferenc.md": "---\ntype: npc\nname: The Stranger\n---\n"})
    reg = SceneRegistry()
    _writer(vault, reg)._seed_introduced_npcs("The stranger nods.", "The stranger waits.")
    assert "npcs/the-stranger.md" not in vault.sheets
    assert reg.in_scene == {"ferenc"}


def test_unnamed_referent_counts_toward_stub_cap() -> None:
    vault = _FakeVault()
    reg = SceneRegistry()
    reply = '"Name\'s Vika." "Call me Kael." The figure is still there.'
    _writer(vault, reg)._seed_introduced_npcs(reply, "The figure watches.")
    assert reg.in_scene == {"vika", "kael"}
    assert "npcs/the-figure.md" not in vault.sheets


def test_auto_exit_gone_drops_dead_npc() -> None:
    dead_state = (
        "---\ntype: npc_state\nstatus: deceased\nlocation: shelter\nin_scene: true\n---\n\nbody\n"
    )
    vault = _FakeVault(sheets={"npcs/vika.state.md": dead_state})
    reg = SceneRegistry()
    reg.enter("vika")
    writer = _writer(vault, reg)

    writer._auto_exit_gone()

    assert "vika" not in reg.in_scene
    fm, _ = frontmatter.split(vault.sheets["npcs/vika.state.md"])
    assert fm is not None
    assert str(fm["in_scene"]).lower() == "false"


# ---- turn-director integration + deterministic hooks ---------------------


def test_apply_patch_dedupes_near_duplicate_bullet() -> None:
    body = (
        "## Recent (last 5 events)\n\n"
        "- T17: Kissed with deliberate intensity, triggering a moment of "
        "unguarded response and lingering uncertainty.\n\n"
        "## Session so far\n\nx\n"
    )
    _, new_body, bullet = _apply_patch(
        {},
        body,
        {
            "recent_bullet": (
                "Kissed with deliberate intensity, triggering a moment of "
                "unguarded response and lingering uncertainty"
            )
        },
        turn=18,
    )
    assert bullet is None
    assert "T18:" not in new_body
    assert new_body.count("Kissed with deliberate intensity") == 1


@pytest.mark.asyncio
async def test_rewrite_throttles_unchanged_session_so_far() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning(json.dumps({"session_so_far": "Lyra has been wary since arrival"}))
    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=2,
        timeout_s=10.0,
    )
    assert out.written is False
    assert out.skipped_reason == "empty_patch"
    assert vault.written_npc_state == []


def test_extract_recent_exchanges_pairs_and_skips_noise() -> None:
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "system", "content": "sheets"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
    ]
    pairs = _extract_recent_exchanges(messages, count=2)
    assert pairs == [("turn 2", "reply 2"), ("turn 1", "reply 1")]


def test_extract_recent_exchanges_unanswered_user_has_no_reply() -> None:
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
    ]
    pairs = _extract_recent_exchanges(messages, count=2)
    assert pairs == [("turn 2", ""), ("turn 1", "reply 1")]


def test_extract_recent_exchanges_drops_the_code_emitted_time_line() -> None:
    # The PC patch must not read the code's "*Day 2, morning.*" as narrated time.
    messages = [
        {"role": "user", "content": "I wait for dawn."},
        {"role": "assistant", "content": "*Day 2, morning.*\n\nThe gate creaks open."},
    ]
    assert _extract_recent_exchanges(messages, count=1) == [
        ("I wait for dawn.", "The gate creaks open.")
    ]


@pytest.mark.asyncio
async def test_turn_without_new_reply_patches_nothing_and_goes_idle() -> None:
    # The previous turn's reply must not be re-applied as this turn's.
    vault = _FakeVault(sheets={"characters/rowan.state.md": _pc_state("gatehouse")})
    reg = SceneRegistry()
    reg.set_pc("rowan")
    client = _client_seq(json.dumps({"recent_bullet": "stale replay"}))
    writer = _turn_writer(vault, reg, client, "reply 1", user="turn 1")
    writer._context.messages.append({"role": "user", "content": "turn 2"})
    await writer.process_frame(LLMContextAssistantTimestampFrame("t"), FrameDirection.DOWNSTREAM)
    client.chat.completions.create.assert_not_awaited()
    assert vault.written_pc_state == []
    assert _phases(writer) == ["idle"]


@pytest.mark.asyncio
async def test_cancelled_writer_still_emits_idle_and_reraises() -> None:
    # Barge-in cancels the process task: CancelledError bypasses
    # `except Exception`, and the client must still get turn.status: idle.
    vault = _FakeVault(sheets={"characters/rowan.state.md": _pc_state("gatehouse")})
    reg = SceneRegistry()
    reg.set_pc("rowan")
    create = AsyncMock(side_effect=asyncio.CancelledError)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    writer = _turn_writer(vault, reg, client, "You wait.")
    with pytest.raises(asyncio.CancelledError):
        await writer.process_frame(
            LLMContextAssistantTimestampFrame("t"), FrameDirection.DOWNSTREAM
        )
    assert _phases(writer) == ["writing_sheets", "idle"]


@pytest.mark.asyncio
async def test_run_feeds_director_repetition_signal() -> None:
    director = DirectorState()
    reply = "The same reply, word for word, twice in a row." * 3
    context = SimpleNamespace(
        messages=[
            {"role": "user", "content": "I wait"},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "I wait"},
            {"role": "assistant", "content": reply},
        ]
    )
    writer = SceneSheetWriter(
        context=context,
        registry=SceneRegistry(),
        vault=_FakeVault(),
        client=_client_returning("{}"),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
        director=director,
    )
    await writer._run()
    assert director.reply_similarity > 0.9
    assert director.user_repeated


_CONSOLIDATION_PAYLOAD = {
    "gloss": "A wary scavenger who trades trust in small, deliberate steps.",
    "speech_register": 'Low and clipped; tests people with half-questions ("You\'re sure?").',
    "hexaco_honesty_humility": 3,
    "hexaco_emotionality": 4,
    "hexaco_extraversion": 2,
    "hexaco_agreeableness": 2,
    "hexaco_conscientiousness": 4,
    "hexaco_openness": 3,
}


@pytest.mark.asyncio
async def test_consolidation_promotes_stub_identity() -> None:
    vault = _FakeVault(
        sheets={
            "npcs/lira.md": _stub_identity_sheet("Lira", "en", "shelter"),
            "npcs/lira.state.md": _OLD_STATE,
        }
    )
    reg = SceneRegistry()
    reg.enter("lira")
    writer = SceneSheetWriter(
        context=SimpleNamespace(messages=[]),
        registry=reg,
        vault=vault,
        client=_client_returning(json.dumps(_CONSOLIDATION_PAYLOAD)),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
    )
    writer._stub_turns["lira"] = 4  # this call is the 5th stub turn

    await writer._maybe_consolidate_stubs("player text", "dm text")

    identity = vault.sheets["npcs/lira.md"]
    assert _STUB_GLOSS_MARKER not in identity
    fm, body = frontmatter.split(identity)
    assert fm is not None
    assert fm["name"] == "Lira"  # canon fields preserved
    assert int(fm["hexaco_emotionality"]) == 4
    assert "A wary scavenger" in body
    assert "## Psychology (HEXACO)" in body
    assert "- emotionality (4)" in body


@pytest.mark.asyncio
async def test_consolidation_rejects_invalid_payload() -> None:
    bad = dict(_CONSOLIDATION_PAYLOAD, hexaco_openness=7)  # out of range
    vault = _FakeVault(
        sheets={
            "npcs/lira.md": _stub_identity_sheet("Lira", "en", "shelter"),
            "npcs/lira.state.md": _OLD_STATE,
        }
    )
    reg = SceneRegistry()
    reg.enter("lira")
    writer = SceneSheetWriter(
        context=SimpleNamespace(messages=[]),
        registry=reg,
        vault=vault,
        client=_client_returning(json.dumps(bad)),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
    )
    writer._stub_turns["lira"] = 4

    await writer._maybe_consolidate_stubs("player text", "dm text")

    assert _STUB_GLOSS_MARKER in vault.sheets["npcs/lira.md"]  # stub survives


def test_validate_consolidation_rules() -> None:
    assert _validate_consolidation(None) is None
    assert _validate_consolidation({}) is None
    assert _validate_consolidation(dict(_CONSOLIDATION_PAYLOAD, gloss="")) is None
    assert _validate_consolidation(dict(_CONSOLIDATION_PAYLOAD, hexaco_openness="high")) is None
    ok = _validate_consolidation(dict(_CONSOLIDATION_PAYLOAD))
    assert ok is not None
    gloss, _speech, hexaco = ok
    assert gloss.startswith("A wary scavenger")
    assert hexaco["hexaco_agreeableness"] == 2


# ---- time_advance clock hint (story engine input) ----------------------


@pytest.mark.asyncio
async def test_rewrite_extracts_time_advance_and_keeps_it_off_the_sheet() -> None:
    vault = _FakeVault()
    client = _client_returning(
        json.dumps({"recent_bullet": "camped for the night", "time_advance": "next_day"})
    )
    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="pc",
        slug="rowan",
        locale="en",
        user_text="we camp until morning",
        assistant_text="You camp. Dawn comes grey and cold.",
        turn=3,
        timeout_s=10.0,
    )
    assert out.time_advance == "next_day"
    assert out.written is True
    _, content = vault.written_pc_state[0]
    assert "time_advance" not in content


@pytest.mark.asyncio
async def test_rewrite_time_advance_only_patch_skips_write_but_keeps_hint() -> None:
    vault = _FakeVault()
    client = _client_returning(json.dumps({"time_advance": "next_segment"}))
    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="pc",
        slug="rowan",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=3,
        timeout_s=10.0,
    )
    assert out.written is False
    assert out.skipped_reason == "empty_patch"
    assert out.time_advance == "next_segment"


@pytest.mark.asyncio
async def test_rewrite_drops_invalid_time_advance_value() -> None:
    vault = _FakeVault()
    client = _client_returning(
        json.dumps({"recent_bullet": "moved on", "time_advance": "warp-speed"})
    )
    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="pc",
        slug="rowan",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=3,
        timeout_s=10.0,
    )
    assert out.time_advance is None
    _, content = vault.written_pc_state[0]
    assert "time_advance" not in content


@pytest.mark.asyncio
async def test_skipped_turn_clears_stale_event_directives() -> None:
    # Turn N fired an event; turn N+1 produces no assistant text and
    # early-returns before the engine hook. The director channel must
    # still be cleared, or N's "[World update]" re-renders in N+2.
    director = DirectorState()
    director.clock_line = "Day 1, evening."
    director.event_directives = ["Marta is missing."]
    writer = SceneSheetWriter(
        context=SimpleNamespace(messages=[]),  # no assistant reply this turn
        registry=SceneRegistry(),
        vault=_FakeVault(),
        client=_client_returning("{}"),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
        director=director,
    )
    await writer._run()
    assert director.event_directives == []
    # The time line is state, not a directive: the next prompt's [Time] keeps it.
    assert director.clock_line == "Day 1, evening."


# ---- voiced-NPC auto-enter (presence vs mention) ------------------------


def test_detect_voiced_names_presence_vs_mention() -> None:
    # Typographic apostrophes on purpose — the model emits them.
    text = (
        "Marta’s expression doesn’t flicker. Kerek nods once from the doorway. "  # noqa: RUF001
        "You should ask Dora about the vote when the hall empties. "
        "Tomi’s fever is worse than yesterday."  # noqa: RUF001
    )
    voiced = detect_voiced_names(text, ["Marta", "Kerek", "Dora", "Tomi", "Vas"])
    assert voiced == ["Marta", "Kerek"]  # possessive+body / gesture verb only


def test_detect_voiced_names_ignores_remote_observation() -> None:
    text = "Far on the western ridge, a glint of glass — Vas watching the wire again."
    assert detect_voiced_names(text, ["Vas"]) == []


def _npc_fixture(name: str, location: str, status: str = "active") -> dict[str, str]:
    slug = name.lower()
    return {
        f"npcs/{slug}.md": f"---\ntype: npc\nlang: en\nname: {name}\n---\n\n# {name}\n",
        f"npcs/{slug}.state.md": (
            f"---\ntype: npc_state\nstatus: {status}\nlocation: {location}\n"
            "in_scene: false\nlast_updated: 2026-07-11\n---\n\n"
            "## Recent (last 5 events)\n\n_(no events yet)_\n"
        ),
    }


def _writer_for_voiced(vault: _FakeVault, reg: SceneRegistry) -> SceneSheetWriter:
    return SceneSheetWriter(
        context=SimpleNamespace(messages=[]),
        registry=reg,
        vault=vault,
        client=_client_returning("{}"),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
    )


_MARTA_VOICED = 'Marta’s expression hardens. "Worse," she says flatly.'  # noqa: RUF001 — the model emits typographic apostrophes


@pytest.mark.asyncio
async def test_voiced_known_npc_is_entered_and_co_located() -> None:
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Marta", "perimeter ridge"),
            "characters/rowan.state.md": _pc_state("the gatehouse"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    await _turn_writer(vault, reg, _client_seq(), _MARTA_VOICED)._run()
    assert "marta" in reg.in_scene
    assert "location: the gatehouse" in vault.sheets["npcs/marta.state.md"]


@pytest.mark.asyncio
async def test_voiced_npc_co_located_against_pc_post_patch_location() -> None:
    # The PC walks into the infirmary where Marta already is: co-locating
    # with the PC's PRE-patch 'gatehouse' overwrote her correct location
    # and the auto-exit then evicted her.
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Marta", "infirmary of ridgeway fallout shelter"),
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    client = _client_seq("{}", json.dumps({"location": "ridgeway fallout shelter infirmary"}))
    await _turn_writer(vault, reg, client, _MARTA_VOICED)._run()
    assert "marta" in reg.in_scene
    assert "location: infirmary of ridgeway fallout shelter" in vault.sheets["npcs/marta.state.md"]


@pytest.mark.asyncio
async def test_auto_entered_npc_stamped_in_scene_when_patch_fails() -> None:
    # Resume rebuilds the registry from disk: a bad_json patch must not
    # leave the entered NPC at in_scene: false.
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Marta", "gatehouse"),
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    await _turn_writer(vault, reg, _client_seq("not json"), _MARTA_VOICED)._run()
    assert "in_scene: true" in vault.sheets["npcs/marta.state.md"]


def test_mentioned_npc_is_not_entered() -> None:
    vault = _FakeVault(sheets=_npc_fixture("Marta", "ridgeway fallout shelter"))
    reg = SceneRegistry()
    writer = _writer_for_voiced(vault, reg)
    writer._auto_enter_voiced_npcs("You should find Marta before the inspection tomorrow.")
    assert reg.in_scene == set()


_LYRA_M = "---\ntype: npc\nname: Lyra Moonwhisper\n---\n"
_LYRA_S = "---\ntype: npc\nname: Lyra Stone\n---\n"


def test_voiced_first_name_enters_multiword_vault_npc() -> None:
    vault = _FakeVault(sheets={"npcs/lyra-moonwhisper.md": _LYRA_M})
    reg = SceneRegistry()
    _writer_for_voiced(vault, reg)._auto_enter_voiced_npcs("Lyra nods toward the gate.")
    assert reg.in_scene == {"lyra-moonwhisper"}


def test_voiced_first_name_needs_presence_and_no_ambiguity() -> None:
    # A bare mention still doesn't count.
    vault = _FakeVault(sheets={"npcs/lyra-moonwhisper.md": _LYRA_M})
    reg = SceneRegistry()
    _writer_for_voiced(vault, reg)._auto_enter_voiced_npcs("Ask Lyra about it tomorrow.")
    assert reg.in_scene == set()
    # Two vault Lyras — even with one already in scene — block the first name.
    vault = _FakeVault(sheets={"npcs/lyra-moonwhisper.md": _LYRA_M, "npcs/lyra-stone.md": _LYRA_S})
    reg = SceneRegistry(in_scene={"lyra-stone"})
    _writer_for_voiced(vault, reg)._auto_enter_voiced_npcs("Lyra nods toward the gate.")
    assert reg.in_scene == {"lyra-stone"}


def test_dead_npc_is_never_pulled_into_scene() -> None:
    vault = _FakeVault(sheets=_npc_fixture("Vas", "perimeter ridge", status="dead"))
    reg = SceneRegistry()
    writer = _writer_for_voiced(vault, reg)
    writer._auto_enter_voiced_npcs("Vas steps out of the dark, smiling.")
    assert reg.in_scene == set()


# ---- off-scene vault NPCs in memory keys ---------------------------------


@pytest.mark.asyncio
async def test_offscene_vault_npc_name_reaches_pc_memory() -> None:
    # The retired leak guard dropped any field naming an off-scene vault
    # NPC unless the name was already in state — which it could never
    # reach. Marathon logs: PC recent_bullet/session_so_far lost every turn.
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Marta", "pharmacy"),
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    client = _client_seq(
        json.dumps(
            {
                "recent_bullet": "Agreed to wait at the gate for Marta",
                "session_so_far": "Rowan holds the gate until Marta returns.",
            }
        )
    )
    await _turn_writer(vault, reg, client, "You settle in by the gate.")._run()
    content = vault.sheets["characters/rowan.state.md"]
    assert "T1: Agreed to wait at the gate for Marta" in content
    assert "until Marta returns" in content


@pytest.mark.asyncio
async def test_rewrite_prompt_pins_the_subject_slug() -> None:
    # Cross-NPC patch scrambling (marathon replay-003: Lyra's summary
    # about Marta): the prompt must name WHO is being updated.
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning("{}")
    await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=2,
        timeout_s=10.0,
    )
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "ONE character: lyra" in prompt
    assert "about lyra and no one else" in prompt


@pytest.mark.asyncio
async def test_rewrite_goal_static_adds_scene_goal_check() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning("{}")
    await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=2,
        timeout_s=10.0,
        goal_static=True,
    )
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "`scene_goal` check" in prompt


@pytest.mark.asyncio
async def test_rewrite_reports_changed_keys() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning(json.dumps({"scene_goal": "flee north", "mood": "afraid"}))
    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="run",
        assistant_text="Lyra bolts for the treeline.",
        turn=2,
        timeout_s=10.0,
    )
    assert set(out.changed_keys) == {"scene_goal", "mood"}


@pytest.mark.asyncio
async def test_rewrite_echoed_unchanged_value_not_counted_as_change() -> None:
    # The model re-sends location unchanged despite the "omit unchanged"
    # rule — an echoed value must not read as movement.
    vault = _FakeVault(sheets={"npcs/lyra.state.md": _OLD_STATE})
    client = _client_returning(json.dumps({"location": "shelter", "mood": "afraid"}))
    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="npc",
        slug="lyra",
        locale="en",
        user_text="wait",
        assistant_text="Lyra stays put.",
        turn=2,
        timeout_s=10.0,
    )
    assert "location" not in out.changed_keys
    assert "mood" in out.changed_keys


# ---- addressed-NPC auto-enter (vocative) ---------------------------------


def test_detect_addressed_names_vocatives_in_quotes() -> None:
    names = ["Marta", "Kerek", "Dora"]
    assert detect_addressed_names('"With me, Marta — match my pace."', names) == ["Marta"]
    assert detect_addressed_names("“Marta, ridge to our right,” you say.", names) == ["Marta"]
    assert detect_addressed_names('"Hold the door, Kerek!" I shout.', names) == ["Kerek"]
    assert detect_addressed_names("“Dora — not now.”", names) == ["Dora"]


def test_detect_addressed_names_ignores_mentions() -> None:
    names = ["Marta"]
    assert detect_addressed_names('"I\'ll tell Marta."', names) == []
    assert detect_addressed_names('"We bunk in Marta\'s tent, then."', names) == []
    assert detect_addressed_names("Marta, wait. You turn away.", names) == []  # not quoted
    assert detect_addressed_names("“Anything moving?” you ask Marta.", names) == []


@pytest.mark.asyncio
async def test_npc_addressed_by_player_is_entered_and_co_located() -> None:
    # 2026-09-25 marathon: the player escorted Marta across the river,
    # talking to her every turn; the DM never voiced her, so she stayed
    # "at the shelter" on disk and an off-screen event fired against it.
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Marta", "ridgeway fallout shelter"),
            "characters/rowan.state.md": _pc_state("river crossing"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    user = '"With me, Marta — match my pace." I step onto the bridge.'
    await _turn_writer(vault, reg, _client_seq(), "The bridge groans underfoot.", user)._run()
    assert "marta" in reg.in_scene
    state = vault.sheets["npcs/marta.state.md"]
    assert "location: river crossing" in state
    assert "in_scene: true" in state


def test_npc_addressed_in_dm_reply_is_entered() -> None:
    vault = _FakeVault(sheets=_npc_fixture("Marta", "ridgeway fallout shelter"))
    reg = SceneRegistry()
    _writer_for_voiced(vault, reg)._auto_enter_voiced_npcs("“Marta, ridge to our right,” you say.")
    assert reg.in_scene == {"marta"}


@pytest.mark.parametrize("status", ["dead", "missing"])
def test_addressed_gone_npc_is_not_entered(status: str) -> None:
    vault = _FakeVault(sheets=_npc_fixture("Marta", "pharmacy district", status=status))
    reg = SceneRegistry()
    _writer_for_voiced(vault, reg)._auto_enter_voiced_npcs("", '"Marta, can you hear me?"')
    assert reg.in_scene == set()


def test_addressed_first_name_needs_no_ambiguity() -> None:
    vault = _FakeVault(sheets={"npcs/lyra-moonwhisper.md": _LYRA_M})
    reg = SceneRegistry()
    _writer_for_voiced(vault, reg)._auto_enter_voiced_npcs("", '"Lyra, hold the gate."')
    assert reg.in_scene == {"lyra-moonwhisper"}
    vault = _FakeVault(sheets={"npcs/lyra-moonwhisper.md": _LYRA_M, "npcs/lyra-stone.md": _LYRA_S})
    reg = SceneRegistry()
    _writer_for_voiced(vault, reg)._auto_enter_voiced_npcs("", '"Lyra, hold the gate."')
    assert reg.in_scene == set()


# ---- with_pc: semantic presence from the PC patch ---------------------------


def test_clean_patch_with_pc_keeps_only_a_list_of_strings() -> None:
    patch = {"with_pc": ["Marta", 3, " ", None, " Kerek "]}
    assert _clean_patch(patch, "pc") == {"with_pc": ["Marta", "Kerek"]}
    assert _clean_patch({"with_pc": "Marta"}, "pc") == {}
    assert _clean_patch({"with_pc": ["Marta"]}, "npc") == {}  # PC patch only


@pytest.mark.asyncio
async def test_rewrite_extracts_with_pc_and_keeps_it_off_the_sheet() -> None:
    vault = _FakeVault()
    client = _client_returning(json.dumps({"recent_bullet": "crossed", "with_pc": ["Marta"]}))
    out = await rewrite_sheet(
        client=client,
        model="test",
        vault=vault,
        kind="pc",
        slug="rowan",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=3,
        timeout_s=10.0,
    )
    assert out.with_pc == ("Marta",)
    _, content = vault.written_pc_state[0]
    assert "with_pc" not in content
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "with_pc" in prompt
    # A with_pc-only patch writes nothing but still reports presence.
    vault = _FakeVault()
    out = await rewrite_sheet(
        client=_client_returning(json.dumps({"with_pc": ["Marta"]})),
        model="test",
        vault=vault,
        kind="pc",
        slug="rowan",
        locale="en",
        user_text="x",
        assistant_text="y",
        turn=3,
        timeout_s=10.0,
    )
    assert out.skipped_reason == "empty_patch"
    assert out.with_pc == ("Marta",)


_ESCORT_USER = "I fall into step beside Marta."
_ESCORT_REPLY = "The bridge groans underfoot as the two of you cross."


@pytest.mark.asyncio
async def test_npc_with_pc_is_entered_settled_and_fed_to_the_story_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2026-09-25 marathon: six escort turns of narration-only presence; no
    # voiced line, no vocative, so Marta's sheet stayed at the shelter.
    engine_kwargs: dict[str, Any] = {}

    def fake_engine(_vault: Any, _reg: Any, **kwargs: Any) -> None:
        engine_kwargs.update(kwargs)

    monkeypatch.setattr("aidm.story_events.run_end_of_turn", fake_engine)
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Marta", "ridgeway fallout shelter"),
            "characters/rowan.state.md": _pc_state("ridgeway fallout shelter"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    client = _client_seq(json.dumps({"location": "river crossing", "with_pc": ["Marta"]}))
    await _turn_writer(vault, reg, client, _ESCORT_REPLY, _ESCORT_USER)._run()
    assert "marta" in reg.in_scene
    state = vault.sheets["npcs/marta.state.md"]
    assert "location: river crossing" in state
    assert "in_scene: true" in state
    assert engine_kwargs["with_pc"] == {"marta"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["dead", "missing"])
async def test_gone_npc_in_with_pc_is_not_entered(status: str) -> None:
    vault = _FakeVault(
        sheets={
            **_npc_fixture("Marta", "ridgeway fallout shelter", status=status),
            "characters/rowan.state.md": _pc_state("river crossing"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    client = _client_seq(json.dumps({"with_pc": ["Marta"]}))
    await _turn_writer(vault, reg, client, _ESCORT_REPLY, _ESCORT_USER)._run()
    assert reg.in_scene == set()
    assert "location: ridgeway fallout shelter" in vault.sheets["npcs/marta.state.md"]


@pytest.mark.asyncio
async def test_ambiguous_with_pc_name_is_ignored() -> None:
    vault = _FakeVault(
        sheets={
            "npcs/lyra-moonwhisper.md": _LYRA_M,
            "npcs/lyra-stone.md": _LYRA_S,
            "characters/rowan.state.md": _pc_state("gatehouse"),
        }
    )
    reg = SceneRegistry()
    reg.set_pc("rowan")
    client = _client_seq(json.dumps({"with_pc": ["Lyra"]}))
    await _turn_writer(vault, reg, client, "Rain on the roof.", "I wait.")._run()
    assert reg.in_scene == set()


# ---- condensed old replies -------------------------------------------------


@pytest.mark.asyncio
async def test_replies_past_the_verbatim_window_become_their_bullets() -> None:
    vault = _FakeVault(sheets={"characters/rowan.state.md": _pc_state("gatehouse")})
    reg = SceneRegistry(in_scene=set(), current_pc_slug="rowan")
    turn_patches = [
        json.dumps({"recent_bullet": "Rowan found the cut cable."}),
        "{}",  # no bullet: this reply has nothing to condense to
        json.dumps({"recent_bullet": "Rowan followed the tracks."}),
        json.dumps({"recent_bullet": "Rowan reached the ridge."}),
    ]
    messages: list[dict[str, Any]] = []
    writer = SceneSheetWriter(
        context=SimpleNamespace(messages=messages),
        registry=reg,
        vault=vault,
        client=_client_seq(*turn_patches),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
        verbatim_replies=2,
    )
    for n in range(1, 5):
        messages += [
            {"role": "user", "content": f"I act {n}."},
            {"role": "assistant", "content": f"Long narration of turn {n}."},
        ]
        await writer._run()

    assert [m["content"] for m in messages] == [
        "I act 1.",
        f"{CONDENSED_REPLY_TAG} Rowan found the cut cable.",
        "I act 2.",
        "Long narration of turn 2.",
        "I act 3.",
        "Long narration of turn 3.",
        "I act 4.",
        "Long narration of turn 4.",
    ]


@pytest.mark.asyncio
async def test_advance_judge_verdict_sets_and_clears_the_stalled_flag() -> None:
    vault = _FakeVault(sheets={"characters/rowan.state.md": _pc_state("gatehouse")})
    reg = SceneRegistry(in_scene=set(), current_pc_slug="rowan")
    director = DirectorState()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "I wait."},
        {"role": "assistant", "content": "She hesitates at the wire."},
        {"role": "user", "content": "Well?"},
        {"role": "assistant", "content": "She hesitates again at the wire."},
    ]
    writer = SceneSheetWriter(
        context=SimpleNamespace(messages=messages),
        registry=reg,
        vault=vault,
        client=_client_seq(),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
        director=director,
        advance_judge=True,
    )
    with patch("aidm.scene_sheet_writer.judge_repeats", AsyncMock(return_value=True)) as judge:
        await writer._run()
    assert director.reply_stalled is True
    assert judge.await_args.kwargs["previous"] == "She hesitates at the wire."
    with patch("aidm.scene_sheet_writer.judge_repeats", AsyncMock(return_value=None)):
        await writer._run()
    assert director.reply_stalled is False  # no verdict never nudges


@pytest.mark.asyncio
async def test_scene_end_verdict_notes_a_scene_change() -> None:
    vault = _FakeVault(sheets={"characters/rowan.state.md": _pc_state("gatehouse")})
    reg = SceneRegistry(in_scene=set(), current_pc_slug="rowan")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Let's go."},
        {"role": "assistant", "content": "You leave the gatehouse for the bridge."},
    ]
    writer = SceneSheetWriter(
        context=SimpleNamespace(messages=messages),
        registry=reg,
        vault=vault,
        client=_client_seq(),
        model="test",
        locale="en",
        timeout_s=10.0,
        emitter=SimpleNamespace(push_frame=AsyncMock()),
        scene_end_judge=True,
    )
    notes: list[int] = []
    with (
        patch("aidm.scene_sheet_writer.judge_scene_ends", AsyncMock(return_value=True)),
        patch.object(
            writer,
            "_run_story_engine",
            AsyncMock(
                side_effect=lambda *_, **__: notes.append(reg.consume_scene_changes()) or False
            ),
        ),
    ):
        await writer._run()
    assert notes == [1]
