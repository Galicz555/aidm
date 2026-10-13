"""Tests for the minimal SceneRegistry."""

from __future__ import annotations

import pytest
from aidm.scene_state import SceneRegistry, place_matches, same_place


# Pairs taken from real marathon auto-exits (docs/bench/*/marathon-50t).
@pytest.mark.parametrize(
    ("npc", "pc"),
    [
        ("hall of ridgeway fallout shelter", "infirmary of ridgeway fallout shelter"),
        ("the gate, her side of the wire", "the gate"),
        ("ridgeway fallout shelter, stores", "ridgeway fallout shelter, workshop alcove"),
        ("main hall of the shelter", "main hall"),
        ("corridor leading to depot", "depot entrance"),
    ],
)
def test_same_place_accepts_rephrasings_of_one_location(npc: str, pc: str) -> None:
    assert same_place(npc, pc)


@pytest.mark.parametrize(
    ("npc", "pc"),
    [
        ("approaching the shelter", "pharmacy district"),
        ("shelter entrance", "south slope"),
        ("the gate", "east yard service exit"),
        ("", "ridgeway fallout shelter"),
    ],
)
def test_same_place_rejects_distinct_locations(npc: str, pc: str) -> None:
    assert not same_place(npc, pc)


def test_place_matches_requires_every_authored_word() -> None:
    assert place_matches("ridgeway fallout shelter", "hall of ridgeway fallout shelter")
    assert not place_matches("ridgeway fallout shelter", "shelter gate")
    assert not place_matches("", "anywhere")


def test_enter_adds_slug_and_active_lists_sorted() -> None:
    reg = SceneRegistry()
    reg.enter("lyra")
    reg.enter("anna")
    reg.enter("zorg")
    assert reg.active() == ["anna", "lyra", "zorg"]


def test_exit_removes_slug() -> None:
    reg = SceneRegistry()
    reg.enter("lyra")
    reg.enter("anna")
    reg.exit("lyra")
    assert reg.active() == ["anna"]


def test_exit_missing_slug_is_noop() -> None:
    reg = SceneRegistry()
    reg.exit("never-entered")
    assert reg.active() == []


def test_enter_idempotent() -> None:
    reg = SceneRegistry()
    reg.enter("lyra")
    reg.enter("lyra")
    assert reg.active() == ["lyra"]


def test_enter_strips_and_rejects_empty() -> None:
    reg = SceneRegistry()
    reg.enter("  lyra  ")
    reg.enter("")
    reg.enter("   ")
    assert reg.active() == ["lyra"]


def test_set_pc_strips_and_clears() -> None:
    reg = SceneRegistry()
    reg.set_pc("  rowan  ")
    assert reg.current_pc_slug == "rowan"
    reg.set_pc(None)
    assert reg.current_pc_slug is None
