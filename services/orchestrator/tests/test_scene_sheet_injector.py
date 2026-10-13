"""Tests for SceneSheetInjector — read-fresh per turn, no cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from aidm.scene_sheet_injector import (
    SCENE_SHEETS_TAG,
    SceneSheetInjector,
    _splice,
    _splice_turn_contract,
)
from aidm.scene_state import SceneRegistry
from aidm.turn_director import TURN_CONTRACT_TAG, DirectorState
from aidm.vault import VaultPathError
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection


@dataclass
class _FakeVault:
    sheets: dict[str, str] = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)

    def read_file(self, path: str, *, max_bytes: int = 200_000) -> str:
        self.reads.append(path)
        if path not in self.sheets:
            raise VaultPathError(f"not a file: {path}")
        return self.sheets[path]


_LYRA = "---\ntype: npc\nname: Lyra\nmood: wary\n---\n\nLyra body."
_MARTA = "---\ntype: npc\nname: Marta\nmood: brisk\n---\n\nMarta body."
_ROWAN = "---\ntype: character\nname: Rowan\n---\n\nRowan body."


def _ctx(messages: list[dict[str, Any]]) -> Any:
    class _Ctx:
        pass

    c = _Ctx()
    c.messages = messages
    return c


def test_splice_inserts_for_active_npc() -> None:
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}]
    _splice(_ctx(messages), [("npc", "lyra", _LYRA)], "en")
    assert messages[1]["content"].startswith(SCENE_SHEETS_TAG)
    assert "Lyra body." in messages[1]["content"]
    assert "### npc: lyra" in messages[1]["content"]


def test_splice_renders_world_events_section_before_sheets() -> None:
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}]
    _splice(
        _ctx(messages),
        [("npc", "lyra", _LYRA)],
        "en",
        directives=["Marta staggers back to the gate, bleeding."],
    )
    content = messages[1]["content"]
    assert "World events" in content
    assert "- Marta staggers back to the gate, bleeding." in content
    # Directives lead the block, sheets follow.
    assert content.index("World events") < content.index("### npc: lyra")


def test_splice_renders_directives_even_with_no_sheets() -> None:
    messages = [{"role": "user", "content": "hi"}]
    _splice(_ctx(messages), [], "en", directives=["The purifier died."])
    assert any(m["role"] == "system" and "The purifier died." in m["content"] for m in messages)


def test_splice_replaces_existing_block() -> None:
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": f"{SCENE_SHEETS_TAG}\nold content"},
        {"role": "user", "content": "hi"},
    ]
    _splice(_ctx(messages), [("npc", "lyra", _LYRA)], "en")
    assert len([m for m in messages if m["content"].startswith(SCENE_SHEETS_TAG)]) == 1
    assert "Lyra body." in messages[1]["content"]


def test_splice_positions_block_before_latest_user_message() -> None:
    """The block goes at the bottom so the history prefix stays KV-cacheable."""
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
    ]
    _splice(_ctx(messages), [("npc", "lyra", _LYRA)], "en")
    assert [m["content"] for m in messages[:3]] == ["persona", "turn 1", "reply 1"]
    assert messages[3]["content"].startswith(SCENE_SHEETS_TAG)
    assert messages[4]["content"] == "turn 2"


def test_splice_moves_stale_block_to_bottom() -> None:
    """A block left mid-history by the previous turn is popped, not refreshed in place."""
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": f"{SCENE_SHEETS_TAG}\nold"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
    ]
    _splice(_ctx(messages), [("npc", "lyra", _LYRA)], "en")
    blocks = [i for i, m in enumerate(messages) if m["content"].startswith(SCENE_SHEETS_TAG)]
    assert blocks == [3]
    assert "Lyra body." in messages[3]["content"]
    assert messages[4]["content"] == "turn 2"


def test_splice_drops_block_when_no_sheets() -> None:
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": f"{SCENE_SHEETS_TAG}\nstale"},
        {"role": "user", "content": "hi"},
    ]
    _splice(_ctx(messages), [], "en")
    assert all(not m["content"].startswith(SCENE_SHEETS_TAG) for m in messages)
    assert len(messages) == 2


def test_pc_sheet_included_alongside_npcs() -> None:
    messages = [{"role": "system", "content": "persona"}]
    _splice(_ctx(messages), [("npc", "lyra", _LYRA), ("pc", "rowan", _ROWAN)], "en")
    payload = messages[1]["content"]
    assert "### npc: lyra" in payload
    assert "### pc: rowan" in payload


@pytest.mark.asyncio
async def test_injector_reads_fresh_every_turn_no_cache() -> None:
    """Sheet writer rewrites on disk; injector must NOT cache stale content."""
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA})
    reg = SceneRegistry()
    reg.enter("lyra")
    inj = SceneSheetInjector(reg, "en", vault)
    ctx = _ctx([{"role": "system", "content": "persona"}])

    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)

    # Three turns, four reads per turn (identity + state + the campaign's
    # feature gate for the First Nightmare block + live beat), no caching.
    assert vault.reads == ["npcs/lyra.md", "npcs/lyra.state.md", "campaign.md", "beat.md"] * 3


@pytest.mark.asyncio
async def test_missing_sheet_silently_skipped() -> None:
    vault = _FakeVault(sheets={})
    reg = SceneRegistry()
    reg.enter("nobody")
    inj = SceneSheetInjector(reg, "en", vault)
    ctx = _ctx([{"role": "system", "content": "persona"}])

    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    assert all(not m["content"].startswith(SCENE_SHEETS_TAG) for m in ctx.messages)


@pytest.mark.asyncio
async def test_block_dropped_when_scene_empties() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA})
    reg = SceneRegistry()
    reg.enter("lyra")
    inj = SceneSheetInjector(reg, "en", vault)
    ctx = _ctx([{"role": "system", "content": "persona"}])

    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    assert any(m["content"].startswith(SCENE_SHEETS_TAG) for m in ctx.messages)

    reg.exit("lyra")
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    assert all(not m["content"].startswith(SCENE_SHEETS_TAG) for m in ctx.messages)


@pytest.mark.asyncio
async def test_turn_contract_spliced_at_prompt_tail() -> None:
    vault = _FakeVault(sheets={})
    reg = SceneRegistry()
    inj = SceneSheetInjector(reg, "en", vault, director=DirectorState())
    ctx = _ctx(
        [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hi"},
        ]
    )
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    assert ctx.messages[-1]["content"].startswith(TURN_CONTRACT_TAG)

    # Next turn: the stale copy is popped, exactly one rides at the tail.
    ctx.messages.append({"role": "assistant", "content": "reply"})
    ctx.messages.append({"role": "user", "content": "next"})
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    contracts = [m for m in ctx.messages if m["content"].startswith(TURN_CONTRACT_TAG)]
    assert len(contracts) == 1
    assert ctx.messages[-1]["content"].startswith(TURN_CONTRACT_TAG)


def test_turn_contract_removed_when_director_disabled() -> None:
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": f"{TURN_CONTRACT_TAG}\nstale"},
        {"role": "user", "content": "hi"},
    ]
    _splice_turn_contract(_ctx(messages), None, "en")
    assert all(not m["content"].startswith(TURN_CONTRACT_TAG) for m in messages)


def test_backlink_footer_stripped() -> None:
    raw = _LYRA + "\n\n# Referenced by\n- characters/rowan.md\n"
    vault = _FakeVault(sheets={"npcs/lyra.md": raw})
    reg = SceneRegistry()
    reg.enter("lyra")
    inj = SceneSheetInjector(reg, "en", vault)
    sheets = inj._collect_sheets()
    assert "# Referenced by" not in sheets[0][2]


# ---- live beat (fresh from disk every turn) -----------------------------


def test_beat_rides_the_sheets_block_fresh_each_read() -> None:
    beat = "---\ntype: beat\n---\n\nThe raid is happening at the south wire."
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA, "beat.md": beat})
    reg = SceneRegistry()
    reg.enter("lyra")
    inj = SceneSheetInjector(reg, "en", vault)
    assert "raid is happening" in inj._read_beat()
    assert "type: beat" not in inj._read_beat()  # frontmatter stripped
    # The story engine rewrites beat.md mid-session → next read sees it.
    vault.sheets["beat.md"] = "---\ntype: beat\n---\n\nThe wire held; dawn comes."
    assert "dawn comes" in inj._read_beat()


def test_splice_includes_beat_even_without_sheets() -> None:
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hello"},
    ]
    _splice(_ctx(messages), [], "en", beat="Marta is missing since dawn.")
    block = next(m for m in messages if m["content"].startswith(SCENE_SHEETS_TAG))
    assert "Current beat" in block["content"]
    assert "Marta is missing since dawn." in block["content"]


def test_missing_beat_changes_nothing() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA})
    reg = SceneRegistry()
    reg.enter("lyra")
    inj = SceneSheetInjector(reg, "en", vault)
    assert inj._read_beat() == ""


def test_turn_contract_lead_placement_lands_before_last_user_message() -> None:
    # Public/API-DM placement: the contract goes BEFORE the latest user
    # message (next to the scene sheets), never at the tail — API models
    # parrot a trailing system message as their reply.
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest action"},
    ]
    _splice_turn_contract(_ctx(messages), DirectorState(), "en", tail=False)
    contracts = [i for i, m in enumerate(messages) if m["content"].startswith(TURN_CONTRACT_TAG)]
    assert len(contracts) == 1
    assert messages[contracts[0] + 1] == {"role": "user", "content": "latest action"}
    assert "message below" in messages[contracts[0]]["content"]


def test_turn_contract_tail_placement_says_above() -> None:
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "action"},
    ]
    _splice_turn_contract(_ctx(messages), DirectorState(), "en", tail=True)
    assert messages[-1]["content"].startswith(TURN_CONTRACT_TAG)
    assert "message above" in messages[-1]["content"]


# ---- per-NPC minds (AIDM_NPC_MINDS_ENABLED) -----------------------------

from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402


def _minds_client(reply_for: Any) -> Any:
    """Fake AsyncOpenAI; `reply_for` maps prompt→content or is a str/callable."""

    async def _create(**kwargs: Any) -> Any:
        prompt = kwargs["messages"][0]["content"]
        content = reply_for(prompt) if callable(reply_for) else reply_for
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    create = AsyncMock(side_effect=_create)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _minds_injector(vault: _FakeVault, registry: SceneRegistry, client: Any) -> SceneSheetInjector:
    inj = SceneSheetInjector(registry, "en", vault, minds_client=client, minds_model="m")
    inj.push_frame = AsyncMock()  # type: ignore[method-assign]
    return inj


@pytest.mark.asyncio
async def test_minds_intents_injected_inside_single_sheets_message() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA, "npcs/marta.md": _MARTA})
    reg = SceneRegistry()
    reg.enter("lyra")
    reg.enter("marta")
    inj = _minds_injector(
        vault, reg, _minds_client("want: hold the gate\nhiding: nothing\nread: wary")
    )
    ctx = _ctx([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    tagged = [m for m in ctx.messages if m["content"].startswith(SCENE_SHEETS_TAG)]
    assert len(tagged) == 1
    assert "[NPC interiors" in tagged[0]["content"]
    assert "- lyra:\n  want: hold the gate" in tagged[0]["content"]
    assert "- marta:\n  want: hold the gate" in tagged[0]["content"]


@pytest.mark.asyncio
async def test_minds_fail_open_when_one_call_raises() -> None:
    def _reply(prompt: str) -> str:
        if "lyra" in prompt.split("\n", 1)[0]:
            raise RuntimeError("boom")
        return "want: log the departure\nhiding: nothing\nread: steady"

    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA, "npcs/marta.md": _MARTA})
    reg = SceneRegistry()
    reg.enter("lyra")
    reg.enter("marta")
    inj = _minds_injector(vault, reg, _minds_client(_reply))
    ctx = _ctx([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    tagged = next(m for m in ctx.messages if m["content"].startswith(SCENE_SHEETS_TAG))
    assert "- marta:\n  want: log the departure" in tagged["content"]
    assert "- lyra:" not in tagged["content"]


@pytest.mark.asyncio
async def test_minds_caps_at_three_npcs() -> None:
    sheets = {f"npcs/n{i}.md": f"---\nname: N{i}\n---\n\nBody {i}." for i in range(4)}
    vault = _FakeVault(sheets=sheets)
    reg = SceneRegistry()
    for i in range(4):
        reg.enter(f"n{i}")
    client = _minds_client("intent")
    inj = _minds_injector(vault, reg, client)
    ctx = _ctx([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    assert client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_minds_knowledge_boundary_one_npc_per_prompt() -> None:
    seen: list[str] = []

    def _reply(prompt: str) -> str:
        seen.append(prompt)
        return "intent"

    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA, "npcs/marta.md": _MARTA})
    reg = SceneRegistry()
    reg.enter("lyra")
    reg.enter("marta")
    inj = _minds_injector(vault, reg, _minds_client(_reply))
    ctx = _ctx([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    lyra_prompt = next(p for p in seen if "Lyra body." in p)
    assert "Marta body." not in lyra_prompt


@pytest.mark.asyncio
async def test_minds_off_when_no_client() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA})
    reg = SceneRegistry()
    reg.enter("lyra")
    inj = SceneSheetInjector(reg, "en", vault)  # no minds client
    inj.push_frame = AsyncMock()  # type: ignore[method-assign]
    ctx = _ctx([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    tagged = next(m for m in ctx.messages if m["content"].startswith(SCENE_SHEETS_TAG))
    assert "[NPC interiors" not in tagged["content"]


def test_splice_renders_time_line_first_even_without_sheets() -> None:
    # The turn contract is the channel the 14B ignores for world info;
    # the in-fiction time rides the sheets block it obeys.
    messages = [{"role": "user", "content": "hi"}]
    _splice(_ctx(messages), [], "en", time_line="Day 3, morning.", directives=["ev"])
    content = messages[0]["content"]
    assert content.startswith(SCENE_SHEETS_TAG)
    assert "[Time] Day 3, morning." in content
    assert content.index("[Time]") < content.index("World events")


@pytest.mark.asyncio
async def test_injector_renders_director_clock_line_in_sheets_block() -> None:
    director = DirectorState()
    director.clock_line = "Day 2, evening."
    inj = SceneSheetInjector(SceneRegistry(), "en", _FakeVault(sheets={}), director=director)
    ctx = _ctx([{"role": "user", "content": "hi"}])
    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    sheets = next(m for m in ctx.messages if m["content"].startswith(SCENE_SHEETS_TAG))
    assert "[Time] Day 2, evening." in sheets["content"]
    contract = next(m for m in ctx.messages if m["content"].startswith(TURN_CONTRACT_TAG))
    assert "Day 2, evening." not in contract["content"]


@pytest.mark.asyncio
async def test_open_threads_render_in_sheets_block_after_time_before_sheets() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA})
    reg = SceneRegistry()
    reg.enter("lyra")
    threads = [f"cause {n} → consequence {n}" for n in range(1, 8)]
    director = DirectorState(clock_line="Day 2, dusk.", open_threads=threads)
    inj = SceneSheetInjector(reg, "en", vault, director=director)
    ctx = _ctx([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])

    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    block = next(m["content"] for m in ctx.messages if m["content"].startswith(SCENE_SHEETS_TAG))
    assert "[Open threads" in block
    assert block.index("[Time]") < block.index("[Open threads") < block.index("### npc: lyra")
    assert "- cause 5 → consequence 5" in block
    assert "cause 6" not in block  # capped at 5


@pytest.mark.asyncio
async def test_no_open_threads_section_when_list_is_empty() -> None:
    vault = _FakeVault(sheets={"npcs/lyra.md": _LYRA})
    reg = SceneRegistry()
    reg.enter("lyra")
    inj = SceneSheetInjector(reg, "en", vault, director=DirectorState())
    ctx = _ctx([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])

    await inj.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
    assert all("Open threads" not in m["content"] for m in ctx.messages)


def test_open_threads_alone_still_render_the_block() -> None:
    messages = [{"role": "user", "content": "hi"}]
    _splice(_ctx(messages), [], "en", open_threads=["the watcher on the ridge → unexplained"])
    assert any("- the watcher on the ridge → unexplained" in m["content"] for m in messages)
