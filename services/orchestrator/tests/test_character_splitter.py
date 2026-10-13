"""Tests for the CharacterStreamSplitter tag parser.

The splitter sits in the LLM → TTS leg, parses ``<npc id="slug">…</npc>``
tags out of streaming ``LLMTextFrame`` chunks, and emits two outputs
per segment: a ``transcript.segment`` event with speaker attribution
and a tag-stripped ``LLMTextFrame`` for downstream TTS.

We exercise the splitter directly via its ``process_frame`` entry
point with a stubbed ``push_frame`` capturing what flows downstream,
so the tests are independent of the broader Pipecat plumbing.
"""

from __future__ import annotations

import pytest
from aidm.character_splitter import CharacterStreamSplitter
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class _CapturingSplitter(CharacterStreamSplitter):
    """Splitter variant that records every pushed frame for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.pushed: list[tuple[Frame, FrameDirection]] = []

    async def push_frame(  # type: ignore[override]
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append((frame, direction))


async def _drive(splitter: _CapturingSplitter, *chunks: str) -> None:
    await splitter.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for chunk in chunks:
        await splitter.process_frame(LLMTextFrame(text=chunk), FrameDirection.DOWNSTREAM)
    await splitter.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)


def _text_segments(splitter: _CapturingSplitter) -> list[str]:
    return [f.text for f, _ in splitter.pushed if isinstance(f, LLMTextFrame)]


def _events(splitter: _CapturingSplitter) -> list[dict[str, object]]:
    return [f.message for f, _ in splitter.pushed if isinstance(f, OutputTransportMessageFrame)]


@pytest.mark.asyncio
async def test_pure_narrator_passes_through_as_dm() -> None:
    sp = _CapturingSplitter()
    await _drive(sp, "A kocsma elhalkul. ", "Egy alak közeledik.")
    segments = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    assert segments == [
        {"type": "transcript.segment", "speaker": "dm", "text": "A kocsma elhalkul. "},
        {"type": "transcript.segment", "speaker": "dm", "text": "Egy alak közeledik."},
    ]
    assert _text_segments(sp) == ["A kocsma elhalkul. ", "Egy alak közeledik."]


@pytest.mark.asyncio
async def test_single_npc_block_emits_with_slug_speaker() -> None:
    sp = _CapturingSplitter()
    await _drive(sp, '<npc id="nova-tessen">Késő van. Ülj le.</npc>')
    seg_events = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    assert seg_events == [
        {
            "type": "transcript.segment",
            "speaker": "npc:nova-tessen",
            "text": "Késő van. Ülj le.",
        }
    ]
    # XML tags are stripped from the audio-bound text.
    assert _text_segments(sp) == ["Késő van. Ülj le."]


@pytest.mark.asyncio
async def test_mixed_narrator_and_npc_emits_in_order() -> None:
    sp = _CapturingSplitter()
    await _drive(
        sp,
        'Belép a fogadóba. <npc id="nova-tessen">Késő van.</npc> Az asztalra mutat.',
    )
    seg_events = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    assert seg_events == [
        {"type": "transcript.segment", "speaker": "dm", "text": "Belép a fogadóba. "},
        {"type": "transcript.segment", "speaker": "npc:nova-tessen", "text": "Késő van."},
        {"type": "transcript.segment", "speaker": "dm", "text": " Az asztalra mutat."},
    ]


@pytest.mark.asyncio
async def test_tag_straddles_chunk_boundary() -> None:
    sp = _CapturingSplitter()
    # The open tag splits across three chunks; the splitter must hold
    # the trailing candidate fragment until the tag completes.
    await _drive(
        sp,
        "Csend. ",
        '<npc id="no',
        'va-tessen">Hagyjátok.',
        "</npc>",
        " A kés a kezében.",
    )
    seg_events = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    assert seg_events == [
        {"type": "transcript.segment", "speaker": "dm", "text": "Csend. "},
        {"type": "transcript.segment", "speaker": "npc:nova-tessen", "text": "Hagyjátok."},
        {"type": "transcript.segment", "speaker": "dm", "text": " A kés a kezében."},
    ]


@pytest.mark.asyncio
async def test_multiple_npcs_each_attributed() -> None:
    sp = _CapturingSplitter()
    await _drive(
        sp,
        '<npc id="elara">Nem szabad.</npc> ',
        '<npc id="nova-tessen">Akkor majd én.</npc>',
    )
    seg_events = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    speakers = [e["speaker"] for e in seg_events]
    assert "npc:elara" in speakers
    assert "npc:nova-tessen" in speakers
    assert speakers == ["npc:elara", "dm", "npc:nova-tessen"]


@pytest.mark.asyncio
async def test_emotion_tags_inside_npc_block_are_preserved() -> None:
    sp = _CapturingSplitter()
    await _drive(sp, '<npc id="nova-tessen">[whispers]Csendben.[/whispers]</npc>')
    seg_events = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    assert seg_events == [
        {
            "type": "transcript.segment",
            "speaker": "npc:nova-tessen",
            "text": "[whispers]Csendben.[/whispers]",
        }
    ]


@pytest.mark.asyncio
async def test_unclosed_npc_block_flushes_on_response_end() -> None:
    # The DM produced a broken response. The splitter should still
    # emit what it has so the player hears *something*.
    sp = _CapturingSplitter()
    await _drive(sp, '<npc id="nova-tessen">Hagyjátok')
    seg_events = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    assert seg_events == [
        {"type": "transcript.segment", "speaker": "npc:nova-tessen", "text": "Hagyjátok"}
    ]


@pytest.mark.asyncio
async def test_bot_end_event_is_emitted_on_response_end() -> None:
    sp = _CapturingSplitter()
    await _drive(sp, "Belépsz a terembe.")
    events = _events(sp)
    assert {"type": "transcript.bot_end"} in events


@pytest.mark.asyncio
async def test_state_resets_between_responses() -> None:
    sp = _CapturingSplitter()
    await _drive(sp, '<npc id="nova-tessen">Egy.</npc>')
    sp.pushed.clear()
    # Fresh response — narrator default applies again.
    await _drive(sp, "Másnap reggel.")
    seg_events = [e for e in _events(sp) if e.get("type") == "transcript.segment"]
    assert seg_events == [{"type": "transcript.segment", "speaker": "dm", "text": "Másnap reggel."}]
