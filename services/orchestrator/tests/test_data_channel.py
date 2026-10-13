"""DataChannelForwarder payload mapping.

We test the frame-to-payload mapping in isolation rather than driving a
full Pipecat pipeline. The frame-flow machinery (push_frame, queues,
observer hooks) is upstream's responsibility — what matters here is
that each frame type the forwarder claims to recognize produces the
expected JSON envelope.
"""

from __future__ import annotations

from aidm.data_channel import DataChannelForwarder
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMTextFrame,
    TranscriptionFrame,
)


def _forwarder() -> DataChannelForwarder:
    return DataChannelForwarder(name="forwarder.test")


def test_user_transcription_maps_to_user_payload() -> None:
    f = _forwarder()
    frame = TranscriptionFrame(text="Hello there", user_id="player", timestamp="")
    payload = f._payload_for(frame)
    assert payload == {"type": "transcript.user", "text": "Hello there"}


def test_user_transcription_strips_whitespace() -> None:
    f = _forwarder()
    frame = TranscriptionFrame(text="  hi  \n", user_id="player", timestamp="")
    payload = f._payload_for(frame)
    assert payload == {"type": "transcript.user", "text": "hi"}


def test_blank_user_transcription_is_dropped() -> None:
    f = _forwarder()
    frame = TranscriptionFrame(text="   ", user_id="player", timestamp="")
    assert f._payload_for(frame) is None


def test_llm_text_chunk_is_ignored() -> None:
    # Assistant-side events are emitted by CharacterStreamSplitter now;
    # the forwarder only handles the user leg.
    f = _forwarder()
    assert f._payload_for(LLMTextFrame(text="A flickering torch ")) is None


def test_unrelated_frame_returns_none() -> None:
    f = _forwarder()
    assert f._payload_for(BotStartedSpeakingFrame()) is None
