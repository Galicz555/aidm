"""SessionContext + locale → Language mapping."""

from __future__ import annotations

import pytest
from aidm.session import SessionContext, session_for_locale
from pipecat.transcriptions.language import Language


def test_hungarian_session_maps_to_language_hu() -> None:
    session = session_for_locale("hu")
    assert session.locale == "hu"
    assert session.language == Language.HU
    assert len(session.session_id) == 8


def test_english_session_maps_to_language_en() -> None:
    session = session_for_locale("en")
    assert session.locale == "en"
    assert session.language == Language.EN


def test_explicit_session_id_preserved() -> None:
    session = session_for_locale("hu", session_id="abc12345")
    assert session.session_id == "abc12345"


def test_unknown_locale_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported locale"):
        session_for_locale("de")


def test_session_context_is_frozen() -> None:
    session = session_for_locale("hu")
    with pytest.raises(AttributeError):
        session.locale = "en"  # type: ignore[misc]


def test_two_default_sessions_get_distinct_ids() -> None:
    a = session_for_locale("hu")
    b = session_for_locale("hu")
    assert a.session_id != b.session_id


def test_session_context_direct_construction() -> None:
    """The dataclass constructor stays public — webrtc signaling will use it."""
    session = SessionContext(locale="en", language=Language.EN, session_id="custom01")
    assert session.locale == "en"
    assert session.session_id == "custom01"
