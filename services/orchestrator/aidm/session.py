"""Per-session runtime context.

``Config`` carries deploy-time knobs (mode, API keys, default locale,
voice-ID map). ``SessionContext`` carries data that varies per session:
the language the current player wants to play in, and a short stable id
for log correlation.

Today the orchestrator runs one session per process (local CLI), and the
context is built once at startup from ``Config.locale``. In Phase 6 the
browser client opens a new session per WebRTC peer-connection, and the
context is built from the locale picked in the start-session form. The
factories that depend on locale (persona, TTS voice) take a
``SessionContext`` so the call sites stay identical across transports.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from pipecat.transcriptions.language import Language


def _new_session_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True, slots=True)
class SessionContext:
    locale: str
    language: Language
    session_id: str = field(default_factory=_new_session_id)
    # Active campaign slug — the vault directory under VAULT_ROOT this
    # session is playing in. The web picker writes it on every offer;
    # the local CLI fills it from VAULT_PATH.name.
    campaign_slug: str = ""
    # Human-readable campaign title (from campaign.md frontmatter).
    # Used by the persona injection so the DM knows which world it's in.
    campaign_title: str = ""
    # Active run slug. Empty string is allowed only for unit tests and
    # the canon-only utility paths; live sessions always carry one.
    run_slug: str = ""
    # Human-readable run title (from runs/<slug>/run.md frontmatter).
    # Used by the persona injection to give the LLM continuity. Empty
    # when the run has no title or when no run is active.
    run_title: str = ""


_LOCALE_TO_LANGUAGE: dict[str, Language] = {
    "hu": Language.HU,
    "en": Language.EN,
}


def session_for_locale(  # noqa: PLR0913 — keyword-only context fields, not positional bloat
    locale: str,
    *,
    session_id: str | None = None,
    campaign_slug: str = "",
    campaign_title: str = "",
    run_slug: str = "",
    run_title: str = "",
) -> SessionContext:
    """Build a session context for ``locale`` ('hu', 'en'). Unknown locales raise."""
    if locale not in _LOCALE_TO_LANGUAGE:
        raise ValueError(
            f"Unsupported locale: {locale!r} (expected one of {sorted(_LOCALE_TO_LANGUAGE)})"
        )
    return SessionContext(
        locale=locale,
        language=_LOCALE_TO_LANGUAGE[locale],
        session_id=session_id or _new_session_id(),
        campaign_slug=campaign_slug.strip(),
        campaign_title=campaign_title.strip(),
        run_slug=run_slug.strip(),
        run_title=run_title.strip(),
    )
