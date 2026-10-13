"""DM persona prompts. Loaded from `prompts/dm.<locale>.md` at runtime so
the prompt can be iterated without code edits or restarts to the build.

An optional `AIDM_PROMPT_FILE` env var overrides the locale-based lookup,
which is useful for A/B-testing alternate persona drafts.

When the active TTS supports ElevenLabs v3 inline emotion tags, the
persona gets a short trailer instructing it to emit them. Emitting tags
when the TTS doesn't understand them produces literal "open bracket
whispers close bracket" speech, hence the gate.

Mode-aware loading: in private mode we prefer `dm.<locale>.private.md`
when present and fall back to the public variant. The private prompt
is tuned for the local model (shorter turns, stronger explicit
tool-use directive).

There used to be a per-session "persona variant" system (gritty / noir
/ romance / brisk). It's gone — the single persona prompt is now
dynamically tuned by the campaign-level ``tone.md`` and the run-level
``beat.md``, baked into ``vault_summary`` by ``vault_tools``. One voice,
many registers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_EMOTION_TAG_TRAILER: dict[str, str] = {
    "hu": (
        "\n\n## Hangulati jelölők\n"
        "A hangszintetizátor érti az inline jelölőket: `[whispers]`, "
        "`[excited]`, `[sad]`, `[laughs]`, `[angry]`. Akkor használd "
        "őket, amikor a karakter érzelmi állapota megváltozik vagy egy "
        "konkrét mondatban más hangszín illik. Sose magyarázd el a "
        "jelölőt — csak írd be a mondat elé."
    ),
    "en": (
        "\n\n## Emotional cues\n"
        "The TTS understands inline tags: `[whispers]`, `[excited]`, "
        "`[sad]`, `[laughs]`, `[angry]`. Use them when the character's "
        "emotional state shifts or a specific line wants a different "
        "tone. Never explain the tag — just place it before the line."
    ),
}


def persona_for(  # noqa: PLR0913 — keyword-only context fields, not positional bloat
    locale: str,
    *,
    emotion_tags: bool = False,
    mode: Literal["public", "private"] = "public",
    campaign_title: str = "",
    run_title: str = "",
    phase: Literal["play", "session-zero"] = "play",
    vault_summary: str = "",
) -> str:
    """Resolve the persona prompt string for this locale/mode.

    ``run_title`` injects a single-line continuity hint at the bottom of
    the persona so the LLM treats the active playthrough as its frame.
    Empty string disables the injection — used by tests, the local CLI
    pre-migration, and the canon-only discovery paths.

    ``phase`` selects the session-zero interviewer prompt when no
    character sheet exists yet for the active run, so the model knows
    it should elicit a sheet before opening any scene.

    ``vault_summary`` is appended verbatim after the persona body —
    callers use it to bake the vault map, authoritative canon files
    (campaign.md / run.md), and the scene-tuning block
    (tone.md / beat.md) into ``system_instruction`` rather than into a
    separate system message in context.
    """
    override = os.environ.get("AIDM_PROMPT_FILE")
    if override:
        path = Path(override).expanduser()
    elif phase == "session-zero":
        path = _resolve_session_zero_path(locale)
    else:
        path = _resolve_prompt_path(locale, mode)
    if not path.is_file():
        raise ValueError(
            f"No persona prompt at {path} (locale={locale!r}, mode={mode!r}, phase={phase!r})"
        )
    text = path.read_text(encoding="utf-8").strip()
    if emotion_tags and phase == "play":
        text += _EMOTION_TAG_TRAILER.get(locale, _EMOTION_TAG_TRAILER["en"])
    if campaign_title or run_title:
        text += _context_line(locale, campaign_title, run_title)
    if vault_summary:
        text += "\n\n" + vault_summary.strip() + "\n"
    return text


def _resolve_session_zero_path(locale: str) -> Path:
    """Pick the session-zero persona for this locale, with English fallback."""
    candidate = _PROMPTS_DIR / f"dm.{locale}.session-zero.md"
    if candidate.is_file():
        return candidate
    return _PROMPTS_DIR / "dm.en.session-zero.md"


def _context_line(locale: str, campaign_title: str, run_title: str) -> str:
    """Format the one-line campaign + run continuity hint."""
    parts: list[str] = []
    if locale == "hu":
        if campaign_title:
            parts.append(f"Aktív kampány: {campaign_title}")
        if run_title:
            parts.append(f"Run: {run_title}")
        return "\n\n" + ". ".join(parts) + "." if parts else ""
    if campaign_title:
        parts.append(f"Active campaign: {campaign_title}")
    if run_title:
        parts.append(f"Run: {run_title}")
    return "\n\n" + ". ".join(parts) + "." if parts else ""


def _resolve_prompt_path(locale: str, mode: str) -> Path:
    """Pick the prompt file for this locale.

    Resolution order:

    1. ``dm.<locale>.private.md``  (private mode)
    2. ``dm.<locale>.md``          (public / fallback)

    Private mode falls through to the public prompt if the private
    variant is missing — keeps the orchestrator bootable on a partial
    install.
    """
    candidates: list[Path] = []
    if mode == "private":
        candidates.append(_PROMPTS_DIR / f"dm.{locale}.private.md")
    candidates.append(_PROMPTS_DIR / f"dm.{locale}.md")
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]  # surface the default path even if missing
