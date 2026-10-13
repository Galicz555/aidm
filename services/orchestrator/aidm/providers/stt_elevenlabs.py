"""ElevenLabs Scribe STT — public-mode speech-to-text.

Scribe v2 outperforms Groq Whisper Large-v3 on Hungarian — verified on a
fantasy-RP test corpus (kultusz / kazamata / proper-noun handling). The
trade-off is latency: Scribe is a REST roundtrip (~400-700ms) versus
Groq Whisper's ~150ms on LPU hardware. For a turn-based DM loop with
2-3s player utterances, the latency cost is acceptable; the win is
fewer mis-transcribed in-world nouns the LLM then can't act on.

Groq Whisper stays available as ``GROQ_STT_MODEL`` for cost-sensitive
runs (Scribe burns ElevenLabs credits per minute of audio).
"""

from __future__ import annotations

import re

import aiohttp
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService

from aidm.config import Config
from aidm.session import SessionContext
from aidm.vault import Vault

# Scribe v2 accepts up to 1000 keyterms per request; we extract them
# entirely from the vault so the bias follows whatever genre the
# campaign is written in — fantasy, sci-fi, noir, contemporary — with
# no hardcoded vocabulary in this file.
_KEYTERM_LIMIT = 1000

# Scribe v2 rejects keyterms with more than 4 spaces (i.e. > 5 words).
# Long phrases like full sentence wiki-link aliases hit this; we drop
# them rather than truncate, since a partial phrase is a worse bias
# than no bias.
_KEYTERM_MAX_SPACES = 4

# Matches Obsidian-style ``[[target]]`` and ``[[target|alias]]`` links.
# The alias (group 2) is the human-spoken form when present; otherwise
# we fall back to the target slug (group 1).
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:\|([^\]]+))?\]\]")


def _slug_to_spoken(token: str) -> str:
    return token.replace("-", " ").replace("_", " ").strip()


def _vault_keyterms(vault: Vault, limit: int = _KEYTERM_LIMIT) -> list[str]:
    """Build Scribe keyterm list from vault content.

    Two genre-agnostic sources:

    * Markdown file basenames — every campaign entity that gets its own
      page contributes a term.
    * Wiki-link targets and aliases — explicit cross-references in the
      vault prose, with the human-readable alias preferred over the
      kebab-case slug.

    Insertion order is preserved (dict-as-ordered-set) so a swap to
    a different vault produces a stable list, useful for caching.
    """
    seen: dict[str, None] = {}

    def _push(token: str) -> bool:
        spoken = _slug_to_spoken(token)
        if spoken and spoken.count(" ") <= _KEYTERM_MAX_SPACES:
            seen.setdefault(spoken, None)
        return len(seen) < limit

    for md in vault.root.rglob("*.md"):
        if not _push(md.stem):
            return list(seen.keys())
        try:
            content = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _WIKI_LINK_RE.finditer(content):
            slug, alias = match.group(1), match.group(2)
            spoken_source = (alias or slug or "").strip()
            if not spoken_source:
                continue
            if not _push(spoken_source):
                return list(seen.keys())
    return list(seen.keys())


def build_elevenlabs_stt(
    config: Config,
    session: SessionContext,
    *,
    http_session: aiohttp.ClientSession,
    vault: Vault | None = None,
) -> ElevenLabsSTTService:
    keyterms = _vault_keyterms(vault) if vault is not None else []
    return ElevenLabsSTTService(
        api_key=config.elevenlabs_api_key,
        aiohttp_session=http_session,
        settings=ElevenLabsSTTService.Settings(
            model=config.elevenlabs_stt_model,
            language=session.language,
            keyterms=keyterms,
        ),
    )
