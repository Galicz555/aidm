"""Runtime configuration loaded from environment variables.

Keys live in `.env` at repo root (see `.env.example`). We deliberately read
plain `os.environ` rather than reaching for pydantic-settings — the surface
is small and the dependency is not worth it yet.

Mode is a startup-time constant. `AIDM_MODE=public` wires cloud providers
(Groq/ElevenLabs Scribe STT, Claude via OpenRouter, ElevenLabs TTS);
`AIDM_MODE=private` wires local ones (Faster-Whisper STT, llama.cpp +
Gemma 4 26B-A4B-it LLM, MOSS-TTS-Nano). Switching modes requires a restart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Mode = Literal["public", "private"]
Transport = Literal["local", "webrtc", "text"]


@dataclass(frozen=True, slots=True)
class Config:
    mode: Mode
    transport: Transport
    locale: str
    audio_input_device: int | None
    audio_output_device: int | None
    # Vault layout: ``vault_root`` is the directory holding all campaigns
    # (one subdirectory per campaign). ``vault_path`` is the *default*
    # campaign — the one selected at startup for the local CLI, and the
    # initial pick in the web picker when localStorage is empty. Switch
    # between campaigns is a per-session runtime decision now (no
    # restart required), so neither of these is process-binding.
    vault_root: Path
    vault_path: Path

    # Public-mode providers
    groq_api_key: str
    groq_stt_model: str
    openrouter_api_key: str
    openrouter_model: str
    elevenlabs_api_key: str
    # Default voice — used when no per-locale override is set. Phase 5
    # callers passed this directly; Phase 6 routes through ``voice_id_for``
    # so locale-specific voices (``_hu``/``_en``) win when configured.
    elevenlabs_voice_id: str
    elevenlabs_voice_id_hu: str
    elevenlabs_voice_id_en: str
    elevenlabs_model: str
    elevenlabs_stt_model: str
    elevenlabs_tts_transport: Literal["ws", "http"]
    stt_provider: Literal["groq", "elevenlabs"]

    # Private-mode endpoints. Defaults match the compose service names
    # on the ``private-net`` internal network; override only when running
    # the local services outside compose (e.g. ad-hoc bench against a
    # port-forwarded llama-server on the host).
    local_llm_base_url: str
    local_llm_model: str
    local_stt_base_url: str
    local_tts_ws_url: str
    local_tts_speaker_wav: str

    # Production deploy knobs. All optional with sensible defaults so test
    # fixtures and local dev don't need to enumerate every field. See
    # .env.example for the full prod recipe.
    allowed_origins: tuple[str, ...] = ()
    public_ip: str = ""
    max_concurrent_sessions: int = 3
    session_token_cap: int = 20_000
    session_tts_char_cap: int = 5_000
    daily_usd_cap: float = 0.0
    budget_db_path: Path = Path("data/budget.db")
    rate_limit_offer: str = "5/minute"
    rate_limit_session: str = "10/minute"
    cf_access_team_domain: str = ""
    cf_access_aud: str = ""
    cf_access_disabled: bool = True
    # TURN relay. STUN alone fails behind symmetric NAT / strict firewalls,
    # so prod deploys should set AIDM_TURN_URL (e.g. turn:turn.example:3478)
    # with credentials. Empty url = STUN-only (fine for LAN/dev).
    turn_url: str = ""
    turn_username: str = ""
    turn_credential: str = ""
    # Default run slug for sessions that don't pick one via requestData.
    # Used by the local CLI (no picker) and as a fallback when the web
    # client omits the field. None = require an explicit choice — the
    # WebRTC offer handler returns 400 instead of silently using a default.
    run_slug: str | None = None

    # --- NPC inline-state-directive failsafe ----------------------------
    # End-of-turn recovery for missed ``[state: <slug> mood=...; goal=...;
    # mask=...]`` directives. When the just-finished assistant text has
    # zero ``[state:`` tokens AND ≥1 active NPC has gone stale, a single
    # extra LLM call asks for ONLY bracket lines and applies them. See
    # ``aidm/npc_state_inline_failsafe.py``. The legacy tool-channel
    # failsafe was retired 2026-06-27 after bench-regress confirmed the
    # ``npc_update_state`` tool channel hit-rate was 0-10% on baseline.
    npc_failsafe_enabled: bool = True
    # 60s default: Qwen3 14B Q5 with 16k-token context typically takes
    # 10-30s per completion. The original 5s wire-up timed out on every
    # retry, neutering the failsafe entirely.
    npc_failsafe_timeout_s: float = 60.0
    # Agentic recon: when the player's message names an entity absent
    # from the fresh scene sheets, force the first completion to be a
    # vault tool call (tool_choice=required; the local provider resets
    # to auto after one dispatch). Code decides WHEN, the model decides
    # WHAT to fetch. Costs one extra completion on affected turns —
    # experiment flag, default off. The persona's tool instructions
    # alone measure 0 tool calls across every 30t/50t replay to date.
    agentic_recon: bool = False
    # Draft→critic pass: on turns with 2+ NPC sheets in scene (the
    # compounding-complexity case the failure profile names — invented
    # relationships, borrowed traits), one extra local completion
    # compares the clean draft against the sheets actually in context
    # and lists contradictions/inventions. A violation triggers one
    # regenerate with those violations named. Only runs when the other
    # guards (tic/para/directive) already found nothing — bounded to at
    # most one extra completion pair per turn. Experiment flag, default
    # off pending a 10t screen.
    critic_pass_enabled: bool = False
    # Directive logit bias: a soft per-turn nudge (not a hard grammar
    # constraint — that's what caused the agentic-recon runaway) toward
    # a live directive's proper-noun anchors (e.g. "Marta"), via
    # llama.cpp's logit_bias on the first leading-space token of each
    # name. Targets directive compliance (~78%) at the sampling layer
    # instead of only catching a miss after the fact. Experiment flag,
    # default off pending a screen.
    directive_logit_bias_enabled: bool = False
    directive_logit_bias_strength: float = 6.0
    # Per-NPC minds: before the DM completion, one small out-of-band
    # completion per in-scene NPC (cap 3), context = ONLY that NPC's own
    # sheet + the last exchange, producing a 1-2 sentence intent the DM
    # weaves. A structural knowledge boundary (the NPC can't see what it
    # didn't witness) and voice anchor. Costs N serial completions ahead
    # of every DM turn — experiment flag, default off (voice-mode latency).
    npc_minds_enabled: bool = False

    # --- Private-mode sampling ------------------------------------------
    # Defaults tuned for the Qwen3 14B instruct swap: low temp
    # + a min_p tail-cut to suppress confabulation (the model inventing
    # non-canon NPCs/places/lore). min_p is llama.cpp-only and has no
    # OpenAI-native field, so llm_local.py routes it through extra_body.
    # For a roleplay finetune raise llm_temperature to ~0.6-0.7 —
    # those mode-collapse at low temp.
    llm_temperature: float = 0.35
    llm_top_p: float = 0.95
    llm_min_p: float = 0.1
    llm_frequency_penalty: float = 0.3
    llm_presence_penalty: float = 0.2

    # --- Per-model harness (services/llm-gateway/profiles/*.env) ----------
    # Defaults = the Qwen3-14B-tuned harness. ``dm_thinking`` flips
    # enable_thinking on the DM request ONLY — side calls (sheet writer,
    # summary, minds, critic) never send it and stay on the gateway's
    # non-thinking default (their small max_tokens came back empty when
    # thinking leaked, 2026-07-09). Each guard_* switches off exactly one
    # pre-release guard in llm_local; ``profile`` is a label for eval output.
    dm_thinking: bool = False
    guard_tics: bool = True
    guard_paragraph: bool = True
    guard_replay: bool = True
    guard_directives: bool = True
    # Rewrite banned phrases in place by backtracking (llm_local) before
    # the excision fallback (2026-09-26 bundle F: excisions 12-20 -> 4-5).
    guard_backtrack: bool = True
    # Cut a reply opening that restates the player's own turn (llm_local).
    guard_echo: bool = True
    # Append this turn's fired world events to the player's message for the
    # DM request only (llm_local); the scene-sheets copy stays.
    events_in_user: bool = True
    # Content-repetition verdict per reply (advance_judge; private mode):
    # nudges the next turn, and after a stalled turn checks the draft too.
    advance_judge: bool = True
    # Judge every draft before release (not only after a stalled turn) and
    # regenerate up to N times while it re-tells the previous beat.
    advance_judge_every_draft: bool = True
    advance_judge_retries: int = 2
    # Scene-end verdict per turn (advance_judge.judge_scene_ends; private
    # mode): a move or time skip ticks the story clock like vault_set_scene.
    scene_end_judge: bool = True
    # Text transport only: how long a session's pipeline stays live with no
    # SSE stream attached. A phone lock outlasts WebRTC's 5 s every time
    # (2026-09-27 playtest lost its session); an idle text pipeline is cheap.
    text_reconnect_grace_s: float = 1800.0
    profile: str = ""
    # DM replies kept verbatim in context; older ones are condensed to
    # their turn's sheet-writer bullets (scene_sheet_writer). 0 = off.
    dm_verbatim_replies: int = 0
    # Prompt-size guard (private mode, HistoryTrimmer): the gateway window
    # (same AIDM_LLM_CTX_SIZE the gateway profile sets) and the target the
    # trimmer holds the whole request under — persona + tools + messages.
    llm_ctx_size: int = 16384
    prompt_token_budget: int = 24000

    def voice_id_for(self, locale: str) -> str:
        """Pick the ElevenLabs voice ID for ``locale``, falling back to the default.

        Same-voice-multiple-languages is fine: ElevenLabs Multilingual v2
        keeps voice identity across languages, so leaving the per-locale
        env vars unset is a valid setup and ``elevenlabs_voice_id`` carries
        every locale. The per-locale slots exist for the deliberate
        choice of *different* voices per language.
        """
        if locale == "hu" and self.elevenlabs_voice_id_hu:
            return self.elevenlabs_voice_id_hu
        if locale == "en" and self.elevenlabs_voice_id_en:
            return self.elevenlabs_voice_id_en
        return self.elevenlabs_voice_id

    @property
    def tts_supports_emotion_tags(self) -> bool:
        """Whether the active TTS understands inline [whispers]/[excited] tags.

        Persona prompts read this to decide whether to emit tags at all —
        emitting them when the TTS doesn't parse them produces literal
        spoken brackets, which sounds awful. Text transport has no TTS
        at all, so the tags would render as literal brackets in the
        transcript panel — also off.
        """
        if self.transport == "text":
            return False
        if self.mode != "public":
            return False
        return self.elevenlabs_model.startswith("eleven_v3")


# Supported locales. The mapping to pipecat ``Language`` lives in
# ``aidm.session``; here we only validate the env-supplied value.
_SUPPORTED_LOCALES: frozenset[str] = frozenset({"hu", "en"})


def load_config() -> Config:
    _load_env_file()

    mode = os.environ.get("AIDM_MODE", "public")
    if mode not in ("public", "private"):
        raise ValueError(f"Unsupported AIDM_MODE: {mode!r} (expected 'public' or 'private')")

    locale = os.environ.get("AIDM_LOCALE", "en")
    if locale not in _SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported AIDM_LOCALE: {locale!r}")

    transport = os.environ.get("AIDM_TRANSPORT", "local")
    if transport not in ("local", "webrtc", "text"):
        raise ValueError(
            f"Unsupported AIDM_TRANSPORT: {transport!r} (expected 'local', 'webrtc' or 'text')"
        )

    return Config(
        mode=mode,  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
        locale=locale,
        audio_input_device=_int_or_none(os.environ.get("AUDIO_INPUT_DEVICE")),
        audio_output_device=_int_or_none(os.environ.get("AUDIO_OUTPUT_DEVICE")),
        vault_root=_resolve_vault_root(
            os.environ.get("AIDM_VAULT_ROOT"),
            os.environ.get("VAULT_PATH"),
        ),
        vault_path=_resolve_vault_path(os.environ.get("VAULT_PATH")),
        groq_api_key=_require_if_public(mode, "GROQ_API_KEY"),
        # whisper-large-v3-turbo is ~5x faster than whisper-large-v3 on Groq,
        # near-identical quality. whisper-large-v3 stays available as fallback
        # for languages where Turbo's quality drops (none we've hit in Hungarian).
        groq_stt_model=os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        openrouter_api_key=_require_if_public(mode, "OPENROUTER_API_KEY"),
        # OpenRouter model slugs follow ``<vendor>/<model>``. claude-sonnet-4.5
        # is the stable default; try anthropic/claude-sonnet-4.6 if your
        # account has access. See https://openrouter.ai/models for the catalog.
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
        elevenlabs_api_key=_require_if_public(mode, "ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.environ.get(
            "ELEVENLABS_VOICE_ID",
            "pNInz6obpgDQGcFmaJgB",
        ),
        # Per-locale overrides — empty string means "fall back to the default".
        # Multilingual v2 keeps voice identity across languages, so leaving
        # these unset is the common case; set them when you want a different
        # voice per language.
        elevenlabs_voice_id_hu=os.environ.get("ELEVENLABS_VOICE_ID_HU", ""),
        elevenlabs_voice_id_en=os.environ.get("ELEVENLABS_VOICE_ID_EN", ""),
        # eleven_multilingual_v2 is the quality default for Hungarian. v3 alpha
        # adds inline emotion tags ([whispers], [excited]) when the account
        # tier permits — flip ELEVENLABS_MODEL=eleven_v3 in .env to try it.
        elevenlabs_model=os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
        # Scribe v2 is the strongest cloud STT on Hungarian we've measured;
        # Groq Whisper Large-v3 stays the fallback (cheaper, faster TTFB).
        elevenlabs_stt_model=os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2"),
        # ws = multi-stream-input WebSocket (low latency, stable on v2/flash/turbo
        # but rejects eleven_v3 with 403). http = chunked-streaming REST endpoint
        # (~500ms extra TTFB, but unlocks eleven_v3's emotional range, multi-speaker
        # dialogue and inline tags).
        elevenlabs_tts_transport=_require_tts_transport(),
        stt_provider=_require_stt_provider(),
        # Private-mode service URLs. Defaults assume the compose
        # service names on the internal ``private-net`` network so the
        # zero-config path works for ``make up-private``; override when
        # running individual services on the host for development.
        local_llm_base_url=os.environ.get("AIDM_LOCAL_LLM_URL", "http://llm-gateway:8002/v1"),
        local_llm_model=os.environ.get("AIDM_LOCAL_LLM_MODEL", "qwen3-14b"),
        local_stt_base_url=os.environ.get("AIDM_LOCAL_STT_URL", "http://stt:8001"),
        local_tts_ws_url=os.environ.get("AIDM_LOCAL_TTS_URL", "ws://tts:8003/v1/tts/stream"),
        local_tts_speaker_wav=os.environ.get("AIDM_LOCAL_TTS_SPEAKER", "/voices/dm_hu.wav"),
        allowed_origins=_parse_origins(os.environ.get("AIDM_ALLOWED_ORIGINS", "")),
        public_ip=os.environ.get("AIDM_PUBLIC_IP", "").strip(),
        max_concurrent_sessions=int(os.environ.get("AIDM_MAX_CONCURRENT_SESSIONS", "3")),
        session_token_cap=int(os.environ.get("AIDM_SESSION_TOKEN_CAP", "20000")),
        session_tts_char_cap=int(os.environ.get("AIDM_SESSION_TTS_CHAR_CAP", "5000")),
        daily_usd_cap=float(os.environ.get("AIDM_DAILY_USD_CAP", "0.30")),
        budget_db_path=Path(os.environ.get("AIDM_BUDGET_DB_PATH", "")).expanduser()
        if os.environ.get("AIDM_BUDGET_DB_PATH")
        else _default_budget_db_path(),
        rate_limit_offer=os.environ.get("AIDM_RATE_LIMIT_OFFER", "5/minute"),
        rate_limit_session=os.environ.get("AIDM_RATE_LIMIT_SESSION", "10/minute"),
        cf_access_team_domain=os.environ.get("CF_ACCESS_TEAM_DOMAIN", "").strip(),
        cf_access_aud=os.environ.get("CF_ACCESS_AUD", "").strip(),
        cf_access_disabled=os.environ.get("CF_ACCESS_DISABLED", "").strip()
        not in ("", "0", "false", "False"),
        turn_url=os.environ.get("AIDM_TURN_URL", "").strip(),
        turn_username=os.environ.get("AIDM_TURN_USERNAME", "").strip(),
        turn_credential=os.environ.get("AIDM_TURN_CREDENTIAL", "").strip(),
        run_slug=(os.environ.get("RUN_SLUG", "").strip() or None),
        npc_failsafe_enabled=_parse_bool(os.environ.get("AIDM_NPC_FAILSAFE_ENABLED"), default=True),
        npc_failsafe_timeout_s=float(os.environ.get("AIDM_NPC_FAILSAFE_TIMEOUT_S", "60.0")),
        agentic_recon=_parse_bool(os.environ.get("AIDM_AGENTIC_RECON"), default=False),
        critic_pass_enabled=_parse_bool(os.environ.get("AIDM_CRITIC_PASS_ENABLED"), default=False),
        directive_logit_bias_enabled=_parse_bool(
            os.environ.get("AIDM_DIRECTIVE_LOGIT_BIAS_ENABLED"), default=False
        ),
        directive_logit_bias_strength=float(
            os.environ.get("AIDM_DIRECTIVE_LOGIT_BIAS_STRENGTH", "6.0")
        ),
        npc_minds_enabled=_parse_bool(os.environ.get("AIDM_NPC_MINDS_ENABLED"), default=False),
        llm_temperature=float(os.environ.get("AIDM_LLM_TEMPERATURE", "0.35")),
        llm_top_p=float(os.environ.get("AIDM_LLM_TOP_P", "0.95")),
        llm_min_p=float(os.environ.get("AIDM_LLM_MIN_P", "0.1")),
        llm_frequency_penalty=float(os.environ.get("AIDM_LLM_FREQUENCY_PENALTY", "0.3")),
        llm_presence_penalty=float(os.environ.get("AIDM_LLM_PRESENCE_PENALTY", "0.2")),
        dm_thinking=_parse_bool(os.environ.get("AIDM_DM_THINKING"), default=False),
        guard_tics=_parse_bool(os.environ.get("AIDM_GUARD_TICS"), default=True),
        guard_paragraph=_parse_bool(os.environ.get("AIDM_GUARD_PARAGRAPH"), default=True),
        guard_replay=_parse_bool(os.environ.get("AIDM_GUARD_REPLAY"), default=True),
        guard_directives=_parse_bool(os.environ.get("AIDM_GUARD_DIRECTIVES"), default=True),
        guard_backtrack=_parse_bool(os.environ.get("AIDM_GUARD_BACKTRACK"), default=True),
        guard_echo=_parse_bool(os.environ.get("AIDM_GUARD_ECHO"), default=True),
        events_in_user=_parse_bool(os.environ.get("AIDM_EVENTS_IN_USER"), default=True),
        advance_judge=_parse_bool(os.environ.get("AIDM_ADVANCE_JUDGE"), default=True),
        advance_judge_every_draft=_parse_bool(
            os.environ.get("AIDM_ADVANCE_JUDGE_EVERY_DRAFT"), default=True
        ),
        advance_judge_retries=int(os.environ.get("AIDM_ADVANCE_JUDGE_RETRIES", "2")),
        scene_end_judge=_parse_bool(os.environ.get("AIDM_SCENE_END_JUDGE"), default=True),
        text_reconnect_grace_s=float(os.environ.get("AIDM_TEXT_RECONNECT_GRACE_S", "1800")),
        profile=os.environ.get("AIDM_PROFILE", "").strip(),
        dm_verbatim_replies=int(os.environ.get("AIDM_DM_VERBATIM_REPLIES", "0")),
        llm_ctx_size=int(os.environ.get("AIDM_LLM_CTX_SIZE", "16384")),
        prompt_token_budget=int(os.environ.get("AIDM_PROMPT_TOKEN_BUDGET", "24000")),
    )


def _parse_origins(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    """Parse a generic boolean env-var. Permissive: 0/false/off/no → False,
    1/true/on/yes → True, anything else (including empty/unset) → ``default``.
    """
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in ("",):
        return default
    if val in ("0", "false", "off", "no"):
        return False
    if val in ("1", "true", "on", "yes"):
        return True
    return default


def _default_budget_db_path() -> Path:
    # Repo-local dev default; production overrides to /data/budget.db.
    return Path(__file__).resolve().parents[3] / "data" / "budget.db"


def _require_stt_provider() -> Literal["groq", "elevenlabs"]:
    value = os.environ.get("STT_PROVIDER", "elevenlabs").lower()
    if value not in ("groq", "elevenlabs"):
        raise ValueError(f"Unsupported STT_PROVIDER: {value!r} (expected 'groq' or 'elevenlabs')")
    return value  # type: ignore[return-value]


def _require_tts_transport() -> Literal["ws", "http"]:
    value = os.environ.get("ELEVENLABS_TTS_TRANSPORT", "ws").lower()
    if value not in ("ws", "http"):
        raise ValueError(
            f"Unsupported ELEVENLABS_TTS_TRANSPORT: {value!r} (expected 'ws' or 'http')"
        )
    return value  # type: ignore[return-value]


def _resolve_vault_path(override: str | None) -> Path:
    """Resolve the *default* campaign vault path.

    Each campaign lives in its own self-contained Obsidian vault under
    ``<vault-root>/<campaign-slug>/``. ``VAULT_PATH`` names the campaign
    the local CLI starts in, and the campaign the web picker selects
    first when localStorage is empty. Browser sessions can switch to
    any sibling campaign at runtime — see ``vault_root``.
    """
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "vault" / "ember-hollow"


def _resolve_vault_root(explicit: str | None, vault_path_override: str | None) -> Path:
    """Resolve the directory that holds every campaign vault.

    Resolution order:

    1. ``AIDM_VAULT_ROOT`` env — explicit override, used when campaigns
       live somewhere other than ``VAULT_PATH``'s sibling tree.
    2. ``VAULT_PATH``'s parent — the common case: ``VAULT_PATH`` names
       one campaign, ``VAULT_PATH/..`` holds all of them.
    3. ``<repo>/vault/`` — the dev default when nothing is set.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    if vault_path_override:
        return Path(vault_path_override).expanduser().resolve().parent
    return Path(__file__).resolve().parents[3] / "vault"


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value and value.strip() else None


def _load_env_file() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env", override=False)


def _require_if_public(mode: str, key: str) -> str:
    """Required in public mode, blank-allowed in private mode.

    Private mode hits no cloud providers, so its env file shouldn't need
    cloud API keys present. Returning "" is safe because the public
    factories are never called when mode == "private".
    """
    value = os.environ.get(key, "")
    if mode == "public" and not value:
        raise RuntimeError(f"Missing required environment variable in public mode: {key}")
    return value
