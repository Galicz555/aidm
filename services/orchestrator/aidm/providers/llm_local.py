"""Local llama.cpp LLM — private-mode response generation.

The ``services/llm-gateway`` container runs ``llama-server`` with a
Qwen3 14B instruct GGUF (Q5_K_M, 16k context), chosen to resist confabulation and use llama.cpp's
native tool-call handler. It speaks an OpenAI-compatible API on port
8002, so we reuse Pipecat's ``OpenAILLMService`` exactly as the
public-mode OpenRouter wiring does — only the ``base_url`` changes.

Persona swap (public ``dm.{locale}.md`` vs private
``dm.{locale}.private.md``) happens in ``aidm.persona`` based on
``config.mode`` — this module just forwards the resolved prompt
string.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from difflib import SequenceMatcher
from typing import Any, NamedTuple, cast

import httpx
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream, DefaultAsyncHttpxClient
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as _ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as _ChoiceDelta
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

from aidm.advance_judge import judge_repeats
from aidm.config import Config
from aidm.persona import persona_for
from aidm.scene_sheet_injector import SCENE_SHEETS_TAG, _content_text, _role_of
from aidm.session import SessionContext
from aidm.story_events import directive_anchor_words, is_time_transition
from aidm.story_events import directive_covered as _directive_covered_fn
from aidm.turn_director import (
    PARA_SHINGLE_WORDS,
    PLAYER_ECHO_THRESHOLD,
    _shingles,
    phrase_text,
    player_echo_ratio,
    similarity,
)

# Per-phase httpx timeouts. ``read`` bounds chunk-arrival on a streaming
# response — the load-bearing one: Pipecat's ``retry_timeout_secs`` only
# bounds the ``chat.completions.create`` coroutine, which returns the
# AsyncStream object as soon as HTTP headers land. After that, chunk
# iteration happens unbounded. Hung gateway mid-stream = forever wait
# unless httpx itself enforces a per-read deadline.
_CONNECT_TIMEOUT_S = 10.0
_READ_TIMEOUT_S = 60.0
_WRITE_TIMEOUT_S = 10.0
_POOL_TIMEOUT_S = 10.0

# Spoken when a completion request fails outright (gateway 400/timeout/
# down). Errors are values inside services; the orchestrator speaks
# in-character apologies — without this the turn dies upstream as an
# ErrorFrame, no end-of-turn frame fires, and every client (browser and
# loop harness alike) hangs waiting for a `turn.status: idle` that never
# comes (observed live: ctx overflow at loop-turn 23, 2026-07-11).
# English-only by decision (docs/decisions/english-only.md).
APOLOGY_TEXT = (
    "*The DM goes quiet for a moment, rubbing their temples.* "
    "I lost the thread of the world just now — give me that once more, "
    "or take your next action, and we pick up exactly where we were."
)


def _apology_chunks(model: str) -> AsyncIterator[ChatCompletionChunk]:
    """A synthetic one-chunk stream carrying the apology.

    Shaped as a real ``ChatCompletionChunk`` so the parent service's
    stream loop processes it like any completion — the full downstream
    turn machinery (splitter → sink → aggregator → sheet writer → idle)
    runs and the session stays alive.
    """

    async def _gen() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="local-llm-apology",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                _ChunkChoice(
                    index=0,
                    delta=_ChoiceDelta(content=APOLOGY_TEXT, role="assistant"),
                    finish_reason="stop",
                )
            ],
        )

    return _gen()


# Appended (then removed) around the single retry dispatch. English-only
# by decision (docs/decisions/english-only.md). Every guard the draft
# failed contributes one section to the same note — a retry that is only
# told about the tic would fix the tic and still miss the directive.
_NOTE_HEADER = "Your previous draft was DISCARDED before the player saw it."
_PHRASE_SECTION = (
    'It reused the overused phrase "{phrase}". Do not use that phrase or any close variant of it.'
)
_PARA_LOOP_SECTION = (
    "It verbatim repeated a passage from a recent turn. "
    "Do not recycle prior prose — advance the scene with fresh narration."
)
_REPLAY_SECTION = (
    "It replayed your previous reply nearly word for word. The player has already "
    "read that — do not restate it; show what happens next."
)
_RESPOND_TO_SECTION = 'The player\'s latest message — respond to THIS:\n"{user}"'
_DIRECTIVE_SECTION = (
    "These world events MUST be woven into your narration as established facts "
    "(they already happened — do not veto, summarize, or ignore them):\n{directives}"
)
_CRITIC_SECTION = (
    "A continuity check found problems: {violations}\nFix these — stay "
    "strictly within what the sheets and the player's message establish, "
    "keep each character's own voice, and don't contradict yourself "
    "within the same reply."
)
_CONTRAST_SECTION = (
    "It leaned on the contrast construction (\"It isn't X. It's Y.\", \"It doesn't X. It "
    'Ys.", "Not X — Y.") more than once. Say what things ARE, directly; use that '
    "construction at most once in the whole reply."
)
_NOTE_FOOTER = "Write a fresh reply to the same player message."

# The "It isn't X. It's Y." / "It doesn't X. It Ys." / "Not X — Y." tic:
# a short negated sentence, then a pronoun-led correction; or a sentence
# opening with "Not X —". Once is a rhetorical choice, more is the
# house style the DM falls into ("It isn't darkness. It's *water*." /
# "The light doesn't flood. It *arrives*.", 2026-09-27 playtest).
_CONTRAST_NEG = r"(?:is|was|are|were|does|did|do|can|wo)(?:n['\u2019]t| not)|cannot"
_CONTRAST_RE = re.compile(
    r"(?:^|(?<=[.!?\n]))[\s\"\u201c*_]*"
    rf"(?:[^.!?;\n—\u2013]{{0,40}}?\b(?i:{_CONTRAST_NEG})\b[^.!?;\n—\u2013]{{0,60}}"
    r"[.!?;—\u2013]+[\s\"\u201d\u201c*_]*(?i:it|this|that|he|she|they|you|we)\b"
    r"|Not\b[^.!?;\n—\u2013]{1,60}?(?:[—\u2013]|--))"
)
# Contrast constructions a single draft may carry before the retry.
_CONTRAST_MAX = 1

# Draft→critic pass. Only fires on turns with 2+ NPC sheets in scene —
# the compounding-complexity case the failure profile names (invented
# relationships, borrowed traits, register drift) — and only when the
# cheaper deterministic guards above already found nothing (bounded to
# one extra completion pair per turn, never stacked on a guard retry).
# Three categories in one call — cheaper than three separate passes,
# and the sheets/draft are already loaded either way:
#   1. sheet contradiction/invention (the original pass)
#   2. self-contradiction within the draft itself (not caught by any
#      other guard — those all compare across turns, never within one)
#   3. voice/register drift — dialogue that doesn't match a sheet's
#      speech_register, piggybacking on the existing scene-tenure
#      voice-check idea but checked every eligible turn, not just long
#      scenes
# The /end goodbye after the session-log tool result: 1-3 sentences.
_FAREWELL_MAX_TOKENS = 256
_CRITIC_MIN_NPCS = 2
_CRITIC_MAX_TOKENS = 350
_CRITIC_TIMEOUT_S = 30.0
_CRITIC_PROMPT_TEMPLATE = (
    "You are a continuity reviewer. Compare the DRAFT narration below "
    "against the CHARACTER SHEETS (ground truth for personality, "
    "relationships, history, location, status, and speech_register). "
    "Flag a violation for any of these three kinds ONLY:\n"
    "1. Contradiction or invention — a claim that contradicts a sheet, "
    "or invents a specific fact about a named character (a relationship, "
    "an event, a possession, a piece of history) with no basis in the "
    "sheets and not stated by the player.\n"
    "2. Self-contradiction — two claims inside the DRAFT itself that "
    "cannot both be true (a location, time, or object stated one way "
    "then another).\n"
    "3. Voice drift — a named character's dialogue that clashes with "
    "their sheet's speech_register (e.g. a terse, guarded character "
    "suddenly speaking in ornate or verbose sentences).\n"
    "Do NOT flag scene-setting, weather, generic reactions, or anything "
    "the player's own message introduced.\n\n"
    "SHEETS:\n{sheets}\n\n"
    "DRAFT:\n{draft}\n\n"
    'Respond with ONLY compact JSON: {{"violations": ["short description", ...]}}. '
    "Empty array if none."
)

# Directive logit bias. A SOFT per-turn nudge, deliberately not a
# grammar/tool_choice constraint (that's what caused the agentic-recon
# runaway — a model that can't converge inside a hard constraint just
# rambles to the token ceiling; a logit bias only adds to the logits,
# it never forces or excludes anything). Only the first leading-space
# token of each directive anchor is biased — that's the far more common
# context (a name appearing mid-sentence after a space) and a longer,
# more specific subword than the bare capital letter alone.
# Simplification: no bias on bare "Name" (no leading space) or continuation
# subwords — covers the common case, not every tokenization path.
_LOGIT_BIAS_TIMEOUT_S = 5.0

# Replay guard. A draft this similar (``turn_director.similarity``) to one of
# the last REPLAY_LOOKBACK assistant replies is a replay — measured
# 0.80-0.96 on the 2026-09-26 marathon's T12-T15 loop, far above natural
# same-scene overlap (~0.2-0.35, see REPLY_SIMILARITY_THRESHOLD). The
# lookback spans HistoryTrimmer's whole 12-turn window: a lookback of 2
# missed 0.97-0.99 copies from 3-5 turns back (2026-09-26 replay-003,
# T11=T6, T41=T38).
_REPLAY_SIMILARITY_MIN = 0.8
_REPLAY_LOOKBACK = 12
# The replay/loop retry samples this much hotter (capped) for that one
# call. Simplification: no seed change — the gateway runs unseeded (llama.cpp
# default = random per request), so every call already gets a new seed;
# hiding the copied reply is the real lever, the bump only widens it.
_LOOP_RETRY_TEMPERATURE_BUMP = 0.25
_LOOP_RETRY_TEMPERATURE_MAX = 1.0

# Banned-phrase / paragraph-loop hits are excised, not regenerated: the
# retry reused the tic 18/30 and the loop 8/13 times on the 2026-09-26
# marathon (replay-002) at ~9-13 s each. Past these bounds the cut would
# gut the reply, so the single retry is used instead.
_EXCISE_MAX_CUT = 0.4
# Backtracking phrase ban (antislop-style, arXiv 2510.15061): a draft
# that trips a banned phrase is cut where the phrase starts and resumed
# from that prefix with the phrase's first token banned for a few tokens
# only, so the model writes around it instead of leaving an excision gap.
_BACKTRACK_MAX = 2
_BACKTRACK_BAN_TOKENS = 4
# Player-echo cut: a leading sentence restates the player's turn when this
# share of its 4+-letter words appear in the player's message (the echo
# swaps person and tense — "I ran" → "you run" — so exact matching misses).
_ECHO_WORD_OVERLAP = 0.6
# Delimits the per-request event block appended to the player's message
# (``_inject_events``); everything before it is the player's own text.
EVENT_NOTE_MARKER = (
    "\n\n[World event — happening in the world right now, NOT the player's action. "
    "Narrate it in THIS reply, woven into what happens next:\n"
)
_LEADING_THINK_RE = re.compile(r"^\s*(?:<think>.*?</think>|</think>)\s*", re.DOTALL)
_EXCISE_MIN_SENTENCES = 2
_SENTENCE_TERMINATORS = ".!?…"
_SENTENCE_TRAILERS = "*)]"
_QUOTE_PAIRS = {'"': '"', "“": "”"}


class _LoggingLocalLLMService(OpenAILLMService):
    """Local-LLM service with bounded chunk-read timeout + dispatch logging.

    Overrides ``create_client`` so the underlying ``httpx.AsyncClient``
    carries a finite ``read`` timeout — that is what bounds the
    chunk-by-chunk read of a streaming response. The parent class's
    ``DefaultAsyncHttpxClient`` carries no read timeout, so a stalled
    llama-gateway leaves Pipecat blocked on the next chunk forever.

    Also carries the **guarded retry**: when the turn director has
    banned phrases live (an established prose tic), the reply is
    buffered and checked; a reused tic or looped paragraph is excised
    sentence-wise (a retry regenerated the same tic most of the time),
    replays and missed event directives are regenerated ONCE —
    enforcement in code instead of hoping the model obeys the contract
    note (measured: it doesn't — replay-005, 2026-07-12). Opening turns
    with no guard armed stream through untouched.

    Optionally carries the **draft→critic pass** (``critic_enabled``):
    on turns with 2+ NPC sheets in scene and no other guard failure, a
    second short completion compares the draft against the sheets
    actually in context and lists contradictions/inventions; a hit
    folds into the same regenerate-once note as the other guards.

    Optionally carries a **directive logit bias** (``logit_bias_enabled``):
    while a directive is live, a soft positive bias on its proper-noun
    anchors nudges generation toward mentioning them, at the sampling
    layer rather than only catching a miss after the fact.
    """

    # Set post-construction by the pipeline (the director is created
    # after the LLM service); None = guard disabled.
    _banned_phrases_provider: Callable[[], list[str]] | None = None
    _event_directives_provider: Callable[[], list[str]] | None = None
    _para_banned_provider: Callable[[], list[str]] | None = None
    _stalled_provider: Callable[[], bool] | None = None
    # True while the /end farewell turn runs (tool_selector.end_session).
    _end_session_provider: Callable[[], bool] | None = None
    # Class-level fallback for instances built via __new__ (tests bypass
    # __init__ entirely) — real construction always goes through __init__.
    _critic_enabled: bool = False
    _logit_bias_enabled: bool = False
    _logit_bias_strength: float = 6.0
    _tokenize_url: str = ""
    _pending_logit_bias: dict[str, int] | None = None
    # True only while the replay/loop retry is in flight (reset in finally).
    _loop_retry_sampling: bool = False
    # Per-guard switches (Config.guard_*); each off disables exactly its guard.
    _guard_tics: bool = True
    _guard_paragraph: bool = True
    _guard_replay: bool = True
    _guard_directives: bool = True
    _guard_backtrack: bool = True
    _guard_echo: bool = True
    _events_in_user: bool = True
    # Pre-release content-repetition check (advance_judge).
    _judge_every_draft: bool = False
    _judge_retries: int = 1
    # Per-dispatch overrides while a backtrack step is in flight.
    _step_logit_ban: dict[str, int] | None = None
    _step_max_tokens: int | None = None
    # True while the farewell leg dispatches: thinking off for that call.
    _farewell_leg: bool = False

    def __init__(  # noqa: PLR0913 — one keyword per per-session flag
        self,
        *args: Any,
        critic_enabled: bool = False,
        logit_bias_enabled: bool = False,
        logit_bias_strength: float = 6.0,
        guard_tics: bool = True,
        guard_paragraph: bool = True,
        guard_replay: bool = True,
        guard_directives: bool = True,
        guard_backtrack: bool = True,
        guard_echo: bool = True,
        events_in_user: bool = True,
        judge_every_draft: bool = False,
        judge_retries: int = 1,
        **kwargs: Any,
    ) -> None:
        gateway_base_url = str(kwargs.get("base_url") or "")
        super().__init__(*args, **kwargs)
        # Static per-session flags, not per-turn state — set once at
        # construction, unlike the provider callbacks above (those exist
        # because DirectorState is built after the LLM service).
        self._critic_enabled = critic_enabled
        self._logit_bias_enabled = logit_bias_enabled
        self._logit_bias_strength = logit_bias_strength
        self._guard_tics = guard_tics
        self._guard_paragraph = guard_paragraph
        self._guard_replay = guard_replay
        self._guard_directives = guard_directives
        self._guard_backtrack = guard_backtrack
        self._guard_echo = guard_echo
        self._events_in_user = events_in_user
        self._judge_every_draft = judge_every_draft
        self._judge_retries = judge_retries
        self._tokenize_url = _tokenize_url(gateway_base_url)
        self._pending_logit_bias = None

    def set_banned_phrases_provider(self, provider: Callable[[], list[str]]) -> None:
        self._banned_phrases_provider = provider

    def set_event_directives_provider(self, provider: Callable[[], list[str]]) -> None:
        self._event_directives_provider = provider

    def set_para_banned_provider(self, provider: Callable[[], list[str]]) -> None:
        self._para_banned_provider = provider

    def set_stalled_provider(self, provider: Callable[[], bool]) -> None:
        """True when the last reply re-told the previous beat (director.reply_stalled)."""
        self._stalled_provider = provider

    def set_end_session_provider(self, provider: Callable[[], bool]) -> None:
        self._end_session_provider = provider

    async def get_chat_completions(self, context: LLMContext) -> AsyncStream[ChatCompletionChunk]:
        if self._end_session_provider is not None and self._end_session_provider():
            return await self._farewell_completion(context)
        banned = (
            self._banned_phrases_provider()
            if self._banned_phrases_provider and self._guard_tics
            else []
        )
        directives = self._event_directives_provider() if self._event_directives_provider else []
        para_banned = (
            self._para_banned_provider()
            if self._para_banned_provider and self._guard_paragraph
            else []
        )
        # Same bias for every dispatch this turn (initial draft AND any
        # guard-triggered retry) — the live directive doesn't change
        # mid-turn, so neither should the nudge toward its anchors.
        self._pending_logit_bias = (
            await self._resolve_logit_bias(directives) if self._logit_bias_enabled else None
        )
        # The logit bias above still sees the directives; only the
        # release guard is switched off.
        guarded = list(directives) if self._guard_directives else []
        prior = _recent_replies(getattr(context, "messages", None)) if self._guard_replay else []
        if not banned and not guarded and not para_banned and not prior:
            return await self._dispatch_logged(context)
        # Buffered mode costs streaming latency on every turn that has a
        # prior reply to replay (i.e. all but the first) or another active
        # guard — only an opening turn with no guards streams through.
        return cast(
            "AsyncStream[ChatCompletionChunk]",
            self._guarded_completion(context, list(banned), guarded, list(para_banned)),
        )

    async def _farewell_completion(self, context: LLMContext) -> AsyncStream[ChatCompletionChunk]:
        """The /end turn, unguarded: the session log leg, then a short goodbye.

        Every guard retry pushes the model to move the story on — the
        advance judge's note is "move the scene forward" — which turned a
        goodbye into two regenerations and a 5872-char continuation
        (2026-09-27 playtest). The log leg (``tool_choice=required``)
        keeps the full token budget so the log JSON is never truncated;
        only the goodbye after it is capped.
        """
        self._pending_logit_bias = None
        if getattr(context, "tool_choice", None) == "required":
            return await self._dispatch_logged(context)
        self._step_max_tokens, self._farewell_leg = _FAREWELL_MAX_TOKENS, True
        try:
            return await self._dispatch_logged(context)
        finally:
            self._step_max_tokens, self._farewell_leg = None, False

    async def _dispatch_logged(self, context: LLMContext) -> AsyncStream[ChatCompletionChunk]:
        """Time the HTTP-to-stream-handle phase so a slow gateway is visible.

        The parent implementation logs ``Generating chat from context``
        at DEBUG and that's the last line we see before a stall. Surface
        the wall-time it took to get the stream object back — that's
        prefill + first-byte. After that the chunks are bounded by the
        httpx ``read`` timeout above.
        """
        started = time.perf_counter()
        msgs = getattr(context, "messages", [])
        logger.info("llm_local: dispatch start (msgs={})", len(msgs))
        forced_recon = getattr(context, "tool_choice", None) == "required"
        try:
            stream = await super().get_chat_completions(context)
        except Exception as exc:
            # Simplification: single-shot fallback, no retry — a ctx-overflow 400
            # retries identically and a hung gateway already burned the
            # httpx timeouts. The apology keeps the turn (and session)
            # alive; the operator reads this WARNING for the real cause.
            logger.warning(
                "llm_local: dispatch FAILED after {:.1f}s — {}: {} — speaking apology instead",
                time.perf_counter() - started,
                type(exc).__name__,
                exc,
            )
            model = str(getattr(self, "model_name", "") or "local")
            return cast("AsyncStream[ChatCompletionChunk]", _apology_chunks(model))
        if forced_recon:
            # tool_choice=required is a one-shot: the injector arms it
            # per turn; leaving it set would force the post-tool
            # narration completion into another tool call, forever.
            context.set_tool_choice(NOT_GIVEN)
            logger.debug("llm_local: tool_choice=required consumed — reset to auto")
        logger.info(
            "llm_local: stream open after {:.1f}s — awaiting chunks",
            time.perf_counter() - started,
        )
        return stream

    async def _guarded_completion(
        self,
        context: LLMContext,
        banned: list[str],
        directives: Sequence[str] = (),
        para_banned: Sequence[str] = (),
    ) -> AsyncIterator[ChatCompletionChunk]:
        """The guarded turn, with this turn's world events riding the player's message.

        Only for the requests of this turn (draft, retry, backtracks): the
        latest user message is restored in ``finally`` — the sheet writer
        and history read it as the player's words.
        """
        restore = self._inject_events(context, directives)
        try:
            async for chunk in self._guarded_body(context, banned, directives, para_banned):
                yield chunk
        finally:
            restore()

    def _inject_events(self, context: LLMContext, directives: Sequence[str]) -> Callable[[], None]:
        """Append the event block to the latest user message; return the undo.

        qwen38 missed 7-8 fired events per marathon with the directive only
        in the ~19k-char scene-sheets block (2026-09-27 Q3/Q4, D7 3-7/20);
        the player's message is where the model looks for what to answer.
        """
        messages = getattr(context, "messages", None)
        if not self._events_in_user or not directives or not isinstance(messages, list):
            return lambda: None
        for message in reversed(messages):
            if _role_of(message) == "user":
                break
        else:
            return lambda: None
        original = message.get("content") if isinstance(message, dict) else None
        if not isinstance(original, str):
            return lambda: None
        lines = "\n".join(f"- {d}" for d in directives)
        message["content"] = f"{original}{EVENT_NOTE_MARKER}{lines}]"

        def restore() -> None:
            message["content"] = original

        return restore

    async def _guarded_body(
        self,
        context: LLMContext,
        banned: list[str],
        directives: Sequence[str] = (),
        para_banned: Sequence[str] = (),
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Buffer the reply; excise tics, regenerate once for anything else.

        A draft that only trips the banned-phrase / paragraph-loop guards
        is released with the offending sentences excised (``_excise``) —
        no second call — unless the cut would exceed ``_EXCISE_MAX_CUT``
        or leave < ``_EXCISE_MIN_SENTENCES``; then it takes the retry.
        Whatever a retry path releases is excised the same way.

        Every failing guard (phrase tic, paragraph loop, directive miss,
        sheet-continuity critique) contributes a section to the single
        retry note. One retry only — a second violation is released
        with a WARNING naming what still failed (bounded latency: at
        most one extra completion pair per turn) so the scorer and
        judge still see the failure.

        The retry note is appended then removed in ``finally`` — it must
        never persist into history.
        """
        stream = await self._dispatch_logged(context)
        chunks = [chunk async for chunk in stream]
        text = _chunks_text(chunks)
        rewritten = await self._backtrack_banned(context, text, [*banned, *para_banned])
        rewritten = self._cut_player_echo(getattr(context, "messages", None), rewritten)
        chunks, text = (
            (chunks, text) if rewritten == text else (_with_text(chunks, rewritten), rewritten)
        )
        chunks, text = await self._advance_until_new(context, chunks, text)

        messages = getattr(context, "messages", None)
        prior = _recent_replies(messages) if self._guard_replay else []
        replay_sim = max((_replay_score(text, reply) for _, reply in prior), default=0.0)
        replay = replay_sim >= _REPLAY_SIMILARITY_MIN

        hit = _first_banned_phrase(text, banned)
        para_hit = _first_banned_phrase(text, para_banned)
        missed = _missed_directives(text, directives)
        contrasts = contrast_count(text) if self._guard_tics else 0
        contrast = contrasts > _CONTRAST_MAX
        clean = hit is None and para_hit is None and not missed and not replay and not contrast

        critic_violations: list[str] = []
        if clean and self._critic_enabled:
            critic_violations = await self._run_critic_pass(context, text)

        if clean and not critic_violations:
            for chunk in chunks:
                yield chunk
            return

        # Repeated 8-grams vs prior replies are the paragraph guard's cut.
        prior_all = _assistant_texts(messages) if self._guard_paragraph else []
        if not replay and not missed and not contrast:
            # Only a tic / paragraph loop failed: cut it out, no second call.
            excised = _safe_excision(text, banned, para_banned, prior_all)
            if excised is not None:
                for chunk in _with_text(chunks, excised):
                    yield chunk
                return

        # A replay/loop retry that sees the reply it copied copies it again
        # (2026-09-26 marathon: "retry still violated" on every loop turn).
        # Hide the copied reply(ies) from that one request and point the
        # model at the player's latest message instead.
        hide = (
            _copied_reply_indices(messages, text if replay else "", para_hit)
            if replay or para_hit
            else []
        )
        latest_user = _latest_user_text(messages) if replay or para_hit else ""

        note_text = _retry_note(
            hit, replay, para_hit, latest_user, missed, critic_violations, contrast=contrast
        )

        logger.info(
            "llm_local: draft failed guard ({}) — regenerating once",
            "; ".join(
                [
                    *_violation_reasons(hit, para_hit, missed, contrasts),
                    *([f"replay sim {replay_sim:.2f}"] if replay else []),
                    *_critic_reasons(critic_violations),
                ]
            ),
        )
        retry_chunks = await self._dispatch_retry(
            context, note_text, hide, hotter=bool(replay or para_hit)
        )
        retry_text = _chunks_text(retry_chunks)
        retry_sim = max((_replay_score(retry_text, reply) for _, reply in prior), default=0.0)
        if replay:
            logger.info(
                "llm_local: replay guard (sim {:.2f}) — {} (retry sim {:.2f})",
                replay_sim,
                "still similar" if retry_sim >= _REPLAY_SIMILARITY_MIN else "retry cleared",
                retry_sim,
            )
        # A fresh draft retried for a lesser guard (banned phrase, missed
        # directive) can come back as a replay — 3 of 15 looping turns on the
        # 2026-09-26 replay. A whole repeated reply is the worse failure:
        # release the draft instead.
        if retry_sim >= _REPLAY_SIMILARITY_MIN and not replay:
            logger.warning(
                "llm_local: retry replayed a prior reply (sim {:.2f}) — releasing the draft",
                retry_sim,
            )
            for chunk in _excise_released(chunks, banned, para_banned, prior_all):
                yield chunk
            return
        retry_chunks = _excise_released(retry_chunks, banned, para_banned, prior_all)
        retry_text = _chunks_text(retry_chunks)
        still = _violation_reasons(
            _first_banned_phrase(retry_text, banned),
            _first_banned_phrase(retry_text, para_banned),
            _missed_directives(retry_text, directives),
            contrast_count(retry_text) if self._guard_tics else 0,
        )
        if retry_sim >= _REPLAY_SIMILARITY_MIN:
            still.append(f"replay sim {retry_sim:.2f}")
        if still:
            logger.warning(
                "llm_local: retry still violated ({}) — releasing as-is", "; ".join(still)
            )
        for chunk in retry_chunks:
            yield chunk

    async def _backtrack_banned(
        self, context: LLMContext, text: str, phrases: Sequence[str]
    ) -> str:
        """``text`` with each banned phrase rewritten in place, up to ``_BACKTRACK_MAX`` times.

        Cut at the earliest phrase, then two prefilled continuations: a
        few tokens with the phrase's first token banned, then the rest
        unbanned. Any failure keeps the text so far — excision downstream
        still catches what is left.
        """
        if not self._guard_backtrack:
            return text
        for _ in range(_BACKTRACK_MAX):
            found = _earliest_phrase(text, phrases)
            if found is None:
                break
            start, phrase = found
            prefix = text[:start].rstrip(" ")
            word = re.match(r"[a-z']+", text[start:], re.IGNORECASE)
            ban = await self._phrase_ban(word.group()) if word else {}
            if not ban:
                break
            step = await self._continue(context, prefix, ban, _BACKTRACK_BAN_TOKENS)
            if not step:
                break
            rest = await self._continue(context, prefix + step, None, None)
            logger.info("llm_local: backtracked banned phrase {!r}", phrase)
            text = prefix + step + rest
        return text

    def _cut_player_echo(self, messages: Any, text: str) -> str:
        """``text`` without an opening that restates the player's turn.

        The next-turn contract line alone did not stop it: 14 echoed turns
        from T22 on (2026-09-26 replay-006).
        """
        if not self._guard_echo:
            return text
        text = _STATE_LEAK_RE.sub("", text).lstrip()
        player = _latest_user_text(messages).split(EVENT_NOTE_MARKER, 1)[0]
        cut = _excise_player_echo(text, player) or _cut_echo_opening(text, player)
        if cut is None:
            return text
        logger.info("llm_local: cut player echo ({} -> {} chars)", len(text), len(cut))
        return cut

    async def _phrase_ban(self, word: str) -> dict[str, int]:
        """-100 bias on the first token of ``word`` in its likely spellings."""
        ban: dict[str, int] = {}
        for variant in {word, f" {word}", word.capitalize(), f" {word.capitalize()}"}:
            token_id = await _first_token_id(self._tokenize_url, variant)
            if token_id is not None:
                ban[str(token_id)] = -100
        return ban

    async def _continue(
        self, context: LLMContext, prefix: str, ban: dict[str, int] | None, max_tokens: int | None
    ) -> str:
        """Continuation of an assistant-prefilled ``prefix`` (llama.cpp prefill); history restored.

        llama-server echoes the prefill ahead of the new tokens (b9976,
        checked live 2026-09-26); strip it so only the continuation returns.
        An empty prefill (phrase at the reply's very start) came back led
        by the Qwen3 template's empty-think closer — ``</think>`` shipped in
        2 of 15 backtracked turns (2026-09-26 replay-005) — so a leading
        think block is stripped first.
        """
        messages = getattr(context, "messages", None)
        if not isinstance(messages, list):
            return ""
        prefill = {"role": "assistant", "content": prefix}
        messages.append(prefill)
        self._step_logit_ban, self._step_max_tokens = ban, max_tokens
        try:
            stream = await self._dispatch_logged(context)
            out = _chunks_text([chunk async for chunk in stream])
            return _LEADING_THINK_RE.sub("", out, count=1).removeprefix(prefix)
        finally:
            self._step_logit_ban = self._step_max_tokens = None
            if prefill in messages:
                messages.remove(prefill)

    async def _dispatch_retry(
        self, context: LLMContext, note_text: str, hide: Sequence[int], *, hotter: bool
    ) -> list[ChatCompletionChunk]:
        """The single retry: note appended, ``hide`` indices removed, both undone in ``finally``.

        ``hotter`` raises temperature for this one call only. Ascending
        re-insert at the original indices restores the exact history —
        nothing about the retry may persist into ``LLMContext``.
        """
        messages = getattr(context, "messages", None)
        note = {"role": "system", "content": note_text}
        hidden: list[tuple[int, Any]] = []
        if isinstance(messages, list):
            hidden = [(i, messages[i]) for i in hide]
            for i in reversed(hide):
                del messages[i]
            messages.append(note)
        self._loop_retry_sampling = hotter
        try:
            retry_stream = await self._dispatch_logged(context)
            return [chunk async for chunk in retry_stream]
        finally:
            self._loop_retry_sampling = False
            if isinstance(messages, list):
                if note in messages:
                    messages.remove(note)
                for i, message in hidden:
                    messages.insert(i, message)

    async def _advance_until_new(
        self, context: LLMContext, chunks: list[ChatCompletionChunk], text: str
    ) -> tuple[list[ChatCompletionChunk], str]:
        """Regenerate while the draft re-tells the previous beat, at most ``_judge_retries`` times.

        Content repetition felt in play (2026-09-27 playtest 3: 8 of 16 replies
        judged as re-telling) — run the check before every release,
        several times if needed. The last draft ships whatever its verdict.
        """
        messages = getattr(context, "messages", None)
        for attempt in range(self._judge_retries):
            if not await self._stalled_again(messages, text):
                break
            logger.info(
                "llm_local: draft re-tells the previous beat — regenerating ({}/{})",
                attempt + 1,
                self._judge_retries,
            )
            player = _latest_user_text(messages).split(EVENT_NOTE_MARKER, 1)[0]
            note = _retry_note(None, False, None, player, (), (), stalled=True)
            chunks = await self._dispatch_retry(context, note, [], hotter=True)
            text = _chunks_text(chunks)
        return chunks, text

    async def _stalled_again(self, messages: Any, draft: str) -> bool:
        """Whether ``draft`` re-tells the previous reply (every draft, or after a stalled turn)."""
        armed = self._judge_every_draft or (
            self._stalled_provider is not None and self._stalled_provider()
        )
        if not armed or not draft.strip():
            return False
        replies = _assistant_texts(messages)
        verdict = await judge_repeats(
            self._client,
            str(getattr(self, "model_name", "") or "local"),
            previous=replies[-1] if replies else "",
            player=_latest_user_text(messages).split(EVENT_NOTE_MARKER, 1)[0],
            current=draft,
        )
        return verdict is True

    async def _run_critic_pass(self, context: LLMContext, draft: str) -> list[str]:
        """One extra completion comparing ``draft`` against the in-scene sheets.

        Selective: skips the LLM call entirely (returns ``[]``) unless
        2+ NPC sheets are in scene this turn — the compounding-
        complexity case the failure profile names. Fails open on any
        error (bad JSON, timeout, gateway error): the critic's own
        failure must never block the player's turn.
        """
        sheets_text = _find_sheets_text(getattr(context, "messages", None))
        if sheets_text is None or _npc_count(sheets_text) < _CRITIC_MIN_NPCS:
            return []
        prompt = _CRITIC_PROMPT_TEMPLATE.format(sheets=sheets_text, draft=draft)
        try:
            completion = await self._client.chat.completions.create(
                model=str(getattr(self, "model_name", "") or "local"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_CRITIC_MAX_TOKENS,
                temperature=0.2,
                timeout=_CRITIC_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning(
                "llm_local: critic pass FAILED — {}: {} — skipping", type(exc).__name__, exc
            )
            return []
        raw = completion.choices[0].message.content or ""
        violations = _extract_violations(raw)
        if violations is None:
            logger.info("llm_local: critic pass returned unparseable JSON — skipping")
            return []
        return violations

    async def _resolve_logit_bias(self, directives: Sequence[str]) -> dict[str, int] | None:
        """Token-id bias map for this turn's live directive anchors.

        One ``/tokenize`` round-trip per distinct anchor word (cheap —
        milliseconds — against the once-per-turn 10-30s completion).
        Fails open: a tokenize error drops that one word, never the
        whole turn.
        """
        if not directives:
            return None
        words: dict[str, None] = {}
        for directive in directives:
            for word in directive_anchor_words(directive):
                words.setdefault(word, None)
        if not words:
            return None
        strength = int(self._logit_bias_strength)
        bias: dict[str, int] = {}
        for word in words:
            token_id = await _first_token_id(self._tokenize_url, f" {word}")
            if token_id is not None:
                bias[str(token_id)] = strength
        if bias:
            logger.info("llm_local: directive logit bias armed for {}", list(words))
        return bias or None

    def build_chat_completion_params(self, params_from_context: Any) -> dict[str, Any]:
        params = super().build_chat_completion_params(params_from_context)
        if self._pending_logit_bias:
            params["logit_bias"] = self._pending_logit_bias
        if self._step_logit_ban:
            params["logit_bias"] = {**(params.get("logit_bias") or {}), **self._step_logit_ban}
        if self._step_max_tokens is not None:
            params["max_tokens"] = self._step_max_tokens
            params.pop("max_completion_tokens", None)
        if self._farewell_leg:
            # A thinking model would spend the capped budget reasoning.
            params["extra_body"] = {
                **(params.get("extra_body") or {}),
                "chat_template_kwargs": {"enable_thinking": False},
            }
        base = params.get("temperature")
        # Simplification: bumps only a numeric temperature; an unset one (NOT_GIVEN,
        # server default) is left alone — every real build sets it via Config.
        if self._loop_retry_sampling and isinstance(base, int | float):
            params["temperature"] = min(
                _LOOP_RETRY_TEMPERATURE_MAX, base + _LOOP_RETRY_TEMPERATURE_BUMP
            )
        return params

    def create_client(  # type: ignore[no-untyped-def]
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs: Any,
    ) -> AsyncOpenAI:
        logger.info(
            "llm_local: client base_url={} timeouts(connect={}s, read={}s, write={}s)",
            base_url,
            _CONNECT_TIMEOUT_S,
            _READ_TIMEOUT_S,
            _WRITE_TIMEOUT_S,
        )
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=1000,
                    keepalive_expiry=None,
                ),
                timeout=httpx.Timeout(
                    connect=_CONNECT_TIMEOUT_S,
                    read=_READ_TIMEOUT_S,
                    write=_WRITE_TIMEOUT_S,
                    pool=_POOL_TIMEOUT_S,
                ),
            ),
        )


# English-only by decision (docs/decisions/english-only.md). Forward
# within the scene — never a time skip (playtest, 2026-09-27).
_STALLED_SECTION = (
    "Your draft re-tells the same beat as your previous reply, which already re-told the one "
    "before it. Move the scene forward WITHIN the scene: an NPC decides or acts, the outcome "
    "of the player's action lands, or something new is revealed. Do not skip time or cut away."
)


def _retry_note(  # noqa: PLR0913 — one flag per guard, straight-line assembly
    hit: str | None,
    replay: bool,
    para_hit: str | None,
    latest_user: str,
    missed: Sequence[str],
    critic_violations: Sequence[str],
    *,
    stalled: bool = False,
    contrast: bool = False,
) -> str:
    """The single retry note: one section per failed guard."""
    sections: list[str] = []
    if hit is not None:
        sections.append(_PHRASE_SECTION.format(phrase=hit))
    if contrast:
        sections.append(_CONTRAST_SECTION)
    if replay:
        sections.append(_REPLAY_SECTION)
    if para_hit is not None:
        sections.append(_PARA_LOOP_SECTION)
    if latest_user:
        sections.append(_RESPOND_TO_SECTION.format(user=latest_user))
    if missed:
        bullets = "\n".join(f"- {d}" for d in missed)
        sections.append(_DIRECTIVE_SECTION.format(directives=bullets))
    if critic_violations:
        bullets = "\n".join(f"- {v}" for v in critic_violations)
        sections.append(_CRITIC_SECTION.format(violations=bullets))
    if stalled:
        sections.append(_STALLED_SECTION)
    return "\n\n".join([_NOTE_HEADER, *sections, _NOTE_FOOTER])


def _chunks_text(chunks: list[ChatCompletionChunk]) -> str:
    """Concatenated text content of a buffered chunk list."""
    parts: list[str] = []
    for chunk in chunks:
        for choice in chunk.choices or []:
            content = getattr(choice.delta, "content", None)
            if content:
                parts.append(content)
    return "".join(parts)


# A draft that is mostly a verbatim slice of a LONGER prior reply scores low
# on whole-text similarity (2026-09-27 playtest: T15 = T14's first 275 of
# 756 chars, ratio ~0.53). Coverage counts the draft's chars that sit in
# matching runs of at least _REPLAY_RUN_CHARS; only drafts this long count.
_REPLAY_COVERAGE_MIN_CHARS = 200
_REPLAY_RUN_CHARS = 20


def _replay_score(draft: str, reply: str) -> float:
    """Replay likeness: whole-text similarity, or how much of ``draft`` is copied from ``reply``."""
    score = similarity(draft, reply)
    d = " ".join(draft.split()).casefold()
    r = " ".join(reply.split()).casefold()
    if len(d) < _REPLAY_COVERAGE_MIN_CHARS or not r:
        return score
    blocks = SequenceMatcher(None, d, r, autojunk=False).get_matching_blocks()
    copied = sum(b.size for b in blocks if b.size >= _REPLAY_RUN_CHARS)
    return max(score, copied / len(d))


def _earliest_phrase(text: str, phrases: Sequence[str]) -> tuple[int, str] | None:
    """``(raw offset, phrase)`` of the banned phrase starting earliest in ``text``.

    Matches ``phrase_text``'s word stream (``[a-z']+``, case-insensitive)
    against the raw text, so the offset points into what the model wrote.
    """
    best: tuple[int, str] | None = None
    for phrase in phrases:
        words = phrase_text(phrase).split()
        if not words:
            continue
        pattern = "(?<![a-z'])" + "[^a-z']+".join(map(re.escape, words)) + "(?![a-z'])"
        match = re.search(pattern, text, re.IGNORECASE)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), phrase)
    return best


def _first_banned_phrase(text: str, banned: Sequence[str]) -> str | None:
    """First banned phrase present in ``text``, tokenized like the detector."""
    if not text:
        return None
    haystack = f" {phrase_text(text)} "
    for phrase in banned:
        needle = phrase_text(phrase)
        if needle and f" {needle} " in haystack:
            return phrase
    return None


def _sentence_units(text: str) -> list[tuple[str, str]]:
    """``(sentence, trailing whitespace)`` pairs; never split inside a quote pair.

    A sentence ends at a terminator (plus trailing ``*)]``) followed by
    whitespace, a closing quote right after a terminator, or a newline.
    A newline also resets an unclosed quote (multi-paragraph dialogue
    reopens its quote each paragraph). Simplification: "Mr. Smith" splits after
    "Mr." — only costs an extra unit, never a cut inside a quote.
    """
    units: list[tuple[str, str]] = []
    n = len(text)
    start = i = len(text) - len(text.lstrip())
    closer = ""
    while i < n:
        ch = text[i]
        end = -1
        if ch == "\n":
            closer, end = "", i
        elif closer:
            if ch == closer:
                closer = ""
                if text[i - 1] in _SENTENCE_TERMINATORS:
                    end = i + 1
        elif ch in _QUOTE_PAIRS:
            closer = _QUOTE_PAIRS[ch]
        elif ch in _SENTENCE_TERMINATORS:
            end = i + 1
            while end < n and text[end] in _SENTENCE_TERMINATORS + _SENTENCE_TRAILERS:
                end += 1
        if end < 0 or (end < n and not text[end].isspace()):
            i += 1
            continue
        ws_end = end
        while ws_end < n and text[ws_end].isspace():
            ws_end += 1
        if text[start:end].strip():
            units.append((text[start:end], text[end:ws_end]))
        start = i = ws_end
    if start < n:
        units.append((text[start:].rstrip(), ""))
    return units


class _Excision(NamedTuple):
    text: str
    dropped: int
    kinds: str  # "banned", "paragraph" or "banned+paragraph"
    cut: float  # share of the draft's characters removed
    kept: int

    @property
    def safe(self) -> bool:
        return self.cut <= _EXCISE_MAX_CUT and self.kept >= _EXCISE_MIN_SENTENCES


def _excise(
    text: str, banned: Sequence[str], para_banned: Sequence[str], prior: Sequence[str]
) -> _Excision | None:
    """Drop every sentence touched by a banned phrase or a repeated 8-gram.

    "Repeated" = a paragraph-banned phrase, or any 8-word shingle shared
    with an assistant reply in ``prior``. Matching runs over the kept
    sentences' word stream (tokenized like the detector), minimal cut
    first: sentences that wholly contain a match go; only a match still
    spanning a boundary afterwards (or formed by a join) drops every
    sentence it touches. ``None`` = nothing to drop.
    """
    units = _sentence_units(text)
    repeated = set().union(*(_shingles(p, PARA_SHINGLE_WORDS) for p in prior))
    targets = [
        ("banned", {phrase_text(p) for p in banned}),
        ("paragraph", {phrase_text(p) for p in para_banned} | repeated),
    ]
    drop: dict[int, str] = {}
    while True:
        words = [
            (w, k)
            for k, (unit, _) in enumerate(units)
            if k not in drop
            for w in phrase_text(unit).split()
        ]
        inside: dict[int, str] = {}
        spanning: dict[int, str] = {}
        for kind, phrases in targets:
            for size in {len(p.split()) for p in phrases if p}:
                for i in range(len(words) - size + 1):
                    window = words[i : i + size]
                    if " ".join(w for w, _ in window) not in phrases:
                        continue
                    touched = {k for _, k in window}
                    hits = inside if len(touched) == 1 else spanning
                    for k in touched:
                        hits.setdefault(k, kind)
        if not inside and not spanning:
            break
        drop.update(inside or spanning)
    if not drop:
        return None
    out: list[str] = []
    sep = ""
    for k, (unit, trailing) in enumerate(units):
        if k in drop:
            # Keep the strongest break (a paragraph break beats a space).
            if trailing.count("\n") > sep.count("\n"):
                sep = trailing
            continue
        if out:
            out.append(sep)
        out.append(unit)
        sep = trailing
    kept = "".join(out)
    kinds = "+".join(kind for kind, _ in targets if kind in drop.values())
    cut = 1 - len(kept) / len(text.strip()) if text.strip() else 0.0
    return _Excision(kept, len(drop), kinds, cut, len(units) - len(drop))


def _tiered_excision(
    text: str, banned: Sequence[str], para_banned: Sequence[str], prior: Sequence[str]
) -> _Excision | None:
    """Every repeated 8-gram out when that is safe, else only the flagged phrases.

    The 2026-09-26 marathon drafts recycle 30-100% of their prose from
    replies still in context; cutting all of it would gut most replies
    (and the retry recycles as much), while the director's own lists
    (<=3 tics, <=2 looped 8-grams) cost 4-31%.
    """
    broad = _excise(text, banned, para_banned, prior)
    if broad is None or broad.safe:
        return broad
    return _excise(text, banned, para_banned, ())


def _safe_excision(
    text: str, banned: Sequence[str], para_banned: Sequence[str], prior: Sequence[str]
) -> str | None:
    """The excised draft when the cut is safe; ``None`` = take the retry."""
    cut = _tiered_excision(text, banned, para_banned, prior)
    if cut is None:
        return None
    if cut.safe:
        logger.info("llm_local: excised {} sentence(s) ({}) — no retry", cut.dropped, cut.kinds)
        return cut.text
    logger.info(
        "llm_local: excised {} sentence(s) ({}) — fallback retry (would cut {:.0%})",
        cut.dropped,
        cut.kinds,
        cut.cut,
    )
    return None


def _with_text(chunks: list[ChatCompletionChunk], text: str) -> list[ChatCompletionChunk]:
    """``chunks`` with their content replaced by ``text`` (other fields kept)."""
    out: list[ChatCompletionChunk] = []
    placed = False
    for chunk in chunks:
        copy = chunk.model_copy(deep=True)
        for choice in copy.choices or []:
            if getattr(choice.delta, "content", None):
                choice.delta.content = None if placed else text
                placed = True
        out.append(copy)
    return out


def _excise_released(
    chunks: list[ChatCompletionChunk],
    banned: Sequence[str],
    para_banned: Sequence[str],
    prior: Sequence[str],
) -> list[ChatCompletionChunk]:
    """Excise a reply the retry path is about to release, when it still trips a guard."""
    text = _chunks_text(chunks)
    if (
        _first_banned_phrase(text, banned) is None
        and _first_banned_phrase(text, para_banned) is None
    ):
        return chunks
    cut = _tiered_excision(text, banned, para_banned, prior)
    if cut is None:
        return chunks
    if not cut.safe:
        logger.info(
            "llm_local: excised {} sentence(s) ({}) — skipped after retry (would cut {:.0%})",
            cut.dropped,
            cut.kinds,
            cut.cut,
        )
        return chunks
    logger.info("llm_local: excised {} sentence(s) ({}) — after retry", cut.dropped, cut.kinds)
    return _with_text(chunks, cut.text)


def _assistant_texts(messages: Any) -> list[str]:
    """Text of every assistant reply in context (the 8-gram repeat source)."""
    if not isinstance(messages, list):
        return []
    return [t for m in messages if _role_of(m) == "assistant" and (t := _content_text(m))]


def _missed_directives(text: str, directives: Sequence[str]) -> list[str]:
    """Event directives not reflected in ``text``; empty when all covered.

    Time transitions are skipped: end-of-turn confirmation in
    ``story_events`` holds/re-renders/abandons them, and gating release on
    a time cue forced a (usually failing) retry on nearly every
    transition turn.
    """
    if not text or not directives:
        return []
    return [
        d for d in directives if not is_time_transition(d) and not _directive_covered_fn(text, d)
    ]


def _recent_replies(messages: Any) -> list[tuple[int, str]]:
    """``(index, text)`` of the last ``_REPLAY_LOOKBACK`` assistant replies with text."""
    if not isinstance(messages, list):
        return []
    found: list[tuple[int, str]] = []
    for i in range(len(messages) - 1, -1, -1):
        if _role_of(messages[i]) == "assistant":
            text = _content_text(messages[i])
            if text:
                found.append((i, text))
                if len(found) == _REPLAY_LOOKBACK:
                    break
    return found


def _copied_reply_indices(messages: Any, replayed_draft: str, para_hit: str | None) -> list[int]:
    """Ascending indices of assistant replies the draft copies from.

    Every assistant reply in context the draft replays (not only the last
    two the guard checks: a loop leaves a run of copies in history, and a
    retry that still sees one copies it — 2026-09-26 T36 retried at 0.99
    with the last two hidden), and any assistant reply carrying the looped
    paragraph. Replies that also
    carry ``tool_calls`` are never hidden — their tool results would be
    orphaned and the chat template would reject the request.
    """
    if not isinstance(messages, list):
        return []
    picked: set[int] = set()
    if replayed_draft:
        picked = {
            i
            for i, m in enumerate(messages)
            if _role_of(m) == "assistant"
            and _replay_score(replayed_draft, _content_text(m)) >= _REPLAY_SIMILARITY_MIN
        }
    if para_hit is not None:
        needle = f" {phrase_text(para_hit)} "
        picked |= {
            i
            for i, m in enumerate(messages)
            if _role_of(m) == "assistant" and needle in f" {phrase_text(_content_text(m))} "
        }
    return sorted(
        i for i in picked if not (isinstance(messages[i], dict) and messages[i].get("tool_calls"))
    )


def _excise_player_echo(text: str, player: str) -> str | None:
    """``text`` minus its leading sentences that restate ``player``; None = no safe cut."""
    if not player or player_echo_ratio(player, text) < PLAYER_ECHO_THRESHOLD:
        return None
    said = {w for w in phrase_text(player).split() if len(w) >= 4}
    units = _sentence_units(text)
    k = 0
    while k < len(units):
        words = [w for w in phrase_text(units[k][0]).split() if len(w) >= 4]
        if not words or sum(w in said for w in words) / len(words) < _ECHO_WORD_OVERLAP:
            break
        k += 1
    if k == 0 or len(units) - k < _EXCISE_MIN_SENTENCES:
        return None
    return "".join(unit + trailing for unit, trailing in units[k:]).strip()


# An opening paragraph that only quotes the player back in italics
# ("The words hang in the air. *Wrong answer.*") — 7 of 18 playtest 3 replies
# (2026-09-27). Short, followed by more text, and its italic span is
# mostly the player's own words.
_ITALIC_SPAN_RE = re.compile(r"\*([^*\n]{1,120})\*")
_ECHO_OPENING_MAX_WORDS = 40
_ECHO_SPAN_OVERLAP = 0.7
# The retired inline state syntax leaking into narration ("[State: …]").
_STATE_LEAK_RE = re.compile(r"^[ \t]*\[State:[^\]\n]*\][ \t]*\n?", re.MULTILINE | re.IGNORECASE)


def _cut_echo_opening(text: str, player: str) -> str | None:
    """``text`` minus an opening paragraph that italic-quotes the player; None = keep."""
    head, sep, rest = text.lstrip().partition("\n\n")
    if not sep or not rest.strip() or len(head.split()) > _ECHO_OPENING_MAX_WORDS:
        return None
    said = set(phrase_text(player).split())
    for span in _ITALIC_SPAN_RE.findall(head):
        words = [w for w in phrase_text(span).split() if len(w) >= 3]
        if words and sum(w in said for w in words) / len(words) >= _ECHO_SPAN_OVERLAP:
            return rest.lstrip()
    return None


def _latest_user_text(messages: Any) -> str:
    """Text of the most recent user message, or ``""``."""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if _role_of(message) == "user":
            return _content_text(message)
    return ""


def contrast_count(text: str) -> int:
    """How many "It isn't X. It's Y." / "Not X — Y." constructions ``text`` carries."""
    return len(_CONTRAST_RE.findall(text))


def _violation_reasons(
    hit: str | None, para_hit: str | None, missed: Sequence[str], contrasts: int = 0
) -> list[str]:
    """Human-readable guard-failure labels for operator logs."""
    reasons: list[str] = []
    if hit is not None:
        reasons.append(f"banned phrase {hit!r}")
    if contrasts > _CONTRAST_MAX:
        reasons.append(f"contrast construction x{contrasts}")
    if para_hit is not None:
        reasons.append(f"paragraph loop {para_hit[:60]!r}")
    if missed:
        reasons.append(f"missed directive(s): {missed[0][:60]!r}")
    return reasons


def _critic_reasons(violations: Sequence[str]) -> list[str]:
    """Human-readable critic-failure label for operator logs."""
    if not violations:
        return []
    return [f"critic: {violations[0][:60]!r}"]


def _find_sheets_text(messages: Any) -> str | None:
    """The current turn's scene-sheets system message content, or None."""
    if not isinstance(messages, list):
        return None
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
        if str(role or "").lower() != "system":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str) and content.startswith(SCENE_SHEETS_TAG):
            return content
    return None


def _npc_count(sheets_text: str) -> int:
    """Number of ``### npc:`` headers — the injector's per-NPC render marker."""
    return sheets_text.count("### npc:")


def _extract_violations(raw: str) -> list[str] | None:
    """Parse ``{"violations": [...]}`` from a possibly-fenced completion.

    Returns ``None`` on any parse failure — the caller treats that as
    "skip," never as "found nothing."
    """
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    violations = parsed.get("violations")
    if not isinstance(violations, list):
        return None
    return [str(v) for v in violations if isinstance(v, str) and v.strip()]


def _tokenize_url(base_url: str) -> str:
    """llama.cpp's raw ``/tokenize`` endpoint next to an OpenAI-compat ``base_url``."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/tokenize"


async def _first_token_id(tokenize_url: str, text: str) -> int | None:
    """First token id llama.cpp assigns to ``text``, or None on any failure."""
    if not tokenize_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=_LOGIT_BIAS_TIMEOUT_S) as client:
            response = await client.post(tokenize_url, json={"content": text})
            response.raise_for_status()
            tokens = response.json().get("tokens")
    except Exception as exc:
        logger.warning(
            "llm_local: tokenize FAILED for {!r} — {}: {}", text, type(exc).__name__, exc
        )
        return None
    if not isinstance(tokens, list) or not tokens or not isinstance(tokens[0], int):
        return None
    return tokens[0]


# llama.cpp's OpenAI-compat surface ignores the ``api_key`` field but
# OpenAI's SDK refuses to construct with an empty string. Any non-empty
# token works; this placeholder makes the intent explicit at the call
# site.
_LOCAL_API_TOKEN = "local-llama-cpp"


def build_local_llm(
    config: Config,
    session: SessionContext,
    *,
    phase: str = "play",
    vault_summary: str = "",
) -> OpenAILLMService:
    return _LoggingLocalLLMService(
        api_key=_LOCAL_API_TOKEN,
        base_url=config.local_llm_base_url,
        critic_enabled=config.critic_pass_enabled,
        logit_bias_enabled=config.directive_logit_bias_enabled,
        logit_bias_strength=config.directive_logit_bias_strength,
        guard_tics=config.guard_tics,
        guard_paragraph=config.guard_paragraph,
        guard_replay=config.guard_replay,
        guard_directives=config.guard_directives,
        guard_backtrack=config.guard_backtrack,
        guard_echo=config.guard_echo,
        events_in_user=config.events_in_user,
        judge_every_draft=config.advance_judge and config.advance_judge_every_draft,
        judge_retries=config.advance_judge_retries if config.advance_judge else 0,
        # retry_timeout_secs bounds the ``chat.completions.create``
        # coroutine — which for streaming requests returns immediately
        # once HTTP headers arrive, so this only catches connect-phase
        # hangs. The chunk-iteration timeout lives on the httpx client
        # (see _LoggingLocalLLMService.create_client).
        retry_timeout_secs=30.0,
        retry_on_timeout=True,
        settings=OpenAILLMService.Settings(
            model=config.local_llm_model,
            system_instruction=persona_for(
                session.locale,
                emotion_tags=config.tts_supports_emotion_tags,
                mode=config.mode,
                campaign_title=session.campaign_title,
                run_title=session.run_title,
                phase=phase,  # type: ignore[arg-type]
                vault_summary=vault_summary,
            ),
            # Sized for 3-6 sentence spoken replies (persona target) plus
            # end-of-session tool chain: vault_write_session_log
            # (200-400 words ≈ 500-600 tokens) + one or two
            # vault_append_note calls (~150 tokens each), without
            # truncating tool-call JSON mid-string.
            max_tokens=4500,
            # Sampling is Config-driven (env-overridable) so the anti-
            # confabulation tuning can be swept without a rebuild. Defaults
            # are tuned for the Qwen3 14B instruct model: low
            # temperature + a min_p tail-cut. See Config.llm_* docstring.
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            # OpenAI-spec siblings to the server-side --repeat-penalty.
            # Frequency hits exact-word repeats ("guarded X" loops);
            # presence nudges into new vocabulary territory turn over turn.
            frequency_penalty=config.llm_frequency_penalty,
            presence_penalty=config.llm_presence_penalty,
            # llama.cpp-only knobs the OpenAI Settings surface can't type.
            # extra["extra_body"] is forwarded verbatim into the request
            # JSON (a bare extra key would become a top-level create()
            # kwarg → TypeError). min_p cuts the low-probability tail where
            # inventions live. enable_thinking is per profile (off for
            # Qwen3-14B): with --reasoning-format deepseek the gateway returns
            # thoughts as ``reasoning_content`` deltas, which Pipecat and
            # every guard here ignore — only ``content`` reaches the
            # splitter, TTS, transcript and LLMContext history. Side calls
            # never send this kwarg and stay non-thinking.
            extra={
                "extra_body": {
                    "min_p": config.llm_min_p,
                    "chat_template_kwargs": {"enable_thinking": config.dm_thinking},
                },
            },
        ),
    )
