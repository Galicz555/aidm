"""Pipecat voice loop wiring.

Event-driven, not pipeline-shaped — Code as Story does NOT apply here.
The Pipeline object is a graph of processors that react to frames; the
list order is dataflow direction, not temporal execution.

**Phase 5 architecture.** Gemini Live is retired. The voice loop is now
a four-stage chain of independent processors:

    transport.input() → VAD → STT → context.user() → LLM → TTS → transport.output() → context.assistant()

Each provider (STT/LLM/TTS) is selected by ``config.mode`` through the
``aidm.providers`` factories. Public mode wires Groq/ElevenLabs Scribe
STT + Claude via OpenRouter + ElevenLabs Multilingual v2; private mode
wires Faster-Whisper Large-v3-Turbo + Gemma 4 26B-A4B via llama.cpp +
MOSS-TTS-Nano on CPU. Same factory surface; only the URLs change.

**Phase 6 split.** Pipeline construction and running are separated so
the WebRTC server can register per-pc_id control hooks (e.g. the user
aggregator's queue for late frame injection) between build and run.
``run_voice_loop`` chains both for the local CLI; ``aidm.server`` calls
them sequentially with a registry write in between.

We add Silero VAD client-side because none of the public providers do
endpointing for us — Groq STT is request/response, not streaming. The
VAD frames are what cut the audio buffer into utterances.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiohttp
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    OutputTransportMessageFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from aidm import frontmatter as _fm
from aidm import story_events
from aidm.audio import INPUT_SAMPLE_RATE
from aidm.budget import DailyBudgetLedger, SessionBudgetGuard
from aidm.character_splitter import CharacterStreamSplitter
from aidm.config import Config, load_config
from aidm.data_channel import DataChannelForwarder
from aidm.history_trimmer import HistoryTrimmer
from aidm.log import current_turn, set_session, set_turn
from aidm.observers import attach_turn_logging, build_latency_observer
from aidm.preflight import check_providers
from aidm.providers import make_llm, make_stt, make_tts, tts_output_sample_rate
from aidm.scene_sheet_injector import SceneSheetInjector
from aidm.scene_sheet_writer import SceneSheetWriter
from aidm.scene_state import SceneRegistry
from aidm.session import SessionContext, session_for_locale
from aidm.session_summary import SessionSummarizer, SummaryEventCallback
from aidm.text_debug import LLMOutputTracer, TextDebugLogger
from aidm.text_sink import TextSink, TurnActivity
from aidm.time_shown import TimeShownProcessor
from aidm.tool_router import ToolSubsetSelector
from aidm.turn_director import DirectorState
from aidm.turn_rollback import Restorable, TurnRollback
from aidm.vault import Vault
from aidm.vault_tools import (
    build_vault_summary,
    default_tools_schema,
    discover_in_scene_npcs,
    discover_pc_slug,
    has_character_sheet,
    wire_vault_tools,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

SinkQueue = asyncio.Queue["dict[str, object] | None"]


@dataclass
class TextPipelineHandle:
    """Per-session entry points for a text-mode pipeline.

    Mirrors ``PipelineHandle`` but without a transport. ``sink_queue``
    is where the ``TextSink`` parks data-channel payloads — the SSE
    response drains it. ``None`` on the queue is the close sentinel.
    """

    worker: PipelineWorker
    user_agg: FrameProcessor
    context: LLMContext
    sink_queue: SinkQueue
    tool_selector: ToolSubsetSelector
    scene_registry: SceneRegistry
    budget_guard: SessionBudgetGuard | None = None
    # What the player saw, for /history after a refresh. The LLMContext
    # can't serve that: the summarizer folds all but the last few turns.
    transcript: list[dict[str, str]] = field(default_factory=list)
    # Regenerate-last-reply support: the turn snapshot + the busy gate.
    rollback: TurnRollback | None = None
    activity: TurnActivity = field(default_factory=TurnActivity)


@dataclass
class PipelineHandle:
    """Per-session entry points used between ``build_pipeline`` and ``run_worker``.

    ``user_agg`` is the LLM-context user aggregator — pushing a
    ``TranscriptionFrame`` to it simulates the player saying something,
    which is what ``/api/session/{pc_id}/end`` needs to nudge the
    persona into writing the session log and saying goodbye. ``context``
    is the shared ``LLMContext`` (system prompt + turn history + tool
    definitions); the WebRTC server stashes a reference to it on peer
    disconnect so a brief reconnect can resume the conversation rather
    than starting from scratch.
    """

    worker: PipelineWorker
    user_agg: FrameProcessor
    context: LLMContext
    tool_selector: ToolSubsetSelector
    budget_guard: SessionBudgetGuard | None = None


async def run_voice_loop() -> None:
    """Local-mode entry point: one process, one session, local audio."""
    config = load_config()
    await check_providers(config)

    # Local CLI has no picker; campaign defaults to whatever VAULT_PATH
    # named (vault_path.name), run defaults to RUN_SLUG or ``default``.
    # Explicit failure beats writing the local session log to the wrong
    # spot if the operator changed vaults without updating RUN_SLUG.
    campaign_slug = config.vault_path.name
    run_slug = (config.run_slug or "default").strip()
    vault = Vault(config.vault_root / campaign_slug, run_slug=run_slug)
    run_title = _read_run_title_for_local(vault)
    campaign_title = _read_campaign_title_for_local(vault)
    session = session_for_locale(
        config.locale,
        campaign_slug=campaign_slug,
        campaign_title=campaign_title,
        run_slug=run_slug,
        run_title=run_title,
    )
    set_session(session.session_id)
    _log_startup(config, session)
    logger.info(
        "vault loaded: {} (campaign={}, run={})",
        vault.root,
        campaign_slug,
        run_slug,
    )

    # The HTTP TTS transport needs an aiohttp session bound to this event
    # loop; the WS transport ignores it. We open one unconditionally so
    # the call site stays the same regardless of transport.
    async with aiohttp.ClientSession() as http_session:
        transport = build_local_transport(config)
        handle = build_pipeline(transport, config, session, vault, http_session)
        await run_worker(handle, session)


def _oob_llm_client(config: Config) -> tuple[AsyncOpenAI, str]:
    """(client, model) for out-of-band completions (sheet writer, NPC minds).

    A plain non-streaming ``AsyncOpenAI`` on the same gateway / OpenRouter
    route as the production LLM — the streaming Pipecat service can't serve
    these one-shot side calls. Each caller builds its own instance (Pipecat
    binds services to event loops; a shared client would couple them).
    """
    from openai import AsyncOpenAI  # noqa: PLC0415 — keep startup cost off cold paths

    if config.mode == "private":
        base_url = config.local_llm_base_url
        api_key = "local-llama-cpp"  # gateway ignores it, OpenAI SDK refuses empty
        model = config.local_llm_model
    else:
        base_url = "https://openrouter.ai/api/v1"
        api_key = config.openrouter_api_key
        model = config.openrouter_model
    return AsyncOpenAI(base_url=base_url, api_key=api_key), model


def _build_sheet_writer(  # noqa: PLR0913 — wiring needs all 7 deps
    config: Config,
    context: LLMContext,
    registry: SceneRegistry,
    vault: Vault,
    locale: str,
    emitter: FrameProcessor,
    director: DirectorState | None = None,
) -> FrameProcessor:
    """Construct the end-of-turn ``SceneSheetWriter``, or its idle stand-in when disabled."""
    if not config.npc_failsafe_enabled:
        return _IdleStatus(emitter)
    client, model = _oob_llm_client(config)
    return SceneSheetWriter(
        context=context,
        registry=registry,
        vault=vault,
        client=client,
        model=model,
        locale=locale,
        timeout_s=config.npc_failsafe_timeout_s,
        emitter=emitter,
        director=director,
        verbatim_replies=config.dm_verbatim_replies,
        advance_judge=config.advance_judge and config.mode == "private",
        scene_end_judge=config.scene_end_judge and config.mode == "private",
        nightmare=config.mode == "private",
    )


def _is_end_of_turn(frame: Frame, direction: FrameDirection) -> bool:
    return direction == FrameDirection.DOWNSTREAM and isinstance(
        frame, LLMContextAssistantTimestampFrame
    )


class _IdleStatus(FrameProcessor):
    """Writer stand-in when ``AIDM_NPC_FAILSAFE_ENABLED`` is off.

    The web client gates input on ``turn.status: idle``, which only the
    writer emits; without it the UI would stay gated forever. Emits via
    the LLM (``emitter``) exactly like the writer, so the event reaches
    TextSink / transport.output upstream of this slot.
    """

    def __init__(self, emitter: FrameProcessor) -> None:
        super().__init__()
        self._emitter = emitter

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if _is_end_of_turn(frame, direction):
            await self._emitter.push_frame(
                OutputTransportMessageFrame(message={"type": "turn.status", "phase": "idle"}),
                FrameDirection.DOWNSTREAM,
            )
        await self.push_frame(frame, direction)


class _TurnCounter(FrameProcessor):
    """Advance the log turn id per player turn (text mode).

    Pipecat's ``on_turn_started`` keys off bot-speaking frames, which a
    text pipeline never emits. Sits right after ``user_agg``: each
    downstream ``LLMContextFrame`` there is one player turn going to the
    LLM (tool re-runs are pushed upstream from the assistant side and
    never pass this slot).
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            set_turn(current_turn() + 1)
            logger.info("turn started")
        await self.push_frame(frame, direction)


class _TranscriptRecorder(FrameProcessor):
    """Append each landed DM reply to the session's UI transcript.

    Directly after ``aggregator.assistant()``: the end-of-turn frame
    arrives right after the reply was added to the context, before the
    trimmer / summarizer rewrite it. Player lines are recorded by the
    text server at the HTTP boundary.
    """

    def __init__(self, context: LLMContext, transcript: list[dict[str, str]]) -> None:
        super().__init__()
        self._context = context
        self._transcript = transcript

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if _is_end_of_turn(frame, direction):
            # push_aggregation appended the reply as the latest assistant
            # message, plain-string content, just before this frame.
            for message in reversed(self._context.messages):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        self._transcript.append({"role": "assistant", "text": content})
                    break
        await self.push_frame(frame, direction)


class _TurnSettled(FrameProcessor):
    """Last in the text tail: an end-of-turn frame here has passed the writer
    and the summarizer processors, so that response is done (``TurnActivity``)."""

    def __init__(self, activity: TurnActivity) -> None:
        super().__init__()
        self._activity = activity

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if _is_end_of_turn(frame, direction):
            self._activity.response_landed()


class _EndAfterFarewell(FrameProcessor):
    """Close the pipeline once the /end farewell turn has fully landed.

    Last in the tail, so the farewell's sheet patches and summary finish
    first. A text leg with a tool call still pending is not the end: its
    result re-runs the LLM for the actual farewell.
    """

    def __init__(self, tool_selector: ToolSubsetSelector, assistant: FrameProcessor) -> None:
        super().__init__()
        self._tool_selector = tool_selector
        self._assistant = assistant

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if (
            _is_end_of_turn(frame, direction)
            and self._tool_selector.end_session
            and not getattr(self._assistant, "has_function_calls_in_progress", False)
        ):
            logger.info("farewell turn landed — ending session")
            await self.push_frame(EndTaskFrame(reason="end-session"), FrameDirection.UPSTREAM)


@dataclass
class _SharedParts:
    """Setup common to both voice and text pipelines.

    Everything here is independent of the transport-specific head (STT +
    VAD + transport.input vs. nothing) and tail (TTS + transport.output
    vs. TextSink). The two ``build_*_pipeline`` callers assemble their
    processor list from these parts plus whatever they need around them.
    """

    llm: FrameProcessor
    context: LLMContext
    aggregator: LLMContextAggregatorPair
    user_agg: FrameProcessor
    scene_registry: SceneRegistry
    director: DirectorState
    tool_selector: ToolSubsetSelector
    budget_guard: SessionBudgetGuard | None
    sheet_writer: FrameProcessor
    sheet_injector: SceneSheetInjector
    time_shown: TimeShownProcessor
    # Vault tool handlers' session memory (sheets inlined into grep results).
    inlined_sheets: set[str]


def _build_shared(
    config: Config,
    session: SessionContext,
    vault: Vault,
    *,
    context: LLMContext | None,
    budget_ledger: DailyBudgetLedger | None,
) -> _SharedParts:
    """Build the LLM + context + scene registry + aggregator stack both pipelines need.

    ``context`` is optional. The default builds a fresh ``LLMContext``;
    pass an existing one to resume a session whose previous transport
    closed (browser reconnect within the grace window). Tools are
    re-attached either way — the LLM adapter reads the schema off the
    context every invocation, so it's a no-op on a resumed context.
    """
    phase = "play" if has_character_sheet(vault) else "session-zero"
    vault_summary = build_vault_summary(vault, locale=session.locale, mode=config.mode)
    llm = make_llm(config, session, phase=phase, vault_summary=vault_summary)

    if context is None:
        context = LLMContext(messages=[])
    scene_registry = SceneRegistry()
    scene_registry.set_pc(discover_pc_slug(vault))
    # Scene presence persists on the NPC state files (in_scene: true/false,
    # writer-stamped); read it back so fresh sessions and grace reconnects
    # resume the scene without waiting on npc_scene_enter calls.
    for slug in discover_in_scene_npcs(vault):
        scene_registry.enter(slug)
    # Turn-director channel: the writer measures (repetition, repeated
    # input, goal fixation) at end-of-turn; the injector renders the
    # findings at the next prompt's tail. Deterministic by design.
    director = DirectorState()
    # Private-mode guarded retry: a reply draft reusing a director-banned
    # phrase is regenerated once before anything streams out (the prompt
    # note alone measurably doesn't suppress tics — replay-005). Duck-
    # typed: the public OpenRouter service has no such hook and needs none.
    set_banned_provider = getattr(llm, "set_banned_phrases_provider", None)
    if callable(set_banned_provider):
        set_banned_provider(lambda: director.banned_phrases)
    set_directives_provider = getattr(llm, "set_event_directives_provider", None)
    if callable(set_directives_provider):
        set_directives_provider(lambda: director.event_directives)
    set_para_provider = getattr(llm, "set_para_banned_provider", None)
    if callable(set_para_provider):
        set_para_provider(lambda: director.paragraph_banned)
    set_stalled_provider = getattr(llm, "set_stalled_provider", None)
    if callable(set_stalled_provider) and config.advance_judge:
        set_stalled_provider(lambda: director.reply_stalled)
    tool_selector = ToolSubsetSelector(vault=vault)
    # The /end farewell turn runs unguarded (no advance/replay/tic retries).
    set_end_session_provider = getattr(llm, "set_end_session_provider", None)
    if callable(set_end_session_provider):
        set_end_session_provider(lambda: tool_selector.end_session)
    inlined_sheets = wire_vault_tools(
        context, llm, vault, scene_registry=scene_registry, locale=session.locale
    )
    aggregator = LLMContextAggregatorPair(context)
    user_agg = aggregator.user()

    minds_client, minds_model = _oob_llm_client(config) if config.npc_minds_enabled else (None, "")
    sheet_injector = SceneSheetInjector(
        scene_registry,
        session.locale,
        vault,
        director=director,
        agentic_recon=config.agentic_recon,
        contract_tail=config.mode != "public",
        minds_client=minds_client,
        minds_model=minds_model,
        minds_timeout_s=config.npc_failsafe_timeout_s,
    )
    # Seeded from disk so an advance on the session's first engine pass
    # is still shown (the director's clock_line starts empty).
    clock = story_events.read_clock(vault)
    time_shown = TimeShownProcessor(
        director, session.locale, initial_clock_line=clock.line() if clock else ""
    )
    return _SharedParts(
        llm=llm,
        context=context,
        aggregator=aggregator,
        user_agg=user_agg,
        scene_registry=scene_registry,
        director=director,
        tool_selector=tool_selector,
        budget_guard=_maybe_make_budget_guard(config, session, budget_ledger),
        sheet_writer=_build_sheet_writer(
            config, context, scene_registry, vault, session.locale, emitter=llm, director=director
        ),
        sheet_injector=sheet_injector,
        time_shown=time_shown,
        inlined_sheets=inlined_sheets,
    )


def _build_summarizer(
    parts: _SharedParts,
    config: Config,
    session: SessionContext,
    vault: Vault,
    *,
    on_event: SummaryEventCallback | None = None,
) -> SessionSummarizer:
    """The summarizer both pipelines use; it sets ``director.open_threads``."""
    return SessionSummarizer(
        context=parts.context,
        config=config,
        session=session,
        vault=vault,
        on_event=on_event,
        director=parts.director,
    )


# Room kept free under the window for the next user message, the turn
# contract and the reply itself.
_PROMPT_RESERVE_TOKENS = 3072


def _history_trimmer(parts: _SharedParts, config: Config) -> HistoryTrimmer:
    """Turn-count trimmer, plus a prompt-size budget in private mode.

    The budget counts what the request carries outside ``context.messages``
    too: the persona (system_instruction) and the full tool schema.
    """
    if config.mode != "private":
        return HistoryTrimmer(parts.context)
    persona = getattr(getattr(parts.llm, "_settings", None), "system_instruction", None) or ""
    tools = [tool.to_default_dict() for tool in default_tools_schema().standard_tools]
    return HistoryTrimmer(
        parts.context,
        token_budget=min(config.prompt_token_budget, config.llm_ctx_size - _PROMPT_RESERVE_TOKENS),
        fixed_chars=len(str(persona)) + len(json.dumps(tools)),
    )


def _post_turn_tail(
    parts: _SharedParts, summarizer: SessionSummarizer, config: Config
) -> list[FrameProcessor]:
    """Processors after ``aggregator.assistant()`` in both pipelines.

    Trimmer (cheap backstop), then the sheet writer, which blocks the
    end-of-turn frame and ungates the UI with ``turn.status: idle``,
    then the summarizer, so a barge-in cancelling the slow summary
    call can never cost that turn's sheet patches. The farewell closer
    goes last so an /end session only stops after all of that.
    """
    return [
        _history_trimmer(parts, config),
        parts.sheet_writer,
        summarizer,
        _EndAfterFarewell(parts.tool_selector, parts.aggregator.assistant()),
    ]


def build_pipeline(  # noqa: PLR0913 — transport + config + session + vault + http + 2 kwargs
    transport: BaseTransport,
    config: Config,
    session: SessionContext,
    vault: Vault,
    http_session: aiohttp.ClientSession,
    *,
    context: LLMContext | None = None,
    budget_ledger: DailyBudgetLedger | None = None,
) -> PipelineHandle:
    """Construct the per-session voice pipeline graph and return its control handle.

    Transport-agnostic on purpose. ``run_voice_loop`` calls this with a
    ``LocalAudioTransport``; the WebRTC server calls it with a
    ``SmallWebRTCTransport``. The pipeline graph is identical either way.
    """
    stt = make_stt(config, session, http_session=http_session, vault=vault)
    tts = make_tts(config, session, http_session=http_session)
    parts = _build_shared(config, session, vault, context=context, budget_ledger=budget_ledger)

    # DataChannelForwarder sits between STT and user_agg to mirror the
    # final TranscriptionFrame as a transcript.user event on the data
    # channel. Assistant-side events (transcript.segment per speaker,
    # transcript.bot_end) are emitted by CharacterStreamSplitter further
    # downstream. Local-mode transports ignore the emitted
    # OutputTransportMessageFrames; WebRTC mode ships them to the
    # browser's data channel.
    processors: list[FrameProcessor] = [
        transport.input(),
        VADProcessor(vad_analyzer=SileroVADAnalyzer()),
        stt,
        DataChannelForwarder(name="forwarder.user"),
        parts.user_agg,
        parts.tool_selector,
        parts.sheet_injector,
        parts.llm,
        parts.time_shown,
        CharacterStreamSplitter(),
        tts,
        transport.output(),
        parts.aggregator.assistant(),
        *_post_turn_tail(parts, _build_summarizer(parts, config, session, vault), config),
    ]
    if parts.budget_guard is not None:
        # Sit between user_agg and ToolSubsetSelector so MetricsFrames
        # (system frames, propagate both ways) are observed, and a tripped
        # TTSSpeakFrame pushed downstream still reaches the TTS.
        processors.insert(5, parts.budget_guard)
    pipeline = Pipeline(processors)

    worker = PipelineWorker(
        pipeline,
        conversation_id=session.session_id,
        observers=[build_latency_observer()],
    )
    attach_turn_logging(worker.turn_tracking_observer)

    return PipelineHandle(
        worker=worker,
        user_agg=parts.user_agg,
        context=parts.context,
        tool_selector=parts.tool_selector,
        budget_guard=parts.budget_guard,
    )


def build_text_pipeline(
    config: Config,
    session: SessionContext,
    vault: Vault,
    *,
    context: LLMContext | None = None,
    budget_ledger: DailyBudgetLedger | None = None,
) -> TextPipelineHandle:
    """Construct the per-session text-mode pipeline graph.

    Same LLM + tool wiring as the voice path; no STT, VAD, TTS or
    transport. Text input enters by ``user_agg.queue_frame`` from the
    HTTP turn endpoint; bot output exits via the ``TextSink`` queue to
    the SSE stream.

    Mode (public/private) selects the LLM backend exactly like the
    voice path. Text mode never invokes STT or TTS providers, so
    private mode here only needs a working local LLM gateway.
    """
    parts = _build_shared(config, session, vault, context=context, budget_ledger=budget_ledger)
    sink_queue: SinkQueue = asyncio.Queue()

    async def _push_debug(payload: dict[str, object]) -> None:
        sink_queue.put_nowait(payload)

    summarizer = _build_summarizer(parts, config, session, vault, on_event=_push_debug)
    transcript: list[dict[str, str]] = []
    activity = TurnActivity(emit=sink_queue.put_nowait)
    rollback = TurnRollback(
        context=parts.context,
        transcript=transcript,
        registry=parts.scene_registry,
        director=parts.director,
        vault=vault,
        processors=[
            p
            for p in (parts.time_shown, parts.sheet_writer, summarizer)
            if isinstance(p, Restorable)
        ],
        summarizer=summarizer,
        inlined_sheets=parts.inlined_sheets,
    )

    # TextDebugLogger sits between SceneSheetInjector and llm so it
    # observes the LLMContextFrame (full prompt, with the scene sheets
    # already injected) → debug.context. The reply frames flow FROM the
    # llm, so debug.bot is emitted by LLMOutputTracer just after it.
    processors: list[FrameProcessor] = [
        parts.user_agg,
        _TurnCounter(),
        parts.tool_selector,
        parts.sheet_injector,
        TextDebugLogger(sink_queue),
        parts.llm,
        parts.time_shown,
        LLMOutputTracer(sink_queue),
        CharacterStreamSplitter(),
        TextSink(sink_queue, activity=activity),
        parts.aggregator.assistant(),
        _TranscriptRecorder(parts.context, transcript),
        *_post_turn_tail(parts, summarizer, config),
        _TurnSettled(activity),
    ]
    if parts.budget_guard is not None:
        # After user_agg so it sees the LLM context flow; no TTS in text
        # mode so the goodbye TTSSpeakFrame is harmless (TextSink ignores it).
        processors.insert(1, parts.budget_guard)
    pipeline = Pipeline(processors)

    # Pipecat's idle watchdog cancels the worker after 5 minutes without
    # BotSpeakingFrame / UserSpeakingFrame — neither of which text mode
    # ever emits (no TTS, no VAD). Setting the timeout to None disables
    # the watchdog so a player reading the DM's reply, thinking, or
    # tabbing away does not silently lose the pipeline.
    worker = PipelineWorker(
        pipeline,
        conversation_id=session.session_id,
        observers=[build_latency_observer()],
        idle_timeout_secs=None,
    )
    attach_turn_logging(worker.turn_tracking_observer)

    return TextPipelineHandle(
        worker=worker,
        user_agg=parts.user_agg,
        context=parts.context,
        sink_queue=sink_queue,
        tool_selector=parts.tool_selector,
        scene_registry=parts.scene_registry,
        budget_guard=parts.budget_guard,
        transcript=transcript,
        rollback=rollback,
        activity=activity,
    )


def _maybe_make_budget_guard(
    config: Config,
    session: SessionContext,
    ledger: DailyBudgetLedger | None,
) -> SessionBudgetGuard | None:
    """Build a guard only in public mode where caps are non-zero.

    Private mode runs on local hardware — no cloud cost — so the guard
    adds nothing. A zero cap also disables the guard so local dev without
    explicit configuration stays untouched.
    """
    if config.mode != "public":
        return None
    if config.session_token_cap <= 0 and config.session_tts_char_cap <= 0:
        return None
    return SessionBudgetGuard(
        token_cap=config.session_token_cap,
        tts_char_cap=config.session_tts_char_cap,
        locale=session.locale,
        ledger=ledger,
        session_id=session.session_id,
    )


async def run_worker(handle: PipelineHandle | TextPipelineHandle, session: SessionContext) -> None:
    """Block on the pipeline worker until it exits or is cancelled."""
    runner = WorkerRunner()
    logger.info("pipeline ready (session={})", session.session_id)
    try:
        await runner.run(handle.worker)
    except asyncio.CancelledError:
        logger.info("pipeline cancelled (session={})", session.session_id)


def build_local_transport(config: Config) -> LocalAudioTransport:
    """Build the PyAudio mic+speaker transport. Sample rates pinned by aidm.audio.

    PipeWire handles BT-mic resampling transparently when the PyAudio
    'pipewire' device is selected (AUDIO_INPUT_DEVICE in .env).
    """
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=tts_output_sample_rate(config),
            input_device_index=config.audio_input_device,
            output_device_index=config.audio_output_device,
        )
    )


def _read_run_title_for_local(vault: Vault) -> str:
    """Best-effort run-title read for the local CLI startup log line."""
    run_root = vault.run_root
    if run_root is None:
        return ""
    run_md = run_root / "run.md"
    if not run_md.is_file():
        return ""
    fm = _fm.parse(run_md.read_text(encoding="utf-8"))
    return str(fm.get("title") or "").strip()


def _read_campaign_title_for_local(vault: Vault) -> str:
    """Best-effort campaign-title read for the local CLI startup."""
    campaign_md = vault.root / "campaign.md"
    if not campaign_md.is_file():
        return ""
    fm = _fm.parse(campaign_md.read_text(encoding="utf-8"))
    return str(fm.get("title") or "").strip()


def _log_startup(config: Config, session: SessionContext) -> None:
    logger.info(
        "starting voice loop: mode={} transport={} locale={} stt={} llm_model={} tts={}/{} "
        "audio_in_device={} audio_out_device={}",
        config.mode,
        config.transport,
        session.locale,
        _stt_label(config),
        _llm_label(config),
        _tts_transport_label(config),
        _tts_model_label(config),
        config.audio_input_device if config.audio_input_device is not None else "default",
        config.audio_output_device if config.audio_output_device is not None else "default",
    )


def _stt_label(config: Config) -> str:
    if config.mode == "private":
        return f"whisper-local:{config.local_stt_base_url}"
    if config.stt_provider == "elevenlabs":
        return f"elevenlabs:{config.elevenlabs_stt_model}"
    return f"groq:{config.groq_stt_model}"


def _llm_label(config: Config) -> str:
    if config.mode == "private":
        return f"local:{config.local_llm_model}"
    return config.openrouter_model


def _tts_transport_label(config: Config) -> str:
    return "ws" if config.mode == "private" else config.elevenlabs_tts_transport


def _tts_model_label(config: Config) -> str:
    return "xtts_v2" if config.mode == "private" else config.elevenlabs_model
