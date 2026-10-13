"""FastAPI HTTP entry point for browser-mode AIDM in text-chat form.

Started by ``aidm/__main__.py`` when ``AIDM_TRANSPORT=text``. Same
session model and reconnect grace as ``aidm.server`` (WebRTC), but the
client speaks JSON over HTTP instead of audio over WebRTC.

Routes:

* ``POST   /api/text/session``             — start (or resume) a session.
* ``POST   /api/text/session/{sid}/turn``  — player input, one line of text.
* ``GET    /api/text/session/{sid}/stream``— SSE: streams DM token deltas,
                                            scene + character side-channel events.
* ``POST   /api/text/session/{sid}/suggest-player-turn`` — generate an
                                            in-character action from the PC
                                            sheet + scene state (one-shot
                                            LLM call; returns ``{text}``).
* ``POST   /api/text/session/{sid}/regenerate`` — discard the last DM
                                            reply (archived under
                                            ``runs/<run>/discarded/``), roll the
                                            world back to before that turn and
                                            re-run the same player line.
* ``POST   /api/text/session/{sid}/end``   — polite end (persona writes
                                            session log + says goodbye).
* ``GET    /api/vault/characters/{slug}``  — character sheet markdown.
* ``GET    /health``                       — liveness.

One process can serve many concurrent sessions; each session owns a
pipeline task, a ``user_agg`` for input injection, and an
``asyncio.Queue`` the SSE stream drains.

Reconnect grace (docs/session-lifecycle.md) keeps the *live* pipeline:
a text pipeline has no transport to lose, and the client only accepts
``resumed_from: "live"`` on reconnect. A session with no SSE stream
attached for ``Config.text_reconnect_grace_s`` (default 30 min — a phone
lock outlasts WebRTC's 5 s every time) is torn down; a re-attach inside the
window cancels that.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from loguru import logger
from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.task import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from aidm import frontmatter as _fm
from aidm import roleplay_prompt
from aidm.budget import DailyBudgetLedger, flush_to_ledger_async, make_ledger
from aidm.config import Config, load_config
from aidm.log import set_session
from aidm.pipeline import SinkQueue, build_text_pipeline, run_worker
from aidm.player_suggest import SuggestionError, suggest_player_action
from aidm.preflight import check_providers
from aidm.server_utils import (
    build_session_vault,
    from_timestamp,
    list_campaigns_on_disk,
    list_runs_on_disk,
    read_campaign_title,
    read_run_title,
    resolve_campaign_dir,
    run_summary,
    seed_run_from_template,
)
from aidm.session import SessionContext, session_for_locale
from aidm.session_resume import (
    ResumeSource,
    build_campaign_grouping,
    list_resumable,
    load_resume_message,
)
from aidm.sessions_registry import END_PHRASES, RECONNECT_GRACE_SECONDS
from aidm.text_sink import TurnActivity
from aidm.tool_router import ToolSubsetSelector
from aidm.turn_rollback import TurnRollback
from aidm.vault import Vault, VaultConflictError, VaultPathError, slugify
from aidm.web_security import (
    build_limiter,
    configure_cors,
    enforce_daily_budget,
    install_rate_limit_handler,
    make_cf_access_verifier,
)

_TOO_MANY_SESSIONS_HU = (
    "Az archívum jelenleg foglalt — túl sok krónika fut egyszerre. Próbáld újra pár perc múlva."
)


@dataclass
class TextSessionEndpoint:
    """Per-session handle the HTTP layer uses to drive a running pipeline."""

    user_agg: FrameProcessor
    sink_queue: SinkQueue
    locale: str
    session_id: str
    # Live reference to the running pipeline's LLMContext (suggest-player-turn).
    context: object
    tool_selector: ToolSubsetSelector
    vault: Vault
    scene_registry: object
    worker: PipelineWorker
    # Player-visible turns for /history; the summarizer folds the context.
    transcript: list[dict[str, str]]
    streams: int = 0
    # Seconds the pipeline stays live with no SSE stream (config).
    grace_s: float = RECONNECT_GRACE_SECONDS
    grace_task: asyncio.Task[None] | None = field(default=None, repr=False)
    # Latest turn's snapshot for /regenerate; None = regenerate unsupported.
    rollback: TurnRollback | None = None
    # A turn is in flight (set at /turn, cleared by TextSink on idle).
    activity: TurnActivity = field(default_factory=TurnActivity)
    # Serialises /turn, /regenerate and /end: each checks busy, snapshots
    # or restores, and queues a turn as one step.
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    if config.transport != "text":
        raise RuntimeError(
            f"aidm.text_server requires AIDM_TRANSPORT=text, got {config.transport!r}"
        )
    await check_providers(config)

    vault = Vault(config.vault_path)
    pipeline_tasks: set[asyncio.Task[None]] = set()
    session_endpoints: dict[str, TextSessionEndpoint] = {}
    ledger: DailyBudgetLedger | None = None
    if config.mode == "public" and config.daily_usd_cap > 0:
        ledger = make_ledger(config.budget_db_path)
        logger.info(
            "budget ledger ready: db={} daily_cap=${:.2f}",
            config.budget_db_path,
            config.daily_usd_cap,
        )

    app.state.config = config
    app.state.vault = vault
    app.state.pipeline_tasks = pipeline_tasks
    app.state.session_endpoints = session_endpoints
    app.state.budget_ledger = ledger

    logger.info(
        "aidm text server ready: mode={} locale_default={} vault={}",
        config.mode,
        config.locale,
        vault.root,
    )
    try:
        yield
    finally:
        for task in pipeline_tasks:
            task.cancel()
        if pipeline_tasks:
            await asyncio.gather(*pipeline_tasks, return_exceptions=True)


def create_app() -> FastAPI:  # noqa: PLR0915 — route registration inline is fine
    """Build the FastAPI app. Called by ``__main__`` and tests."""
    config = load_config()
    app = FastAPI(lifespan=_lifespan)
    configure_cors(app, config)
    limiter = build_limiter()
    install_rate_limit_handler(app, limiter)
    cf_access = make_cf_access_verifier(config)
    rate_limit_session = config.rate_limit_session

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/info")
    async def info() -> dict[str, str]:
        # Cheap unauthenticated probe the web client calls once on load
        # to render the mode badge. Persona variants are gone — the DM
        # tunes itself from tone.md + beat.md in the active campaign / run.
        config: Config = app.state.config
        return {
            "mode": config.mode,
            "transport": config.transport,
        }

    @app.get("/api/sessions")
    async def sessions(
        run: str = Query(""), _auth: None = Depends(cf_access)
    ) -> dict[str, list[dict[str, object]]]:
        # Legacy flat listing kept for backward compatibility. Now
        # run-scoped: the listing covers the active run's sessions/
        # subtree only (other runs of the same campaign don't bleed in).
        # Falls back to config.run_slug (env) when ?run= is absent.
        cfg: Config = app.state.config
        active_run = (run or cfg.run_slug or "").strip()
        if not active_run:
            return {"sessions": []}
        try:
            vault = Vault(cfg.vault_path, run_slug=active_run)
        except VaultPathError:
            return {"sessions": []}
        entries = list_resumable(vault)
        return {"sessions": [_entry_dict(e) for e in entries]}

    @app.get("/api/campaigns")
    async def campaigns(_auth: None = Depends(cf_access)) -> list[dict[str, str]]:
        """Campaign inventory for the run picker.

        Mirrors the contract on the voice server: a flat list of
        ``{slug, title}`` for the dropdown. Session listings live on
        ``/api/sessions/grouping`` — different concern, different path.
        """
        cfg: Config = app.state.config
        return list_campaigns_on_disk(cfg.vault_root)

    @app.get("/api/campaigns/{slug}/runs")
    async def list_runs(slug: str, _auth: None = Depends(cf_access)) -> list[dict[str, object]]:
        cfg: Config = app.state.config
        campaign_dir = resolve_campaign_dir(cfg.vault_root, slug)
        return list_runs_on_disk(campaign_dir)

    @app.post("/api/campaigns/{slug}/runs", status_code=201)
    async def create_run(
        slug: str, request: Request, _auth: None = Depends(cf_access)
    ) -> dict[str, object]:
        cfg: Config = app.state.config
        campaign_dir = resolve_campaign_dir(cfg.vault_root, slug)
        payload = await _safe_json(request)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="title is required")
        title = str(payload.get("title", "")).strip()
        if not title:
            raise HTTPException(status_code=400, detail="title is required")
        status_value = str(payload.get("status", "active")).strip() or "active"
        if status_value not in {"active", "completed", "archived"}:
            raise HTTPException(
                status_code=400,
                detail="status must be one of active, completed, archived",
            )
        run_slug = slugify(title)
        if not run_slug:
            raise HTTPException(status_code=400, detail="title produces an empty slug")
        runs_root = campaign_dir / "runs"
        run_dir = runs_root / run_slug
        if run_dir.exists():
            raise HTTPException(status_code=409, detail=f"run already exists: {run_slug}")
        run_dir.mkdir(parents=True, exist_ok=False)
        for sub in ("characters", "arcs", "sessions", "notes"):
            (run_dir / sub).mkdir(exist_ok=True)
        seed_run_from_template(run_dir)
        fm = {
            "type": "run",
            "title": title,
            "status": status_value,
            "created": datetime.now(tz=UTC).date().isoformat(),
            "parent_campaign": slug,
        }
        (run_dir / "run.md").write_text(
            _fm.serialize(fm, f"# {title}\n"),
            encoding="utf-8",
        )
        logger.info("run created: campaign={} slug={} title={!r}", slug, run_slug, title)
        return run_summary(run_dir)

    @app.get("/api/sessions/grouping")
    async def sessions_grouping(
        campaign: str = Query(""),
        run: str = Query(""),
        _auth: None = Depends(cf_access),
    ) -> dict[str, object]:
        """Session-resume picker: campaigns + their sessions + orphans.

        Previously served under ``/api/campaigns``; renamed because the
        picker semantics ate that URL. Same response shape — only the
        path moved. ``?campaign=`` picks which campaign vault to read,
        ``?run=`` selects which run's sessions to surface within it.
        Both default to the env-configured fallback if omitted.
        """
        cfg: Config = app.state.config
        active_campaign = (campaign or cfg.vault_path.name).strip()
        active_run = (run or cfg.run_slug or "").strip()
        try:
            vault = Vault(
                resolve_campaign_dir(cfg.vault_root, active_campaign),
                run_slug=active_run or None,
            )
        except VaultPathError:
            vault = app.state.vault  # canon-only fallback (no sessions surfaced)
        grouping = build_campaign_grouping(vault)
        return {
            "campaigns": [
                {
                    "slug": c.slug,
                    "title": c.title,
                    "status": c.status,
                    "tone": c.tone,
                    "lang": c.lang,
                    "path": c.path,
                    "sessions": [_entry_dict(s) for s in c.sessions],
                }
                for c in grouping.campaigns
            ],
            "orphans": [_entry_dict(e) for e in grouping.orphans],
        }

    @app.post("/api/text/session")
    @limiter.limit(rate_limit_session)
    async def start_session(request: Request, _auth: None = Depends(cf_access)) -> dict[str, str]:
        cfg: Config = app.state.config
        payload = await _safe_json(request)
        config: Config = cfg
        locale = _locale_from_payload(payload, config.locale)

        # Hot-reattach first, before the budget and session caps — a
        # refresh or network blip rejoins a pipeline that already counts
        # against them. Carries only the session_id, no run_id; the
        # browser repopulates the transcript via /history.
        requested_sid = _session_id_from_payload(payload)
        session_endpoints: dict[str, TextSessionEndpoint] = app.state.session_endpoints
        if requested_sid and requested_sid in session_endpoints:
            live = session_endpoints[requested_sid]
            if live.streams == 0:
                _arm_grace(live)  # a fresh window to reopen the SSE stream
            logger.info(
                "text session hot-reattach: session={} locale={}",
                live.session_id,
                live.locale,
            )
            return {
                "session_id": live.session_id,
                "locale": live.locale,
                "resumed_from": "live",
            }

        enforce_daily_budget(app.state.budget_ledger, cfg.daily_usd_cap)
        active = sum(1 for t in app.state.pipeline_tasks if not t.done())
        if cfg.max_concurrent_sessions > 0 and active >= cfg.max_concurrent_sessions:
            raise HTTPException(status_code=503, detail=_TOO_MANY_SESSIONS_HU)

        # Fresh start — validate (campaign, run) and build the session.
        campaign_slug = _campaign_slug_from_payload(payload, config.vault_path.name)
        run_slug = _run_slug_from_payload(payload, config.run_slug)
        if not campaign_slug:
            raise HTTPException(status_code=400, detail="campaign_id is required")
        if not run_slug:
            raise HTTPException(
                status_code=400,
                detail="run_id is required (no RUN_SLUG fallback configured)",
            )
        campaign_dir = resolve_campaign_dir(config.vault_root, campaign_slug)
        try:
            vault = Vault(campaign_dir, run_slug=run_slug)
        except VaultPathError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        run_title = read_run_title(vault)
        campaign_title = read_campaign_title(vault)
        try:
            session = session_for_locale(
                locale,
                campaign_slug=campaign_slug,
                campaign_title=campaign_title,
                run_slug=run_slug,
                run_title=run_title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        resume_request = _resume_from_payload(payload)
        ctx_kwargs: dict[str, object] = {}
        if resume_request is not None:
            source, entry_id = resume_request
            try:
                resume_body = load_resume_message(vault, source, entry_id, locale)
            except VaultPathError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            # Pre-seed the new LLMContext with the resume body as a
            # system message. The persona prompt still wraps it
            # (system_instruction on the LLM service); this is the
            # second system message and provides the campaign
            # memory.
            ctx_kwargs["context"] = _build_resume_context(resume_body)
            resumed_from = f"{source}:{entry_id}"
            logger.info(
                "text session start with resume: session={} from={} locale={}",
                session.session_id,
                resumed_from,
                locale,
            )
        else:
            resumed_from = ""
            logger.info(
                "text session start: session={} locale={}",
                session.session_id,
                locale,
            )

        await _spawn_pipeline(app, session, locale, ctx_kwargs)
        return {
            "session_id": session.session_id,
            "locale": locale,
            "resumed_from": resumed_from,
        }

    @app.get("/api/text/session/{session_id}/history")
    async def history(session_id: str, _auth: None = Depends(cf_access)) -> dict[str, object]:
        # Page refreshed mid-session: the browser stashed session_id in
        # localStorage, the start-session route accepted the hot-reattach,
        # now we hand back the session's own transcript (not the
        # LLMContext, which the summarizer folds) so the page repaints
        # exactly where the player left off.
        endpoint = _require_endpoint(app, session_id)
        return {
            "session_id": endpoint.session_id,
            "locale": endpoint.locale,
            "turns": list(endpoint.transcript),
        }

    @app.post("/api/text/session/{session_id}/turn", status_code=202)
    async def turn(
        session_id: str, request: Request, _auth: None = Depends(cf_access)
    ) -> dict[str, str]:
        endpoint = _require_endpoint(app, session_id)
        payload = await _safe_json(request)
        text = ""
        if isinstance(payload, dict):
            value = payload.get("text")
            if isinstance(value, str):
                text = value.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        async with endpoint.turn_lock:
            # One turn at a time server-side too: a second tab or API client
            # posting mid-turn would snapshot half-written state and reset
            # the journal the running turn is writing into.
            if endpoint.activity.busy:
                raise HTTPException(status_code=409, detail="a turn is still in flight")
            endpoint.activity.begin()
            if endpoint.rollback is not None and not endpoint.tool_selector.end_session:
                try:
                    # capture() first waits for a summary still running from the
                    # previous turn: it splices the context and appends to the
                    # notes file, and landing after the snapshot it would be
                    # rolled back with this turn — or survive a restore of it.
                    # A turn can wait up to the summary timeout here.
                    await endpoint.rollback.capture(text)
                except Exception:
                    # Fail open: the turn plays, it just can't be regenerated.
                    logger.exception("turn snapshot failed — regenerate off for this turn")
                    endpoint.rollback.discard()
            await _queue_player_turn(endpoint, text)
        return {"status": "accepted"}

    @app.post("/api/text/session/{session_id}/regenerate", status_code=202)
    async def regenerate(session_id: str, _auth: None = Depends(cf_access)) -> dict[str, str]:
        endpoint = _require_endpoint(app, session_id)
        rollback = endpoint.rollback
        async with endpoint.turn_lock:
            if endpoint.activity.busy:
                raise HTTPException(status_code=409, detail="a turn is still in flight")
            if rollback is None or rollback.snapshot is None or endpoint.tool_selector.end_session:
                raise HTTPException(status_code=409, detail="no turn to regenerate")
            snapshot = rollback.snapshot
            endpoint.activity.begin()
            queued = False
            try:
                await rollback.settle()
                if rollback.snapshot is not snapshot:  # a sheet/beat PUT dropped it meanwhile
                    raise HTTPException(status_code=409, detail="no turn to regenerate")
                turn_number, reply = rollback.turn_number(), rollback.discarded_reply()
                _restore_or_raise(rollback)
                archived = _archive_discarded(
                    endpoint.vault, turn_number, snapshot.user_text, reply
                )
                logger.info(
                    "regenerate: turn {} rolled back, reply archived to {}", turn_number, archived
                )
                await _queue_player_turn(endpoint, snapshot.user_text)
                queued = True
            finally:
                if not queued:
                    endpoint.activity.release()
        return {"status": "accepted"}

    @app.post("/api/text/session/{session_id}/suggest-player-turn")
    async def suggest_player_turn(
        session_id: str, _auth: None = Depends(cf_access)
    ) -> dict[str, str]:
        endpoint = _require_endpoint(app, session_id)
        cfg: Config = app.state.config
        try:
            text = await suggest_player_action(
                cfg,
                endpoint.vault,
                endpoint.scene_registry,  # type: ignore[arg-type]
                endpoint.context,
                endpoint.locale,
            )
        except SuggestionError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        return {"text": text}

    @app.get("/api/text/session/{session_id}/stream")
    async def stream(session_id: str, _auth: None = Depends(cf_access)) -> StreamingResponse:
        endpoint = _require_endpoint(app, session_id)
        queue = endpoint.sink_queue
        session_endpoints: dict[str, TextSessionEndpoint] = app.state.session_endpoints

        async def event_source() -> AsyncIterator[bytes]:
            endpoint.streams += 1
            _disarm_grace(endpoint)
            try:
                while True:
                    payload = await queue.get()
                    if payload is None:
                        break
                    data = json.dumps(payload, ensure_ascii=False)
                    yield f"data: {data}\n\n".encode()
            finally:
                # Browser gone (disconnect cancels us) or the pipeline
                # closed. A still-live session gets the reconnect grace.
                endpoint.streams -= 1
                if endpoint.streams == 0 and session_endpoints.get(session_id) is endpoint:
                    _arm_grace(endpoint)

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/text/session/{session_id}/end", status_code=202)
    async def end_session(session_id: str, _auth: None = Depends(cf_access)) -> dict[str, str]:
        # The pipeline closes itself after the farewell turn lands
        # (pipeline._EndAfterFarewell); the SSE stream then gets its
        # close sentinel.
        endpoint = _require_endpoint(app, session_id)
        if endpoint.tool_selector.end_session:
            raise HTTPException(status_code=409, detail="the session is already ending")
        # end_session is read per LLM dispatch: set mid-turn, the turn's
        # pending tool re-run would take the capped, unguarded farewell path.
        if not await endpoint.activity.wait_settled(_END_WAIT_S):
            raise HTTPException(status_code=409, detail="a turn is still in flight")
        async with endpoint.turn_lock:
            if endpoint.activity.busy:
                raise HTTPException(status_code=409, detail="a turn is still in flight")
            endpoint.tool_selector.end_session = True
            if endpoint.rollback is not None:
                endpoint.rollback.discard()  # the farewell turn is never regenerated
            phrase = END_PHRASES.get(endpoint.locale, END_PHRASES["en"])
            endpoint.activity.begin()
            await _queue_player_turn(endpoint, phrase, source="end_session")
        logger.info(
            "text end-signal injected: session={} phrase={!r}",
            endpoint.session_id,
            phrase,
        )
        return {"status": "accepted"}

    @app.get("/api/vault/tone", response_class=PlainTextResponse)
    async def tone_get(
        campaign: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> str:
        """Campaign-level tone modulator (free-form prose). Empty body
        when the campaign has no ``tone.md``."""
        vault = _build_campaign_vault(app.state.config, campaign)
        try:
            return vault.read_file("tone.md")
        except VaultPathError:
            return ""

    @app.put("/api/vault/tone", status_code=204)
    async def tone_put(
        request: Request,
        campaign: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> Response:
        vault = _build_campaign_vault(app.state.config, campaign)
        body = (await request.body()).decode("utf-8", errors="replace")
        try:
            vault.write_tone(body)
        except VaultPathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info("tone.md written: campaign={} bytes={}", campaign, len(body))
        return Response(status_code=204)

    @app.get("/api/vault/beat", response_class=PlainTextResponse)
    async def beat_get(
        campaign: str = Query(..., min_length=1),
        run: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> str:
        """Active-run beat modulator (free-form prose). Empty body
        when the run has no ``beat.md`` yet."""
        vault = build_session_vault(app.state.config, campaign, run)
        try:
            return vault.read_file("beat.md")
        except VaultPathError:
            return ""

    @app.put("/api/vault/beat", status_code=204)
    async def beat_put(
        request: Request,
        campaign: str = Query(..., min_length=1),
        run: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> Response:
        vault = build_session_vault(app.state.config, campaign, run)
        body = (await request.body()).decode("utf-8", errors="replace")
        try:
            vault.write_beat(body)
        except VaultPathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _drop_snapshots_on(app, vault)
        logger.info("beat.md written: campaign={} run={} bytes={}", campaign, run, len(body))
        return Response(status_code=204)

    @app.get("/api/vault/characters/{slug}", response_class=PlainTextResponse)
    async def character_sheet(
        slug: str,
        response: Response,
        campaign: str = Query(..., min_length=1),
        run: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> str:
        # Vault._safe enforces vault-root containment so traversal raises.
        vault = build_session_vault(app.state.config, campaign, run)
        rel = f"characters/{slug}.md"
        try:
            text = vault.read_file(rel)
        except VaultPathError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # Last-Modified anchors optimistic concurrency on the PUT side.
        run_root = vault.run_root
        path = (run_root / rel) if run_root is not None else (vault.root / rel)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime is not None:
            response.headers["Last-Modified"] = format_datetime(from_timestamp(mtime), usegmt=True)
        return text

    @app.put("/api/vault/characters/{slug}", status_code=204)
    async def update_character_sheet(
        slug: str,
        request: Request,
        campaign: str = Query(..., min_length=1),
        run: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> Response:
        """Overwrite a character sheet from the browser editor.

        Optimistic concurrency: if the caller supplied
        ``If-Unmodified-Since``, the request fails with 409 when the
        file's mtime is newer (i.e. the DM wrote in between). The
        write itself goes through ``Vault.write_character_sheet`` so
        the server-side auto-fill (``last_updated``) still applies.
        """
        vault = build_session_vault(app.state.config, campaign, run)
        run_root = vault.run_root
        assert run_root is not None
        path = run_root / f"characters/{slug}.md"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="character not found")
        if_unmodified = request.headers.get("if-unmodified-since")
        if if_unmodified:
            try:
                supplied = parsedate_to_datetime(if_unmodified).timestamp()
            except (TypeError, ValueError, IndexError) as exc:
                raise HTTPException(
                    status_code=400, detail="malformed If-Unmodified-Since"
                ) from exc
            current = path.stat().st_mtime
            # Allow a 1-second slack — HTTP-date resolution is one second
            # but the filesystem mtime is sub-second.
            if current > supplied + 1.0:
                raise HTTPException(
                    status_code=409,
                    detail="character sheet changed since last read",
                )
        body = (await request.body()).decode("utf-8", errors="replace")
        try:
            vault.write_character_sheet(slug, body)
        except VaultPathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _drop_snapshots_on(app, vault)
        return Response(status_code=204)

    @app.get(
        "/api/vault/characters/{slug}/roleplay-prompt",
        response_class=PlainTextResponse,
    )
    async def character_roleplay_prompt(
        slug: str,
        campaign: str = Query(..., min_length=1),
        run: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> str:
        """Compose a paste-ready LLM persona prompt for the character.

        Pure string composition over the character sheet (no LLM call).
        The browser exposes a button that opens a modal with this text;
        the user copies it into another tool / fresh conversation.
        """
        vault = build_session_vault(app.state.config, campaign, run)
        try:
            sheet = vault.read_file(f"characters/{slug}.md")
        except VaultPathError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # Strip the auto-appended "# Referenced by" footer if present
        # so it doesn't bleed into the generated persona prompt.
        sheet = sheet.split("\n# Referenced by\n", 1)[0]
        return roleplay_prompt.build(sheet)

    return app


def _build_campaign_vault(config: Config, campaign_slug: str) -> Vault:
    """Build a Vault rooted at vault_root/<campaign> with no active run.

    Used by tone.md GET/PUT which are campaign-canon, not run-scoped.
    """
    campaign_dir = resolve_campaign_dir(config.vault_root, campaign_slug)
    try:
        return Vault(campaign_dir)
    except VaultPathError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _spawn_pipeline(
    app: FastAPI,
    session: SessionContext,
    locale: str,
    ctx_kwargs: Mapping[str, object],
) -> None:
    config: Config = app.state.config
    # Per-session Vault bound to the session's (campaign, run) pair; the
    # LLM tools see this run's overlay, not the canon-only state stored
    # on app.state. Falls back to the default campaign for legacy callers.
    campaign_slug = (session.campaign_slug or config.vault_path.name).strip()
    vault = Vault(
        resolve_campaign_dir(config.vault_root, campaign_slug),
        run_slug=session.run_slug or None,
    )
    session_endpoints: dict[str, TextSessionEndpoint] = app.state.session_endpoints

    set_session(session.session_id)

    ledger: DailyBudgetLedger | None = app.state.budget_ledger
    handle = build_text_pipeline(
        config,
        session,
        vault,
        budget_ledger=ledger,
        **ctx_kwargs,  # type: ignore[arg-type]
    )
    endpoint = TextSessionEndpoint(
        user_agg=handle.user_agg,
        sink_queue=handle.sink_queue,
        locale=locale,
        session_id=session.session_id,
        context=handle.context,
        tool_selector=handle.tool_selector,
        vault=vault,
        scene_registry=handle.scene_registry,
        worker=handle.worker,
        transcript=handle.transcript,
        grace_s=config.text_reconnect_grace_s,
        rollback=handle.rollback,
        activity=handle.activity,
    )
    session_endpoints[session.session_id] = endpoint

    async def run() -> None:
        try:
            await run_worker(handle, session)
        except Exception:
            logger.exception("text pipeline crashed: session={}", session.session_id)
        finally:
            session_endpoints.pop(session.session_id, None)
            _disarm_grace(endpoint)
            # Drop the SSE close sentinel so any attached stream returns
            # cleanly instead of hanging on get().
            handle.sink_queue.put_nowait(None)
            if handle.budget_guard is not None:
                await flush_to_ledger_async(handle.budget_guard)
            logger.info("text session end: session={}", session.session_id)

    task = asyncio.create_task(run(), name=f"aidm-text-pipeline-{session.session_id}")
    pipeline_tasks: set[asyncio.Task[None]] = app.state.pipeline_tasks
    pipeline_tasks.add(task)
    task.add_done_callback(pipeline_tasks.discard)
    # The client opens the SSE stream right after this returns; one that
    # never does is torn down like any other disconnect.
    _arm_grace(endpoint)


# /end waits this long for an in-flight turn (writer passes run up to a
# minute per sheet) before refusing.
_END_WAIT_S = 300.0


def _restore_or_raise(rollback: TurnRollback) -> None:
    """Roll the turn back, or map the failure to its HTTP status (nothing replayed)."""
    try:
        rollback.restore()
    except VaultConflictError as exc:
        logger.warning("regenerate refused, nothing rolled back: {}", exc)
        raise HTTPException(status_code=409, detail=f"conflict: {exc}") from exc
    except VaultPathError as exc:
        # Incomplete: the files not restored stay journaled, memory is
        # untouched and nothing is replayed — a retry finishes the job.
        logger.error("regenerate: rollback FAILED, retry to finish: {}", exc)
        raise HTTPException(status_code=500, detail=f"regenerate failed: {exc}") from exc


def _archive_discarded(vault: Vault, turn: int, player_text: str, reply: str) -> str:
    """Archive after a successful restore only (a retried restore must not archive twice)."""
    try:
        return vault.write_discarded_reply(_discarded_reply_markdown(turn, player_text, reply))
    except (OSError, VaultPathError) as exc:
        logger.error("regenerate: rolled back, but the discarded reply was NOT archived: {}", exc)
        return "(not archived)"


def _drop_snapshots_on(app: FastAPI, vault: Vault) -> None:
    """A browser edit landed on this run: no live session may roll its files back over it."""
    session_endpoints: dict[str, TextSessionEndpoint] = app.state.session_endpoints
    for endpoint in session_endpoints.values():
        same_run = endpoint.vault.root == vault.root and endpoint.vault.run_slug == vault.run_slug
        if same_run and endpoint.rollback is not None and endpoint.rollback.snapshot is not None:
            endpoint.rollback.discard()
            logger.info(
                "regenerate snapshot dropped (browser edit): session={}", endpoint.session_id
            )


async def _queue_player_turn(
    endpoint: TextSessionEndpoint, text: str, *, source: str = "turn"
) -> None:
    """Record the player line for /history and hand it to the pipeline."""
    endpoint.transcript.append({"role": "user", "text": text})
    await endpoint.user_agg.queue_frame(
        TranscriptionFrame(
            text=text, user_id=f"aidm.text_server.{source}", timestamp=str(time.time())
        ),
        FrameDirection.DOWNSTREAM,
    )


def _discarded_reply_markdown(turn: int, player_text: str, reply: str) -> str:
    fm = {
        "turn": turn,
        "discarded_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    return _fm.serialize(fm, f"## Player\n\n{player_text}\n\n## Discarded reply\n\n{reply}\n")


def _arm_grace(endpoint: TextSessionEndpoint) -> None:
    """(Re)start the reconnect window for a session with no SSE stream attached."""
    _disarm_grace(endpoint)
    endpoint.grace_task = asyncio.create_task(
        _end_after_grace(endpoint), name=f"aidm-text-grace-{endpoint.session_id}"
    )


def _disarm_grace(endpoint: TextSessionEndpoint) -> None:
    if endpoint.grace_task is not None:
        endpoint.grace_task.cancel()
        endpoint.grace_task = None


async def _end_after_grace(endpoint: TextSessionEndpoint) -> None:
    await asyncio.sleep(endpoint.grace_s)
    if endpoint.tool_selector.end_session:
        # The player pressed End and left: let the farewell turn (log
        # write, sheet patches, summary) finish, then close.
        logger.info("text session detached while ending: session={}", endpoint.session_id)
        await endpoint.worker.stop_when_done()  # type: ignore[no-untyped-call]
    else:
        # Mid-turn drop: cancel, no replay (docs/session-lifecycle.md).
        logger.info("text session reconnect grace expired: session={}", endpoint.session_id)
        await endpoint.worker.cancel(reason="client disconnected")


def _require_endpoint(app: FastAPI, session_id: str) -> TextSessionEndpoint:
    session_endpoints: dict[str, TextSessionEndpoint] = app.state.session_endpoints
    endpoint = session_endpoints.get(session_id)
    if endpoint is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return endpoint


async def _safe_json(request: Request) -> object:
    try:
        return await request.json()
    except (ValueError, json.JSONDecodeError):
        return None


def _locale_from_payload(payload: object, default: str) -> str:
    if isinstance(payload, dict):
        value = payload.get("locale")
        if isinstance(value, str):
            return value
    return default


def _run_slug_from_payload(payload: object, default: str | None) -> str:
    """Pull ``run_id`` from the start-session payload; fall back to RUN_SLUG env."""
    if isinstance(payload, dict):
        value = payload.get("run_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return (default or "").strip()


def _campaign_slug_from_payload(payload: object, default: str) -> str:
    """Pull ``campaign_id`` from the start-session payload; fall back to default campaign."""
    if isinstance(payload, dict):
        value = payload.get("campaign_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default.strip()


def _entry_dict(e: object) -> dict[str, object]:
    """Serialize a ResumeEntry to a JSON-safe dict.

    Defined as ``object``-typed for the import-graph cost (avoiding a
    runtime import of ResumeEntry just for an isinstance check); the
    dataclass attributes accessed below are stable.
    """
    return {
        "id": e.id,  # type: ignore[attr-defined]
        "source": e.source,  # type: ignore[attr-defined]
        "path": e.path,  # type: ignore[attr-defined]
        "title": e.title,  # type: ignore[attr-defined]
        "mtime": e.mtime,  # type: ignore[attr-defined]
        "snippet": e.snippet,  # type: ignore[attr-defined]
        "campaign": e.campaign,  # type: ignore[attr-defined]
        "date": e.date,  # type: ignore[attr-defined]
        "characters": list(e.characters),  # type: ignore[attr-defined]
        "locations": list(e.locations),  # type: ignore[attr-defined]
    }


def _resume_from_payload(payload: object) -> tuple[ResumeSource, str] | None:
    if not isinstance(payload, dict):
        return None
    block = payload.get("resume_from")
    if not isinstance(block, dict):
        return None
    source = block.get("source")
    entry_id = block.get("id")
    if source not in ("log", "running"):
        return None
    if not isinstance(entry_id, str) or not entry_id.strip():
        return None
    return (source, entry_id.strip())


def _build_resume_context(resume_body: str) -> object:
    # Lazy import keeps the test_text_server fast path from pulling
    # the full pipecat aggregator module just to build a fake handle.
    from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: PLC0415

    return LLMContext(messages=[{"role": "system", "content": resume_body}])


def _session_id_from_payload(payload: object) -> str | None:
    if isinstance(payload, dict):
        value = payload.get("session_id")
        if isinstance(value, str):
            return value
    return None
