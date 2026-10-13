"""FastAPI HTTP/WebRTC entry point for browser-mode AIDM.

Started by ``aidm/__main__.py`` when ``AIDM_TRANSPORT=webrtc``. Speaks
two things:

* HTTP signaling: ``POST /api/offer`` accepts a WebRTC SDP offer and
  returns the orchestrator's SDP answer. ``POST /api/offer/{pc_id}/
  candidate`` carries trickle-ICE candidates (browser → server).
* The session lifecycle described in ``docs/session-lifecycle.md``: one
  ``SmallWebRTCConnection`` = one session. The connection's ``closed``
  event tears down the pipeline task.

The pipeline graph itself is built by ``aidm.pipeline.run_pipeline`` —
identical to the local-mode path, only the transport is different.

Why a single shared ``aiohttp.ClientSession`` and ``Vault``: both are
process-level resources. The aiohttp session pools TCP connections to
ElevenLabs/OpenRouter; opening one per peer would multiply TLS
handshakes and break keepalive. The vault is read-only.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from typing import Any

import aiohttp
from aiortc.rtcconfiguration import RTCIceServer
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse
from loguru import logger
from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from aidm import frontmatter, roleplay_prompt
from aidm.audio import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE_TRANSPORT
from aidm.budget import DailyBudgetLedger, flush_to_ledger_async, make_ledger
from aidm.config import Config, load_config
from aidm.log import set_session
from aidm.pipeline import build_pipeline, run_worker
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
from aidm.session import session_for_locale
from aidm.sessions_registry import (
    END_PHRASES,
    RECONNECT_GRACE_SECONDS,
    PendingContext,
    claim_pending,
    is_pending,
    stash_pending,
    sweep_pending,
)
from aidm.tool_router import ToolSubsetSelector
from aidm.vault import Vault, VaultPathError, slugify
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
class SessionEndpoint:
    """Per-pc_id handle the HTTP layer uses to nudge a running pipeline."""

    user_agg: FrameProcessor
    locale: str
    session_id: str
    tool_selector: ToolSubsetSelector


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    if config.transport != "webrtc":
        raise RuntimeError(f"aidm.server requires AIDM_TRANSPORT=webrtc, got {config.transport!r}")
    await check_providers(config)

    http_session = aiohttp.ClientSession()
    # Per-session Vault instances are built in _spawn_pipeline with the
    # active (campaign_slug, run_slug) pair. The default-campaign Vault
    # here exists for legacy callers + ``app.state.vault`` references —
    # eventually those should switch to the per-session pattern.
    vault = Vault(config.vault_path)
    ice_servers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
    if config.turn_url:
        ice_servers.append(
            RTCIceServer(
                urls=config.turn_url,
                username=config.turn_username or None,
                credential=config.turn_credential or None,
            )
        )
    elif not config.cf_access_disabled:
        logger.warning(
            "STUN-only ICE in non-dev posture: clients behind symmetric NAT "
            "will fail to establish media. Set AIDM_TURN_URL to a TURN relay."
        )
    request_handler = SmallWebRTCRequestHandler(ice_servers=ice_servers)
    pipeline_tasks: set[asyncio.Task[None]] = set()
    session_endpoints: dict[str, SessionEndpoint] = {}
    pending_contexts: dict[str, PendingContext] = {}
    ledger: DailyBudgetLedger | None = None
    if config.mode == "public" and config.daily_usd_cap > 0:
        ledger = make_ledger(config.budget_db_path)
        logger.info(
            "budget ledger ready: db={} daily_cap=${:.2f}",
            config.budget_db_path,
            config.daily_usd_cap,
        )

    app.state.config = config
    app.state.http_session = http_session
    app.state.vault = vault
    app.state.request_handler = request_handler
    app.state.pipeline_tasks = pipeline_tasks
    app.state.session_endpoints = session_endpoints
    app.state.pending_contexts = pending_contexts
    app.state.budget_ledger = ledger

    sweeper = asyncio.create_task(sweep_pending(pending_contexts), name="aidm-pending-sweeper")

    logger.info(
        "aidm webrtc server ready: mode={} locale_default={} vault={} "
        "max_sessions={} cf_access={} public_ip={}",
        config.mode,
        config.locale,
        vault.root,
        config.max_concurrent_sessions,
        "off" if config.cf_access_disabled else (config.cf_access_team_domain or "unconfigured"),
        config.public_ip or "<auto>",
    )
    try:
        yield
    finally:
        sweeper.cancel()
        for task in pipeline_tasks:
            task.cancel()
        if pipeline_tasks:
            await asyncio.gather(*pipeline_tasks, return_exceptions=True)
        await asyncio.gather(sweeper, return_exceptions=True)
        await request_handler.close()  # type: ignore[no-untyped-call]
        await http_session.close()


def create_app() -> FastAPI:  # noqa: PLR0915 — route registration inline is fine
    """Build the FastAPI app. Called by ``__main__`` and tests."""
    config = load_config()
    app = FastAPI(lifespan=_lifespan)
    configure_cors(app, config)
    limiter = build_limiter()
    install_rate_limit_handler(app, limiter)
    cf_access = make_cf_access_verifier(config)
    rate_limit_offer = config.rate_limit_offer

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

    @app.post("/api/offer")
    @limiter.limit(rate_limit_offer)
    async def offer(request: Request, _auth: None = Depends(cf_access)) -> dict[str, str] | None:
        cfg: Config = app.state.config
        # Daily budget kill-switch — reads SQLite, ~ms latency.
        enforce_daily_budget(app.state.budget_ledger, cfg.daily_usd_cap)
        # Concurrent session cap: pipeline_tasks holds running sessions.
        active = sum(1 for t in app.state.pipeline_tasks if not t.done())
        if cfg.max_concurrent_sessions > 0 and active >= cfg.max_concurrent_sessions:
            raise HTTPException(status_code=503, detail=_TOO_MANY_SESSIONS_HU)

        payload = await request.json()
        try:
            rtc_request = SmallWebRTCRequest.from_dict(payload)
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid offer: {exc}") from exc

        # Pre-validate (campaign, run) selection so we don't accept the
        # SDP and then tear it down a beat later when the pipeline can't
        # find them. Only a live reconnect (unexpired pending context)
        # skips the check — the pair is pinned by the resumed
        # SessionContext. An expired or unknown session_id starts fresh,
        # so it must carry a valid pair like any other start.
        if not is_pending(
            _session_id_from_request(rtc_request.request_data), app.state.pending_contexts
        ):
            campaign_slug = _campaign_slug_from_request(
                rtc_request.request_data, cfg.vault_path.name
            )
            run_slug = _run_slug_from_request(rtc_request.request_data, cfg.run_slug)
            if not campaign_slug:
                raise HTTPException(status_code=400, detail="campaign_id is required")
            if not run_slug:
                raise HTTPException(
                    status_code=400,
                    detail="run_id is required (no RUN_SLUG fallback configured)",
                )
            build_session_vault(cfg, campaign_slug, run_slug)

        handler: SmallWebRTCRequestHandler = app.state.request_handler

        async def on_new_connection(connection: SmallWebRTCConnection) -> None:
            await _spawn_pipeline(app, connection, rtc_request.request_data)

        answer = await handler.handle_web_request(rtc_request, on_new_connection)
        # Echo session_id back so the browser can pass it as
        # requestData.session_id on a reconnect attempt. The
        # session_endpoints registry was populated by _spawn_pipeline
        # under the same pc_id.
        if answer is not None:
            session_endpoints: dict[str, SessionEndpoint] = app.state.session_endpoints
            endpoint = session_endpoints.get(answer.get("pc_id", ""))
            if endpoint is not None:
                answer["session_id"] = endpoint.session_id
        return answer

    @app.post("/api/offer/{pc_id}/candidate")
    async def candidate(
        pc_id: str, request: Request, _auth: None = Depends(cf_access)
    ) -> dict[str, str]:
        payload = await request.json()
        try:
            patch = SmallWebRTCPatchRequest(
                pc_id=pc_id,
                candidates=payload.get("candidates", []),
            )
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid candidate: {exc}") from exc
        handler: SmallWebRTCRequestHandler = app.state.request_handler
        await handler.handle_patch_request(patch)
        return {"status": "ok"}

    @app.post("/api/session/{pc_id}/end", status_code=202)
    async def end_session(pc_id: str, _auth: None = Depends(cf_access)) -> dict[str, str]:
        # The browser's End-session button hits this when the player is
        # done. We do NOT close the peer connection here — we flip the
        # tool selector into end-session mode (only the session-log
        # tool is on the menu) and push a synthetic player utterance
        # ("köszi, vége" / "thanks, we're done") so the persona writes
        # the session log and says a brief farewell. The browser waits
        # for the next transcript.bot_end on the data channel, then
        # disconnects.
        session_endpoints: dict[str, SessionEndpoint] = app.state.session_endpoints
        endpoint = session_endpoints.get(pc_id)
        if endpoint is None:
            raise HTTPException(status_code=404, detail="unknown pc_id")
        endpoint.tool_selector.end_session = True
        phrase = END_PHRASES.get(endpoint.locale, END_PHRASES["en"])
        await endpoint.user_agg.queue_frame(
            TranscriptionFrame(
                text=phrase,
                user_id="aidm.server.end_session",
                timestamp=str(time.time()),
            ),
            FrameDirection.DOWNSTREAM,
        )
        logger.info(
            "webrtc end-signal injected: pc_id={} session={} phrase={!r}",
            pc_id,
            endpoint.session_id,
            phrase,
        )
        return {"status": "accepted"}

    @app.get("/api/campaigns")
    async def list_campaigns(_auth: None = Depends(cf_access)) -> list[dict[str, str]]:
        """Enumerate every campaign vault under ``vault_root``.

        A "campaign vault" is any direct subdirectory of ``vault_root``
        that holds a ``campaign.md`` at its top level. The picker uses
        the returned title for display; the slug is the directory name
        and is what flows into ``requestData.campaign_id`` on
        ``/api/offer``.
        """
        cfg: Config = app.state.config
        return list_campaigns_on_disk(cfg.vault_root)

    @app.get("/api/campaigns/{slug}/runs")
    async def list_runs(slug: str, _auth: None = Depends(cf_access)) -> list[dict[str, Any]]:
        cfg: Config = app.state.config
        campaign_dir = resolve_campaign_dir(cfg.vault_root, slug)
        return list_runs_on_disk(campaign_dir)

    @app.post("/api/campaigns/{slug}/runs", status_code=201)
    async def create_run(
        slug: str, request: Request, _auth: None = Depends(cf_access)
    ) -> dict[str, Any]:
        cfg: Config = app.state.config
        campaign_dir = resolve_campaign_dir(cfg.vault_root, slug)
        payload = await request.json()
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
            frontmatter.serialize(fm, f"# {title}\n"),
            encoding="utf-8",
        )
        logger.info("run created: campaign={} slug={} title={!r}", slug, run_slug, title)
        return run_summary(run_dir)

    @app.get("/api/vault/characters/{slug}", response_class=PlainTextResponse)
    async def character_sheet(
        slug: str,
        response: Response,
        campaign: str = Query(..., min_length=1),
        run: str = Query(..., min_length=1),
        _auth: None = Depends(cf_access),
    ) -> str:
        # The browser fetches this after a character.change data-channel
        # event so the side panel can render the full markdown sheet.
        # Vault._safe enforces vault-root containment so a slug like
        # '../../etc/passwd' resolves outside the root and raises.
        vault = build_session_vault(app.state.config, campaign, run)
        rel = f"characters/{slug}.md"
        try:
            text = vault.read_file(rel)
        except VaultPathError as exc:
            # 404 covers both "outside vault" and "file missing" — the
            # browser treats both the same way ("no sheet to render").
            raise HTTPException(status_code=404, detail=str(exc)) from exc
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
        vault = build_session_vault(app.state.config, campaign, run)
        run_root = vault.run_root
        assert run_root is not None  # run query param is required
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
        vault = build_session_vault(app.state.config, campaign, run)
        try:
            sheet = vault.read_file(f"characters/{slug}.md")
        except VaultPathError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        sheet = sheet.split("\n# Referenced by\n", 1)[0]
        return roleplay_prompt.build(sheet)

    return app


async def _spawn_pipeline(  # noqa: PLR0915 — session build + handler wiring inline
    app: FastAPI,
    connection: SmallWebRTCConnection,
    request_data: Any | None,
) -> None:
    """Build a SessionContext + transport + run the pipeline as a background task.

    The peer connection drives the session lifecycle: when it closes, the
    runner sees ``CancelledError`` (or the transport's own EOF) and the
    task exits. We track the task on app.state so the lifespan teardown
    can cancel any survivors.
    """
    config: Config = app.state.config
    http_session: aiohttp.ClientSession = app.state.http_session
    pending_contexts: dict[str, PendingContext] = app.state.pending_contexts

    resumed = claim_pending(_session_id_from_request(request_data), pending_contexts)
    if resumed is not None:
        session = resumed.session
        locale = session.locale
        resume_kwargs: dict[str, Any] = {"context": resumed.context}
        # Resume uses the (campaign, run) from the previously-captured
        # SessionContext. The browser may have sent fresh ids, but
        # reconnect semantics say the in-flight pair wins (a mid-session
        # swap would lose continuity).
        vault = build_session_vault(config, session.campaign_slug, session.run_slug)
        logger.info(
            "webrtc session resume: pc_id={} session={} locale={} campaign={} run={}",
            connection.pc_id,
            session.session_id,
            locale,
            session.campaign_slug,
            session.run_slug,
        )
    else:
        locale = _locale_from_request(request_data, config.locale)
        campaign_slug = _campaign_slug_from_request(request_data, config.vault_path.name)
        run_slug = _run_slug_from_request(request_data, config.run_slug)
        if not campaign_slug:
            raise HTTPException(
                status_code=400,
                detail="campaign_id is required",
            )
        if not run_slug:
            raise HTTPException(
                status_code=400,
                detail="run_id is required (no RUN_SLUG fallback configured)",
            )
        vault = build_session_vault(config, campaign_slug, run_slug)
        run_title = read_run_title(vault)
        campaign_title = read_campaign_title(vault)
        session = session_for_locale(
            locale,
            campaign_slug=campaign_slug,
            campaign_title=campaign_title,
            run_slug=run_slug,
            run_title=run_title,
        )
        resume_kwargs = {}
        logger.info(
            "webrtc session start: pc_id={} session={} locale={} campaign={} run={}",
            connection.pc_id,
            session.session_id,
            locale,
            campaign_slug,
            run_slug,
        )
    # set_session writes to the loguru contextvar; per-task isolation is
    # handled by loguru's contextvars mode, so logs from concurrent peer
    # sessions don't bleed into each other.
    set_session(session.session_id)

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            # 48 kHz on the WebRTC seam — aiortc's opus encoder hard-requires
            # it. The TTS provider stays on its native rate; pipecat's output
            # transport upsamples 24 kHz → 48 kHz before the opus encoder.
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE_TRANSPORT,
        ),
    )

    ledger: DailyBudgetLedger | None = app.state.budget_ledger
    handle = build_pipeline(
        transport,
        config,
        session,
        vault,
        http_session,
        budget_ledger=ledger,
        **resume_kwargs,
    )
    endpoint = SessionEndpoint(
        user_agg=handle.user_agg,
        locale=locale,
        session_id=session.session_id,
        tool_selector=handle.tool_selector,
    )
    session_endpoints: dict[str, SessionEndpoint] = app.state.session_endpoints
    session_endpoints[connection.pc_id] = endpoint

    # Pipecat's smallwebrtc transport reports a peer close only through
    # this event; without it the pipeline lingers until the idle watchdog.
    @transport.event_handler("on_client_disconnected")  # type: ignore[untyped-decorator]
    async def _on_client_disconnected(_transport: object, _connection: object) -> None:
        if handle.tool_selector.end_session:
            # End pressed, then the browser hung up: let the farewell
            # turn (log write, sheet patches, summary) finish, then close.
            await handle.worker.stop_when_done()  # type: ignore[no-untyped-call]
        else:
            # Drop without End: cancel in-flight work (no replay); the
            # context is stashed below for the reconnect grace.
            await handle.worker.cancel(reason="peer disconnected")

    async def run() -> None:
        try:
            await run_worker(handle, session)
        except Exception:
            logger.exception(
                "webrtc pipeline crashed: pc_id={} session={}",
                connection.pc_id,
                session.session_id,
            )
        finally:
            session_endpoints.pop(connection.pc_id, None)
            # Stash the LLMContext so a quick browser reconnect resumes
            # the conversation instead of starting from scratch. The
            # sweeper purges expired entries. An ended session is not
            # resumable. See docs/session-lifecycle.md.
            if not handle.tool_selector.end_session:
                stash_pending(pending_contexts, session, handle.context)
            if handle.budget_guard is not None:
                await flush_to_ledger_async(handle.budget_guard)
            logger.info(
                "webrtc session end: pc_id={} session={} (held {:.0f}s for resume)",
                connection.pc_id,
                session.session_id,
                RECONNECT_GRACE_SECONDS,
            )

    task = asyncio.create_task(run(), name=f"aidm-pipeline-{session.session_id}")
    pipeline_tasks: set[asyncio.Task[None]] = app.state.pipeline_tasks
    pipeline_tasks.add(task)
    task.add_done_callback(pipeline_tasks.discard)


def _session_id_from_request(request_data: Any | None) -> str | None:
    if not isinstance(request_data, dict):
        return None
    session_id = request_data.get("session_id")
    return session_id if isinstance(session_id, str) else None


def _locale_from_request(request_data: Any | None, default: str) -> str:
    """Pull ``locale`` from the offer's ``requestData`` blob, fall back to default.

    The browser may post ``{ "sdp": ..., "type": "offer",
    "requestData": { "locale": "en" } }`` to start an English session.
    Unknown / malformed values fall back rather than 400ing — keeps the
    signaling path forgiving; the session_for_locale call below will
    raise if the locale is truly unsupported.
    """
    if isinstance(request_data, dict):
        candidate = request_data.get("locale")
        if isinstance(candidate, str):
            return candidate
    return default


def _run_slug_from_request(request_data: Any | None, default: str | None) -> str:
    """Pull ``run_id`` from the offer's ``requestData`` blob, fall back to RUN_SLUG env.

    Returns the empty string if neither is present; the caller decides
    how to surface the failure. The picker UI always supplies one; the
    fallback covers the local-CLI / scripted-test paths.
    """
    if isinstance(request_data, dict):
        candidate = request_data.get("run_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return (default or "").strip()


def _campaign_slug_from_request(request_data: Any | None, default: str) -> str:
    """Pull ``campaign_id`` from the offer's ``requestData`` blob.

    Falls back to the default campaign (the one ``VAULT_PATH`` names)
    so the local CLI / legacy clients that don't send a campaign keep
    working.
    """
    if isinstance(request_data, dict):
        candidate = request_data.get("campaign_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return default.strip()
