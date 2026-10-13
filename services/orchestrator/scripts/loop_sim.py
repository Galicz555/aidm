"""Closed-loop text-mode harness — real server, cloud PC simulator.

Drives the *production* text server (``AIDM_TRANSPORT=text``,
``AIDM_LOG_LEVEL=DEBUG``) end to end: a cloud model plays Rowan from
his sheet and a scenario beat script, the DM pipeline answers, and every
SSE side-band event is captured for scoring. No parallel offline
pipeline — what's measured is what ships. (``eval_llm.py`` stays the
transcript-pinned regression harness; this one closes the loop.)

The DM defaults to the private pipeline (``AIDM_MODE=private``, local
llm-gateway). ``--dm-model <openrouter-id>`` spawns the server in
``AIDM_MODE=public`` with that model as the DM instead. ``--sim-model``
takes an Anthropic id (default) or an OpenRouter ``vendor/model`` slug.

The simulator and this script are **test infrastructure**, not product:
the private-mode "no cloud calls during gameplay" promise applies to the
private DM pipeline, which stays fully local.

Per replay:

1. Copy the campaign vault into a sandbox (``_sandbox_vault`` pattern —
   the checked-in vault is never touched; the pristine ``eval-*`` runs
   ARE the reset).
2. Spawn ``python -m aidm`` against the sandbox (or ``--attach`` to a
   running server), healthcheck it and the llama.cpp gateway.
3. ``POST /api/text/session`` → ``GET .../stream`` (SSE, captured to
   ``events.jsonl``) → N turns of simulator → ``POST .../turn`` → wait
   for ``turn.status: idle`` (the writer blocks end-of-turn until sheet
   patches + the story engine land) → ``POST .../end``.
4. Snapshot the sandbox run subtree + write ``manifest.json`` and
   ``transcript.json``.

Output layout (mirrors docs/bench conventions)::

    docs/bench/<date>/loop/<scenario>/replay-NNN/
        manifest.json  transcript.json  events.jsonl  server.log
        vault-snapshot/

Usage::

    uv run python services/orchestrator/scripts/loop_sim.py \\
        services/orchestrator/scripts/loop_scenarios/smoke-5t.yaml \\
        [--turns N] [--replays N] [--attach URL] [--sim-model ID] [--dm-model ID]

``--sim-model local`` simulates the player on the CPU ``sim-llm`` service
(``docker compose --profile eval up -d sim-llm``, Gemma 3 4B) — free, no
key; e.g. ``make loop-sim-50 LOOP_ARGS="--sim-model local"``.

Needs: ``ANTHROPIC_API_KEY`` when the sim model is an Anthropic id;
``OPENROUTER_API_KEY`` when the sim model is an OpenRouter slug or
``--dm-model`` is set. Private DM runs (no ``--dm-model``) also need the
llm-gateway up (``docker compose --profile private up``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_THIS.parents[1]))  # services/orchestrator → aidm importable
sys.path.insert(0, str(_THIS.parent))  # scripts dir → eval_llm importable

# Same .env the orchestrator loads — so ANTHROPIC_API_KEY (and the
# gateway URL) can live there instead of the shell. Shell env wins.
load_dotenv(_REPO_ROOT / ".env", override=False)

from aidm.turn_director import TURN_CONTRACT_TAG  # noqa: E402
from eval_llm import _sandbox_vault  # noqa: E402

DEFAULT_SIM_MODEL = "claude-sonnet-5"
# `--sim-model local`: the player is simulated by the CPU `sim-llm` compose
# service (Gemma 3 4B, a different family from the Qwen DM, so no self-play) —
# free, no API key. Simplification: a 4B sim is weaker than a frontier one; compare code
# changes against a local-sim baseline, not against Anthropic-sim runs.
LOCAL_SIM = "local"
SIM_MAX_TOKENS = 2048  # thinking models consume tokens before text output; 220 was too tight
# A simulator turn this similar to any of its own prior outputs is
# degenerate — re-roll (≤2), then accept but flag it for the scorer.
SIM_DUP_THRESHOLD = 0.8
SIM_MAX_REROLLS = 2
SIM_PRIOR_WINDOW = 5
# Retries for a transient cloud-API error (5xx/429/timeout) on one sim
# completion. The PC simulator is Anthropic/OpenRouter, which flap; a
# single unretried 5xx aborts a whole 50-turn replay.
SIM_API_RETRIES = 4
_HISTORY_WINDOW = 3  # (player, dm) exchange pairs shown to the simulator

DEFAULT_TURN_TIMEOUT_S = 600.0
_HEALTH_TIMEOUT_S = 60.0
_END_DRAIN_TIMEOUT_S = 180.0
# Quiet period after a turn.status idle before the turn counts as done —
# long enough to catch the writer pass of a tool-call continuation
# starting, short enough not to drag out a 50-turn replay.
IDLE_GRACE_S = 5.0
# No DM text at all this long after posting a turn means the completion
# came back empty (upstream flake) and the turn will never end — re-post
# once instead of burning the whole turn timeout. Covers the private
# 14B's slowest tool-round + first-token path with room to spare.
FIRST_TEXT_TIMEOUT_S = 120.0


# --- scenario -------------------------------------------------------------


@dataclass(frozen=True)
class Beat:
    intent: str
    until_turn: int


@dataclass(frozen=True)
class Scenario:
    id: str
    run_slug: str
    turns: int
    beats: tuple[Beat, ...]


def load_scenario(path: Path) -> Scenario:
    """Parse a loop_scenarios/*.yaml file; raises ValueError on bad shape."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"scenario file is not a mapping: {path}")
    scn = spec.get("scenario") or {}
    beats_raw = spec.get("beats") or []
    beats = tuple(
        Beat(intent=str(b["intent"]), until_turn=int(b["until_turn"]))
        for b in beats_raw
        if isinstance(b, dict) and b.get("intent")
    )
    if not beats:
        raise ValueError(f"scenario has no beats: {path}")
    scenario = Scenario(
        id=str(scn.get("id") or path.stem),
        run_slug=str(scn.get("run_slug") or ""),
        turns=int(scn.get("turns") or 0),
        beats=beats,
    )
    if not scenario.run_slug or scenario.turns <= 0:
        raise ValueError(f"scenario needs run_slug and turns > 0: {path}")
    return scenario


def current_beat(beats: tuple[Beat, ...], turn: int) -> str:
    """The active beat intent for a 1-based turn (last beat covers overflow)."""
    for beat in beats:
        if turn <= beat.until_turn:
            return beat.intent
    return beats[-1].intent


# --- player simulator -------------------------------------------------------


def _strip_contract_tag(dm_text: str) -> str:
    """Remove the injected turn-contract prefix before feeding to the simulator.

    The model occasionally echoes the tag at the start of its reply
    (sometimes behind leading whitespace); the simulator Claude gets
    confused by the HTML comment and produces degenerate follow-up
    inputs, cascading into empty player text and HTTP 400.
    """
    stripped = dm_text.lstrip()
    if stripped.startswith(TURN_CONTRACT_TAG):
        return stripped[len(TURN_CONTRACT_TAG) :].lstrip()
    return dm_text


def is_degenerate(candidate: str, priors: list[str], threshold: float = SIM_DUP_THRESHOLD) -> bool:
    """True when the candidate near-duplicates any prior simulator output."""
    norm = " ".join(candidate.split()).casefold()
    if not norm:
        return True
    for prior in priors:
        prior_norm = " ".join(prior.split()).casefold()
        if prior_norm and SequenceMatcher(None, norm, prior_norm).ratio() >= threshold:
            return True
    return False


def build_sim_prompt(
    *,
    pc_sheet: str,
    beat_intent: str,
    history: list[tuple[str, str]],
    priors: list[str],
) -> str:
    """One-shot prompt for the PC simulator. Pure string assembly."""
    exchange_lines: list[str] = []
    for player, dm in history[-_HISTORY_WINDOW:]:
        exchange_lines.append(f"Player: {player}")
        exchange_lines.append(f"DM: {dm}")
    exchanges = "\n".join(exchange_lines) or "(session start — no prior turns)"
    avoid = "\n".join(f"- {p}" for p in priors[-SIM_PRIOR_WINDOW:]) or "(none yet)"
    return (
        "You are playing Rowan, the player character in a tabletop RPG "
        "session run by an AI DM. Produce his NEXT action or line, exactly "
        "as the player would type it.\n\n"
        f"Character sheet (identity + current state):\n```\n{pc_sheet}\n```\n\n"
        f"Your current goal as the player (pursue it, but react to the DM "
        f"first):\n{beat_intent}\n\n"
        f"Recent exchange:\n{exchanges}\n\n"
        f"Your previous inputs — do NOT repeat or rephrase any of them:\n{avoid}\n\n"
        "Rules:\n"
        "- 1-3 sentences, first person, in Rowan's voice.\n"
        "- React to the DM's latest beat concretely; advance your goal.\n"
        "- Stay consistent with the sheet: HEXACO, inventory, bonds, fears.\n"
        "- No DM narration, no meta commentary, no option lists.\n"
        "- Output ONLY the player input, nothing else."
    )


async def simulate_player(  # noqa: PLR0913 — simulator context is naturally wide
    client: Any,
    model: str,
    *,
    pc_sheet: str,
    beat_intent: str,
    history: list[tuple[str, str]],
    priors: list[str],
) -> tuple[str, int, bool]:
    """Return ``(player_text, rerolls, degenerate)`` for this turn.

    No sampling parameters: Sonnet 5 (like Opus 4.8) rejects non-default
    ``temperature``/``top_p`` with a 400. Re-roll variety comes from the
    prompt instead — each retry names the rejected output explicitly.
    """
    base_prompt = build_sim_prompt(
        pc_sheet=pc_sheet, beat_intent=beat_intent, history=history, priors=priors
    )
    text = ""
    last_nonempty = ""
    for attempt in range(SIM_MAX_REROLLS + 1):
        prompt = base_prompt
        if attempt and text:
            prompt += (
                f"\n\nYour draft {text!r} was rejected as a near-duplicate of a "
                "previous input. Produce a clearly DIFFERENT action — different "
                "verb, different target, or different tack on the goal."
            )
        text = _unwikilink(await _sim_complete(client, model, prompt))
        if text:
            last_nonempty = text
        if text and not is_degenerate(text, priors):
            return text, attempt, False
    # All attempts degenerate — release the last non-empty draft rather than
    # returning "" (which aborts the replay with HTTP 400).
    return last_nonempty, SIM_MAX_REROLLS, True


def _is_transient(exc: Exception) -> bool:
    """A 5xx / 429 / timeout worth retrying — the cloud sim API flaps."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status == 429):
        return True
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connection", "InternalServer", "Overloaded"))


_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def _unwikilink(text: str) -> str:
    """Small sims copy the PC sheet's Obsidian links verbatim: [[a-b|Alias]] → Alias."""
    return _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1).replace("-", " "), text)


async def _sim_complete(client: Any, model: str, prompt: str) -> str:
    """One simulator completion, retrying transient cloud API errors.

    A single 5xx anywhere in a 50-turn replay otherwise crashes the whole
    run — Anthropic/OpenRouter both flap intermittently. Bounded retry with
    linear backoff; a non-transient error (400/auth) still raises.
    """
    for attempt in range(SIM_API_RETRIES + 1):
        try:
            if model == LOCAL_SIM:
                resp = await client.chat.completions.create(
                    model="sim",  # llama-server --alias; one model per server
                    max_tokens=SIM_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                return (resp.choices[0].message.content or "").strip()
            if "/" in model:  # OpenRouter slug (vendor/model)
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=SIM_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                    # Reasoning models burn the whole token budget thinking
                    # and return empty content for a 1-3 sentence task —
                    # OpenRouter drops the param for models without it.
                    extra_body={"reasoning": {"enabled": False}},
                )
                return (resp.choices[0].message.content or "").strip()
            response = await client.messages.create(
                model=model,
                max_tokens=SIM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            ).strip()
        except Exception as exc:
            if attempt >= SIM_API_RETRIES or not _is_transient(exc):
                raise
            print(
                f"[sim] transient API error ({type(exc).__name__}); "
                f"retry {attempt + 1}/{SIM_API_RETRIES}",
                file=sys.stderr,
            )
            await asyncio.sleep(2.0 * (attempt + 1))
    return ""  # unreachable — loop either returns or raises


# --- SSE capture -------------------------------------------------------------


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one ``data: {...}`` SSE line; None for keepalives/garbage."""
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:") :].strip()
    if not payload:
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class SseCapture:
    """Drains the session SSE stream to events.jsonl + an in-process queue.

    The server-side pipeline (and its sink queue) survives a client
    disconnect, so a fresh GET resumes the feed. Any close — clean EOF
    or transport error — triggers one reconnect attempt UNLESS the
    harness has called ``expect_close()`` (just before ``POST .../end``):
    only then is a clean close the end-of-session sentinel. A silent
    mid-run EOF would otherwise stop capture and misreport every later
    turn as a timeout.
    """

    def __init__(self, base_url: str, session_id: str, out_path: Path) -> None:
        self._url = f"{base_url}/api/text/session/{session_id}/stream"
        self._out_path = out_path
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.closed = asyncio.Event()
        self._expect_close = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="loop-sim-sse")

    def expect_close(self) -> None:
        """Harness is about to end the session — a clean close is final."""
        self._expect_close = True

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        attempts = 0
        with self._out_path.open("a", encoding="utf-8") as fh:
            while attempts < 3:
                attempts += 1
                try:
                    async with (
                        httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client,
                        client.stream("GET", self._url) as response,
                    ):
                        async for line in response.aiter_lines():
                            event = parse_sse_line(line)
                            if event is None:
                                continue
                            fh.write(
                                json.dumps({"ts": time.time(), **event}, ensure_ascii=False) + "\n"
                            )
                            fh.flush()
                            await self.queue.put(event)
                    if self._expect_close:
                        break  # end-of-session sentinel drained
                    print(f"[sse] stream closed mid-run; reconnect {attempts}/3", file=sys.stderr)
                    await asyncio.sleep(1.0)
                except (httpx.HTTPError, httpx.StreamError) as exc:
                    print(f"[sse] stream dropped ({exc}); reconnect {attempts}/3", file=sys.stderr)
                    await asyncio.sleep(1.0)
        self.closed.set()


async def collect_turn(
    sse: SseCapture,
    *,
    timeout_s: float,
    grace_s: float = IDLE_GRACE_S,
    first_text_s: float = FIRST_TEXT_TIMEOUT_S,
) -> tuple[str, bool]:
    """Drain SSE until the turn is genuinely over; return ``(dm_text, done)``.

    A tool-calling DM can produce several full LLM responses per player
    turn (narration, tool call, continuation), and the sheet writer runs
    a full ``writing_sheets`` → ``idle`` cycle after each one — so the
    first ``idle`` is not proof the turn ended. The turn is done when an
    ``idle`` has arrived AND the stream then stays quiet for ``grace_s``;
    any later event (another response streaming, another writer pass)
    voids the idle and the wait resumes.

    An empty LLM completion (upstream flake) produces no text, no writer
    pass and no ``idle`` — the turn would hang forever. If no DM text at
    all arrives within ``first_text_s``, give up early with
    ``("", False)`` so the caller can re-post the turn instead of
    burning the whole ``timeout_s``.

    ``dm_text`` joins every ``debug.bot`` payload of the turn — a
    tool-calling DM's reply is split across them, and keeping only the
    last one would drop the earlier narration from the transcript.
    ``transcript.segment`` chunks are the fallback so a debug-channel
    regression degrades instead of aborting.
    """
    now = time.monotonic()
    deadline = now + timeout_s
    text_deadline = now + first_text_s
    bots: list[str] = []
    segments: list[str] = []
    idle = False
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            break
        if idle:
            wait_s = min(grace_s, remaining)
        elif bots or segments:
            wait_s = remaining
        else:
            remaining_text = text_deadline - now
            if remaining_text <= 0:
                break  # stalled: no DM text at all — let the caller retry
            wait_s = min(remaining, remaining_text)
        try:
            event = await asyncio.wait_for(sse.queue.get(), timeout=wait_s)
        except TimeoutError:
            break  # every wait is clamped to whichever deadline applies
        etype = event.get("type")
        if etype == "debug.bot":
            bots.append(str(event.get("text") or ""))
        elif etype == "transcript.segment":
            segments.append(str(event.get("text") or ""))
        idle = etype == "turn.status" and event.get("phase") == "idle"
    dm_text = "\n\n".join(b for b in bots if b.strip()).strip() or "".join(segments).strip()
    return dm_text, idle


# --- server process -----------------------------------------------------------


_PORT_FREE_TIMEOUT_S = 15.0


class PortBusyError(RuntimeError):
    """The harness port is taken before our server starts."""


@dataclass
class ServerHandle:
    base_url: str
    process: subprocess.Popen[bytes] | None = None
    log_file: Any = None

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        if self.log_file is not None:
            self.log_file.close()


def _wait_port_free(port: int, timeout_s: float | None = None) -> None:
    """Return once nothing listens on ``port``; else PortBusyError.

    SO_REUSEADDR lets the probe past TIME_WAIT sockets (a server stopped a
    second ago — both 2026-09-27 Gemma-4-31B runs aborted on that) while a live
    listener still refuses it; the wait covers a server still shutting down.
    """
    deadline = time.monotonic() + (_PORT_FREE_TIMEOUT_S if timeout_s is None else timeout_s)
    while True:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise PortBusyError(
                        f"port {port} already in use — stop the stale server first"
                    ) from exc
        time.sleep(1.0)


def start_server(
    *, sandbox_campaign: Path, run_slug: str, port: int, log_path: Path, dm_model: str = ""
) -> ServerHandle:
    """Spawn ``python -m aidm`` against the sandbox vault, stderr → log file.

    Refuses a busy port: a stale server left there answers /health for the
    new one, which dies on "address already in use" (2026-09-27: a whole
    marathon ran against a server orphaned hours earlier).
    """
    _wait_port_free(port)
    env = dict(os.environ)
    env.update(
        {
            "AIDM_MODE": "public" if dm_model else "private",
            "AIDM_TRANSPORT": "text",
            "AIDM_LOG_LEVEL": "DEBUG",
            "AIDM_VAULT_ROOT": str(sandbox_campaign.parent),
            "VAULT_PATH": str(sandbox_campaign),
            "RUN_SLUG": run_slug,
            "AIDM_SERVER_HOST": "127.0.0.1",
            "AIDM_SERVER_PORT": str(port),
            "CF_ACCESS_DISABLED": "1",
        }
    )
    if dm_model:
        env.update(
            {
                "OPENROUTER_MODEL": dm_model,
                # Bench run: the turn count bounds cost. The production
                # budget fuses would otherwise 503 replay 2+ of the day
                # (daily USD cap) or end a marathon session mid-run
                # (session token cap).
                "AIDM_DAILY_USD_CAP": "0",
                "AIDM_SESSION_TOKEN_CAP": "0",
                "AIDM_SESSION_TTS_CHAR_CAP": "0",
            }
        )
    env.setdefault("AIDM_LOCAL_LLM_URL", "http://localhost:8002/v1")
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "aidm"],
        cwd=str(_THIS.parents[1]),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return ServerHandle(base_url=f"http://127.0.0.1:{port}", process=process, log_file=log_file)


async def wait_health(base_url: str, timeout_s: float = _HEALTH_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    return False


def _sim_llm_url() -> str:
    return os.environ.get("AIDM_SIM_LLM_URL", "http://localhost:8004/v1").rstrip("/")


async def check_gateway(url: str | None = None) -> bool:
    url = url or os.environ.get("AIDM_LOCAL_LLM_URL", "http://localhost:8002/v1").rstrip("/")
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{url}/models")
        except httpx.HTTPError:
            return False
    return response.status_code == 200


async def check_openrouter_tools(model: str) -> bool:
    """True when the OpenRouter model advertises tool calling.

    Vault tools are load-bearing for the DM — abort before burning a
    session on a model that can't call them.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get("https://openrouter.ai/api/v1/models")
        except httpx.HTTPError:
            return False
    if response.status_code != 200:
        return False
    for entry in response.json().get("data", []):
        if entry.get("id") == model:
            return "tools" in (entry.get("supported_parameters") or [])
    return False


# --- replay -------------------------------------------------------------------


@dataclass
class TurnLog:
    turn: int
    player: str
    dm: str
    latency_s: float
    sim_rerolls: int
    degenerate: bool


@dataclass
class ReplayReport:
    scenario_id: str
    run_slug: str
    turns_requested: int
    sim_model: str
    dm_mode: str = "private"
    dm_model: str = ""
    turns: list[TurnLog] = field(default_factory=list)
    aborted_at_turn: int | None = None
    abort_reason: str | None = None
    wall_clock_s: float = 0.0


def _read_pc_sheet(sandbox_campaign: Path, run_slug: str) -> str:
    """Rowan's identity + state, straight from the sandbox run."""
    chars = sandbox_campaign / "runs" / run_slug / "characters"
    parts: list[str] = []
    for name in ("rowan.md", "rowan.state.md"):
        path = chars / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8").rstrip())
    return "\n\n".join(parts)


async def run_replay(  # noqa: PLR0911, PLR0912, PLR0913, PLR0915 — orchestration is naturally wide
    *,
    scenario: Scenario,
    turns: int,
    out_dir: Path,
    port: int,
    attach: str | None,
    sim_model: str,
    dm_model: str,
    turn_timeout_s: float,
) -> ReplayReport:
    report = ReplayReport(
        scenario_id=scenario.id,
        run_slug=scenario.run_slug,
        turns_requested=turns,
        sim_model=sim_model,
        dm_mode="public" if dm_model else "private",
        dm_model=dm_model or os.environ.get("AIDM_LOCAL_LLM_MODEL", "qwen3-14b"),
    )
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)

    campaign_slug = os.environ.get("AIDM_LOOP_CAMPAIGN", "ember-hollow")
    real_root = _REPO_ROOT / "vault"
    sandbox_root = _sandbox_vault(real_root, campaign_slug)
    sandbox_campaign = sandbox_root / campaign_slug

    server: ServerHandle | None = None
    sse: SseCapture | None = None
    current_turn = 0
    try:
        if attach:
            server = ServerHandle(base_url=attach.rstrip("/"))
        else:
            try:
                server = start_server(
                    sandbox_campaign=sandbox_campaign,
                    run_slug=scenario.run_slug,
                    port=port,
                    log_path=out_dir / "server.log",
                    dm_model=dm_model,
                )
            except PortBusyError as exc:
                report.abort_reason = str(exc)
                report.aborted_at_turn = 0
                return report
        if not await wait_health(server.base_url):
            report.abort_reason = "server never became healthy (see server.log)"
            report.aborted_at_turn = 0
            return report
        if dm_model:
            if not await check_openrouter_tools(dm_model):
                report.abort_reason = (
                    f"{dm_model} not found on OpenRouter or lacks tool support "
                    "(the DM needs vault tools)"
                )
                report.aborted_at_turn = 0
                return report
        elif not await check_gateway():
            report.abort_reason = (
                "llm-gateway unreachable — start it with "
                "`docker compose --profile private up -d llm-gateway`"
            )
            report.aborted_at_turn = 0
            return report

        if sim_model == LOCAL_SIM:
            if not await check_gateway(_sim_llm_url()):
                report.abort_reason = (
                    f"sim LLM unreachable at {_sim_llm_url()} — start it with "
                    "`docker compose --profile eval up -d sim-llm`"
                )
                report.aborted_at_turn = 0
                return report
            from openai import AsyncOpenAI  # noqa: PLC0415 — provider picked at runtime

            sim_client: Any = AsyncOpenAI(base_url=_sim_llm_url(), api_key="local-llama-cpp")
        elif "/" in sim_model:  # OpenRouter slug (vendor/model)
            from openai import AsyncOpenAI  # noqa: PLC0415 — provider picked at runtime

            sim_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            from anthropic import AsyncAnthropic  # noqa: PLC0415 — provider picked at runtime

            sim_client = AsyncAnthropic()
        async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
            response = await client.post(
                "/api/text/session",
                json={
                    "campaign_id": campaign_slug,
                    "run_id": scenario.run_slug,
                    "locale": "en",
                },
            )
            response.raise_for_status()
            session_id = str(response.json()["session_id"])
            print(f"[session] {session_id} ({scenario.id}, {turns} turns)", file=sys.stderr)

            sse = SseCapture(server.base_url, session_id, out_dir / "events.jsonl")
            sse.start()

            history: list[tuple[str, str]] = []
            priors: list[str] = []
            for turn in range(1, turns + 1):
                current_turn = turn
                pc_sheet = _read_pc_sheet(sandbox_campaign, scenario.run_slug)
                beat_intent = current_beat(scenario.beats, turn)
                player_text, rerolls, degenerate = await simulate_player(
                    sim_client,
                    sim_model,
                    pc_sheet=pc_sheet,
                    beat_intent=beat_intent,
                    history=history,
                    priors=priors,
                )
                if not player_text.strip():
                    report.aborted_at_turn = turn
                    report.abort_reason = f"turn {turn}: simulator produced empty player text"
                    return report
                priors.append(player_text)
                print(f"[turn {turn}/{turns}] player: {player_text[:80]!r}", file=sys.stderr)

                turn_started = time.monotonic()
                # An empty completion (upstream flake) yields no text and
                # no idle; one re-post recovers it — the server treats a
                # second POST as a plain interruption.
                for post_attempt in (1, 2):
                    response = await client.post(
                        f"/api/text/session/{session_id}/turn", json={"text": player_text}
                    )
                    response.raise_for_status()
                    dm_text, turn_done = await collect_turn(sse, timeout_s=turn_timeout_s)
                    if turn_done or dm_text or post_attempt == 2:
                        break
                    print(
                        f"[turn {turn}/{turns}] no DM text after "
                        f"{FIRST_TEXT_TIMEOUT_S:.0f}s — re-posting once",
                        file=sys.stderr,
                    )
                latency = time.monotonic() - turn_started
                if not turn_done:
                    report.aborted_at_turn = turn
                    report.abort_reason = f"turn {turn} never reached idle in {turn_timeout_s:.0f}s"
                    report.turns.append(
                        TurnLog(turn, player_text, dm_text, latency, rerolls, degenerate)
                    )
                    return report
                if not dm_text:
                    report.aborted_at_turn = turn
                    report.abort_reason = f"turn {turn} produced no DM reply"
                    report.turns.append(
                        TurnLog(turn, player_text, dm_text, latency, rerolls, degenerate)
                    )
                    return report

                history.append((player_text, _strip_contract_tag(dm_text)))
                report.turns.append(
                    TurnLog(turn, player_text, dm_text, latency, rerolls, degenerate)
                )
                print(
                    f"[turn {turn}/{turns}] dm ({latency:.0f}s): {dm_text[:80]!r}", file=sys.stderr
                )

            # From here a clean SSE close is the end-of-session sentinel,
            # not a mid-run drop to reconnect around.
            sse.expect_close()
            response = await client.post(f"/api/text/session/{session_id}/end")
            response.raise_for_status()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(sse.closed.wait(), timeout=_END_DRAIN_TIMEOUT_S)
        return report
    except Exception as exc:
        # Never write a clean-looking manifest for a crashed replay —
        # a truncated artifact would otherwise be scored as a short,
        # successful run.
        report.aborted_at_turn = current_turn
        report.abort_reason = f"crash: {type(exc).__name__}: {exc}"
        print(f"[replay] crashed: {report.abort_reason}", file=sys.stderr)
        return report
    finally:
        report.wall_clock_s = time.monotonic() - started
        if sse is not None:
            await sse.stop()
        if server is not None:
            server.stop()
        if not attach:
            # An attached server never touched our sandbox — a snapshot
            # of it would be pristine, misleading data.
            _snapshot(sandbox_campaign, scenario.run_slug, out_dir)
        _write_artifacts(report, out_dir)
        shutil.rmtree(sandbox_root, ignore_errors=True)


def _snapshot(sandbox_campaign: Path, run_slug: str, out_dir: Path) -> None:
    source = sandbox_campaign / "runs" / run_slug
    if not source.is_dir():
        return
    target = out_dir / "vault-snapshot"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)


def _write_artifacts(report: ReplayReport, out_dir: Path) -> None:
    transcript = [
        {
            "turn": t.turn,
            "player": t.player,
            "dm": t.dm,
            "latency_s": round(t.latency_s, 2),
            "sim_rerolls": t.sim_rerolls,
            "degenerate": t.degenerate,
        }
        for t in report.turns
    ]
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "scenario_id": report.scenario_id,
        "run_slug": report.run_slug,
        "turns_requested": report.turns_requested,
        "turns_completed": len([t for t in report.turns if t.dm]),
        "sim_model": report.sim_model,
        "dm_mode": report.dm_mode,
        "dm_model": report.dm_model,
        "dm_transport": "text",
        "aborted_at_turn": report.aborted_at_turn,
        "abort_reason": report.abort_reason,
        "wall_clock_s": round(report.wall_clock_s, 1),
        "per_turn": [
            {"turn": t.turn, "latency_s": round(t.latency_s, 2), "sim_rerolls": t.sim_rerolls}
            for t in report.turns
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --- entry point ----------------------------------------------------------------


async def _main_async(args: argparse.Namespace) -> int:
    if "/" not in args.sim_model and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is required (Anthropic PC simulator)", file=sys.stderr)
        return 2
    needs_openrouter = "/" in args.sim_model or bool(args.dm_model)
    if needs_openrouter and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required (OpenRouter sim/DM model)", file=sys.stderr)
        return 2
    scenario = load_scenario(args.scenario)
    turns = args.turns or scenario.turns
    out_root = args.out or (
        _REPO_ROOT / "docs" / "bench" / date.today().isoformat() / "loop" / scenario.id
    )
    exit_code = 0
    # Number replays past whatever already exists — re-invoking the same
    # scenario on the same day must never overwrite an earlier artifact.
    existing = 0
    if out_root.is_dir():
        for entry in out_root.glob("replay-*"):
            suffix = entry.name.removeprefix("replay-")
            if suffix.isdigit():
                existing = max(existing, int(suffix))
    for replay in range(1, args.replays + 1):
        out_dir = out_root / f"replay-{existing + replay:03d}"
        print(f"==> {scenario.id} replay {replay}/{args.replays} → {out_dir}", file=sys.stderr)
        report = await run_replay(
            scenario=scenario,
            turns=turns,
            out_dir=out_dir,
            port=args.port,
            attach=args.attach,
            sim_model=args.sim_model,
            dm_model=args.dm_model,
            turn_timeout_s=args.turn_timeout,
        )
        status = "OK" if report.aborted_at_turn is None else f"ABORTED@{report.aborted_at_turn}"
        print(
            f"{status}  {scenario.id} replay {replay}: "
            f"{len(report.turns)} turns, {report.wall_clock_s:.0f}s"
            + (f" — {report.abort_reason}" if report.abort_reason else ""),
            flush=True,
        )
        if report.aborted_at_turn is not None:
            exit_code = 1
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="loop_scenarios/*.yaml scenario file")
    parser.add_argument("--turns", type=int, default=0, help="Override scenario turn count")
    parser.add_argument("--replays", type=int, default=1, help="Replays (fresh sandbox each)")
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default docs/bench)")
    parser.add_argument("--port", type=int, default=8123, help="Port for the spawned server")
    parser.add_argument(
        "--attach",
        type=str,
        default="",
        help="Attach to a running text server URL instead of spawning (debug only — "
        "the attached server must already point at the intended vault/run)",
    )
    parser.add_argument("--sim-model", type=str, default=DEFAULT_SIM_MODEL)
    parser.add_argument(
        "--dm-model",
        type=str,
        default="",
        help="OpenRouter model id for the DM; spawns the server in AIDM_MODE=public "
        "(no effect with --attach)",
    )
    parser.add_argument("--turn-timeout", type=float, default=DEFAULT_TURN_TIMEOUT_S)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
