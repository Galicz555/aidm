"""Auto session-summary writer.

Long sessions overflow the local LLM's context window. This module
periodically asks the LLM to compress the older portion of the
conversation into a 200-400 word summary, and rewrites the
``LLMContext.messages`` so the next turn enters with:

    [system prompt, summary-as-system, ...last K turns]

instead of the full raw history. The summary is rolling: each run
folds the previous summary into the new one, so exactly one summary
message lives in the context. It also lands in
``vault/notes/session-<id>.md`` via the existing vault layer, so it
survives the process and feeds the Phase B persistent-resume feature.

Triggers per ``AIDM_SUMMARY_EVERY_N_TURNS`` (default 5) completed
turns, or on context pressure. Pipeline position: after
``aggregator.assistant()``, before the history trimmer, on
``LLMContextAssistantTimestampFrame`` — Pipecat's ``on_turn_ended``
never fires mid-session in text mode (no bot-speaking frames), which
left this module dead there.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from loguru import logger
from openai import AsyncOpenAI
from pipecat.frames.frames import EndFrame, Frame, LLMContextAssistantTimestampFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from aidm.config import Config
from aidm.session import SessionContext
from aidm.session_resume import RESUME_TAG
from aidm.story_events import _word_stems
from aidm.turn_director import DirectorState
from aidm.vault import Vault, VaultPathError

DEFAULT_TURNS_BEFORE_SUMMARY = 5
DEFAULT_TURNS_TO_KEEP = 4
# Below this many user turns in the "older" portion the summarization
# is not worth a round-trip — leave the context alone.
MIN_OLD_USER_TURNS = 2
# Hard cap on the summary tokens. Local LLMs can wander; capping keeps
# the summary tight enough that it does not itself blow the context.
# 400 words of prose (~530 tokens) plus the open-threads section.
SUMMARY_MAX_TOKENS = 750
MAX_OPEN_THREADS = 5
_THREADS_HEADER = "Open threads:"
# The header line alone (markdown-decorated or "Open threads: none"); a
# prose sentence that merely starts with the words does not match.
_THREADS_HEADER_RE = re.compile(
    r"^[#* \t]*open threads[* \t]*(?::[^\n]*)?$", re.IGNORECASE | re.MULTILINE
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
_NON_THREADS = {"none", "n/a", "nothing"}
# ASCII arrow, but not the "-->" that closes an age marker.
_ASCII_ARROW_RE = re.compile(r"(?<!-)->")
# A thread the summarizer carries verbatim this many summaries (~15 turns)
# is dropped unless the excerpt just summarized still mentions it — the
# local 14B copies the prior list forward instead of resolving it.
THREAD_MAX_AGE = 3
# Persisted per bullet so a resume / WebRTC rebuild keeps the age; absent = 0.
_AGE_MARK_RE = re.compile(r"\s*<!--aidm:age=(\d+)-->$")
# Capitalized words in a thread's cause side: its subject (usually a name).
_SUBJECT_RE = re.compile(r"\b[A-Z][a-zA-Z\u00C0-\u017F]{3,}")
# Approximate characters per token. 4 is the canonical rule of thumb
# for English; Hungarian is a bit denser (3-3.5) so we err low to
# trigger compression earlier rather than later.
APPROX_CHARS_PER_TOKEN = 3.5
# Pressure threshold: when the estimated prompt size exceeds this
# fraction of the configured context window, the summarizer fires
# regardless of turn count.
DEFAULT_PRESSURE_RATIO = 0.7
_SUMMARY_TAG = "<!--aidm:session-summary-->"

SummaryEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class SessionSummarizer(FrameProcessor):
    """Per-session summarization coordinator.

    Sits after the sheet writer and runs each summary in its own task, so
    the end-of-turn frame passes straight through and user-turn
    interruptions (which cancel a processor's frame task) can't kill it.
    The next turn may land mid-call — the live-list splice keeps it. The
    turn counter only resets on success so a failed call is retried on
    the next turn.
    """

    def __init__(  # noqa: PLR0913 — context + config + session + vault + 2 shared channels
        self,
        *,
        context: LLMContext,
        config: Config,
        session: SessionContext,
        vault: Vault,
        on_event: SummaryEventCallback | None = None,
        director: DirectorState | None = None,
    ) -> None:
        super().__init__()
        self._context = context
        self._config = config
        self._session = session
        self._vault = vault
        self._on_event = on_event
        self._director = director or DirectorState()
        # A resumed session (notes-file memory) or a WebRTC grace rebuild
        # (stashed context) carries the last thread list in its text.
        self._thread_ages: dict[str, int] = {}
        # Keys of aged-out threads: the 14B re-lists them verbatim, which would
        # restart them at age 0 and cycle. Simplification: in-memory, a resume forgets them.
        self._retired_threads: set[str] = set()
        if not self._director.open_threads:
            carried = _carried_threads(context.messages or [])
            self._director.open_threads = [t for t, _ in carried]
            self._thread_ages = {_thread_key(t): age for t, age in carried}
        self._client = _build_summary_client(config)
        self._model = _summary_model_name(config)
        self._turns_since_summary = 0
        self._summary_index = 0
        self._task: asyncio.Task[None] | None = None
        self._turns_before = _int_env("AIDM_SUMMARY_EVERY_N_TURNS", DEFAULT_TURNS_BEFORE_SUMMARY)
        self._turns_to_keep = _int_env("AIDM_SUMMARY_KEEP_LAST_N", DEFAULT_TURNS_TO_KEEP)
        # Pressure only guards the local gateway's window (compose default
        # 16384). API models have far larger windows; 0 disables pressure
        # there and the turn trigger alone paces summaries.
        self._ctx_size = _int_env("AIDM_LLM_CTX_SIZE", 16384) if config.mode == "private" else 0
        self._pressure_ratio = _float_env("AIDM_SUMMARY_PRESSURE_RATIO", DEFAULT_PRESSURE_RATIO)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, LLMContextAssistantTimestampFrame
        ):
            # Own task, not awaited here: every text-mode user turn broadcasts an
            # interruption that cancels this processor's frame task, which silently
            # killed every mid-session summary (2026-09-25 marathon: 0 of ~9).
            if self._task is not None and not self._task.done():
                self._turns_since_summary += 1
            else:
                self._task = asyncio.create_task(self._summarize_logged())
        elif isinstance(frame, EndFrame) and self._task is not None:
            await self._task  # let the farewell summary land before shutdown
        await self.push_frame(frame, direction)

    async def settle(self) -> None:
        """Wait out an in-flight summary so its context splice and note write land.

        ``asyncio.wait`` neither raises the task's outcome nor forwards a
        cancellation of the caller into the summary.
        """
        if self._task is not None and not self._task.done():
            await asyncio.wait([self._task])

    def snapshot(self) -> object:
        """Counters a text regenerate rolls back; call after ``settle``."""
        return (
            self._turns_since_summary,
            self._summary_index,
            dict(self._thread_ages),
            set(self._retired_threads),
        )

    def restore(self, state: object) -> None:
        turns, index, ages, retired = cast("tuple[int, int, dict[str, int], set[str]]", state)
        self._turns_since_summary, self._summary_index = turns, index
        self._thread_ages, self._retired_threads = dict(ages), set(retired)

    async def _summarize_logged(self) -> None:
        try:
            await self.maybe_summarize(self._context)
        except Exception as exc:
            logger.warning("session summary crashed end-of-turn: {}: {}", type(exc).__name__, exc)

    async def maybe_summarize(self, context: LLMContext) -> None:
        self._turns_since_summary += 1
        # Two independent triggers: hit the turn count, OR the
        # estimated context-fill ratio is past the pressure threshold.
        # The second guard catches runaway-prose sessions that would
        # otherwise march straight into a 400-token overflow before the
        # turn-counter fires.
        turn_trigger = self._turns_since_summary >= self._turns_before
        pressure_trigger = self._under_pressure(context)
        if not (turn_trigger or pressure_trigger):
            return
        if pressure_trigger and not turn_trigger:
            logger.info(
                "summary triggered by context pressure (~{} tok / {} ctx)",
                self._estimate_tokens(context),
                self._ctx_size,
            )
        if await self._summarize(context):
            self._turns_since_summary = 0
            self._summary_index += 1

    def _under_pressure(self, context: LLMContext) -> bool:
        if self._ctx_size <= 0:
            return False
        estimate = self._estimate_tokens(context)
        return estimate >= int(self._ctx_size * self._pressure_ratio)

    def _estimate_tokens(self, context: LLMContext) -> int:
        # Char-count over all messages divided by the rule-of-thumb
        # chars-per-token. Cheap (no tokenizer dep) and only needs to
        # be approximately right — we err low so the threshold trips
        # earlier rather than later.
        total_chars = 0
        for message in getattr(context, "messages", None) or []:
            total_chars += len(_render_message_for_estimate(message))
        return int(total_chars / APPROX_CHARS_PER_TOKEN)

    async def _summarize(self, context: LLMContext) -> bool:
        messages = list(context.messages or [])
        split = _find_split_index(messages, keep_last_user_turns=self._turns_to_keep)
        if split is None:
            logger.debug(
                "summary skipped: not enough older turns (messages={}, keep={})",
                len(messages),
                self._turns_to_keep,
            )
            return False

        head = messages[:split]
        tail = messages[split:]
        old_user_turns = _count_user_messages(head)
        if old_user_turns < MIN_OLD_USER_TURNS:
            logger.debug(
                "summary skipped: only {} old user turns (<{} required)",
                old_user_turns,
                MIN_OLD_USER_TURNS,
            )
            return False

        system_messages = _take_leading_system_messages(head)
        # Before the first summary, the resumed session's memory is the
        # prior — otherwise this session's notes file forgets it.
        prior_text = next(
            (_unwrap_summary(m) for m in system_messages if _is_summary(m)), ""
        ) or next((_unwrap_summary(m) for m in system_messages if _is_resume_memory(m)), "")
        prior, _ = _split_threads(prior_text)  # threads ride separately, from the director
        prev_threads = list(self._director.open_threads)
        kept_system = [m for m in system_messages if not _is_summary(m)]
        try:
            raw = await self._ask_llm_to_summarize(head, prior, prev_threads)
        except Exception:
            logger.exception("session summary failed: LLM call raised")
            return False
        prose, parsed = _split_threads(raw)
        if not prose:
            logger.warning("session summary failed: completion had no prose before the threads")
            return False
        # Simplification: a missing/empty section keeps the old list, so threads can never
        # drop to zero via a summary — a resolve-everything completion reads as
        # "no section". Add an explicit "none" sentinel if that shows in runs.
        candidates = parsed or prev_threads
        prev_ages = {k: self._thread_ages.get(k, 0) for k in map(_thread_key, prev_threads)}
        threads, ages, aged_out = _age_threads(
            candidates, prev_ages, _excerpt_text(head), self._retired_threads
        )
        summary_text = _render_summary(prose, threads, ages)

        # Splice against the LIVE list, not the snapshot: a barge-in (voice)
        # can append a turn during the summary call. Only an unchanged head
        # prefix is safe to replace; anything after it is kept as-is.
        current = context.messages
        if len(current) < split or any(
            a is not b for a, b in zip(current[:split], head, strict=True)
        ):
            logger.info("session summary discarded: history head changed mid-summary")
            return False
        summary_system: Any = {
            "role": "system",
            "content": _wrap_summary_for_context(summary_text, self._session.locale),
        }
        context.set_messages([*kept_system, summary_system, *current[split:]])
        self._director.open_threads = threads
        self._thread_ages = ages
        self._retired_threads = (self._retired_threads | set(map(_thread_key, aged_out))) - set(
            ages
        )
        # The free per-run metric: grep this line instead of paying for a judge.
        logger.info(
            "open_threads: {} open ({} dropped since last, {} aged out){}",
            len(threads),
            len(set(prev_threads) - set(candidates)),
            len(aged_out),
            "" if parsed else " — thread section missing or empty, kept previous",
        )

        # Persist to vault. Failure is non-fatal — the in-memory rewrite
        # already happened; vault is a nice-to-have for cross-session
        # resume (Phase B step 3).
        try:
            vault_path = self._vault.append_note(
                f"session-{self._session.session_id}",
                summary_text,
            )
            logger.info(
                "session summary written: turns={} keep={} → {}",
                old_user_turns,
                self._turns_to_keep,
                vault_path,
            )
        except VaultPathError as exc:
            logger.warning("session summary not persisted: {}", exc)

        await self._emit_event(
            {
                "type": "debug.summary",
                "index": self._summary_index + 1,
                "summarized_turns": old_user_turns,
                "kept_turns": _count_user_messages(tail),
                "summary": summary_text,
                "open_threads": threads,
            }
        )
        return True

    async def _ask_llm_to_summarize(self, head: list[Any], prior: str, threads: list[str]) -> str:
        # Render the head into a plain chat transcript the LLM can
        # compress. We strip role-internal structure (tool calls etc.)
        # since the summary is for narrative continuity, not for
        # replaying tool exchanges.
        transcript = _render_transcript(head)
        prompt = _summary_prompt(self._session.locale, transcript, prior, threads)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=SUMMARY_MAX_TOKENS,
            timeout=self._config.npc_failsafe_timeout_s,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("summary LLM returned empty content")
        return text

    async def _emit_event(self, payload: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        result = self._on_event(payload)
        if asyncio.iscoroutine(result):
            await result


def _build_summary_client(config: Config) -> AsyncOpenAI:
    if config.mode == "private":
        return AsyncOpenAI(api_key="local-llama-cpp", base_url=config.local_llm_base_url)
    return AsyncOpenAI(
        api_key=config.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _summary_model_name(config: Config) -> str:
    return config.local_llm_model if config.mode == "private" else config.openrouter_model


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        logger.warning("invalid float for {}={!r}, using default {}", name, value, default)
        return default


def _render_message_for_estimate(message: Any) -> str:
    """Cheap char-count of a message's content for token estimation."""
    raw = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(raw)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        logger.warning("invalid int for {}={!r}, using default {}", name, value, default)
        return default


def _find_split_index(messages: list[Any], *, keep_last_user_turns: int) -> int | None:
    """Index where ``head`` ends and ``tail`` (kept-as-is) begins.

    We walk from the end backward and count user-role messages. The
    split lands right before the Kth-from-last user message so the
    tail covers exactly ``keep_last_user_turns`` user turns plus their
    accompanying assistant/tool replies.
    """
    if keep_last_user_turns <= 0:
        return None
    seen_users = 0
    for i in range(len(messages) - 1, -1, -1):
        role = _role_of(messages[i])
        if role == "user":
            seen_users += 1
            if seen_users == keep_last_user_turns:
                # The head must hold at least one user turn for the
                # summary to be meaningful — otherwise return None and
                # let the next on_turn_ended retry.
                if _count_user_messages(messages[:i]) == 0:
                    return None
                return i
    return None


def _count_user_messages(messages: list[Any]) -> int:
    return sum(1 for m in messages if _role_of(m) == "user")


def _take_leading_system_messages(messages: list[Any]) -> list[Any]:
    """Return the contiguous system-role messages at the head of the list.

    The system prompt always sits at index 0; some adapters add a
    secondary system message for tool guidance. Either way we want to
    preserve them as-is so the persona doesn't get rewritten by the
    summary.
    """
    leading: list[Any] = []
    for message in messages:
        if _role_of(message) != "system":
            break
        leading.append(message)
    return leading


def _role_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def _render_transcript(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        role = _role_of(message)
        if role == "system":
            continue  # the persona is constant; do not feed it to itself
        content = _content_of(message)
        if not content:
            continue
        label = {"user": "PLAYER", "assistant": "DM", "tool": "TOOL"}.get(role, role.upper())
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def _content_of(message: Any) -> str:
    raw = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p.strip() for p in parts if p.strip())
    return str(raw).strip()


# English-only for both locales (docs/decisions/english-only.md); the
# Hungarian prose instructions above it stay as they were.
_THREADS_INSTRUCTIONS = (
    "After the prose summary, add a separate section: the line "
    f'"{_THREADS_HEADER}" followed by at most {MAX_OPEN_THREADS} bullets, one '
    'line each, in the form "- cause → pending consequence". List only what '
    "is still unresolved at the end of the excerpt: promises and "
    "foreshadowing not yet paid off, pending threats, open questions, and "
    "invented characters or entities whose role is still open. Name people, "
    "places, numbers and facts exactly as the excerpt states them. The "
    "prose-only rule applies to the summary, not to this section.\n\n"
)


def _threads_block(threads: Sequence[str]) -> str:
    if not threads:
        return _THREADS_INSTRUCTIONS
    bullets = "\n".join(f"- {t}" for t in threads)
    return (
        "Open threads carried from earlier in the session. Drop every thread "
        "the excerpt resolved; keep every thread it did not resolve (reword "
        "one only if the excerpt moved it forward).\n"
        f"=== Earlier open threads ===\n{bullets}\n=== End ===\n\n"
        f"{_THREADS_INSTRUCTIONS}"
    )


def _summary_prompt(
    locale: str, transcript: str, prior: str = "", threads: Sequence[str] = ()
) -> str:
    threads_part = _threads_block(threads)
    if locale == "hu":
        prior_block = (
            "A session eddigi összefoglalója (ez a részlet ELŐTT történt). "
            "Dolgozd bele: tartsd meg ami még számít, a részletben lezárult "
            "szálakat már ne említsd nyitottként.\n"
            f"=== Korábbi összefoglaló ===\n{prior}\n=== Vége ===\n\n"
            if prior
            else ""
        )
        return (
            f"{prior_block}"
            "Az alábbi egy szerepjáték-session részlete egy mesélő (DM) és egy "
            "játékos között. Írj EGY összefoglalót a session egészéről eddig, "
            "magyarul, 200-400 szóban, hogy a "
            "mesélő a következő fordulóban folytatni tudja anélkül hogy "
            "elveszítené a fonalat.\n\n"
            "A summary fókuszáljon:\n"
            "- Kik szerepelnek (játékos karaktere, NPC-k névvel)\n"
            "- Hol vagyunk most (helyszín)\n"
            "- A főbb események időrendben\n"
            "- A játékos fontos döntései és azok következményei\n"
            "- Nyitott szálak, függő fenyegetések, megoldatlan rejtélyek\n"
            "- A karakter állapota (sebek, lelki állapot, fontos tárgyak)\n\n"
            "Ne találj ki semmit. Csak amit a részlet tényleg tartalmaz.\n"
            "Próza, NEM listák. Múlt időben.\n\n"
            f"{threads_part}"
            "=== Session-részlet ===\n"
            f"{transcript}\n"
            "=== Vége ==="
        )
    prior_block = (
        "Summary of the session so far (it happened BEFORE the excerpt). "
        "Fold it in: keep what still matters, and do not list threads the "
        "excerpt resolved as still open.\n"
        f"=== Earlier summary ===\n{prior}\n=== End ===\n\n"
        if prior
        else ""
    )
    return (
        f"{prior_block}"
        "Below is a tabletop RPG session excerpt between a DM and a player. "
        "Write ONE summary of the whole session so far, in English, "
        "200-400 words, so the DM can continue the "
        "next turn without losing the thread.\n\n"
        "The summary should cover:\n"
        "- Who is involved (player character, NPCs by name)\n"
        "- Where we are now (location)\n"
        "- Main events in order\n"
        "- The player's key decisions and their consequences\n"
        "- Open threads, pending threats, unsolved mysteries\n"
        "- The character's state (wounds, mood, important items)\n\n"
        "Do not invent anything. Only what the excerpt actually contains.\n"
        "Prose, NOT lists. Past tense.\n\n"
        f"{threads_part}"
        "=== Session excerpt ===\n"
        f"{transcript}\n"
        "=== End ==="
    )


def _wrap_summary_for_context(summary: str, locale: str) -> str:
    if locale == "hu":
        return (
            f"{_SUMMARY_TAG}\n"
            "[A session korábbi részének automatikus összefoglalója — kezeld "
            "úgy mintha a saját emléked lenne, ne idézz belőle szó szerint, "
            "de használd a folytonossághoz:]\n\n"
            f"{summary}"
        )
    return (
        f"{_SUMMARY_TAG}\n"
        "[Automated summary of the earlier session — treat it as your own "
        "memory; do not quote it verbatim, use it for continuity:]\n\n"
        f"{summary}"
    )


def _is_summary(message: Any) -> bool:
    return _role_of(message) == "system" and _content_of(message).startswith(_SUMMARY_TAG)


def _is_resume_memory(message: Any) -> bool:
    return _role_of(message) == "system" and _content_of(message).startswith(RESUME_TAG)


def _unwrap_summary(message: Any) -> str:
    return _content_of(message).partition("\n\n")[2]


def _split_threads(text: str) -> tuple[str, list[str] | None]:
    """Split a summary into (prose, threads); threads is None without a section.

    The last "Open threads" header line wins; bullets after it (``-``,
    ``*``, ``•``, ``1.``) are the threads, capped at MAX_OPEN_THREADS.
    """
    headers = list(_THREADS_HEADER_RE.finditer(text))
    if not headers:
        return text.strip(), None
    threads: list[str] = []
    for line in text[headers[-1].end() :].splitlines():
        match = _BULLET_RE.match(line)
        if match and match.group(1).strip(" .").lower() not in _NON_THREADS:
            threads.append(_ASCII_ARROW_RE.sub("→", match.group(1)))
    return text[: headers[-1].start()].strip(), threads[:MAX_OPEN_THREADS]


def _render_summary(prose: str, threads: Sequence[str], ages: dict[str, int] | None = None) -> str:
    """The persisted/in-context form; ``_carried_threads`` reads it back, ages too."""
    if not threads:
        return prose
    ages = ages or {}
    bullets = "\n".join(
        f"- {t}" + (f" <!--aidm:age={a}-->" if (a := ages.get(_thread_key(t), 0)) else "")
        for t in threads
    )
    return f"{prose}\n\n{_THREADS_HEADER}\n{bullets}"


def _carried_threads(messages: list[Any]) -> list[tuple[str, int]]:
    """(thread, age) from a rolling summary, else from resume memory, else none."""
    leading = _take_leading_system_messages(messages)
    carriers = [m for m in leading if _is_summary(m)] + [m for m in leading if _is_resume_memory(m)]
    for message in carriers:
        threads = _split_threads(_unwrap_summary(message))[1]
        if threads:
            return [_unmark_age(t) for t in threads]
    return []


def _unmark_age(bullet: str) -> tuple[str, int]:
    match = _AGE_MARK_RE.search(bullet)
    return (bullet[: match.start()], int(match.group(1))) if match else (bullet, 0)


def _thread_key(thread: str) -> str:
    """Wording identity: case, spacing, curly quotes and a final period don't count."""
    return " ".join(thread.replace("\u2019", "'").casefold().split()).rstrip(".")


def _excerpt_text(head: Sequence[Any]) -> str:
    """The summarized turns' text, without role labels ("PLAYER" would match threads)."""
    return "\n".join(_content_of(m) for m in head if _role_of(m) != "system")


def _age_threads(
    threads: Sequence[str],
    prev_ages: dict[str, int],
    excerpt: str,
    retired: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[str], dict[str, int], list[str]]:
    """Age verbatim-carried threads; drop stale ones. Returns (kept, ages, aged_out).

    A thread whose key is in ``prev_ages`` was carried verbatim and ages by one;
    a new or reworded one starts at 0. At THREAD_MAX_AGE it is dropped unless
    ``excerpt`` still mentions it. A ``retired`` (previously aged-out) thread the
    model re-lists verbatim is dropped at once unless the excerpt mentions it.
    """
    excerpt_stems = _word_stems(excerpt)
    kept: list[str] = []
    ages: dict[str, int] = {}
    aged_out: list[str] = []
    for thread in threads:
        key = _thread_key(thread)
        age = prev_ages[key] + 1 if key in prev_ages else 0
        stale = age >= THREAD_MAX_AGE or key in retired
        if stale and not _thread_mentioned(thread, excerpt_stems):
            aged_out.append(thread)
            continue
        kept.append(thread)
        ages[key] = age
    return kept, ages, aged_out


def _thread_mentioned(thread: str, excerpt_stems: set[str]) -> bool:
    """The excerpt still touches the thread's cause side (text before the arrow).

    With a capitalized subject ("Lyra's approach"), that subject plus one more
    cause word must appear; without one, half the cause words. The consequence
    side is speculation the prose never repeats, and whole-thread fractional
    overlap (``story_events.directive_covered``) passes on a single generic word
    ("closer") in a 5-turn excerpt — replay 2026-09-26 marathon-50t kept a
    departed NPC alive that way.
    Simplification: 5-char stem overlap, no synonyms — a thread paraphrased entirely
    ("the radio hiss" for "static on the radio") ages out; it can be re-listed.
    """
    cause = thread.partition("→")[0]
    stems = _word_stems(cause)
    subject = _word_stems(" ".join(_SUBJECT_RE.findall(cause)))
    hits = stems & excerpt_stems
    if subject:
        return bool(subject & hits) and bool(hits - subject)
    return 0 < len(stems) <= 2 * len(hits)
