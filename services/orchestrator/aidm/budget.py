"""Cost guard for public-mode cloud API spend.

Two layers:

1. **Per-session cap** — a frame processor watches ``MetricsFrame``s, sums
   LLM tokens and TTS characters for the current session, and when either
   crosses ``AIDM_SESSION_TOKEN_CAP`` / ``AIDM_SESSION_TTS_CHAR_CAP`` it
   pushes a brief Hungarian goodbye and an ``EndTaskFrame`` downstream. The
   session ends, the player gets one closing sentence, no further API spend.

2. **Daily USD cap** — a SQLite ledger (one row per UTC date) accumulates
   estimated $-spend from every session's measured usage at teardown. New
   sessions check the ledger; over budget → ``BudgetExceededError`` which
   the HTTP layer turns into a 503 with a Hungarian message.

Pricing is approximate by design. We track conservatively (worst-case) and
the limits are budget *fuses*, not accounting — being off by 30% only changes
how soon the fuse trips. The numbers below are taken from public price lists
as of 2026-06; bump them when providers change pricing.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    MetricsFrame,
    StartFrame,
    TTSSpeakFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Conservative pricing in USD per unit. We always use the most expensive
# realistic backend so the fuse trips on time even if the actual provider
# is cheaper. Update when provider prices move.
_LLM_INPUT_USD_PER_1K = 0.003  # Claude Sonnet 4.5 input
_LLM_OUTPUT_USD_PER_1K = 0.015  # Claude Sonnet 4.5 output
_TTS_USD_PER_1K_CHARS = 0.30  # ElevenLabs Multilingual v2 Creator tier worst case
_STT_USD_PER_MINUTE = 0.004  # Groq whisper-large-v3-turbo


class BudgetExceededError(Exception):
    """Raised when a new session would push the daily ledger over the cap."""


_GOODBYES: dict[str, str] = {
    "hu": "A krónika ezen fejezete véget ért, kapitány. Térj vissza később.",
    "en": "This chapter of the chronicle ends here. Return another time.",
}


class DailyBudgetLedger:
    """SQLite-backed daily $-spend tracker. Thread-safe via a single lock.

    One row per UTC date. ``add_spend`` and ``current_spend`` are the only
    operations; the HTTP layer reads, sessions write at teardown.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS daily_spend ("
                "  date TEXT PRIMARY KEY,"
                "  usd REAL NOT NULL"
                ")"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, isolation_level=None)

    def current_spend_usd(self) -> float:
        today = _utc_today()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT usd FROM daily_spend WHERE date = ?", (today,)).fetchone()
        return float(row[0]) if row else 0.0

    def add_spend(self, usd: float) -> float:
        """Add ``usd`` to today's row, return the new total."""
        today = _utc_today()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO daily_spend(date, usd) VALUES(?, ?) "
                "ON CONFLICT(date) DO UPDATE SET usd = usd + excluded.usd",
                (today, usd),
            )
            row = conn.execute("SELECT usd FROM daily_spend WHERE date = ?", (today,)).fetchone()
        return float(row[0])

    def check_cap(self, cap_usd: float) -> None:
        """Raise BudgetExceededError if today's spend already meets/exceeds cap."""
        current = self.current_spend_usd()
        if current >= cap_usd:
            raise BudgetExceededError(f"daily USD cap reached: {current:.4f} >= {cap_usd:.4f}")


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


class SessionBudgetGuard(FrameProcessor):
    """Per-session token/char accumulator. Trips the session when over cap.

    Sits anywhere downstream of LLM/TTS so it sees their ``MetricsFrame``s
    (Pipecat emits these system-frames upstream too, hence the direction-
    agnostic handling). On trip:

      1. Push a ``TTSSpeakFrame`` with the locale-appropriate goodbye.
      2. Push an ``EndTaskFrame`` downstream so the pipeline winds down.

    On teardown (via ``flush_to_ledger``), commit the measured $-spend to
    the daily ledger.
    """

    def __init__(
        self,
        *,
        token_cap: int,
        tts_char_cap: int,
        locale: str,
        ledger: DailyBudgetLedger | None,
        session_id: str,
    ) -> None:
        super().__init__()
        self._token_cap = token_cap
        self._char_cap = tts_char_cap
        self._locale = locale
        self._ledger = ledger
        self._session_id = session_id
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._tts_chars = 0
        self._tripped = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, MetricsFrame):
            self._absorb_metrics(frame.data)
            await self.push_frame(frame, direction)
            await self._maybe_trip(direction)
            return

        await self.push_frame(frame, direction)

    def _absorb_metrics(self, data: Sequence[object]) -> None:
        for item in data:
            if isinstance(item, LLMUsageMetricsData):
                self._prompt_tokens += item.value.prompt_tokens
                self._completion_tokens += item.value.completion_tokens
            elif isinstance(item, TTSUsageMetricsData):
                self._tts_chars += item.value

    async def _maybe_trip(self, direction: FrameDirection) -> None:
        if self._tripped:
            return
        total_tokens = self._prompt_tokens + self._completion_tokens
        if total_tokens < self._token_cap and self._tts_chars < self._char_cap:
            return
        self._tripped = True
        logger.warning(
            "budget cap tripped: session={} tokens={}/{} tts_chars={}/{}",
            self._session_id,
            total_tokens,
            self._token_cap,
            self._tts_chars,
            self._char_cap,
        )
        goodbye = _GOODBYES.get(self._locale, _GOODBYES["en"])
        # Push downstream regardless of incoming direction — these are
        # commands aimed at TTS + pipeline worker.
        await self.push_frame(TTSSpeakFrame(goodbye), FrameDirection.DOWNSTREAM)
        await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)

    def estimated_session_usd(self) -> float:
        prompt = self._prompt_tokens / 1000.0 * _LLM_INPUT_USD_PER_1K
        completion = self._completion_tokens / 1000.0 * _LLM_OUTPUT_USD_PER_1K
        tts = self._tts_chars / 1000.0 * _TTS_USD_PER_1K_CHARS
        return prompt + completion + tts

    def flush_to_ledger(self) -> None:
        if self._ledger is None:
            return
        cost = self.estimated_session_usd()
        if cost <= 0:
            return
        # Lock + sqlite write is sync; this is called from session teardown
        # which is fine to block briefly.
        new_total = self._ledger.add_spend(cost)
        logger.info(
            "budget flushed: session={} tokens_in={} tokens_out={} tts_chars={} "
            "session_usd={:.4f} daily_total_usd={:.4f}",
            self._session_id,
            self._prompt_tokens,
            self._completion_tokens,
            self._tts_chars,
            cost,
            new_total,
        )


def make_ledger(db_path: Path) -> DailyBudgetLedger:
    """Build the singleton ledger. Cheap; safe to call once per process."""
    return DailyBudgetLedger(db_path)


async def flush_to_ledger_async(guard: SessionBudgetGuard) -> None:
    """asyncio-friendly wrapper — runs the sync sqlite write off the loop."""
    await asyncio.to_thread(guard.flush_to_ledger)
