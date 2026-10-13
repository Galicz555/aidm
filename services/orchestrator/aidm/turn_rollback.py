"""Text-mode "regenerate the last DM reply": snapshot a turn, roll it back.

``/turn`` captures, before the player's frame is queued, everything the
turn can change: the ``LLMContext`` messages + tool menu, the UI
transcript length, the ``SceneRegistry``, the ``DirectorState``, the
private bookkeeping of the stateful processors (``Restorable``), the
tool handlers' inlined-sheet memory — and it opens a ``Vault`` write
journal, so every file the turn touches keeps its pre-image. Only the
latest turn's snapshot is kept; regenerating again reuses it.

Mutable holders are restored IN PLACE: the LLM service's guard
providers close over the live ``DirectorState``, the endpoint and the
recorder share the transcript list, and the aggregators share the
context — a replaced object would silently fork the session.

Not a pipeline stage; driven by the text server between turns, never
while one is in flight (the server's ``TurnActivity`` gate).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

from pipecat.processors.aggregators.llm_context import LLMContext

from aidm.scene_state import SceneRegistry
from aidm.session_summary import SessionSummarizer
from aidm.turn_director import DirectorState
from aidm.vault import Vault, VaultPathError


@runtime_checkable
class Restorable(Protocol):
    """A processor whose private per-session state a regenerate rolls back."""

    def snapshot(self) -> object: ...

    def restore(self, state: object) -> None: ...


@dataclass(frozen=True)
class TurnSnapshot:
    user_text: str
    messages: list[Any]
    tools: Any
    tool_choice: Any
    transcript_len: int
    registry: SceneRegistry
    director: DirectorState
    processors: tuple[tuple[Restorable, object], ...]
    inlined_sheets: frozenset[str]


class TurnRollback:
    """Owns the latest turn's snapshot and the session vault's journal."""

    def __init__(  # noqa: PLR0913 — one reference per piece of turn state
        self,
        *,
        context: LLMContext,
        transcript: list[dict[str, str]],
        registry: SceneRegistry,
        director: DirectorState,
        vault: Vault,
        processors: list[Restorable],
        summarizer: SessionSummarizer | None,
        inlined_sheets: set[str],
    ) -> None:
        self._context = context
        self._transcript = transcript
        self._registry = registry
        self._director = director
        self._vault = vault
        self._processors = processors
        self._summarizer = summarizer
        self._inlined_sheets = inlined_sheets
        self.snapshot: TurnSnapshot | None = None

    async def settle(self) -> None:
        """Let a pending summary land, so it is inside (not after) a snapshot or restore."""
        if self._summarizer is not None:
            await self._summarizer.settle()

    async def capture(self, user_text: str) -> None:
        """Snapshot the world before ``user_text`` runs; journal its disk writes from here."""
        await self.settle()
        self.snapshot = TurnSnapshot(
            user_text=user_text,
            messages=copy.deepcopy(self._context.messages),
            tools=self._context.tools,
            tool_choice=self._context.tool_choice,
            transcript_len=len(self._transcript),
            registry=copy.deepcopy(self._registry),
            director=copy.deepcopy(self._director),
            processors=tuple((p, copy.deepcopy(p.snapshot())) for p in self._processors),
            inlined_sheets=frozenset(self._inlined_sheets),
        )
        self._vault.begin_journal()

    def discard(self) -> None:
        """No regenerate from here on (the /end turn): drop the snapshot, stop journaling."""
        self.snapshot = None
        self._vault.end_journal()

    def discarded_reply(self) -> str:
        """The DM text the snapshot's turn produced (what a regenerate throws away)."""
        if self.snapshot is None:
            return ""
        landed = self._transcript[self.snapshot.transcript_len + 1 :]
        return "\n\n".join(e["text"] for e in landed if e.get("role") == "assistant")

    def turn_number(self) -> int:
        """1-based player turn of the snapshot, counted on the UI transcript."""
        if self.snapshot is None:
            return 0
        before = self._transcript[: self.snapshot.transcript_len]
        return 1 + sum(1 for e in before if e.get("role") == "user")

    def restore(self) -> None:
        """Put disk and memory back to the snapshot; a fresh journal starts for the re-run.

        Disk first: any ``Vault.restore_journal`` failure (a conflicting
        outside write, a staging error, an incomplete commit) propagates
        with memory untouched, so nothing replays over a half-restored
        vault and a retry can finish what stayed journaled.
        """
        snap = self.snapshot
        if snap is None:
            raise VaultPathError("no turn snapshot to restore")
        self._vault.restore_journal()
        self._context.set_messages(copy.deepcopy(snap.messages))
        self._context.set_tools(snap.tools)
        self._context.set_tool_choice(snap.tool_choice)
        del self._transcript[snap.transcript_len :]
        _copy_fields(copy.deepcopy(snap.registry), into=self._registry)
        _copy_fields(copy.deepcopy(snap.director), into=self._director)
        for processor, state in snap.processors:
            processor.restore(copy.deepcopy(state))
        self._inlined_sheets.clear()
        self._inlined_sheets.update(snap.inlined_sheets)
        self._vault.begin_journal()


def _copy_fields(source: object, *, into: object) -> None:
    for f in fields(source):  # type: ignore[arg-type]
        setattr(into, f.name, getattr(source, f.name))
