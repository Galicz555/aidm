"""Loguru configuration with per-session / per-turn context.

ContextVars carry session_id and turn_id across async boundaries. Loguru's
patcher injects them into every record so the format string can render them
without manual `.bind()` at each call site.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
# A one-slot mutable holder, not a plain int: every Pipecat processor runs
# in its own task with a *copy* of the context, so a plain ``set`` in one
# processor never reaches its siblings. ``set_session`` installs the
# holder before the pipeline tasks spawn; they all share the reference.
_turn_id: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar("turn_id", default=None)


def set_session(session_id: str) -> None:
    _session_id.set(session_id)
    _turn_id.set([0])


def set_turn(turn_id: int) -> None:
    holder = _turn_id.get()
    if holder is None:
        _turn_id.set([turn_id])
    else:
        holder[0] = turn_id


def current_turn() -> int:
    holder = _turn_id.get()
    return holder[0] if holder is not None else 0


def configure(level: str = "INFO") -> None:
    """Three log tiers:

    * ``INFO``  — turn anchors: user input, reply timing, writer results.
    * ``DEBUG`` — + turn narrative (aggregators, injector, dispatch phases).
      Full prompt dumps, per-chunk emits and HTTP wire logs are excluded
      so a session log stays readable.
    * ``TRACE`` — the firehose: everything, including pipecat's full
      context dumps and httpx/httpcore/openai wire-level logs.
    """
    level = level.upper()
    trace = level == "TRACE"
    logger.remove()
    logger.configure(patcher=_inject_context)  # type: ignore[arg-type]
    logger.add(
        sys.stderr,
        level=level,
        # Pipecat dumps the full message history ("Generating chat from
        # context [...]" — tens of KB) at DEBUG. Only TRACE shows it;
        # our TextDebugLogger already carries the same payload to the
        # browser debug panel.
        filter=None if trace else _drop_context_dumps,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> "
            "<level>{level: <7}</level> "
            "<cyan>[{extra[session_id]}/t{extra[turn_id]}]</cyan> "
            "<level>{message}</level>"
        ),
    )
    # Bridge stdlib loggers into the loguru sink. At TRACE they come in
    # at DEBUG (request options, connect_tcp, chunk reads — the works);
    # at DEBUG and below only warnings/errors surface, because our own
    # dispatch/writer logs already cover the happy path.
    if level in ("DEBUG", "TRACE"):
        bridge_level = logging.DEBUG if trace else logging.WARNING
        _bridge_stdlib_loggers(("httpx", "httpcore", "openai"), bridge_level)


def _drop_context_dumps(record: Record) -> bool:
    return "Generating chat from context" not in record["message"]


def _bridge_stdlib_loggers(names: tuple[str, ...], bridge_level: int) -> None:
    """Route the named stdlib loggers into loguru at the given level."""

    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            level_name = logger.level(record.levelname).name if record.levelname else "DEBUG"
            logger.opt(depth=6, exception=record.exc_info).log(
                level_name,
                "[{}] {}",
                record.name,
                record.getMessage(),
            )

    handler = _InterceptHandler()
    for name in names:
        std = logging.getLogger(name)
        std.handlers = [handler]
        std.setLevel(bridge_level)
        std.propagate = False


def _inject_context(record: dict[str, Any]) -> None:
    record["extra"].setdefault("session_id", _session_id.get())
    record["extra"].setdefault("turn_id", current_turn())
