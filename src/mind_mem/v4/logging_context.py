"""v4 structured logging context (round 4 audit, DeepSeek 9.75→10 gap).

Adds correlation-ID + key-value context propagation across v4
operations so log lines can be grouped by request, agent, or
workspace without parsing free-text messages.

Two surfaces:

    LogContext              thread-local (contextvar-backed) stack of
                            key→value bindings. Push/pop manually or
                            with the ``with_context`` context manager.

    @with_correlation_id    decorator that auto-attaches a UUID4
                            correlation_id to the wrapped function's
                            context. Useful for top-level entry
                            points (recall, consolidate, evict).

Read the active context at log time with :func:`current_context()`.
The dict is a snapshot — modifying it does not change the stack.

The module integrates cleanly with :mod:`logging` via a
:class:`StructuredLogFilter`: install it on the root logger and every
log record gains a ``ctx`` attribute carrying the current bindings,
ready to be formatted as JSON by an upstream handler.

This is dependency-free (stdlib + contextvars) so the v4 surface
stays importable on a fresh install.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

__all__ = [
    "LogContext",
    "current_context",
    "with_context",
    "with_correlation_id",
    "StructuredLogFilter",
]


_ctx_stack: contextvars.ContextVar[tuple[dict[str, Any], ...]] = contextvars.ContextVar("v4_log_ctx", default=())


class LogContext:
    """Push / pop key-value bindings on the contextvar-backed stack."""

    @staticmethod
    def push(**bindings: Any) -> contextvars.Token:
        """Push a new frame; returns a token for :meth:`pop`."""
        current = _ctx_stack.get()
        new = current + (dict(bindings),)
        return _ctx_stack.set(new)

    @staticmethod
    def pop(token: contextvars.Token) -> None:
        """Pop the frame associated with ``token``."""
        _ctx_stack.reset(token)


def current_context() -> dict[str, Any]:
    """Return the merged active context as a dict.

    Newer frames override older ones for the same key. Always returns
    a fresh dict so callers can mutate it safely.
    """
    merged: dict[str, Any] = {}
    for frame in _ctx_stack.get():
        merged.update(frame)
    return merged


@contextmanager
def with_context(**bindings: Any) -> Iterator[dict[str, Any]]:
    """Context manager: push bindings on enter, pop on exit.

    ::

        with with_context(workspace="/tmp/ws", agent_id="A"):
            recall(...)
    """
    token = LogContext.push(**bindings)
    try:
        yield current_context()
    finally:
        LogContext.pop(token)


def with_correlation_id(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that wraps the function in a ``with_context`` block
    bearing a fresh ``correlation_id`` UUID4.

    If the active context already carries a ``correlation_id`` from
    an outer call, the decorator preserves it (no overwrite). This
    lets nested calls inherit the parent's correlation ID for
    end-to-end traces.

    Works on both ``def`` and ``async def`` targets — the wrapper is
    chosen at decoration time via :func:`inspect.iscoroutinefunction`
    so async frameworks (FastAPI handlers, ``asyncio.run`` entry
    points) see the correct callable shape and ``await`` it correctly.
    Without this, an ``async def`` target would silently degrade into
    a sync function returning an unawaited coroutine.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def _inner_async(*args: Any, **kwargs: Any) -> Any:
            existing = current_context().get("correlation_id")
            cid = existing or str(uuid.uuid4())
            with with_context(correlation_id=cid):
                return await fn(*args, **kwargs)

        return _inner_async

    @functools.wraps(fn)
    def _inner(*args: Any, **kwargs: Any) -> Any:
        existing = current_context().get("correlation_id")
        cid = existing or str(uuid.uuid4())
        with with_context(correlation_id=cid):
            return fn(*args, **kwargs)

    return _inner


class StructuredLogFilter(logging.Filter):
    """Logging filter that attaches the active context to each record.

    Install on the root logger:

    ::

        logging.getLogger().addFilter(StructuredLogFilter())

    Every log record gains a ``ctx`` attribute (dict) that downstream
    JSON / structured handlers can serialize alongside the message.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.ctx = current_context()  # type: ignore[attr-defined]
        return True
