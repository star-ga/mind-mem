#!/usr/bin/env python3
"""Per-unit timeouts that actually preempt, for long benchmark runs.

Why this module exists
----------------------
``benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md`` records, as one of the two
engineering blockers on a full 500-question run, that::

    signal.alarm per-question timeout cannot preempt native
    (torch/sqlite/regex) stages, so pathological haystacks are not
    skippable in-process.

That is not a tuning problem, it is a structural one. CPython runs a Python
signal handler from the *eval loop*, between bytecodes. A C call that holds
the GIL and does not return -- a catastrophic-backtracking regex, a tensor
op, a sqlite query on a pathological plan -- never gets back to the eval
loop, so the handler never runs. ``SIGALRM`` is delivered, the flag is set,
and nothing happens. The process has to be killed from outside.

A thread does not help either: Python has no way to kill a thread, so a
watchdog thread can observe the hang and still not end it.

What actually preempts is a separate process and a signal the process
cannot ignore. :func:`run_with_hard_timeout` runs the unit of work in a
child, waits out the deadline, and then ``SIGKILL``s the child's whole
process group -- descendants included, because a worker pool that outlives
its parent is the same hang wearing a different pid.

Not a redesign of the scorer: the caller still decides what the unit of
work is and how it is scored. This module only owns "run it, or give up on
it in bounded time, and say which happened".
"""

from __future__ import annotations

import multiprocessing
import os
import queue as _queue
import signal
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

#: The unit finished and returned a value.
OK = "ok"
#: The unit raised. The exception is reported, not re-raised: one bad
#: question must not end a 500-question run.
ERROR = "error"
#: The deadline passed with the child still alive; the child was killed.
TIMEOUT = "timeout"
#: The child died without reporting -- a segfault, an OOM kill, a native
#: abort. Distinguished from ERROR because no Python exception exists.
CRASHED = "crashed"

#: How long to wait for a killed child to be reaped before giving up on it.
_REAP_SECONDS = 10.0
#: Parent poll interval while waiting on the child.
_POLL_SECONDS = 0.02


@dataclass(frozen=True)
class Outcome:
    """What became of one unit of work. Never a bare value -- always a status.

    ``status`` is the load-bearing field: a caller that reads ``value``
    without reading ``status`` cannot tell a real ``None`` from a hang, and
    that confusion is exactly how a timed-out question becomes a zero in a
    recall average.
    """

    status: str
    value: Any = None
    error: str = ""
    elapsed_s: float = 0.0
    exitcode: int | None = None
    killed: bool = False

    @property
    def ok(self) -> bool:
        return self.status == OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 3),
            "exitcode": self.exitcode,
            "killed": self.killed,
        }


def _child(result_q: "multiprocessing.Queue[Any]", func: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    """Run ``func`` in the child and post the outcome back.

    ``setsid`` first: it makes this process its own group leader, so the
    parent can kill the whole subtree with one ``killpg``. Without it a
    hung native stage that had already forked workers leaves them running
    after the parent "recovered".
    """
    try:
        os.setsid()
    except OSError:  # pragma: no cover - already a leader, or no setsid
        pass
    try:
        result_q.put((OK, func(*args, **kwargs), ""))
    except BaseException:  # noqa: BLE001 - the child reports, never crashes silently
        result_q.put((ERROR, None, traceback.format_exc()))


def _kill_tree(proc: "multiprocessing.process.BaseProcess") -> bool:
    """SIGKILL the child and every descendant it leads. Returns True if reaped."""
    pid = proc.pid
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # The child may not have reached setsid yet, or may already be
            # gone. Fall back to killing just the child.
            try:
                proc.kill()
            except (ProcessLookupError, OSError, ValueError):  # pragma: no cover
                pass
    proc.join(_REAP_SECONDS)
    return not proc.is_alive()


def run_with_hard_timeout(
    func: Callable[..., Any],
    timeout_s: float,
    *args: Any,
    start_method: str = "spawn",
    **kwargs: Any,
) -> Outcome:
    """Run ``func(*args, **kwargs)`` in a child, killing it after ``timeout_s``.

    ``func``, its arguments and its return value must be picklable --
    ``spawn`` is the default start method because the stack this harness
    drives loads CUDA, and CUDA does not survive ``fork``. Pass
    ``start_method="fork"`` only where that is known not to apply.

    Returns an :class:`Outcome`; never raises for a failure of ``func``.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    # ``Any``: typeshed types ``get_context`` as ``BaseContext``, which does
    # not declare ``Process``/``Queue`` even though every real context has
    # them. Narrowing here rather than ignoring the error at each use.
    ctx: Any = multiprocessing.get_context(start_method)
    result_q: "multiprocessing.Queue[Any]" = ctx.Queue()
    proc = ctx.Process(target=_child, args=(result_q, func, args, kwargs), daemon=False)
    started = time.monotonic()
    proc.start()
    deadline = started + timeout_s

    payload: tuple[str, Any, str] | None = None
    timed_out = False
    while True:
        try:
            payload = result_q.get_nowait()
            break
        except _queue.Empty:
            pass
        if not proc.is_alive():
            # Exited. Give the queue a moment to flush a result posted just
            # before exit, then accept that there is none (a crash).
            try:
                payload = result_q.get(timeout=0.5)
            except _queue.Empty:
                payload = None
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(_POLL_SECONDS)

    if timed_out:
        reaped = _kill_tree(proc)
        elapsed = time.monotonic() - started
        return Outcome(
            status=TIMEOUT,
            error=f"unit exceeded {timeout_s:g}s and was killed" + ("" if reaped else " (child did not reap)"),
            elapsed_s=elapsed,
            exitcode=proc.exitcode,
            killed=True,
        )

    proc.join(_REAP_SECONDS)
    elapsed = time.monotonic() - started
    if payload is None:
        return Outcome(
            status=CRASHED,
            error=f"child exited with code {proc.exitcode} without reporting a result",
            elapsed_s=elapsed,
            exitcode=proc.exitcode,
        )
    status, value, error = payload
    return Outcome(status=status, value=value, error=error, elapsed_s=elapsed, exitcode=proc.exitcode)


# ---------------------------------------------------------------------------
# Hang fixtures -- the proof that this mechanism preempts what signal.alarm
# does not. They live here, not in the test file, because ``spawn`` has to
# import the target's module by name in a fresh interpreter.
# ---------------------------------------------------------------------------


#: A recursive CTE with a bound no run will reach. SQLite evaluates it
#: entirely inside its own C loop, so the interpreter never reaches a
#: bytecode boundary and a pending Python signal handler is never called.
#: ``count(*)`` keeps it CPU-bound rather than memory-bound -- a fixture that
#: OOMs the box would "prove" the timeout for the wrong reason.
_HANG_SQL = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 1000000000000) SELECT count(*) FROM c"


def native_sqlite_hang(*_args: Any, **_kwargs: Any) -> str:  # pragma: no cover - killed, never returns
    """Burn effectively forever inside SQLite's C query loop.

    ``sqlite3`` is one of the three stages ``LONGMEMEVAL_FINDINGS`` named as
    unskippable, and it is unskippable for the structural reason: the query
    runs in C, and a Python ``SIGALRM`` handler only runs from the eval
    loop, which this call never returns to.

    Measured on this interpreter (2026-09-03, CPython 3.14): with
    ``signal.alarm(1)`` armed, the handler does NOT run and the process has
    to be killed from outside. Note that a catastrophic-backtracking regex
    is NOT a valid fixture here -- CPython's ``sre`` polls for signals, so
    the alarm handler does fire on it. The fixture has to be a call that
    genuinely never checks.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(_HANG_SQL).fetchone()
    return "never reached"


def quick_value(value: Any = 42) -> Any:
    """Return promptly -- the control that proves the runner runs anything."""
    return value


def raises_value_error(message: str = "boom") -> None:
    """Raise -- the control that proves an error is reported, not swallowed."""
    raise ValueError(message)
