"""The per-question timeout must preempt a NATIVE hang, not just a Python one.

``benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md`` names this as one of the two
blockers on a full LongMemEval-S run: ``signal.alarm`` cannot stop a stage that
is inside a C call, because a Python signal handler only runs from the eval
loop and a C call that holds the GIL never returns to it.

These tests are paired on purpose. The first is the **positive control for the
defect**: it shows ``signal.alarm`` genuinely failing to preempt, so the second
test is measuring a real recovery rather than reciting one. A "the new
mechanism works" test with no demonstration that the old one does not would
pass just as happily against a hang that was never a hang.
"""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - fixed argv, shell=False, local interpreter
import sys
import textwrap
import time

import pytest

from benchmarks.hard_timeout import (
    CRASHED,
    ERROR,
    OK,
    TIMEOUT,
    native_sqlite_hang,
    quick_value,
    raises_value_error,
    run_with_hard_timeout,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "setsid") or sys.platform.startswith("win"),
    reason="process-group kill is POSIX-only; the Windows path uses a different primitive",
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Generous enough that a loaded shared box does not fail the test, small
#: enough that a working alarm would obviously beat it.
_ALARM_GRACE_S = 12.0


_ALARM_SCRIPT = textwrap.dedent(
    """
    import signal, sqlite3, sys

    from benchmarks.hard_timeout import _HANG_SQL

    def _handler(signum, frame):
        # If this ever runs, signal.alarm preempted the native stage and
        # the premise of the new mechanism would be wrong.
        print("ALARM_HANDLER_RAN", flush=True)
        sys.exit(7)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(1)
    print("ENTERING_NATIVE", flush=True)
    sqlite3.connect(":memory:").execute(_HANG_SQL).fetchone()
    print("RETURNED", flush=True)
    """
)


def test_signal_alarm_cannot_preempt_a_native_hang() -> None:
    """The defect, demonstrated: a 1s alarm does not end a native regex hang.

    Run in a child so the hang cannot take the test session with it. The
    child is expected NOT to exit on its own; we kill it and assert that is
    what happened.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_REPO_ROOT, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    proc = subprocess.Popen(  # nosec B603 - fixed argv, shell=False
        [sys.executable, "-c", _ALARM_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    try:
        proc.communicate(timeout=_ALARM_GRACE_S)
        pytest.fail(
            "the alarm-based timeout returned within "
            f"{_ALARM_GRACE_S:g}s -- this fixture no longer hangs natively, so "
            "the recovery test below would be vacuous. Fix the fixture, do not "
            "relax this assertion."
        )
    except subprocess.TimeoutExpired:
        # This is the expected outcome: 1s alarm, >12s later still running.
        pass
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
            proc.kill()
        proc.wait(timeout=10)

    assert proc.returncode != 7, "the alarm handler ran; the native-hang premise is broken"


def test_hard_timeout_recovers_from_the_same_native_hang() -> None:
    """The fix: the identical hang is preempted, in bounded time."""
    t0 = time.monotonic()
    outcome = run_with_hard_timeout(native_sqlite_hang, 2.0)
    elapsed = time.monotonic() - t0

    assert outcome.status == TIMEOUT, outcome.as_dict()
    assert outcome.killed is True
    # Bounded recovery: the deadline plus reaping, nowhere near the
    # _ALARM_GRACE_S the alarm-based version blew straight through.
    assert elapsed < _ALARM_GRACE_S, f"recovery took {elapsed:.1f}s"


def test_the_killed_child_is_actually_gone() -> None:
    """A recovery that leaves the hang running has not recovered anything."""
    outcome = run_with_hard_timeout(native_sqlite_hang, 1.5)
    assert outcome.status == TIMEOUT
    assert outcome.exitcode is not None, "child was never reaped"
    # SIGKILL shows up as a negative exit code equal to -SIGKILL.
    assert outcome.exitcode == -signal.SIGKILL, outcome.as_dict()


def test_a_normal_unit_returns_its_value() -> None:
    """The runner has to run things, not only kill them."""
    outcome = run_with_hard_timeout(quick_value, 30.0, "hello")
    assert outcome.status == OK
    assert outcome.value == "hello"
    assert outcome.killed is False


def test_an_exception_is_reported_not_raised() -> None:
    """One bad question reports; it does not end a 500-question run."""
    outcome = run_with_hard_timeout(raises_value_error, 30.0, "kaboom")
    assert outcome.status == ERROR
    assert "kaboom" in outcome.error
    assert "ValueError" in outcome.error


def test_status_names_are_distinct() -> None:
    """A caller must be able to tell a hang from a crash from a real None."""
    assert len({OK, ERROR, TIMEOUT, CRASHED}) == 4


def test_zero_timeout_is_refused() -> None:
    """A non-positive deadline is a caller bug, not a silent no-op."""
    with pytest.raises(ValueError):
        run_with_hard_timeout(quick_value, 0)
