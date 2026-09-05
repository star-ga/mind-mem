"""Shared test fixtures.

Currently one thing: a real governance admission for tests that exercise
``BlockStore.write_block`` directly.

Since the write invariant landed, ``write_block`` refuses any write with no
open admission. That is the point of it. Storage-layer unit tests still need
to call it directly, so they open a **real** admission through the gate --
a genuine chain entry, not a stubbed receipt.

There is deliberately no test-only constructor, no wildcard receipt and no
environment escape hatch in ``src/``. An invariant with a bypass reserved for
tests is not an invariant, and this project has already been bitten twice by
checks that reported success over work they never inspected.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from typing import Iterator

import pytest

# The two benchmark entry-point scripts moved out of the wheel in 5.0.0
# (src/mind_mem/bench/{locomo_suite,recompaction_bench}.py -> benchmarks/).
# They are still real, tested logic -- they reproduce the published LoCoMo and
# recompaction numbers -- so their tests keep running; they just import by
# module name from benchmarks/ instead of through the package.
_BENCHMARKS = pathlib.Path(__file__).resolve().parents[1] / "benchmarks"
if _BENCHMARKS.is_dir() and str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))


@pytest.fixture
def admitted(tmp_path) -> Iterator[None]:
    """Open a real governance admission for the duration of one test.

    Uses ``admit_proposal``, whose receipt covers any block id for its scope --
    the same ambient authority ``apply_engine`` runs under while applying an
    approved proposal. It is a real ``admit`` call writing a real chain entry;
    nothing here forges a receipt.
    """
    from mind_mem.governance_gate import get_gate

    workspace = str(tmp_path)
    gate = get_gate(workspace)
    if gate is None:  # pragma: no cover - defensive
        yield
        return
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            gate.admit_proposal(
                proposal_id="TEST-ADMISSION",
                content="[]",
                actor="pytest",
            )
        )
        yield


@pytest.fixture
def admit_delete():
    """Open a real ``DELETE`` scope around one block id, for storage tests.

    The delete-side twin of :func:`admitted`, and needed for the same reason:
    since the delete invariant landed, ``BlockStore.delete_block`` refuses any
    removal with no ``admit_delete`` scope open, and a WRITE receipt is
    explicitly not transferable to a delete. Storage-layer tests that remove a
    row directly therefore have to open the real scope, exactly as the MCP and
    HTTP delete doors do.

    Measured 2026-09-02: six such tests were red on the ``postgres backend``
    job -- the only job that runs them, because they gate on
    ``MIND_MEM_TEST_PG_DSN`` and skip everywhere else, which is why a fully
    green local gate never saw them.

    Like :func:`admitted`, this mints a genuine receipt through the gate and
    writes a genuine chain entry. There is no test-only bypass, and there must
    not be one: an invariant with an escape hatch reserved for tests is not an
    invariant.
    """
    from mind_mem.governance_gate import get_gate

    @contextlib.contextmanager
    def _scope(workspace: str, block_id: str, *, rationale: str = "storage-layer test removal") -> Iterator[None]:
        gate = get_gate(workspace)
        with gate.admit_delete(str(block_id), rationale=rationale, actor="pytest"):
            yield

    return _scope


#: Hard ceiling that aborts the session mid-test.
#:
#: The per-test check below cannot help while a SINGLE test is leaking: it
#: only runs at teardown, and the test that took this box down never got
#: there. Measured 2026-09-04: one test reached 30,935 threads inside its own
#: body, and both full runs that hit it deadlocked -- unable to create the
#: threads they needed to finish, so teardown never ran and no failure was
#: ever reported. Eight times the per-test budget, and still two orders of
#: magnitude below the point where the machine stops being able to fork.
_THREAD_CEILING = int(os.environ.get("MIND_MEM_TEST_THREAD_CEILING", "1024"))

#: Where the watchdog records why it stopped a run. A SIGKILLed pytest flushes
#: nothing, so without this the only evidence is the exit code.
_CEILING_LOG = pathlib.Path(__file__).resolve().parents[1] / ".pytest-thread-ceiling.log"


#: Polled by the out-of-process watchdog below.
_WATCHDOG_SOURCE = """
import os, signal, sys, time
pid, ceiling, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
status = "/proc/%d/status" % pid
while True:
    time.sleep(0.25)
    try:
        with open(status) as fh:
            live = next(int(l.split()[1]) for l in fh if l.startswith("Threads:"))
    except (OSError, StopIteration, ValueError):
        break                      # the run ended; nothing to guard
    if live <= ceiling:
        continue
    msg = (
        "\\n*** thread ceiling exceeded: %d OS threads (ceiling %d). A test is "
        "creating threads faster than it retires them; stopping the run so this "
        "ends in a failure rather than a machine that cannot fork. Raise "
        "MIND_MEM_TEST_THREAD_CEILING to change the bound. ***\\n" % (live, ceiling)
    )
    # fd 2 is pytest's capture file by the time this runs, and a SIGKILLed
    # run never flushes it -- so the reason goes to a real file as well, or
    # the only thing anyone sees is exit code 137.
    try:
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(msg)
    except OSError:
        pass
    os.write(2, msg.encode())
    try:
        os.kill(pid, signal.SIGINT)   # pytest reports this, naming the test
    except OSError:
        break
    time.sleep(10.0)
    try:
        os.kill(pid, 0)
    except OSError:
        break                      # it took the interrupt; a report is coming
    os.write(2, b"*** still running 10s after SIGINT; killing ***\\n")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    break
"""


@pytest.fixture(autouse=True, scope="session")
def _abort_session_on_runaway_threads() -> Iterator[None]:
    """Stop a runaway test before it takes the machine, not after.

    ``_fail_on_leaked_threads`` names the culprit, but only once the test
    returns. A test that fans out threads faster than it finishes never gets
    there, so the protection has to be able to fire from outside it.

    It also has to fire from outside the *process*. An in-process watchdog
    thread was tried first and is not reliable under exactly the conditions it
    exists for: measured 2026-09-04, a watchdog sampling every 0.25s never ran
    while one test took the interpreter from 1,845 to 7,505 threads in twelve
    seconds, because a thread waiting on a condition does not get scheduled
    against thousands of others contending for the GIL. Signals have the same
    problem -- CPython runs the handler in the main thread, which is equally
    starved -- so SIGINT is only the first ask, and SIGKILL is what makes the
    guarantee real. Either way the run exits non-zero and the reason is on
    stderr.

    Linux-only, because it reads ``/proc/<pid>/status``. Elsewhere the per-test
    check is the whole protection: it still turns a silent leak into a red
    run, it just cannot cut one short mid-test.
    """
    if not os.path.exists("/proc/self/status"):
        yield
        return

    watchdog = subprocess.Popen(
        [sys.executable, "-c", _WATCHDOG_SOURCE, str(os.getpid()), str(_THREAD_CEILING), str(_CEILING_LOG)],
        stdout=subprocess.DEVNULL,
    )
    try:
        yield
    finally:
        watchdog.terminate()
        try:
            watchdog.wait(timeout=5)
        except subprocess.TimeoutExpired:
            watchdog.kill()
            watchdog.wait()


#: How far above the session's own starting thread count the suite may drift
#: before a test is held responsible.
#:
#: Deliberately an ABSOLUTE bound against a session baseline rather than a
#: per-test delta. A per-test delta only catches a test that leaks a lot at
#: once; it is blind to the shape that actually took this machine down, which
#: is a small leak repeated thousands of times. Measured 2026-09-04: one
#: pytest process running a single file reached **34,024 live threads**, and a
#: second reached 40,222, between them consuming every thread the box could
#: create -- ``RuntimeError: can't start new thread`` from an unrelated
#: two-thread script, swap fully exhausted, and both runs deadlocked because
#: they could no longer create the threads they needed to finish. Neither run
#: reported a failure while doing it.
#:
#: 128 is generous: the suite legitimately starts watcher, inbox and HTTP
#: daemon threads and several concurrency tests hold workers across an
#: assertion. It is also two orders of magnitude below the numbers above, so
#: an accumulating leak trips inside the run that introduces it.
_THREAD_BUDGET = int(os.environ.get("MIND_MEM_TEST_THREAD_BUDGET", "128"))

#: A thread still winding down when its test returned is not a leak. Bounded
#: so a genuine leak still fails promptly rather than costing 2s per test:
#: the wait is only ever paid on the way to a failure.
_THREAD_SETTLE_SECONDS = 2.0


@pytest.fixture(scope="session")
def _thread_baseline() -> int:
    """Live threads before this session ran anything of its own."""
    return threading.active_count()


@pytest.fixture(autouse=True)
def _fail_on_leaked_threads(_thread_baseline: int, request) -> Iterator[None]:
    """Fail the test that leaks threads, instead of the machine that runs it.

    A test that leaks threads reports PASS. That is the whole problem: the
    cost lands on whoever runs the suite next, as an unrelated-looking
    ``can't start new thread``, a wedged Bash tool, or a box that has to be
    torn down -- never as a red test naming the file that did it.

    The bound is checked against a session baseline rather than against the
    count at this test's own entry, because the leak that matters accumulates:
    one abandoned worker per operation is under any sane per-test delta and
    still reaches five figures over a full run. Charging the drift to the
    first test that crosses the line names a culprit early instead of blaming
    whichever test happened to run last.

    Not a substitute for a leak-specific test. This is the backstop that makes
    *silent* leaking impossible -- the property that a passing suite must not
    hand the next run a process it cannot fork in.
    """
    yield

    leaked = threading.active_count() - _thread_baseline
    if leaked <= _THREAD_BUDGET:
        return
    deadline = time.monotonic() + _THREAD_SETTLE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(0.05)
        leaked = threading.active_count() - _thread_baseline
        if leaked <= _THREAD_BUDGET:
            return

    alive = [t for t in threading.enumerate() if t.is_alive()]
    sample = ", ".join(sorted({t.name for t in alive})[:8])
    pytest.fail(
        f"thread leak: {leaked} threads above the session baseline "
        f"({threading.active_count()} live, budget {_THREAD_BUDGET}) after "
        f"{request.node.nodeid}. Leaking threads while passing is how a green "
        f"suite exhausts the machine's fork capacity. Live thread names: {sample}",
        pytrace=False,
    )


@pytest.fixture(autouse=True, scope="session")
def _child_processes_speak_utf8() -> Iterator[None]:
    """Make every subprocess this suite spawns write UTF-8.

    Many tests run ``mm_cli`` through ``subprocess.run(..., encoding="utf-8")``.
    On Windows a child Python writes stdout in the console codepage (cp1252),
    so any non-ASCII character the CLI prints comes back as bytes the parent
    cannot decode. The decode happens on subprocess's reader THREAD, where the
    UnicodeDecodeError does not propagate -- ``run`` returns with
    ``stdout=None`` and the test dies on ``"..." in None`` with a TypeError
    that says nothing about encoding. Measured 2026-08-29: an em dash in
    ``mm resume`` output (0x97 in cp1252) failed every Windows and macOS matrix
    row while passing on Linux, whose default is already UTF-8.

    Asking the child for UTF-8 is the other half of the parent already asking to
    decode UTF-8; without it the pair only agrees by accident of platform.
    Ten test modules spawn ``mm_cli`` and none of them set this, so it belongs
    here rather than in one helper.
    """
    import os

    previous = {k: os.environ.get(k) for k in ("PYTHONUTF8", "PYTHONIOENCODING")}
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _contain_bare_tempfiles(tmp_path_factory, monkeypatch):
    """Keep every bare ``tempfile`` call inside pytest's own tmp tree.

    115 call sites in this suite reach for ``tempfile.mkdtemp(prefix="mm_...")``
    without a ``dir=`` and without cleanup, so each one leaves a workspace
    behind in ``/tmp`` forever. Measured on this machine: **12,141 orphaned
    directories holding 469,413 inodes** — 57% of every inode on the tmpfs.
    That is not a tidiness problem. When the inode table fills, ``shutil.copy2``
    starts failing inside ``init_workspace``, and the suite reports dozens of
    unrelated-looking errors and ERRORs at collection; the run that found this
    died mid-traceback because pytest could no longer write its own output.

    Redirecting ``tempfile.tempdir`` is a single fix for all of them, and for
    every future one: ``mkdtemp``/``NamedTemporaryFile``/``mkstemp`` consult it
    when no explicit ``dir=`` is given, so unchanged call sites now land under
    ``tmp_path`` and pytest reaps them on its normal schedule. A call site that
    passes ``dir=`` explicitly is unaffected.

    The alternative — editing 115 sites — fixes today's leaks and none of
    tomorrow's, because nothing would stop the 116th.
    """
    # Deliberately NOT inside ``tmp_path``: several tests scan their own
    # ``tmp_path`` for stray files and would report our scratch as a leak
    # from the code under test. A sibling directory isolates without
    # polluting the thing being measured.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path_factory.mktemp("bare-tempfiles")))
    yield


@pytest.fixture(autouse=True)
def _isolate_operator_home_state(tmp_path_factory, monkeypatch):
    """Never let a test read or write the operator's real ``~/.mind-mem``.

    ``model_gate._registry_path`` falls back to ``~/.mind-mem`` when
    ``MIND_MEM_GATE_REGISTRY`` is unset, and ``_promotion_ledger_path`` sits
    beside it. So a plain ``WeightRegistry()`` in a test wrote the operator's
    live ``~/.mind-mem/model_promotions.json`` — 29 KB of it on this machine,
    last modified *by a test run*. Two sessions running the suite at once then
    fought over one file, which is why
    ``test_v28_completion::test_weight_registry_revert`` failed in full runs
    and passed alone.

    The docstring on ``_promotion_ledger_path`` says no test can write the
    operator's real ledger by forgetting a second env var — true, and beside
    the point: the default path needs no forgetting at all. Both env vars are
    pinned here so the guarantee holds without every test remembering.

    This is containment, not a behaviour change: a test that sets either
    variable itself still wins, because monkeypatch.setenv here runs first.
    """
    # Sibling of ``tmp_path``, not a child, for the same reason as above:
    # a test scanning its own tmp_path must not see this directory.
    home = tmp_path_factory.mktemp("operator-home")
    monkeypatch.setenv("MIND_MEM_GATE_REGISTRY", str(home / "model_gate.json"))
    monkeypatch.setenv("MIND_MEM_PROMOTION_LEDGER", str(home / "model_promotions.json"))
    yield
