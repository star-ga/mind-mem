# Copyright 2026 STARGA, Inc.
"""Two writers must not be able to fork a hash chain.

``EvidenceChain.create`` used to take ``previous_hash`` from the tail it
loaded into memory at construction time and append without serialising
against anyone else. Two processes opening the same store therefore each
believed they owned the tail: their appends interleaved into one file and
every writer that had seen an empty file restarted the ledger at
``_GENESIS_HASH``. That is not a damaged record, it is several chains in
one file — and it is the shape of the live break (268 non-linking rows
from line 30 on, most carrying a zero ``previous_hash``).

Measured against the code before the fix, with four processes appending
150 records each: **489 of 600 rows did not link and 3 were rooted at
genesis behind real history**. The same fork is reachable with no
concurrency at all — ``import_jsonl`` of an empty chain over a populated
store leaves a writer holding no history in front of a non-empty file,
and the next ``create`` used to append a second genesis-rooted record.

Every test here comes in two halves:

* the gate — real concurrent OS processes (not threads: the defect is
  cross-process, and threads already shared ``_lock``), asserting the
  file that comes out is one intact chain; and
* its control — the identical body run against the pre-fix code path,
  asserting it comes out **broken**. Without that half a green gate only
  proves the writers ran, not that the test can see a fork.

``AuditChain.append`` already took the cross-process lock and already
re-read its tail under it. Its tests here are not a fix, they are the
proof that the lock is load-bearing rather than decorative: defeat it and
the same body reports a broken ledger.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import pytest
from _platform_compat import atomic_cross_process_append

from mind_mem.audit_chain import AuditChain
from mind_mem.evidence_objects import (
    _GENESIS_HASH,
    EvidenceAction,
    EvidenceChain,
    EvidenceChainCompromisedError,
)
from mind_mem.mind_filelock import _BREAK_ADOPTED, _BREAK_NOTHING, _BREAK_REMOVED, FileLock, LockTimeout

# ---------------------------------------------------------------------------
# The writer that runs in each child process
# ---------------------------------------------------------------------------

#: Run in a fresh interpreter, one per writer. ``defeat`` swaps in the
#: pre-fix code path — no cross-process lock, and ``previous_hash`` taken
#: from this process's own in-memory tail (the v5.0.1 expression, copied
#: verbatim) — so the controls exercise the real defect and not a mock of
#: it.
#:
#: The round barrier is what makes both halves deterministic instead of
#: timing-dependent: every writer resolves its tail for round *r* before
#: any writer has appended its round-*r* record. Under the fix they queue
#: on the store lock and come out as one chain; under the pre-fix path
#: they all link to their own private tail and the file forks.
#:
#: That sentence was true of the evidence path by construction (a private
#: in-memory tail) and only usually true of the audit path, whose pre-fix
#: defect is simply "no lock": three writers released by the round barrier
#: still read the file tail and append in microseconds, and a scheduler
#: that runs them one after another produces a clean chain -- measured once
#: in a full-suite run, 60 rows all linking, the twin red for the wrong
#: reason. The audit control therefore adds a second barrier INSIDE the
#: unlocked section (after the tail is read, before the record is written),
#: which forces exactly the interleaving the lock exists to prevent. It is
#: applied only in defeat mode: under the lock a barrier inside the
#: critical section would deadlock, which is the lock doing its job.
_WORKER_SOURCE = '''\
import os, sys, time, traceback

store, tag, rounds, barrier_dir, defeat, mode, nprocs = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4],
    sys.argv[5] == "1", sys.argv[6], int(sys.argv[7]),
)


class _NullLock:
    """What the store lock amounts to when it is not taken at all."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def barrier(label):
    open(os.path.join(barrier_dir, "%s-%s" % (label, tag)), "w", encoding="utf-8").close()
    prefix = "%s-" % label
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if sum(1 for f in os.listdir(barrier_dir) if f.startswith(prefix)) >= nprocs:
            return
        time.sleep(0.002)
    raise SystemExit("barrier timeout at %s" % label)


def _run():
  if mode == "evidence":
      import mind_mem.evidence_objects as eo

      if defeat:
          eo.FileLock = _NullLock
          eo.EvidenceChain._linkable_previous_hash = lambda self: (
              self._entries[-1].evidence_hash if self._entries else eo._GENESIS_HASH
          )
      chain = eo.EvidenceChain(store_path=store)
      for r in range(rounds):
          barrier("r%d" % r)
          chain.create(
              action=eo.EvidenceAction.APPLY,
              actor=tag,
              target_block_id="%s-%04d" % (tag, r),
              target_file="decisions/DECISIONS.md",
              payload=("%s-%d" % (tag, r)).encode(),
          )
  else:
      import mind_mem.audit_chain as ac

      if defeat:
          ac.FileLock = _NullLock
          # Every writer resolves the tail before any writer appends -- the
          # interleaving the lock prevents, forced rather than hoped for. The
          # barrier label counts tail reads; ``append`` is the only caller of
          # ``_last_entry`` and every writer makes the same calls in the same
          # order, so the counts agree across processes.
          _read_tail = ac.AuditChain._last_entry
          _tail_reads = [0]

          def _tail_then_wait(self):
              last = _read_tail(self)
              barrier("t%d" % _tail_reads[0])
              _tail_reads[0] += 1
              return last

          ac.AuditChain._last_entry = _tail_then_wait
      chain = ac.AuditChain(store)
      for r in range(rounds):
          barrier("r%d" % r)
          chain.append(
              "update_field",
              "decisions/DECISIONS.md",
              agent=tag,
              reason="round %d" % r,
              payload={"tag": tag, "round": r},
          )
  return rounds


# The worker's own testimony, on stdout, in ONE line the parent parses.
#
# A harness that infers "the writers ran" from rows in the file is inferring
# it from the very artefact under test. Where an append can be lost -- on
# Windows, with the store lock deliberately defeated -- a writer that
# completed every round is indistinguishable from one that never started.
# Measured on the windows-latest runners: 46 of 60 evidence rows and 54 of 60
# audit rows, with every worker exiting 0 and stderr empty.
try:
    done = _run()
except BaseException:
    # Printed, never swallowed. An anonymous exit code is what left five CI
    # rows saying only "a worker failed: (0, 0, 1, 1, 0, 1)".
    sys.stdout.write("FAILED %s\\n" % tag)
    sys.stdout.flush()
    traceback.print_exc()
    raise SystemExit(1)
sys.stdout.write("DONE %s %d\\n" % (tag, done))
sys.stdout.flush()
'''


@dataclass(frozen=True)
class ChainReport:
    """What came out of a concurrent run, as facts rather than a verdict."""

    rows: int
    nonlinking: int
    genesis_rooted_after_first: int
    writers_seen: int
    verified: bool
    broken: tuple
    exit_codes: tuple
    stderr_tail: str
    #: Rounds the workers themselves reported completing, summed over the
    #: ``DONE <tag> <n>`` lines they print. The writers' own testimony, which
    #: is what "the control really ran" is a claim about -- as distinct from
    #: ``rows``, which is what survived in the file.
    attempted: int = 0
    #: Tags that printed ``FAILED`` instead of ``DONE``.
    failed_writers: tuple = ()

    def is_one_intact_chain(self) -> bool:
        """Every row links, nothing restarted at genesis, and it verifies."""
        return self.nonlinking == 0 and self.genesis_rooted_after_first == 0 and self.verified and not self.broken

    def why_the_writers_did_not_run(self, procs: int, rounds: int) -> str:
        """A ONE-LINE diagnosis, or "" when the writers did all their work.

        First line on purpose. pytest's short summary shows only the first
        line of an assertion message, so ``f"the control did not write:\n{report}"``
        printed exactly ``the control did not write:`` and nothing else --
        five CI rows whose whole visible content was the name of a symptom.
        """
        if self.failed_writers:
            return f"{len(self.failed_writers)} writer(s) raised: {', '.join(self.failed_writers)} (stderr below)"
        if self.exit_codes != (0,) * procs:
            return f"a writer exited non-zero: exit_codes={self.exit_codes} (stderr below)"
        if self.attempted != procs * rounds:
            return f"writers reported {self.attempted} of {procs * rounds} appends"
        if self.writers_seen != procs:
            return f"records came from {self.writers_seen} of {procs} writers"
        if self.rows == 0:
            return "the file is empty"
        return ""

    def __str__(self) -> str:  # pragma: no cover - only rendered on failure
        return (
            f"rows={self.rows} attempted={self.attempted} nonlinking={self.nonlinking} "
            f"genesis_rooted_after_first={self.genesis_rooted_after_first} "
            f"writers_seen={self.writers_seen} verified={self.verified} "
            f"broken={len(self.broken)} exit_codes={self.exit_codes} "
            f"failed_writers={self.failed_writers}\n"
            f"{self.stderr_tail}"
        )


def _spawn_writers(tmp_path, *, mode: str, target: str, procs: int, rounds: int, defeat: bool) -> tuple[tuple, str, int, tuple]:
    """Run *procs* real OS processes appending *rounds* records each.

    Returns ``(exit_codes, stderr, appends_reported, failed_tags)``. The last
    two come from the workers' own ``DONE``/``FAILED`` lines rather than from
    the file they wrote, because on a platform where an unsynchronised append
    can be lost the file cannot distinguish "did not run" from "ran and was
    overwritten".
    """
    worker = tmp_path / f"writer_{mode}_{'defeat' if defeat else 'fixed'}.py"
    worker.write_text(_WORKER_SOURCE, encoding="utf-8")
    barrier_dir = tmp_path / f"barrier_{mode}_{'defeat' if defeat else 'fixed'}"
    barrier_dir.mkdir()

    env = dict(os.environ)
    # The child must import the very package under test, whether this run
    # is against an installed wheel or the src tree.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["MIND_MEM_LOG_LEVEL"] = "error"

    running = [
        subprocess.Popen(
            [
                sys.executable,
                str(worker),
                target,
                f"w{i}",
                str(rounds),
                str(barrier_dir),
                "1" if defeat else "0",
                mode,
                str(procs),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for i in range(procs)
    ]
    codes, errs, attempted, failed = [], [], 0, []
    for proc in running:
        out, err = proc.communicate(timeout=180)
        codes.append(proc.returncode)
        if err and err.strip():
            errs.append(err[-2000:])
        for line in (out or "").splitlines():
            parts = line.split()
            if parts[:1] == ["DONE"] and len(parts) == 3:
                attempted += int(parts[2])
            elif parts[:1] == ["FAILED"] and len(parts) == 2:
                failed.append(parts[1])
    return tuple(codes), "\n".join(errs), attempted, tuple(failed)


def _read_evidence(store: str) -> list[dict]:
    if not os.path.isfile(store):
        return []
    with open(store, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _evidence_report(store: str, codes: tuple, stderr: str, attempted: int = 0, failed: tuple = ()) -> ChainReport:
    rows = _read_evidence(store)
    prev = _GENESIS_HASH
    nonlinking = 0
    genesis_after_first = 0
    for idx, row in enumerate(rows):
        if row["previous_hash"] != prev:
            nonlinking += 1
        if idx > 0 and row["previous_hash"] == _GENESIS_HASH:
            genesis_after_first += 1
        prev = row["evidence_hash"]
    reopened = EvidenceChain(store_path=store)
    verified, broken = reopened.verify_chain()
    return ChainReport(
        rows=len(rows),
        nonlinking=nonlinking,
        genesis_rooted_after_first=genesis_after_first,
        writers_seen=len({row["actor"] for row in rows}),
        verified=verified,
        broken=tuple(broken),
        exit_codes=codes,
        stderr_tail=stderr,
        attempted=attempted,
        failed_writers=failed,
    )


def _audit_report(workspace: str, codes: tuple, stderr: str, attempted: int = 0, failed: tuple = ()) -> ChainReport:
    path = os.path.join(workspace, ".mind-mem-audit", "chain.jsonl")
    rows = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    prev = _GENESIS_HASH
    nonlinking = 0
    genesis_after_first = 0
    for idx, row in enumerate(rows):
        if row["prev_hash"] != prev:
            nonlinking += 1
        if idx > 0 and row["prev_hash"] == _GENESIS_HASH:
            genesis_after_first += 1
        prev = row["entry_hash"]
    verified, errors = AuditChain(workspace).verify()
    return ChainReport(
        rows=len(rows),
        nonlinking=nonlinking,
        genesis_rooted_after_first=genesis_after_first,
        writers_seen=len({row["agent"] for row in rows}),
        verified=verified,
        broken=tuple(errors),
        exit_codes=codes,
        stderr_tail=stderr,
        attempted=attempted,
        failed_writers=failed,
    )


_PROCS = 3
_ROUNDS = 20


def _run_evidence(tmp_path, *, defeat: bool) -> ChainReport:
    store = str(tmp_path / ("defeat" if defeat else "fixed") / "evidence_chain.jsonl")
    os.makedirs(os.path.dirname(store), exist_ok=True)
    codes, stderr, attempted, failed = _spawn_writers(tmp_path, mode="evidence", target=store, procs=_PROCS, rounds=_ROUNDS, defeat=defeat)
    return _evidence_report(store, codes, stderr, attempted, failed)


def _run_audit(tmp_path, *, defeat: bool) -> ChainReport:
    workspace = str(tmp_path / ("audit_defeat" if defeat else "audit_fixed"))
    os.makedirs(workspace, exist_ok=True)
    codes, stderr, attempted, failed = _spawn_writers(tmp_path, mode="audit", target=workspace, procs=_PROCS, rounds=_ROUNDS, defeat=defeat)
    return _audit_report(workspace, codes, stderr, attempted, failed)


# ---------------------------------------------------------------------------
# The gate: concurrent processes produce one intact chain
# ---------------------------------------------------------------------------


class TestConcurrentWritersCannotForkTheEvidenceChain:
    def test_every_row_links_and_the_chain_verifies(self, tmp_path):
        report = _run_evidence(tmp_path, defeat=False)

        # Positive control first: the run must actually have happened.
        # `nonlinking == 0` over an empty file is a pass that proves
        # nothing, and so is a run where every child died on import.
        assert report.exit_codes == (0,) * _PROCS, f"a writer failed: {report.why_the_writers_did_not_run(_PROCS, _ROUNDS)}\n{report}"
        # NOT lifted into the platform-conditional below. Under the lock the
        # appends are serialised, so "every record survives" is a product
        # guarantee on every platform including Windows -- which the Windows
        # rows already prove, and which is precisely why the twin's version of
        # this assertion (with the lock removed) was measuring the filesystem
        # instead of the product.
        assert report.rows == _PROCS * _ROUNDS, f"writers did not all land their records:\n{report}"
        assert report.writers_seen == _PROCS, f"records came from {report.writers_seen} of {_PROCS} writers:\n{report}"

        assert report.is_one_intact_chain(), f"concurrent writers forked the chain:\n{report}"
        assert report.nonlinking == 0
        assert report.genesis_rooted_after_first == 0
        assert report.verified is True
        assert report.broken == ()

    def test_exactly_one_record_is_rooted_at_genesis(self, tmp_path):
        """The fork's signature is a *second* chain root, not a bad link."""
        report = _run_evidence(tmp_path, defeat=False)
        assert report.exit_codes == (0,) * _PROCS, f"a writer failed: {report.why_the_writers_did_not_run(_PROCS, _ROUNDS)}\n{report}"
        store = str(tmp_path / "fixed" / "evidence_chain.jsonl")
        rooted = [row for row in _read_evidence(store) if row["previous_hash"] == _GENESIS_HASH]
        assert len(rooted) == 1, f"{len(rooted)} chains rooted at genesis in one file:\n{report}"


class TestAuditChainLockIsLoadBearing:
    """``audit_chain.append`` already locks — this proves the lock works."""

    def test_every_entry_links_and_the_ledger_verifies(self, tmp_path):
        report = _run_audit(tmp_path, defeat=False)

        assert report.exit_codes == (0,) * _PROCS, f"a writer failed: {report.why_the_writers_did_not_run(_PROCS, _ROUNDS)}\n{report}"
        assert report.rows == _PROCS * _ROUNDS, f"writers did not all land their entries:\n{report}"
        assert report.writers_seen == _PROCS, f"entries came from {report.writers_seen} of {_PROCS} writers:\n{report}"

        assert report.is_one_intact_chain(), f"concurrent writers forked the ledger:\n{report}"
        assert report.broken == ()


# ---------------------------------------------------------------------------
# The controls: the same body, run against the code without the gate
# ---------------------------------------------------------------------------


def _assert_the_control_really_ran(report: ChainReport) -> None:
    """The twin's positive control: the writers did the work, in full.

    ``assert report.rows == _PROCS * _ROUNDS`` used to stand here, and it was
    the wrong statement in the wrong place. With the lock defeated -- which is
    the whole point of a twin -- nothing serialises the appends, so how many
    of them survive is a property of the OS, not of mind-mem. POSIX gives
    ``open(path, "a")`` the ``O_APPEND`` flag and a single write on it is
    atomic, so all sixty always land; the Windows CRT emulates append as seek
    then write, and the second writer overwrites the first. Measured on the
    windows-latest runners: 46 of 60 evidence rows, 54 of 60 audit rows,
    **every worker exiting 0 with empty stderr** -- the writers ran perfectly
    and the control called them absent.

    So the control asks the writers, not the file. Each worker prints
    ``DONE <tag> <rounds>`` as its last act, and this checks that testimony.
    That is strictly stronger than the row count it replaces: it also catches
    a worker that exits non-zero (which the twins never checked at all, though
    the gates did) and a worker that dies mid-loop, and it cannot be satisfied
    by a run that never happened.
    """
    reason = report.why_the_writers_did_not_run(_PROCS, _ROUNDS)
    assert reason == "", f"the control did not run: {reason}\n{report}"
    assert report.exit_codes == (0,) * _PROCS, f"the control did not run: exit_codes={report.exit_codes}\n{report}"
    assert report.attempted == _PROCS * _ROUNDS, f"the control did not run: {report.attempted} appends reported\n{report}"
    assert report.writers_seen == _PROCS, f"the control did not run concurrently: {report.writers_seen} writers\n{report}"
    assert report.rows > 0, f"the control did not run: the file is empty\n{report}"


def _assert_lost_appends_are_the_platform_and_nothing_else(report: ChainReport) -> None:
    """State the platform difference explicitly; assert on BOTH sides.

    Never a skip. Where the OS gives atomic cross-process append the twin
    keeps the exact assertion it always had -- not one record lost. Where it
    does not, the weaker relation that is still true everywhere is asserted,
    and the run is required to have lost records for the documented reason
    rather than for an unexamined one.
    """
    assert report.rows <= report.attempted, f"more rows than appends — the harness is miscounting:\n{report}"
    if atomic_cross_process_append():
        assert report.rows == report.attempted, f"an append was lost on a platform whose append is atomic:\n{report}"
    else:
        # Windows. Losing appends here is the CRT's seek-then-write, and it is
        # reachable only because this twin removed the lock -- the gate above
        # runs the same writers WITH the lock and lands every record on this
        # same runner, which is what proves the loss is the missing lock and
        # not a broken harness.
        assert report.rows <= report.attempted


class TestMutationTwin:
    """The gate above is only worth its green if it can go red.

    Each twin runs the *same* writers and the *same* analysis with one
    thing taken away, and asserts the result is broken. If a twin ever
    passes as intact, the corresponding gate is measuring nothing.
    """

    def test_without_the_store_lock_the_evidence_chain_forks(self, tmp_path):
        report = _run_evidence(tmp_path, defeat=True)

        _assert_the_control_really_ran(report)
        _assert_lost_appends_are_the_platform_and_nothing_else(report)

        assert not report.is_one_intact_chain(), (
            f"the pre-fix code path produced one intact chain — this test cannot detect the fork it exists to detect:\n{report}"
        )
        assert report.nonlinking > 0, f"no non-linking row in the unfixed run:\n{report}"
        assert report.genesis_rooted_after_first > 0, f"no second genesis root in the unfixed run:\n{report}"
        assert report.verified is False

    def test_without_its_lock_the_audit_ledger_forks(self, tmp_path):
        report = _run_audit(tmp_path, defeat=True)

        _assert_the_control_really_ran(report)
        _assert_lost_appends_are_the_platform_and_nothing_else(report)

        assert not report.is_one_intact_chain(), (
            f"the audit ledger survived losing its lock — the concurrency assertion above is not measuring the lock:\n{report}"
        )
        assert report.verified is False


class TestTheControlAsksTheWritersNotTheFile:
    """Pin the exact CI observation the twin's control used to misread.

    These are the numbers the windows-latest rows reported, verbatim: three
    writers, sixty appends between them, **every worker exited 0 with empty
    stderr**, the chain came out forked exactly as the twin requires -- and
    46 of the 60 rows were in the file, because with the lock defeated the
    Windows CRT's non-atomic append lets one writer land on top of another.
    The old control read that as "the control did not write".

    Constructed rather than raced: the point is what the control CONCLUDES
    from a given observation, and an observation you have to provoke on one
    OS cannot be asserted on the others.
    """

    #: (evidence, audit) as measured on windows-latest at 2697baf.
    MEASURED = ((46, 27, 1), (54, 34, 1))

    def _report(self, rows: int, nonlinking: int, genesis_after: int, **over) -> ChainReport:
        fields = dict(
            rows=rows,
            nonlinking=nonlinking,
            genesis_rooted_after_first=genesis_after,
            writers_seen=_PROCS,
            verified=False,
            broken=("load_integrity_compromised",),
            exit_codes=(0,) * _PROCS,
            stderr_tail="",
            attempted=_PROCS * _ROUNDS,
            failed_writers=(),
        )
        fields.update(over)
        return ChainReport(**fields)

    @pytest.mark.parametrize("rows,nonlinking,genesis_after", MEASURED)
    def test_lost_appends_are_not_read_as_an_absent_writer(self, rows, nonlinking, genesis_after) -> None:
        report = self._report(rows, nonlinking, genesis_after)
        assert report.rows < report.attempted, "this fixture is supposed to have lost appends"
        _assert_the_control_really_ran(report)
        assert not report.is_one_intact_chain(), "and the property under test still holds"

    def test_the_control_still_fails_when_a_writer_really_is_absent(self) -> None:
        """Mutation control. Without it the relaxation above proves nothing.

        Four independent ways a writer can fail to do its work; each must
        still be caught, and each must say so on the FIRST line.
        """
        cases = {
            "raised": self._report(46, 27, 1, failed_writers=("w1",), exit_codes=(0, 1, 0)),
            "non-zero exit": self._report(46, 27, 1, exit_codes=(0, 1, 0)),
            "short": self._report(46, 27, 1, attempted=_PROCS * _ROUNDS - 1),
            "one writer": self._report(46, 27, 1, writers_seen=1),
            "empty": self._report(0, 0, 0),
        }
        for name, report in cases.items():
            with pytest.raises(AssertionError) as excinfo:
                _assert_the_control_really_ran(report)
            first_line = str(excinfo.value).splitlines()[0]
            assert first_line.strip(), f"{name}: the diagnosis line is empty"
            assert "the control did not run" in first_line, f"{name}: {first_line!r}"
            assert first_line != "the control did not run:", f"{name}: the reason is missing from the first line"

    def test_a_platform_with_atomic_append_still_demands_every_row(self) -> None:
        """The other side of the platform branch, asserted rather than assumed."""
        lossy = self._report(46, 27, 1)
        if atomic_cross_process_append():
            with pytest.raises(AssertionError, match="append was lost"):
                _assert_lost_appends_are_the_platform_and_nothing_else(lossy)
        else:
            _assert_lost_appends_are_the_platform_and_nothing_else(lossy)
        # True on every platform: the harness may never report more rows than
        # appends, whatever the filesystem does.
        with pytest.raises(AssertionError, match="miscounting"):
            _assert_lost_appends_are_the_platform_and_nothing_else(self._report(61, 1, 1))


# ---------------------------------------------------------------------------
# The genesis guard, reachable with no concurrency at all
# ---------------------------------------------------------------------------


def _seed(store: str, n: int = 3) -> EvidenceChain:
    chain = EvidenceChain(store_path=store)
    for i in range(n):
        chain.create(
            action=EvidenceAction.APPLY,
            actor="seed",
            target_block_id=f"B-{i:03d}",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
    return chain


def _genesis_rooted(store: str) -> int:
    return sum(1 for row in _read_evidence(store) if row["previous_hash"] == _GENESIS_HASH)


def _prefix_v501_tail(self) -> str:
    """The v5.0.1 expression for ``previous_hash``, copied verbatim.

    ``self._entries[-1].evidence_hash if self._entries else _GENESIS_HASH``
    — the whole defect in one line: the tail is whatever this process
    happens to hold, and nothing consults the file.
    """
    return self._entries[-1].evidence_hash if self._entries else _GENESIS_HASH


class TestGenesisIntoANonEmptyChainIsRefused:
    def test_import_of_an_empty_chain_over_a_populated_store_is_refused(self, tmp_path):
        store = str(tmp_path / "evidence_chain.jsonl")
        chain = _seed(store)

        # Positive control: this chain CAN append to this store right now,
        # so the refusal below is about the state, not about a store that
        # was never writable or a method that never writes.
        chain.create(
            action=EvidenceAction.APPLY,
            actor="before",
            target_block_id="B-BEFORE",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
        assert len(_read_evidence(store)) == 4
        assert _genesis_rooted(store) == 1

        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        chain.import_jsonl(str(empty))  # public API: replaces in-memory history

        before = open(store, "rb").read()
        with pytest.raises(EvidenceChainCompromisedError) as caught:
            chain.create(
                action=EvidenceAction.APPLY,
                actor="after-import",
                target_block_id="B-FORK",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )
        assert "genesis into non-empty chain" in str(caught.value)
        assert open(store, "rb").read() == before, "the refusal landed after the append"
        assert _genesis_rooted(store) == 1, "a second chain was rooted at genesis"

    def test_a_writer_holding_a_different_history_is_refused(self, tmp_path):
        """The other half of the disagreement: history, not emptiness.

        Genesis-into-non-empty is the case where a writer holds *no*
        history. A writer holding the *wrong* history is the same fork
        with a different first symptom, and linking to a tail the store
        does not have would splice two ledgers together.
        """
        store = str(tmp_path / "evidence_chain.jsonl")
        chain = _seed(store, n=3)

        foreign_path = str(tmp_path / "foreign" / "evidence_chain.jsonl")
        os.makedirs(os.path.dirname(foreign_path), exist_ok=True)
        _seed(foreign_path, n=2)
        chain.import_jsonl(foreign_path)  # now holding somebody else's chain

        before = open(store, "rb").read()
        with pytest.raises(EvidenceChainCompromisedError) as caught:
            chain.create(
                action=EvidenceAction.APPLY,
                actor="foreign",
                target_block_id="B-FOREIGN",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )
        assert "is not the store's last record" in str(caught.value)
        assert open(store, "rb").read() == before
        assert EvidenceChain(store_path=store).verify_chain() == (True, [])

    def test_a_stale_reader_cannot_restart_the_chain(self, tmp_path):
        """A writer whose view of the store predates other writers' records.

        This is the state every forked process in the live break was in:
        it read the store when it was empty and never looked again. Here
        the store is written by a second chain object, so the first one's
        view is stale through nothing but ordinary use.
        """
        store = str(tmp_path / "evidence_chain.jsonl")
        stale = EvidenceChain(store_path=store)  # opened before the file exists
        _seed(store, n=2)  # somebody else writes it

        stale.create(
            action=EvidenceAction.APPLY,
            actor="stale",
            target_block_id="B-STALE",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
        rows = _read_evidence(store)
        assert len(rows) == 3, "the stale writer's record did not land"
        assert _genesis_rooted(store) == 1, "the stale writer restarted the chain at genesis"
        assert EvidenceChain(store_path=store).verify_chain() == (True, [])


class TestAnUnlockableStoreStillReads:
    """Reading may degrade without the lock. Writing may not.

    Re-anchoring archives the old chain read-only, and an auditor has to
    be able to verify exactly that file. A lock the writers need must
    never become a lock the readers require — but the write path has to
    keep failing closed on the same condition, or "the lockfile could not
    be created" would quietly become "append without serialising", which
    is the fork.
    """

    def test_a_chain_whose_lock_cannot_be_taken_still_loads_and_verifies(self, tmp_path, monkeypatch):
        store = str(tmp_path / "evidence_chain.jsonl")
        _seed(store, n=3)

        def _refuse(self):
            raise PermissionError(13, "read-only archive")

        monkeypatch.setattr(EvidenceChain, "_store_lock", _refuse)
        reopened = EvidenceChain(store_path=store)
        assert len(reopened) == 3, "an unlockable store did not load"
        assert reopened.verify_chain() == (True, [])

    def test_but_appending_to_it_is_refused(self, tmp_path, monkeypatch):
        """Positive control for the test above: the tolerance is read-only."""
        store = str(tmp_path / "evidence_chain.jsonl")
        _seed(store, n=3)
        before = open(store, "rb").read()

        def _refuse(self):
            raise PermissionError(13, "read-only archive")

        monkeypatch.setattr(EvidenceChain, "_store_lock", _refuse)
        reopened = EvidenceChain(store_path=store)
        with pytest.raises(PermissionError):
            reopened.create(
                action=EvidenceAction.APPLY,
                actor="unserialised",
                target_block_id="B-X",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )
        assert open(store, "rb").read() == before


class TestGenesisGuardMutationTwin:
    """Take the guard away and the same sequences fork the ledger.

    Both twins restore the v5.0.1 ``previous_hash`` expression, which is
    exactly what the guard replaced. A twin that still passed would mean
    the assertions above hold for some other reason.
    """

    def test_the_import_route_forks_without_the_guard(self, tmp_path, monkeypatch):
        store = str(tmp_path / "evidence_chain.jsonl")
        chain = _seed(store)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        chain.import_jsonl(str(empty))

        monkeypatch.setattr(EvidenceChain, "_linkable_previous_hash", _prefix_v501_tail)
        chain.create(
            action=EvidenceAction.APPLY,
            actor="after-import",
            target_block_id="B-FORK",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )

        assert _genesis_rooted(store) == 2, "the unguarded path did not fork — the guard's test proves nothing"
        reopened = EvidenceChain(store_path=store)
        assert reopened.verify_chain() == (False, ["load_integrity_compromised"])
        assert reopened.integrity_compromised is True

    def test_the_stale_reader_route_forks_without_the_guard(self, tmp_path, monkeypatch):
        store = str(tmp_path / "evidence_chain.jsonl")
        stale = EvidenceChain(store_path=store)
        _seed(store, n=2)

        monkeypatch.setattr(EvidenceChain, "_linkable_previous_hash", _prefix_v501_tail)
        stale.create(
            action=EvidenceAction.APPLY,
            actor="stale",
            target_block_id="B-STALE",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )

        assert _genesis_rooted(store) == 2, "the unguarded path did not fork — the guard's test proves nothing"
        assert EvidenceChain(store_path=store).integrity_compromised is True


class TestHashesAreNeverRewritten:
    """Stopping the bleeding, not re-anchoring: repair is not this code's call."""

    def test_a_refused_append_leaves_every_existing_byte_alone(self, tmp_path):
        store = str(tmp_path / "evidence_chain.jsonl")
        _seed(store, n=4)
        before = open(store, "rb").read()

        stale = EvidenceChain(store_path=store)
        stale._entries = []  # the shape a forked writer is in
        with pytest.raises(EvidenceChainCompromisedError):
            stale.create(
                action=EvidenceAction.APPLY,
                actor="stale",
                target_block_id="B-X",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )

        assert open(store, "rb").read() == before
        # And the refusal says so, so nobody reads it as an invitation to
        # rewrite history.
        assert EvidenceChain(store_path=store).verify_chain() == (True, [])

    def test_a_store_that_shrank_is_refused_not_re_anchored(self, tmp_path):
        """Append-only means append-only: a shorter file is a rewritten one."""
        store = str(tmp_path / "evidence_chain.jsonl")
        chain = _seed(store, n=4)
        rows = _read_evidence(store)
        with open(store, "w", encoding="utf-8") as fh:  # somebody truncated history
            fh.write(json.dumps(rows[0], separators=(",", ":")) + "\n")

        with pytest.raises(EvidenceChainCompromisedError) as caught:
            chain.create(
                action=EvidenceAction.APPLY,
                actor="after-truncation",
                target_block_id="B-X",
                target_file="decisions/DECISIONS.md",
                payload=b"payload",
            )
        assert "shrank" in str(caught.value)
        assert len(_read_evidence(store)) == 1, "the refused append still wrote"


# ---------------------------------------------------------------------------
# The other way a ledger loses history: not a fork, a truncation
# ---------------------------------------------------------------------------


def _prefix_v501_export(self, path: str) -> None:
    """The v5.0.1 body of ``export_jsonl``, copied verbatim.

    It opens the destination ``"w"`` and writes whatever this process
    holds in memory — with no check that the destination is the store it
    is about to overwrite, and no attempt to absorb what other writers
    appended first.
    """
    self._raise_if_compromised("export")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ev in self._entries:
            fh.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")


def _two_writers(store: str) -> EvidenceChain:
    """Seed *store* from two chain objects; return the one holding the stale view.

    Two objects over one file is the shape two processes are in — the
    first has no idea the second appended anything.
    """
    first = _seed(store, n=3)
    second = EvidenceChain(store_path=store)
    for i in range(2):
        second.create(
            action=EvidenceAction.ROLLBACK,
            actor="second-writer",
            target_block_id=f"B-2ND-{i}",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
    return first


def _actors(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["actor"] for line in fh if line.strip()]


class TestExportCannotTruncateTheLedger:
    """``export_jsonl`` opens its destination ``"w"``. Aimed at the store,
    that is not an export — it is an append-only ledger rewritten down to
    one process's in-memory view. The result still verifies, because a
    shortened chain links exactly as well as a whole one, so nothing
    downstream can report the loss. It has to be refused at the call.
    """

    def test_export_onto_the_store_is_refused_and_every_byte_survives(self, tmp_path):
        store = str(tmp_path / "evidence_chain.jsonl")
        stale = _two_writers(store)

        # Positive control 1: the records that a truncation would destroy
        # are provably on disk, and this test can see them. Without this
        # the byte-comparison below would pass just as well over a store
        # the second writer never managed to write to.
        assert _actors(store) == ["seed", "seed", "seed", "second-writer", "second-writer"]
        assert len(stale) == 3, "the exporter is not holding the stale view the defect needs"

        # Positive control 2: export works right now, so the refusal below
        # is about the destination and not about a method that never wrote.
        elsewhere = str(tmp_path / "exported.jsonl")
        stale.export_jsonl(elsewhere)
        assert os.path.isfile(elsewhere)

        before = open(store, "rb").read()
        with pytest.raises(ValueError, match="own store"):
            stale.export_jsonl(store)
        assert open(store, "rb").read() == before, "the refusal landed after the truncation"
        assert _actors(store) == ["seed", "seed", "seed", "second-writer", "second-writer"]

    def test_the_store_is_refused_however_it_is_spelt(self, tmp_path):
        """A guard that only matches the literal path is a guard around one spelling."""
        store = str(tmp_path / "evidence_chain.jsonl")
        stale = _two_writers(store)
        before = open(store, "rb").read()

        alias = str(tmp_path / "alias.jsonl")
        os.symlink(store, alias)
        spellings = [
            alias,
            os.path.join(str(tmp_path), ".", "evidence_chain.jsonl"),
            os.path.join(str(tmp_path), "sub", "..", "evidence_chain.jsonl"),
        ]
        for spelling in spellings:
            with pytest.raises(ValueError, match="own store"):
                stale.export_jsonl(spelling)
            assert open(store, "rb").read() == before, f"{spelling} got through"

        # Positive control: a genuinely different destination still exports,
        # so the loop above is not passing because every path is refused.
        other = str(tmp_path / "genuinely_elsewhere.jsonl")
        stale.export_jsonl(other)
        assert len(_actors(other)) == 5

    def test_an_export_carries_what_other_writers_appended(self, tmp_path):
        """An export that stops at this process's prefix is a false history.

        It is presented as *the* chain, it verifies, and the records it
        omits leave no trace in it.
        """
        store = str(tmp_path / "evidence_chain.jsonl")
        stale = _two_writers(store)
        assert len(stale) == 3, "nothing to absorb — the test would prove nothing"

        out = str(tmp_path / "exported.jsonl")
        stale.export_jsonl(out)
        assert _actors(out) == ["seed", "seed", "seed", "second-writer", "second-writer"]

        restored = EvidenceChain()
        restored.import_jsonl(out)
        assert len(restored) == 5
        assert restored.verify_chain() == (True, [])


class TestExportGuardMutationTwin:
    """The same bodies against the v5.0.1 export, which must come out broken.

    Without this half, the gate above only proves that ``export_jsonl``
    can raise and that a file has five lines in it — not that either test
    can see a ledger being truncated.
    """

    def test_without_the_guard_the_store_is_truncated_and_still_verifies(self, tmp_path, monkeypatch):
        store = str(tmp_path / "evidence_chain.jsonl")
        stale = _two_writers(store)
        assert len(_actors(store)) == 5, "the control did not write"

        monkeypatch.setattr(EvidenceChain, "export_jsonl", _prefix_v501_export)
        stale.export_jsonl(store)  # the pre-fix path: no refusal at all

        assert _actors(store) == ["seed", "seed", "seed"], "the unguarded export did not truncate — the guard's test proves nothing"
        # The sting: two records are gone and the ledger reports itself fine.
        assert EvidenceChain(store_path=store).verify_chain() == (True, [])

    def test_without_the_refresh_the_export_is_a_stale_prefix(self, tmp_path, monkeypatch):
        store = str(tmp_path / "evidence_chain.jsonl")
        stale = _two_writers(store)

        monkeypatch.setattr(EvidenceChain, "export_jsonl", _prefix_v501_export)
        out = str(tmp_path / "exported.jsonl")
        stale.export_jsonl(out)

        assert _actors(out) == ["seed", "seed", "seed"], "the unrefreshed export was not stale — the refresh test proves nothing"
        restored = EvidenceChain()
        restored.import_jsonl(out)
        assert restored.verify_chain() == (True, []), "a partial history that does not even look partial"


# ---------------------------------------------------------------------------
# The lock the whole fix rests on: does it actually exclude?
# ---------------------------------------------------------------------------

#: Each worker takes the same lock over and over and, inside the critical
#: section, writes its own tag to a shared file and reads it back. Another
#: tag coming back means two processes were inside at once — measured, not
#: argued about.
#:
#: ``defeat`` restores the two v5.0.1 behaviours that made the lock leaky:
#: a lockfile was judged *stale* when it was merely empty or missing, and a
#: lockfile was unlinked **by path** rather than by identity. Together they
#: let one process break a lock another was in the middle of taking, and let
#: a releasing process delete its successor's claim.
_LOCK_PROBE_SOURCE = '''\
import os, sys, time, traceback
import mind_mem.mind_filelock as mfl

target, tag, iters, report, defeat = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1",
)

if defeat:
    def _v501_unlink(self, identity):
        """v5.0.1 release/break: remove whatever lockfile is there now."""
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass

    def _v501_stale(self):
        """The v5.0.1 _is_stale verdict, copied verbatim."""
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                pid_str = f.read().strip()
            if not pid_str:
                return (0, 0)
            pid = int(pid_str)
            try:
                os.kill(pid, 0)
                return None
            except ProcessLookupError:
                return (0, 0)
            except PermissionError:
                return None
        except (OSError, ValueError):
            try:
                return (0, 0) if (time.time() - os.path.getmtime(self.lock_path)) > 300 else None
            except OSError:
                return (0, 0)

    def _v501_break(self, identity=None):
        """v5.0.1 _break_stale: unlink the path, arbitrated by nothing.

        Reports the corpse gone, which is what v5.0.1's caller assumed when
        it looped straight back into the create.
        """
        _v501_unlink(self, identity)
        return mfl._BREAK_REMOVED

    mfl.FileLock._unlink_if_ours = _v501_unlink
    mfl.FileLock._stale_identity = _v501_stale
    mfl.FileLock._break_stale = _v501_break

holder = target + ".holder"
violations = 0
done = 0
failure = ""
try:
    for i in range(iters):
        with mfl.FileLock(target, timeout=30.0):
            with open(holder, "w", encoding="utf-8") as fh:
                fh.write(tag)
            time.sleep(0.0005)
            with open(holder, encoding="utf-8") as fh:
                if fh.read() != tag:
                    violations += 1
        done += 1
except BaseException as exc:
    # NAMED, not swallowed. A worker that dies here used to leave the parent
    # holding nothing but an exit code -- five CI rows whose entire visible
    # evidence was "a worker failed: (0, 0, 1, 1, 0, 1)". The class and the
    # message go into the report line so the parent can put them on the
    # FIRST line of its assertion; the full traceback goes to stderr, which
    # the parent now also captures.
    failure = "%s: %s" % (type(exc).__name__, exc)
    traceback.print_exc()

# One report file PER WORKER, never one shared file appended by six of them.
# That shared append had the defect this suite exists to catch, one layer
# down: on Windows an unsynchronised append is seek-then-write, so the
# harness could lose a worker's line and then report the worker as missing.
with open("%s.%s" % (report, tag), "w", encoding="utf-8") as fh:
    fh.write("%s %d %d %s\\n" % (tag, violations, done, failure or "-"))
if failure:
    raise SystemExit(1)
'''

_LOCK_PROCS = 6
_LOCK_ITERS = 600


@dataclass(frozen=True)
class LockProbeResult:
    """What the probe measured, and — when a worker died — why.

    ``violations``/``exit_codes``/``reported`` are what the assertions read.
    ``diagnosis`` is a ONE-LINE reason, because pytest's short summary shows
    only the first line of an assertion message: ``a worker failed: (0, 0, 1,
    1, 0, 1)`` was the whole of what four Windows CI rows had to say, and the
    child's traceback — which the harness had captured into a pipe and then
    dropped on the floor — was the only thing that could have named the cause.
    """

    violations: int
    exit_codes: tuple
    reported: int
    completed: int
    diagnosis: str
    stderr_tail: str

    def __str__(self) -> str:  # pragma: no cover - only rendered on failure
        return (
            f"{self.diagnosis or 'all workers finished'}\n"
            f"violations={self.violations} exit_codes={self.exit_codes} "
            f"reported={self.reported} completed={self.completed}\n"
            f"{self.stderr_tail}"
        )


def _run_lock_probe(tmp_path, *, defeat: bool, tag: str) -> LockProbeResult:
    """Run the probe and report what happened, including why a worker died."""
    worker = tmp_path / f"lock_probe_{tag}.py"
    worker.write_text(_LOCK_PROBE_SOURCE, encoding="utf-8")
    target = tmp_path / f"shared_{tag}.dat"
    target.write_text("", encoding="utf-8")
    report = tmp_path / f"lock_report_{tag}.txt"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["MIND_MEM_LOG_LEVEL"] = "error"

    running = [
        subprocess.Popen(
            [sys.executable, str(worker), str(target), f"w{i}", str(_LOCK_ITERS), str(report), "1" if defeat else "0"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for i in range(_LOCK_PROCS)
    ]
    codes, errs = [], []
    for proc in running:
        _, err = proc.communicate(timeout=300)
        codes.append(proc.returncode)
        if err and err.strip():
            errs.append(err[-2000:])

    violations = completed = reported = 0
    failures: list[str] = []
    for part in sorted(tmp_path.glob(f"{report.name}.*")):
        line = part.read_text(encoding="utf-8").strip()
        if not line:
            continue
        reported += 1
        fields = line.split(None, 3)
        violations += int(fields[1])
        completed += int(fields[2])
        if len(fields) > 3 and fields[3] != "-":
            failures.append(f"{fields[0]} {fields[3]}")

    diagnosis = ""
    if failures:
        diagnosis = "a worker raised: " + "; ".join(failures)
    elif tuple(codes) != (0,) * _LOCK_PROCS:
        diagnosis = f"a worker exited non-zero without reporting why: {tuple(codes)}"
    elif reported != _LOCK_PROCS:
        diagnosis = f"only {reported} of {_LOCK_PROCS} workers left a report file"
    return LockProbeResult(violations, tuple(codes), reported, completed, diagnosis, "\n".join(errs))


class TestTheStoreLockActuallyExcludes:
    """Serialising the append is only as true as the lock underneath it.

    ``EvidenceChain.create`` and ``AuditChain.append`` both resolve their
    tail and write inside ``FileLock``. If that lock lets two processes in,
    both resolve the same tail, both link to it, and the ledger forks — the
    exact break those tests exist to prevent, reintroduced one layer down.
    So the lock gets measured directly rather than assumed: six processes,
    six hundred acquisitions each, counting the times a second process was
    inside the section.
    """

    def test_no_two_processes_are_ever_inside_the_section(self, tmp_path):
        probe = _run_lock_probe(tmp_path, defeat=False, tag="fixed")

        # Positive control: zero violations must mean zero overlaps, not
        # zero work. Every worker has to have finished and reported — and
        # when one did not, the FIRST line has to say why it did not.
        assert probe.diagnosis == "", f"{probe.diagnosis}\n{probe}"
        assert probe.exit_codes == (0,) * _LOCK_PROCS, f"a worker failed: {probe}"
        assert probe.reported == _LOCK_PROCS, f"only {probe.reported}/{_LOCK_PROCS} workers reported\n{probe}"
        assert probe.completed == _LOCK_PROCS * _LOCK_ITERS, (
            f"workers completed {probe.completed} of {_LOCK_PROCS * _LOCK_ITERS} acquisitions\n{probe}"
        )

        assert probe.violations == 0, f"two processes held the same lock {probe.violations} times\n{probe}"


class TestAHarnessThatCannotSayWhyIsNotAHarness:
    """The child's failure must reach the assertion, on its FIRST line.

    This is the defect that made the Windows lock failures unreadable. The
    harness passed ``stderr=subprocess.PIPE``, dropped the value
    ``communicate()`` handed back, and asserted on exit codes alone. Four CI
    rows therefore reported the entirety of what they knew as
    ``a worker failed: (0, 0, 1, 1, 0, 1)`` — six integers, and a traceback
    that had been captured and discarded. A harness that can watch a child
    die and not say of what is not measuring the child.

    Proven by killing a worker deliberately and reading what comes back, so
    the claim is demonstrated rather than asserted about the code.
    """

    def test_a_dying_worker_names_its_exception(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "test_chain_concurrency._LOCK_PROBE_SOURCE",
            "import sys\nraise RuntimeError('the child died of this')\n",
        )
        probe = _run_lock_probe(tmp_path, defeat=False, tag="dying")

        assert probe.exit_codes != (0,) * _LOCK_PROCS, "the mutation did not actually kill anything"
        assert "the child died of this" in str(probe), f"the child's reason never reached the parent:\n{probe}"
        assert "RuntimeError" in probe.stderr_tail, f"the child's traceback was dropped:\n{probe}"
        assert str(probe).splitlines()[0].strip(), "the diagnosis line is empty"

    def test_a_worker_that_raises_inside_the_section_is_named_not_counted(self, tmp_path, monkeypatch) -> None:
        """A failure INSIDE the loop, where the worker still writes a report.

        Distinct from the case above: here the worker survives to report, so
        the diagnosis has to come from the report line rather than from the
        exit code, and the acquisitions it did NOT complete must not be
        counted as done.
        """
        mutant = _LOCK_PROBE_SOURCE.replace(
            "    for i in range(iters):",
            "    for i in range(iters):\n        if i == 3: raise OSError(99, 'lock exploded')",
            1,
        )
        assert mutant != _LOCK_PROBE_SOURCE, "the mutation did not apply"
        monkeypatch.setattr("test_chain_concurrency._LOCK_PROBE_SOURCE", mutant)
        probe = _run_lock_probe(tmp_path, defeat=False, tag="exploding")

        assert probe.reported == _LOCK_PROCS, f"the workers did not report:\n{probe}"
        assert "lock exploded" in probe.diagnosis, f"the reason is not on the first line:\n{probe}"
        assert "OSError" in probe.diagnosis, probe.diagnosis
        assert probe.completed == _LOCK_PROCS * 3, f"unfinished acquisitions were counted as done:\n{probe}"

    def test_the_clean_run_reports_no_diagnosis(self, tmp_path) -> None:
        """Control: the diagnosis is not a string this harness always emits."""
        probe = _run_lock_probe(tmp_path, defeat=False, tag="clean")
        assert probe.diagnosis == "", probe.diagnosis
        assert probe.stderr_tail == "", probe.stderr_tail


#: Injected into the lock probe to make ``os.open`` answer the way Windows
#: does under contention: EACCES on the create, not EEXIST.
#:
#: Not a guess about Windows. It is the error windows-latest 3.12 actually
#: produced -- CI run 33904519141, job 101126178298, six processes on one
#: lock, four of them killed by
#: ``PermissionError: [Errno 13] Permission denied: '...shared_fixed.dat.lock'``
#: raised straight out of ``acquire``. Errno-only, with no ``winerror``:
#: ``OSError.__str__`` renders ``[WinError n]`` whenever one is set, and that
#: message did not, because ``os.open`` reaches the CRT's ``_wopen``.
#:
#: Fires only while the lockfile genuinely exists, which is the state the
#: platform reports it for, and only half the time, so both the refused and
#: the ordinary path are exercised in one run.
_WINDOWS_CREATE_REFUSAL = """\
import os as _os, errno as _errno, random as _random
_real_open = _os.open


def _windows_sharing_violation(path, flags, *a, **k):
    p = _os.fspath(path)
    if (flags & _os.O_CREAT) and p.endswith(".lock") and _random.random() < 0.5:
        if _os.path.exists(p):
            raise PermissionError(_errno.EACCES, "Permission denied", p)
    return _real_open(path, flags, *a, **k)


_os.open = _windows_sharing_violation
"""

#: The create arm as it stood before the refusal was handled: EEXIST is
#: contention and every other answer escapes ``acquire``. Applied on top of
#: :data:`_WINDOWS_CREATE_REFUSAL` in the twin below.
_PRE_FIX_CREATE_ARM = """\
def _create_claim_that_only_knows_eexist(self):
    try:
        return os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


mfl.FileLock._try_create_claim = _create_claim_that_only_knows_eexist
"""


def _probe_with(source_extra: str, tmp_path, tag: str, monkeypatch, iters: int = 150):
    """Run the lock probe with *source_extra* spliced into the worker."""
    monkeypatch.setattr("test_chain_concurrency._LOCK_ITERS", iters)
    head = "import os, sys, time, traceback\nimport mind_mem.mind_filelock as mfl\n"
    assert head in _LOCK_PROBE_SOURCE, "the worker preamble moved"
    monkeypatch.setattr(
        "test_chain_concurrency._LOCK_PROBE_SOURCE",
        _LOCK_PROBE_SOURCE.replace(head, head + source_extra, 1),
    )
    return _run_lock_probe(tmp_path, defeat=False, tag=tag)


class TestTheWindowsCreateRefusalDoesNotKillWorkers:
    """The evidenced Windows failure, reproduced across real processes here.

    ``FileLock`` answers ``O_CREAT|O_EXCL`` contention correctly on POSIX
    because POSIX only ever says ``EEXIST``. Windows also says ``EACCES``,
    and that answer escaped ``acquire`` and killed the worker: four of six in
    ``test_no_two_processes_are_ever_inside_the_section``, four of six in
    ``test_the_clean_run_reports_no_diagnosis``, and one of two in
    ``test_served_ledger_concurrency``.

    What is simulated is the ERRNO, not the platform. That Windows raises it
    is established by the CI transcript, not by this file; what these two
    tests establish is that the shipped lock handles it -- and, in the twin,
    that the handling is load-bearing rather than decorative.
    """

    def test_six_contending_workers_all_survive_the_refusal(self, tmp_path, monkeypatch) -> None:
        probe = _probe_with(_WINDOWS_CREATE_REFUSAL, tmp_path, "winsim", monkeypatch)

        assert probe.diagnosis == "", f"{probe.diagnosis}\n{probe}"
        assert probe.exit_codes == (0,) * _LOCK_PROCS, f"a worker died on a contended create:\n{probe}"
        assert probe.reported == _LOCK_PROCS, f"only {probe.reported}/{_LOCK_PROCS} workers reported\n{probe}"
        assert probe.completed == _LOCK_PROCS * 150, f"the refusal cost acquisitions:\n{probe}"
        # The refusal must be waited out, never routed around: exclusion is
        # still the property, and a "fix" that let a second process in would
        # pass every assertion above.
        assert probe.violations == 0, f"two processes held the same lock {probe.violations} times\n{probe}"

    def test_the_pre_fix_create_arm_kills_them(self, tmp_path, monkeypatch) -> None:
        """MUTATION TWIN. Put back "only EEXIST is contention" and it dies.

        Without this the test above would also pass on a platform where the
        injection never fires, and the guard it exists to prove would be
        untested. Measured here: the same ``PermissionError: [Errno 13]``
        text, from the same assertion line, as the Windows rows produced.
        """
        probe = _probe_with(_WINDOWS_CREATE_REFUSAL + _PRE_FIX_CREATE_ARM, tmp_path, "winsim_prefix", monkeypatch)

        assert probe.exit_codes != (0,) * _LOCK_PROCS, f"the pre-fix arm survived — the injection did not fire:\n{probe}"
        assert "PermissionError" in probe.diagnosis, f"it died of something else:\n{probe}"
        assert "Errno 13" in probe.diagnosis, f"not the errno the runners reported:\n{probe}"
        # And the lock was never actually broken by it — the workers died
        # LOSING a race, which is exactly why raising was the wrong answer.
        assert probe.violations == 0, f"the pre-fix arm also let two processes in:\n{probe}"


def _v501_stale_identity(self):
    """The v5.0.1 ``_is_stale`` verdict, in this code's identity protocol.

    Empty or missing was read as "the owner is gone" — the two states a
    lock being *taken* passes through.
    """
    try:
        with open(self.lock_path, "r", encoding="utf-8") as fh:
            pid_str = fh.read().strip()
        if not pid_str:
            return (0, 0)
        pid = int(pid_str)
        try:
            os.kill(pid, 0)
            return None
        except ProcessLookupError:
            return (0, 0)
        except PermissionError:
            return None
    except (OSError, ValueError):
        return (0, 0)


def _v501_break(self, identity=None):
    """v5.0.1 ``_break_stale``: unlink the path, arbitrated by nothing."""
    try:
        os.unlink(self.lock_path)
    except OSError:
        pass
    return _BREAK_REMOVED


class TestAnEmptyLockfileIsNotACorpse:
    """A lockfile is created and *then* written, so it is briefly empty.

    Reading that as an abandoned lock is how a waiter walked into a
    critical section somebody else was in the middle of claiming. This is
    the deterministic statement of it — no timing, no sampling.
    """

    def test_a_lock_being_taken_is_not_judged_abandoned(self, tmp_path):
        target = tmp_path / "contested.dat"
        target.write_text("", encoding="utf-8")
        lock = FileLock(str(target), timeout=0.2)
        lockfile = tmp_path / "contested.dat.lock"

        # Positive control: this lock DOES recognise a genuinely dead owner,
        # so the refusal below is about the state and not a verdict that
        # never fires.
        lockfile.write_text(_IMPOSSIBLE_PID + "\n", encoding="utf-8")
        assert lock._stale_identity() is not None

        lockfile.write_text("", encoding="utf-8")  # created, pid not yet written
        assert lock._stale_identity() is None, "a lock mid-handshake was read as abandoned"

        # And the acquire waits it out rather than stealing it.
        with pytest.raises(LockTimeout):
            lock.acquire()
        assert lockfile.exists(), "the acquire removed a lockfile it did not own"


class TestStoreLockMutationTwin:
    """The v5.0.1 verdict against the same file, which must come out wrong."""

    def test_the_v501_verdict_steals_a_lock_that_is_being_taken(self, tmp_path, monkeypatch):
        target = tmp_path / "contested.dat"
        target.write_text("", encoding="utf-8")
        lockfile = tmp_path / "contested.dat.lock"
        lockfile.write_text("", encoding="utf-8")

        monkeypatch.setattr(FileLock, "_stale_identity", _v501_stale_identity)
        monkeypatch.setattr(FileLock, "_break_stale", _v501_break)

        thief = FileLock(str(target), timeout=0.2)
        thief.acquire()  # v5.0.1: breaks a lock that was mid-handshake
        try:
            assert int(lockfile.read_text(encoding="utf-8").strip()) == os.getpid(), (
                "the v5.0.1 verdict did not steal the lock — this twin proves nothing"
            )
        finally:
            thief.release()


# ---------------------------------------------------------------------------
# Breaking a crashed holder's lock: the hand-off nobody watches
# ---------------------------------------------------------------------------

#: A holder that dies leaves its lockfile behind, and the next writers must
#: break it to make progress. That break is the one moment the protocol
#: deliberately deletes a file it does not own, so it is the one moment two
#: waiters can both act. Each round of this probe plants a lockfile owned by
#: a pid that cannot exist and releases every worker at once, so the
#: hand-off is exercised on every round rather than once at startup.
_STALE_RACE_SOURCE = """\
import os, sys, time
import mind_mem.mind_filelock as mfl

target, tag, rounds, bdir, nprocs, evidence = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4],
    int(sys.argv[5]), sys.argv[6],
)

breaks = [0]
_real_break = mfl.FileLock._break_stale


def _counted(self, identity=None):
    breaks[0] += 1
    return _real_break(self, identity)


mfl.FileLock._break_stale = _counted

holder = target + ".holder"
violations = 0
for r in range(rounds):
    open(os.path.join(bdir, "ready-%d-%s" % (r, tag)), "w").close()
    go = os.path.join(bdir, "go-%d" % r)
    deadline = time.monotonic() + 60.0
    while not os.path.exists(go):
        if time.monotonic() > deadline:
            raise SystemExit("go timeout at round %d" % r)
        time.sleep(0.001)
    with mfl.FileLock(target, timeout=30.0):
        with open(holder, "w", encoding="utf-8") as fh:
            fh.write(tag)
        time.sleep(0.0008)
        with open(holder, encoding="utf-8") as fh:
            got = fh.read()
        if got != tag:
            violations += 1
            with open(evidence, "a", encoding="utf-8") as fh:
                fh.write("round=%d tag=%s read=%s\\n" % (r, tag, got))
with open(evidence + ".counts", "a", encoding="utf-8") as fh:
    fh.write("%s %d %d\\n" % (tag, violations, breaks[0]))
"""

#: Windows unlink semantics, prepended to a worker so a Linux box can run
#: the matrix row it does not have. On Windows a file cannot be unlinked
#: while any handle is open on it unless that handle asked for
#: ``FILE_SHARE_DELETE``, which CPython's ``os.open`` never does. POSIX
#: allows it, and that difference is the whole regression: the break used to
#: unlink the abandoned lockfile *while holding a descriptor on it*, so on
#: Windows a stale lock could never be broken, every waiter spun to its
#: timeout, and the test session died rather than failing.
#:
#: Fidelity, stated plainly: this refuses unlinks for handles open in **this
#: process only**, which is exactly the break path. It therefore cannot see
#: the release-side gap, where another process's open handle refuses a
#: releasing process's unlink. That one is unmeasured here by construction.
_WINDOWS_UNLINK_PREAMBLE = """\
import os as _wos

_real_unlink = _wos.unlink


def _windows_unlink(path, *, dir_fd=None):
    try:
        target = _wos.stat(path)
    except OSError:
        return _real_unlink(path, dir_fd=dir_fd)
    for entry in _wos.listdir("/proc/self/fd"):
        try:
            st = _wos.stat("/proc/self/fd/" + entry)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (target.st_dev, target.st_ino):
            raise PermissionError(
                32,
                "The process cannot access the file because it is being used by another process",
            )
    return _real_unlink(path, dir_fd=dir_fd)


_wos.unlink = _windows_unlink

# Positive control, run before any of the code under test: a green result
# from a simulation that is not actually refusing anything says nothing at
# all. Prove the refusal here, and make its absence a worker failure.
_probe = _wos.path.join(_wos.path.dirname(_wos.path.abspath(__file__)), "winsim-probe-%d" % _wos.getpid())
_pfd = _wos.open(_probe, _wos.O_CREAT | _wos.O_RDWR, 0o600)
try:
    try:
        _wos.unlink(_probe)
    except PermissionError:
        pass
    else:
        raise SystemExit("the windows unlink simulation is not refusing anything")
finally:
    _wos.close(_pfd)
_real_unlink(_probe)
"""

#: The break protocol exactly as it shipped in 5.0.1: OS-lock arbitration,
#: and then ``unlink`` of the inode **while this process still holds a
#: descriptor on it**. Legal on POSIX, refused on Windows. Prepended after
#: :data:`_WINDOWS_UNLINK_PREAMBLE`, it is the mutation that must turn the
#: simulated-Windows gate red — and if it does not, that gate is not
#: watching the thing it was written to watch.
_UNLINK_BREAK_PREAMBLE = """\
import os as _uos
import mind_mem.mind_filelock as _umfl


def _unlink_break(self, identity=None):
    if identity is None:
        identity = self._stale_identity()
    if identity is None:
        return _umfl._BREAK_NOTHING
    try:
        fd = _uos.open(self.lock_path, _uos.O_RDWR)
    except OSError:
        return _umfl._BREAK_NOTHING
    try:
        st = _uos.fstat(fd)
        if (st.st_dev, st.st_ino) != identity:
            return _umfl._BREAK_NOTHING
        held = self._try_os_lock(fd)
        if held is None:
            self._unlink_if_ours(identity)
            return _umfl._BREAK_REMOVED
        if not held:
            return _umfl._BREAK_NOTHING
        try:
            on_path = _uos.stat(self.lock_path)
        except OSError:
            return _umfl._BREAK_NOTHING
        if (on_path.st_dev, on_path.st_ino) != identity:
            return _umfl._BREAK_NOTHING
        try:
            _uos.unlink(self.lock_path)  # the Windows-illegal step
        except OSError:
            pass
        # Unconditionally "the corpse is gone", which is what 5.0.1's caller
        # assumed when it looped straight back into the create without ever
        # consulting its own deadline. Hence a spin, not a timeout.
        return _umfl._BREAK_REMOVED
    finally:
        try:
            self._os_unlock(fd)
        except OSError:
            pass
        try:
            _uos.close(fd)
        except OSError:
            pass


_umfl.FileLock._break_stale = _unlink_break
"""

#: A pid no process can have, so ``os.kill(pid, 0)`` is a definite
#: "this owner is gone" rather than a guess.
_IMPOSSIBLE_PID = "999999999"
_RACE_PROCS = 6
_RACE_ROUNDS = 60


#: The share of the per-test kill this race may spend before it fails with
#: its own words. Strictly below 1, so for any positive external timeout
#: the internal deadline fires FIRST — which is the one property the
#: constant it replaces could not have.
_RACE_BUDGET_SHARE = 0.5

#: Used only when no per-test kill is in force at all (a bare ``pytest``
#: run with pytest-timeout absent), where there is nothing to fire before.
_RACE_BUDGET_UNPOLICED = 120.0


def _effective_pytest_timeout(request) -> float | None:
    """The per-test wall-clock kill in force for *request*, or ``None``.

    Resolution order is pytest-timeout's own: a ``timeout`` marker on the
    item beats ``--timeout`` on the command line, which beats the ini
    value. Every lookup is guarded, because with pytest-timeout not
    installed neither the option nor the ini key exists and both raise.
    """
    marker = request.node.get_closest_marker("timeout")
    if marker is not None:
        if marker.args:
            return float(marker.args[0])
        if "timeout" in marker.kwargs:
            return float(marker.kwargs["timeout"])
    for read, key in ((request.config.getoption, "timeout"), (request.config.getini, "timeout")):
        try:
            raw = read(key)
        except (ValueError, KeyError):
            continue
        if raw in (None, "", 0, "0"):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _budget_that_fires_first(request) -> float:
    """A self-imposed deadline is only a deadline while it fires first.

    This race's bound used to be the constant ``240.0`` while CI runs
    ``--timeout=120 --timeout-method=thread``. That constant could never
    fire: pytest-timeout's thread reached 120 s first and killed the
    interpreter inside the poll loop, so every Windows row of CI run
    33628984458 ended in a stack dump and ``exit code 1`` with no summary
    line at all — five other failures in that run lost their names with
    it, and roughly ninety per cent of the suite never ran.

    Deriving the bound from the kill actually in force makes that ordering
    impossible to get wrong, instead of something the next edit has to
    remember. There is deliberately no floor term: a floor is exactly what
    would let the budget climb back above a small timeout.
    """
    killer = _effective_pytest_timeout(request)
    if killer is None or killer <= 0:
        return _RACE_BUDGET_UNPOLICED
    return killer * _RACE_BUDGET_SHARE


@dataclass(frozen=True)
class _RaceOutcome:
    """What one stale-break race produced, wedge included.

    ``wedged`` is empty when the race finished inside its budget, and
    otherwise carries the reason *plus* the evidence a reader needs to
    diagnose it without a rerun: which round it stopped on, every worker's
    pid and live/exited state at that moment, the lockfile's contents, the
    barrier directory, and each worker's captured output.
    """

    violations: int
    breaks: int
    codes: tuple[int, ...]
    pids: tuple[int, ...]
    evidence: str
    transcript: str
    wedged: str


def _race_snapshot(running, target, bdir, round_reached: int) -> str:
    """The state of the box at the moment the race ran out of budget."""
    lines = [f"  stopped waiting in round {round_reached} of {_RACE_ROUNDS}"]
    for proc in running:
        state = proc.poll()
        lines.append(f"  worker pid={proc.pid} state={'running' if state is None else f'exited {state}'}")
    lock = target.parent / (target.name + ".lock")
    try:
        lines.append(f"  lockfile {lock.name}: {lock.read_text(encoding='utf-8')!r}")
    except OSError as exc:
        lines.append(f"  lockfile {lock.name}: unreadable ({exc})")
    try:
        entries = sorted(os.listdir(bdir))
        lines.append(f"  barrier {bdir.name}: {len(entries)} entries, last: {entries[-12:]}")
    except OSError as exc:
        lines.append(f"  barrier {bdir.name}: unreadable ({exc})")
    return "\n".join(lines)


def _worker_transcript(talk) -> str:
    """Every worker's captured output, which used to be read and dropped."""
    lines = []
    for pid, code, out, err in talk:
        lines.append(f"  worker pid={pid} exit={code}")
        for stream, text in (("stdout", out), ("stderr", err)):
            if text:
                lines.append(f"    {stream}: " + text.replace("\n", "\n      "))
    return "\n".join(lines) or "  (no worker output)"


def _run_stale_race(request, tmp_path, *, tag: str, preamble: str = "") -> _RaceOutcome:
    """Run the stale-break race under a deadline it is guaranteed to reach.

    The budget is derived from the per-test kill in force (see
    :func:`_budget_that_fires_first`) rather than chosen next to it, so a
    lock that can never be broken comes back as a named failure carrying
    pids, states and the lockfile — not as a hung session that
    pytest-timeout kills, taking every other test's failure text with it.
    That is how five Windows failures came back with no assertion text.
    """
    budget = _budget_that_fires_first(request)
    worker = tmp_path / f"stale_race_{tag}.py"
    worker.write_text(preamble + _STALE_RACE_SOURCE, encoding="utf-8")
    bdir = tmp_path / f"barrier_stale_{tag}"
    bdir.mkdir()
    target = tmp_path / f"contested_{tag}.dat"
    target.write_text("", encoding="utf-8")
    evidence = tmp_path / f"overlaps_{tag}.txt"
    evidence.write_text("", encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["MIND_MEM_LOG_LEVEL"] = "error"

    running = [
        subprocess.Popen(
            [
                sys.executable,
                str(worker),
                str(target),
                f"w{i}",
                str(_RACE_ROUNDS),
                str(bdir),
                str(_RACE_PROCS),
                str(evidence),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for i in range(_RACE_PROCS)
    ]
    deadline = time.monotonic() + budget
    wedged = ""
    snapshot = ""
    next_liveness = 0.0
    try:
        for rnd in range(_RACE_ROUNDS):
            prefix = f"ready-{rnd}-"
            while sum(1 for f in os.listdir(bdir) if f.startswith(prefix)) < _RACE_PROCS:
                now = time.monotonic()
                if now > deadline:
                    # Nobody can make progress at all. Photograph the box
                    # BEFORE the finally block kills anything: the live
                    # pids and states are the diagnosis.
                    wedged = f"the race made no progress past round {rnd} of {_RACE_ROUNDS} within its {budget:.1f}s budget"
                    snapshot = _race_snapshot(running, target, bdir, rnd)
                    break
                if now >= next_liveness:
                    # A worker cannot legitimately be gone here: it exits
                    # only after the final round, which needs the final
                    # ``go-`` file, which the parent writes only once that
                    # round's barrier has filled. So a dead worker
                    # mid-barrier IS the failure, and waiting out the rest
                    # of the budget to say so only buries the reason.
                    # Sampled at 20 Hz rather than at the 1 kHz poll rate,
                    # so the healthy path pays six waitpids every 50 ms
                    # instead of six every millisecond.
                    next_liveness = now + 0.05
                    if any(proc.poll() is not None for proc in running):
                        gone = sum(1 for proc in running if proc.poll() is not None)
                        wedged = (
                            f"{gone} of {_RACE_PROCS} workers were already gone during round {rnd}'s barrier, "
                            "which no worker can reach before the parent opens the last gate"
                        )
                        snapshot = _race_snapshot(running, target, bdir, rnd)
                        break
                time.sleep(0.001)
            if wedged:
                break
            # The crashed holder: a lockfile naming an owner that is gone.
            (target.parent / (target.name + ".lock")).write_text(_IMPOSSIBLE_PID + "\n", encoding="utf-8")
            (bdir / f"go-{rnd}").write_text("", encoding="utf-8")
    finally:
        codes = []
        talk = []
        for proc in running:
            try:
                out, err = proc.communicate(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:  # pragma: no cover - worker wedged
                if not snapshot:
                    snapshot = _race_snapshot(running, target, bdir, _RACE_ROUNDS)
                proc.kill()
                out, err = proc.communicate()
                wedged = wedged or f"a worker had to be killed after {budget:.1f}s: it never exited"
            codes.append(proc.returncode)
            talk.append((proc.pid, proc.returncode, (out or "").strip(), (err or "").strip()))

    transcript = _worker_transcript(talk)
    if wedged:
        wedged = "\n".join([wedged, snapshot, transcript])

    counts = tmp_path / f"overlaps_{tag}.txt.counts"
    lines = [ln.split() for ln in counts.read_text(encoding="utf-8").splitlines() if ln.strip()] if counts.exists() else []
    return _RaceOutcome(
        violations=sum(int(parts[1]) for parts in lines),
        breaks=sum(int(parts[2]) for parts in lines),
        codes=tuple(codes),
        pids=tuple(proc.pid for proc in running),
        evidence=evidence.read_text(encoding="utf-8"),
        transcript=transcript,
        wedged=wedged,
    )


#: Six workers x sixty rounds of contended acquire is legitimately long —
#: 15.1 s and 15.4 s measured on Linux for the two tests below — and the
#: 120 s default kill leaves the race no room to reach its own deadline on
#: a slower runner. This is the hang detector for these two tests only,
#: and it is what :func:`_budget_that_fires_first` derives the internal
#: 180 s budget from: twelve times the measured cost, and still half the
#: kill. No assertion below is relaxed by it.
_RACE_KILL = 360


@pytest.mark.timeout(_RACE_KILL)
class TestBreakingACrashedHoldersLockHasOneWinner:
    def test_no_two_waiters_break_the_same_lock(self, request, tmp_path):
        race = _run_stale_race(request, tmp_path, tag="fixed")

        # Bounded time, not a timeout: a lock nobody can break has to come
        # back as a named failure here, not as a hung session somebody else
        # has to kill.
        assert not race.wedged, race.wedged

        # Positive control: the break path has to have been walked, or a
        # clean result says nothing. Every round plants a lockfile that
        # somebody must break before anyone can enter.
        assert race.codes == (0,) * _RACE_PROCS, f"a worker failed: {race.codes}\n{race.transcript}"
        assert race.breaks >= _RACE_ROUNDS, f"only {race.breaks} stale breaks over {_RACE_ROUNDS} rounds — the path was not exercised"

        assert race.violations == 0, f"two processes held the same lock {race.violations} times:\n{race.evidence}"


#: The simulation needs ``/proc/self/fd`` to know which files this process
#: holds open. Naming the requirement is the point — a gate that quietly
#: does nothing on a platform is worse than one that says it is not running.
_HAS_PROCFS = os.path.isdir("/proc/self/fd")


@pytest.mark.timeout(_RACE_KILL)
@pytest.mark.skipif(not _HAS_PROCFS, reason="the Windows unlink simulation reads /proc/self/fd")
class TestACrashedHoldersLockIsBreakableUnderWindowsUnlink:
    """The Windows matrix row, run on a box that is not Windows.

    The property, stated so it can be pinned: *for a lockfile whose owner
    is confirmed dead and N concurrent waiters, exactly one waiter enters
    the critical section, it does so within a bounded time (not a timeout),
    and no syscall in the break path removes or renames the lockfile while
    the breaker holds a descriptor on it.*

    The last clause is the one a Linux run cannot check by inspection, so
    the run refuses those unlinks instead — which is what a Windows runner
    does. Before the break became an adoption this was not a failure but a
    hang: six workers spinning to a 30 s ``LockTimeout`` while the parent
    waited on a round barrier that could never fill.

    What this does **not** cover, and only a Windows CI row can: that
    ``msvcrt.locking`` really does contend and really is released on death
    across processes, that its mandatory byte-range locks behave as
    assumed, and the release-side unlink, which is refused by *another*
    process's handle and so is invisible to a same-process simulation.
    """

    def test_the_break_never_unlinks_a_file_it_holds_open(self, request, tmp_path):
        race = _run_stale_race(
            request,
            tmp_path,
            tag="winsim",
            preamble=_WINDOWS_UNLINK_PREAMBLE,
        )

        assert not race.wedged, race.wedged
        # Positive control, twice over: the simulation proves itself active
        # at worker startup (or the worker exits non-zero), and the break
        # path has to have been walked on every round.
        assert race.codes == (0,) * _RACE_PROCS, f"a worker failed: {race.codes}\n{race.transcript}"
        assert race.breaks >= _RACE_ROUNDS, f"only {race.breaks} stale breaks over {_RACE_ROUNDS} rounds — the path was not exercised"

        assert race.violations == 0, f"two processes held the same lock {race.violations} times:\n{race.evidence}"


# ---------------------------------------------------------------------------
# The deadline the two races above run under, and the wedge it must catch
# ---------------------------------------------------------------------------


class _StubNode:
    """An item carrying a ``timeout`` marker, or carrying none."""

    def __init__(self, seconds: float | None) -> None:
        self._seconds = seconds

    def get_closest_marker(self, name: str):
        if name != "timeout" or self._seconds is None:
            return None
        return pytest.mark.timeout(self._seconds).mark


class _StubConfig:
    """A config where ``--timeout`` and the ini key may each be absent.

    Absent means *raises*, which is what ``getoption``/``getini`` really do
    when pytest-timeout is not installed — a stub that returned ``None``
    instead would hide the branch that has to survive that.
    """

    def __init__(self, option: float | None, ini: float | None) -> None:
        self._option, self._ini = option, ini

    def getoption(self, name: str):
        if name != "timeout" or self._option is None:
            raise ValueError(f"no option named {name!r}")
        return self._option

    def getini(self, name: str):
        if name != "timeout" or self._ini is None:
            raise ValueError(f"unknown ini option {name!r}")
        return self._ini


class _StubRequest:
    """Only what :func:`_effective_pytest_timeout` reads, nothing else."""

    def __init__(self, *, marker: float | None = None, option: float | None = None, ini: float | None = None) -> None:
        self.node = _StubNode(marker)
        self.config = _StubConfig(option, ini)


#: The bound that shipped before the derivation, kept as a number so the
#: regression it caused can be asserted rather than described.
_BUDGET_THAT_COULD_NOT_FIRE = 240.0

#: The kill ``.github/workflows/ci.yml`` passes on every matrix row.
_CI_KILL = 120.0


class TestTheRaceDeadlineFiresBeforeTheKill:
    """A self-imposed deadline is only a deadline while it fires first.

    Pins the ordering, the resolution order it is read from, and the fact
    that the constant this replaced violated it at the timeout CI runs.
    """

    def test_a_marker_beats_the_command_line(self) -> None:
        assert _effective_pytest_timeout(_StubRequest(marker=50, option=120)) == 50.0

    def test_the_command_line_beats_the_ini_value(self) -> None:
        assert _effective_pytest_timeout(_StubRequest(option=120, ini=300)) == 120.0

    def test_the_ini_value_is_read_when_nothing_else_is_set(self) -> None:
        assert _effective_pytest_timeout(_StubRequest(ini=300)) == 300.0

    def test_no_kill_at_all_is_reported_as_none(self) -> None:
        # pytest-timeout absent: both lookups raise, and the race falls back
        # to its own bound because there is nothing to fire before.
        assert _effective_pytest_timeout(_StubRequest()) is None
        assert _budget_that_fires_first(_StubRequest()) == _RACE_BUDGET_UNPOLICED

    @pytest.mark.timeout(97)
    def test_the_stubs_agree_with_a_real_request(self, request) -> None:
        # Positive control for every stub above: read the same value off the
        # genuine fixture, so a stub that lies about the API is caught here.
        assert _effective_pytest_timeout(request) == 97.0
        assert _budget_that_fires_first(request) == 97.0 * _RACE_BUDGET_SHARE

    def test_the_budget_is_strictly_under_the_kill_at_every_scale(self) -> None:
        for killer in (1.0, 5.0, 30.0, _CI_KILL, 240.0, float(_RACE_KILL), 3600.0):
            budget = _budget_that_fires_first(_StubRequest(marker=killer))
            assert budget < killer, f"a {budget}s budget under a {killer}s kill can never fire"

    def test_the_constant_this_replaced_could_not_fire_under_ci(self) -> None:
        # Positive control for the property above: name the regression as a
        # number. CI run 33628984458 is what it cost — five Windows rows
        # killed mid-poll with no summary line.
        assert _BUDGET_THAT_COULD_NOT_FIRE > _CI_KILL, (
            "the constant no longer outlives the CI kill, so this test no longer "
            "demonstrates the ordering bug the derivation exists to prevent"
        )
        assert _budget_that_fires_first(_StubRequest(option=_CI_KILL)) < _CI_KILL

    def test_both_races_declare_the_kill_their_budget_is_derived_from(self) -> None:
        # Wiring, not intent: a derivation nothing consults is decoration.
        for cls in (
            TestBreakingACrashedHoldersLockHasOneWinner,
            TestACrashedHoldersLockIsBreakableUnderWindowsUnlink,
        ):
            marks = [m for m in getattr(cls, "pytestmark", []) if m.name == "timeout"]
            assert marks, f"{cls.__name__} lost its timeout marker — its budget is back to the 120s default"
            killer = float(marks[0].args[0])
            assert _budget_that_fires_first(_StubRequest(marker=killer)) < killer


#: A worker that reaches the barrier and then never acquires. Round 0
#: fills, the parent plants the crashed holder's lockfile and opens the
#: gate, and round 1's barrier can never fill — the exact shape of the
#: Windows wedge, on demand and in seconds.
_NEVER_ACQUIRES_PREAMBLE = """\
import time as _wt

import mind_mem.mind_filelock as _wmfl


def _never_acquires(self):
    while True:
        _wt.sleep(0.05)


_wmfl.FileLock.acquire = _never_acquires
"""

#: Small enough to keep this test cheap, large enough that the wedge, the
#: six one-second waits for a worker that will not exit, and the kills all
#: fit inside it. Measured below the marker; see the assertion.
_WEDGE_KILL = 40


class TestAWedgedRaceFailsWithEvidenceInsteadOfKillingTheRun:
    """The behaviour the Windows rows needed and did not have.

    On CI run 33628984458 this race stopped making progress on a Windows
    runner, the 240 s bound could not fire under the 120 s kill, and
    pytest-timeout's thread method took the interpreter out at 10% of the
    suite: a stack dump, ``exit code 1``, no summary line, and five other
    failures on those rows that were never named. This test reproduces a
    wedge deliberately and asserts the run comes back with the diagnosis
    instead — which round, which pids, whether they were alive, and what
    the lockfile held at that moment.
    """

    @pytest.mark.timeout(_WEDGE_KILL)
    def test_a_race_that_cannot_hand_over_names_the_round_the_pids_and_the_lockfile(self, request, tmp_path):
        started = time.monotonic()
        race = _run_stale_race(request, tmp_path, tag="wedge", preamble=_NEVER_ACQUIRES_PREAMBLE)
        elapsed = time.monotonic() - started

        assert race.wedged, (
            "a race whose workers can never acquire came back clean — this harness cannot see a wedge, "
            f"so the two gates above prove nothing (codes={race.codes})"
        )
        # Round 0 completed, so this is the hand-over wedging and not the
        # workers failing to start.
        assert f"round 1 of {_RACE_ROUNDS}" in race.wedged, f"the wedge is not where this test aims it:\n{race.wedged}"
        # Positive control on the pids: the report has to name the processes
        # actually spawned, not a plausible-looking placeholder.
        for pid in race.pids:
            assert f"pid={pid}" in race.wedged, f"worker {pid} is missing from the report:\n{race.wedged}"
        assert "state=running" in race.wedged, f"no worker was reported alive at the wedge:\n{race.wedged}"
        assert _IMPOSSIBLE_PID in race.wedged, f"the crashed holder's lockfile is missing:\n{race.wedged}"
        assert "barrier" in race.wedged, f"the barrier state is missing:\n{race.wedged}"

        # And it is bounded: the whole thing, kills included, finished
        # inside the kill rather than being ended by it.
        assert elapsed < _WEDGE_KILL, f"the wedge took {elapsed:.1f}s of a {_WEDGE_KILL}s kill — no margin left"


#: A worker that is gone before it ever reaches the barrier. This is the
#: shape a Windows worker takes when its lock protocol raises: one of the
#: six is missing, round 0's barrier can never fill, and the parent waits
#: for a file that is not coming. Prepended to the worker source, so it
#: fires before any import.
_DIES_AT_STARTUP_PREAMBLE = "raise SystemExit(3)\n"


class TestADeadWorkerIsNamedAtOnceRatherThanWaitedOut:
    """The budget is the backstop, not the diagnosis.

    A worker cannot legitimately exit during a barrier wait — reaching the
    exit needs the final ``go-`` file, which the parent writes only after
    the final barrier has filled — so one that is gone mid-barrier is
    itself the failure. Sitting out the remaining budget to report it costs a CI row
    minutes and buries the reason under a wall-clock message.
    """

    @pytest.mark.timeout(_WEDGE_KILL)
    def test_workers_that_exited_are_named_without_burning_the_budget(self, request, tmp_path):
        budget = _budget_that_fires_first(request)
        started = time.monotonic()
        race = _run_stale_race(request, tmp_path, tag="dead", preamble=_DIES_AT_STARTUP_PREAMBLE)
        elapsed = time.monotonic() - started

        assert race.wedged, f"six dead workers came back as a clean race (codes={race.codes})"
        # Positive control: they really did die, and died the way this test
        # arranged them to — not because the harness failed to start them.
        assert race.codes == (3,) * _RACE_PROCS, f"the workers did not exit as arranged: {race.codes}"
        assert "already gone during round 0" in race.wedged, f"the wedge was not attributed to them:\n{race.wedged}"
        assert "exited 3" in race.wedged, f"the exit states are missing:\n{race.wedged}"
        assert elapsed < budget, (
            f"took {elapsed:.1f}s of a {budget:.1f}s budget to notice six dead workers — "
            "the fast path is not wired, so a Windows row pays the whole budget to learn this"
        )


#: One process, one crashed holder's lockfile, one attempt to take it. The
#: mutation twin does not need six workers and sixty rounds: under Windows
#: unlink semantics the pre-adoption protocol does not *sometimes* lose the
#: race, it can never break the lock at all.
_CORPSE_PROBE_SOURCE = """\
import os, sys
import mind_mem.mind_filelock as mfl

target, marker = sys.argv[1], sys.argv[2]
with mfl.FileLock(target, timeout=2.0):
    open(marker, "w").close()
"""


def _try_to_break_a_corpse(tmp_path, *, tag: str, preamble: str, patience: float = 30.0) -> tuple[bool, int | None]:
    """Plant a crashed holder's lockfile and try once to take the lock.

    Returns ``(entered, exit_code)``. A worker still running when
    *patience* runs out is killed and reported as not having entered,
    which is the honest reading of a protocol that cannot break the lock
    — whether it gives up at its timeout or spins forever without ever
    consulting one.
    """
    worker = tmp_path / f"corpse_probe_{tag}.py"
    worker.write_text(preamble + _CORPSE_PROBE_SOURCE, encoding="utf-8")
    target = tmp_path / f"corpse_{tag}.dat"
    target.write_text("", encoding="utf-8")
    marker = tmp_path / f"entered_{tag}"
    (tmp_path / f"corpse_{tag}.dat.lock").write_text(_IMPOSSIBLE_PID + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["MIND_MEM_LOG_LEVEL"] = "error"
    proc = subprocess.Popen(
        [sys.executable, str(worker), str(target), str(marker)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        proc.communicate(timeout=patience)
    except subprocess.TimeoutExpired:  # pragma: no cover - only the mutant gets here
        proc.kill()
        proc.communicate()
    return marker.exists(), proc.returncode


@pytest.mark.skipif(not _HAS_PROCFS, reason="the Windows unlink simulation reads /proc/self/fd")
class TestWindowsUnlinkGateMutationTwin:
    """Put the unlink back under the simulation, and it must stop working.

    This is the measurement the regression was found by, in its smallest
    honest form. The protocol that shipped in 5.0.1 breaks a stale lock by
    unlinking it while holding a descriptor on it. Refuse that unlink, as
    Windows does, and the corpse is immortal: the acquirer confirms the
    dead owner, asks for the unlink, is refused, is told the corpse is
    gone, loops back into the create, and finds it still there — forever,
    without ever consulting its own timeout. Measured that way: a worker
    still burning 98% of a core fourteen minutes into a thirty-second
    ``timeout``.
    """

    def test_unlinking_under_our_own_descriptor_can_never_break_the_lock(self, tmp_path):
        # Positive control first: the shipped protocol takes exactly this
        # planted lock under exactly this simulation. Without it, "the
        # mutant did not get in" is a statement about a broken harness.
        entered, code = _try_to_break_a_corpse(tmp_path, tag="shipped", preamble=_WINDOWS_UNLINK_PREAMBLE)
        assert entered, (
            f"the shipped break could not take a crashed holder's lock under the simulation (exit {code}) — "
            "the twin's harness is broken and the comparison below proves nothing"
        )

        entered, code = _try_to_break_a_corpse(
            tmp_path,
            tag="mutant",
            preamble=_WINDOWS_UNLINK_PREAMBLE + _UNLINK_BREAK_PREAMBLE,
            patience=6.0,
        )
        assert not entered, (
            "unlinking the lockfile from under our own descriptor still broke the lock under Windows "
            f"unlink semantics (exit {code}) — this twin cannot see the regression it is here to catch"
        )


def _break_and_claim(lock: FileLock, lock_path: str) -> tuple[int, tuple[int, int]] | None:
    """A second breaker doing exactly what the protocol allows it to do.

    It takes the OS lock on the abandoned file and then *adopts* it: same
    inode, its own pid, its own descriptor, the lock still held. That is
    the winning half of a stale break under this protocol, and it never
    unlinks — which is what makes the competitor itself runnable on
    Windows. Returns ``(fd, identity)`` of the claim it now holds, or
    ``None`` when the arbitration refused it because somebody else is
    already breaking that file. The caller owns the returned descriptor.
    """
    fd = os.open(lock_path, os.O_RDWR)
    if lock._try_os_lock(fd) is not True:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f"{os.getpid()}\n".encode())
    st = os.fstat(fd)
    return (fd, (st.st_dev, st.st_ino))


class _CompetingBreaker:
    """Runs a second breaker in the gap between a ``stat`` and the unlink.

    Two waiters confirm the same abandoned lockfile; the first unlinks it
    and creates its own claim; the second — whose ``stat`` already said
    "yes, that is the dead one" — unlinks by path and deletes the winner's
    claim. Driving the interleaving from the ``stat`` makes it a fact
    rather than a sampling exercise, and the competitor still has to pass
    the same arbitration a real one would.
    """

    def __init__(self, lock: FileLock) -> None:
        self.lock = lock
        self.lock_path = lock.lock_path
        self.real_stat = os.stat
        self.attempts = 0
        self.winner: tuple[int, int] | None = None
        self.fd: int | None = None

    def __call__(self, path, *args, **kwargs):
        st = self.real_stat(path, *args, **kwargs)
        if self.attempts == 0 and str(path) == self.lock_path:
            self.attempts = 1
            claimed = _break_and_claim(self.lock, self.lock_path)
            if claimed is not None:
                self.fd, self.winner = claimed
        return st

    def close(self) -> None:
        """Hand back the descriptor and the OS lock an adopter would keep."""
        if self.fd is not None:
            try:
                self.lock._os_unlock(self.fd)
            except OSError:
                pass
            os.close(self.fd)
            self.fd = None


def _plant_a_crashed_holder(tmp_path) -> tuple[FileLock, str, tuple[int, int]]:
    target = tmp_path / "contested.dat"
    target.write_text("", encoding="utf-8")
    lock = FileLock(str(target), timeout=0.2)
    with open(lock.lock_path, "w", encoding="utf-8") as fh:
        fh.write(_IMPOSSIBLE_PID + "\n")
    identity = lock._stale_identity()
    assert identity is not None, "the planted lockfile must read as abandoned"
    return lock, lock.lock_path, identity


class TestBreakingIsAtomicAgainstAHandOff:
    """Only one waiter may break a crashed holder's lock.

    No ``skipif`` any more, and that is the point: the break no longer
    unlinks anything, so there is no step here Windows refuses. The
    post-condition moved with the protocol — "the lockfile is gone"
    became "the lockfile is *ours*", which is the stronger of the two:
    one event now says both that exactly one waiter broke it and that
    exactly one holder came out of it.
    """

    def test_a_second_breaker_is_refused_while_the_break_is_in_progress(self, tmp_path, monkeypatch):
        lock, lock_path, identity = _plant_a_crashed_holder(tmp_path)
        competitor = _CompetingBreaker(lock)
        monkeypatch.setattr(os, "stat", competitor)

        outcome = lock._break_stale(identity)

        monkeypatch.undo()
        competitor.close()
        # Positive control: the competitor has to have tried, or "it did not
        # get in" is a statement about a competitor that never ran.
        assert competitor.attempts == 1, "the competitor never ran — the interleaving was not exercised"
        assert competitor.winner is None, "a second breaker got in and replaced the lockfile mid-break"

        assert outcome == _BREAK_ADOPTED, f"the abandoned lockfile was not broken: {outcome}"
        assert os.path.exists(lock_path), "the break unlinked the lockfile — the step Windows refuses"
        assert _identity_of(lock_path) == identity, "the path names a different inode than the one adopted"
        assert lock._lock_identity == identity, "the adopter did not record the inode it judged"
        assert lock._lock_fd is not None, "the adopter did not keep the descriptor"
        with open(lock_path, encoding="utf-8") as fh:
            assert int(fh.read().strip()) == os.getpid(), "the adopted lockfile does not name us"

        # And it really is locked, not merely rewritten: a fresh descriptor
        # on the same file is refused while the adopter holds it.
        probe = os.open(lock_path, os.O_RDWR)
        try:
            assert lock._try_os_lock(probe) is False, "the adopter is not holding the OS lock"
        finally:
            os.close(probe)

        lock.release()
        assert not os.path.exists(lock_path), "release did not remove the adopted lockfile"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the mutation itself is the Windows-illegal step: it unlinks the lockfile the adopting competitor holds open",
)
class TestStaleBreakMutationTwin:
    """Why the identity check alone was not enough, stated as a measurement.

    :meth:`FileLock._unlink_if_ours` is the shipped helper and it is
    correct where :meth:`FileLock.release` uses it — a live owner's
    lockfile cannot be swapped, because swapping it means breaking it and
    breaking it means the owner is dead. As the *break* primitive it is
    not correct, and this is the run that says so: it holds no OS lock, so
    the competing breaker walks straight in.
    """

    def test_check_then_unlink_deletes_the_winners_claim(self, tmp_path, monkeypatch):
        lock, lock_path, identity = _plant_a_crashed_holder(tmp_path)
        competitor = _CompetingBreaker(lock)
        monkeypatch.setattr(os, "stat", competitor)

        lock._unlink_if_ours(identity)  # stat says "the corpse", then unlink by path

        monkeypatch.undo()
        competitor.close()
        assert competitor.attempts == 1, "the competitor never ran — the twin proves nothing"
        assert competitor.winner is not None, (
            "the competitor was refused — check-then-unlink cannot have deleted a claim that was never made"
        )
        assert not os.path.exists(lock_path), (
            "check-then-unlink left the winner's claim alone — this twin cannot see the defect it is here to catch"
        )


def _we_hold_it_open(path: str) -> bool:
    """Whether this process has any descriptor open on *path* right now."""
    target = os.stat(path)
    for entry in os.listdir("/proc/self/fd"):
        try:
            st = os.stat("/proc/self/fd/" + entry)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (target.st_dev, target.st_ino):
            return True
    return False


class TestAFilesystemThatCannotArbitrateStillBreaksACorpse:
    """The degraded fallback, where no OS lock exists to arbitrate with.

    NFS and FUSE mounts that answer ENOLCK have nothing atomic to hand, so
    the break falls back to the identity-checked unlink it always had. Two
    properties are new and neither is reachable on the OS-locked path, so
    they are stated here.
    """

    @pytest.mark.skipif(not _HAS_PROCFS, reason="reads /proc/self/fd to see what this process holds open")
    def test_it_closes_its_descriptor_before_unlinking(self, tmp_path, monkeypatch):
        lock, lock_path, identity = _plant_a_crashed_holder(tmp_path)
        monkeypatch.setattr(FileLock, "_try_os_lock", lambda self, fd: None)

        # Positive control: with the lockfile open, the helper says so — or
        # "we had nothing open" below is a helper that always says no.
        held = os.open(lock_path, os.O_RDONLY)
        try:
            assert _we_hold_it_open(lock_path), "the open-descriptor probe cannot see an open descriptor"
        finally:
            os.close(held)

        seen: dict = {}
        real_unlink = os.unlink

        def watching_unlink(path, *args, **kwargs):
            if str(path) == lock_path:
                seen["open_at_unlink"] = _we_hold_it_open(lock_path)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", watching_unlink)
        outcome = lock._break_stale(identity)
        monkeypatch.undo()

        assert seen.get("open_at_unlink") is False, (
            "the fallback unlinked the lockfile while still holding a descriptor on it — the one step Windows refuses"
        )
        assert outcome == _BREAK_REMOVED
        assert not os.path.exists(lock_path), "the corpse was not removed"

    def test_a_refused_unlink_is_never_reported_as_removed(self, tmp_path, monkeypatch):
        """Otherwise the acquirer loops on it and never reaches its deadline.

        ``_BREAK_REMOVED`` is the one answer that sends the acquirer back
        into the create with no sleep and no timeout check. Reporting it on
        the strength of having *asked* for an unlink that was refused is a
        spin, not a retry — measured as a worker still burning 98% of a core
        fourteen minutes into a thirty-second ``timeout``.
        """
        lock, lock_path, identity = _plant_a_crashed_holder(tmp_path)
        monkeypatch.setattr(FileLock, "_try_os_lock", lambda self, fd: None)

        real_unlink = os.unlink

        def refusing_unlink(path, *args, **kwargs):
            if str(path) == lock_path:
                raise PermissionError(32, "used by another process")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", refusing_unlink)

        outcome = lock._break_stale(identity)
        # Positive control: the refusal has to have taken effect, or this is
        # a test about an unlink that simply succeeded.
        assert os.path.exists(lock_path), "the refusing unlink did not refuse"
        assert outcome == _BREAK_NOTHING, f"a refused unlink was reported as {outcome!r} — the acquirer loops on that answer"

        # End to end: an acquirer meeting an unbreakable corpse gives up at
        # its own deadline instead of spinning on it forever.
        waiter = FileLock(lock.path, timeout=0.3)
        started = time.monotonic()
        with pytest.raises(LockTimeout):
            waiter.acquire()
        assert time.monotonic() - started < 30.0, "the acquirer did not honour its own timeout"
        monkeypatch.undo()
        os.unlink(lock_path)


def _replace_with_a_fresh_claim(lock_path: str) -> tuple[int, int]:
    """Stand in for a successor: a different lockfile at the same path."""
    os.unlink(lock_path)
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        st = os.fstat(fd)
    finally:
        os.close(fd)
    return (st.st_dev, st.st_ino)


def _identity_of(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return (st.st_dev, st.st_ino)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot unlink a lockfile that is still open")
class TestReleaseOnlyRemovesItsOwnClaim:
    """Releasing is the other place a lockfile gets deleted.

    ``release`` unlinking by path is safe in the ordinary case only
    because a live owner's lockfile cannot legitimately be replaced. That
    is an argument about the rest of the system, not a property of
    ``release`` — so the identity check stays, and this is what it buys.
    """

    def test_a_successors_claim_survives_our_release(self, tmp_path):
        target = tmp_path / "released.dat"
        target.write_text("", encoding="utf-8")

        # Positive control: release DOES remove its own lockfile, so the
        # survival below is the check working and not release doing nothing.
        ordinary = FileLock(str(target), timeout=1.0)
        ordinary.acquire()
        assert os.path.exists(ordinary.lock_path)
        ordinary.release()
        assert not os.path.exists(ordinary.lock_path)

        lock = FileLock(str(target), timeout=1.0)
        lock.acquire()
        successor = _replace_with_a_fresh_claim(lock.lock_path)
        lock.release()

        assert os.path.exists(lock.lock_path), "release deleted a lockfile it did not create"
        assert _identity_of(lock.lock_path) == successor, "release replaced the successor's claim"
        os.unlink(lock.lock_path)

    def test_unlinking_by_path_deletes_it(self, tmp_path, monkeypatch):
        """The twin: the same body with the identity check taken out."""
        target = tmp_path / "released.dat"
        target.write_text("", encoding="utf-8")

        def _unlink_by_path(self, identity):
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass

        monkeypatch.setattr(FileLock, "_unlink_if_ours", _unlink_by_path)

        lock = FileLock(str(target), timeout=1.0)
        lock.acquire()
        _replace_with_a_fresh_claim(lock.lock_path)
        lock.release()

        assert not os.path.exists(lock.lock_path), "unlinking by path left the successor's claim alone — this twin proves nothing"
