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
_WORKER_SOURCE = '''\
import os, sys, time

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


def barrier(r):
    open(os.path.join(barrier_dir, "r%d-%s" % (r, tag)), "w").close()
    prefix = "r%d-" % r
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if sum(1 for f in os.listdir(barrier_dir) if f.startswith(prefix)) >= nprocs:
            return
        time.sleep(0.002)
    raise SystemExit("barrier timeout at round %d" % r)


if mode == "evidence":
    import mind_mem.evidence_objects as eo

    if defeat:
        eo.FileLock = _NullLock
        eo.EvidenceChain._linkable_previous_hash = lambda self: (
            self._entries[-1].evidence_hash if self._entries else eo._GENESIS_HASH
        )
    chain = eo.EvidenceChain(store_path=store)
    for r in range(rounds):
        barrier(r)
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
    chain = ac.AuditChain(store)
    for r in range(rounds):
        barrier(r)
        chain.append(
            "update_field",
            "decisions/DECISIONS.md",
            agent=tag,
            reason="round %d" % r,
            payload={"tag": tag, "round": r},
        )
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

    def is_one_intact_chain(self) -> bool:
        """Every row links, nothing restarted at genesis, and it verifies."""
        return self.nonlinking == 0 and self.genesis_rooted_after_first == 0 and self.verified and not self.broken

    def __str__(self) -> str:  # pragma: no cover - only rendered on failure
        return (
            f"rows={self.rows} nonlinking={self.nonlinking} "
            f"genesis_rooted_after_first={self.genesis_rooted_after_first} "
            f"writers_seen={self.writers_seen} verified={self.verified} "
            f"broken={len(self.broken)} exit_codes={self.exit_codes}\n"
            f"{self.stderr_tail}"
        )


def _spawn_writers(tmp_path, *, mode: str, target: str, procs: int, rounds: int, defeat: bool) -> tuple[tuple, str]:
    """Run *procs* real OS processes appending *rounds* records each."""
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
        )
        for i in range(procs)
    ]
    codes, errs = [], []
    for proc in running:
        _, err = proc.communicate(timeout=180)
        codes.append(proc.returncode)
        errs.append(err[-800:] if err else "")
    return tuple(codes), "\n".join(e for e in errs if e.strip())


def _read_evidence(store: str) -> list[dict]:
    if not os.path.isfile(store):
        return []
    with open(store, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _evidence_report(store: str, codes: tuple, stderr: str) -> ChainReport:
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
    )


def _audit_report(workspace: str, codes: tuple, stderr: str) -> ChainReport:
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
    )


_PROCS = 3
_ROUNDS = 20


def _run_evidence(tmp_path, *, defeat: bool) -> ChainReport:
    store = str(tmp_path / ("defeat" if defeat else "fixed") / "evidence_chain.jsonl")
    os.makedirs(os.path.dirname(store), exist_ok=True)
    codes, stderr = _spawn_writers(tmp_path, mode="evidence", target=store, procs=_PROCS, rounds=_ROUNDS, defeat=defeat)
    return _evidence_report(store, codes, stderr)


def _run_audit(tmp_path, *, defeat: bool) -> ChainReport:
    workspace = str(tmp_path / ("audit_defeat" if defeat else "audit_fixed"))
    os.makedirs(workspace, exist_ok=True)
    codes, stderr = _spawn_writers(tmp_path, mode="audit", target=workspace, procs=_PROCS, rounds=_ROUNDS, defeat=defeat)
    return _audit_report(workspace, codes, stderr)


# ---------------------------------------------------------------------------
# The gate: concurrent processes produce one intact chain
# ---------------------------------------------------------------------------


class TestConcurrentWritersCannotForkTheEvidenceChain:
    def test_every_row_links_and_the_chain_verifies(self, tmp_path):
        report = _run_evidence(tmp_path, defeat=False)

        # Positive control first: the run must actually have happened.
        # `nonlinking == 0` over an empty file is a pass that proves
        # nothing, and so is a run where every child died on import.
        assert report.exit_codes == (0,) * _PROCS, f"a writer failed:\n{report}"
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
        assert report.exit_codes == (0,) * _PROCS, f"a writer failed:\n{report}"
        store = str(tmp_path / "fixed" / "evidence_chain.jsonl")
        rooted = [row for row in _read_evidence(store) if row["previous_hash"] == _GENESIS_HASH]
        assert len(rooted) == 1, f"{len(rooted)} chains rooted at genesis in one file:\n{report}"


class TestAuditChainLockIsLoadBearing:
    """``audit_chain.append`` already locks — this proves the lock works."""

    def test_every_entry_links_and_the_ledger_verifies(self, tmp_path):
        report = _run_audit(tmp_path, defeat=False)

        assert report.exit_codes == (0,) * _PROCS, f"a writer failed:\n{report}"
        assert report.rows == _PROCS * _ROUNDS, f"writers did not all land their entries:\n{report}"
        assert report.writers_seen == _PROCS, f"entries came from {report.writers_seen} of {_PROCS} writers:\n{report}"

        assert report.is_one_intact_chain(), f"concurrent writers forked the ledger:\n{report}"
        assert report.broken == ()


# ---------------------------------------------------------------------------
# The controls: the same body, run against the code without the gate
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """The gate above is only worth its green if it can go red.

    Each twin runs the *same* writers and the *same* analysis with one
    thing taken away, and asserts the result is broken. If a twin ever
    passes as intact, the corresponding gate is measuring nothing.
    """

    def test_without_the_store_lock_the_evidence_chain_forks(self, tmp_path):
        report = _run_evidence(tmp_path, defeat=True)

        # Same positive control as the gate: the writers really ran.
        assert report.rows == _PROCS * _ROUNDS, f"the control did not write:\n{report}"
        assert report.writers_seen == _PROCS, f"the control did not run concurrently:\n{report}"

        assert not report.is_one_intact_chain(), (
            f"the pre-fix code path produced one intact chain — this test cannot detect the fork it exists to detect:\n{report}"
        )
        assert report.nonlinking > 0, f"no non-linking row in the unfixed run:\n{report}"
        assert report.genesis_rooted_after_first > 0, f"no second genesis root in the unfixed run:\n{report}"
        assert report.verified is False

    def test_without_its_lock_the_audit_ledger_forks(self, tmp_path):
        report = _run_audit(tmp_path, defeat=True)

        assert report.rows == _PROCS * _ROUNDS, f"the control did not write:\n{report}"
        assert report.writers_seen == _PROCS, f"the control did not run concurrently:\n{report}"

        assert not report.is_one_intact_chain(), (
            f"the audit ledger survived losing its lock — the concurrency assertion above is not measuring the lock:\n{report}"
        )
        assert report.verified is False


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
import os, sys, time
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
for i in range(iters):
    with mfl.FileLock(target, timeout=30.0):
        with open(holder, "w", encoding="utf-8") as fh:
            fh.write(tag)
        time.sleep(0.0005)
        with open(holder, encoding="utf-8") as fh:
            if fh.read() != tag:
                violations += 1
with open(report, "a", encoding="utf-8") as fh:
    fh.write("%s %d\\n" % (tag, violations))
'''

_LOCK_PROCS = 6
_LOCK_ITERS = 600


def _run_lock_probe(tmp_path, *, defeat: bool, tag: str) -> tuple[int, tuple, int]:
    """Return (violations, exit codes, workers that reported)."""
    worker = tmp_path / f"lock_probe_{tag}.py"
    worker.write_text(_LOCK_PROBE_SOURCE, encoding="utf-8")
    target = tmp_path / f"shared_{tag}.dat"
    target.write_text("", encoding="utf-8")
    report = tmp_path / f"lock_report_{tag}.txt"
    report.write_text("", encoding="utf-8")

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
        )
        for i in range(_LOCK_PROCS)
    ]
    codes = []
    for proc in running:
        proc.communicate(timeout=300)
        codes.append(proc.returncode)
    lines = [ln for ln in report.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sum(int(ln.split()[1]) for ln in lines), tuple(codes), len(lines)


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
        violations, codes, reported = _run_lock_probe(tmp_path, defeat=False, tag="fixed")

        # Positive control: zero violations must mean zero overlaps, not
        # zero work. Every worker has to have finished and reported.
        assert codes == (0,) * _LOCK_PROCS, f"a worker failed: {codes}"
        assert reported == _LOCK_PROCS, f"only {reported}/{_LOCK_PROCS} workers reported"

        assert violations == 0, f"two processes held the same lock {violations} times"


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


def _run_stale_race(tmp_path, *, tag: str, preamble: str = "", budget: float = 240.0) -> tuple[int, int, tuple, str, str]:
    """Return (violations, stale_breaks, exit codes, evidence, wedged).

    *budget* is a wall-clock bound on the whole race, and *wedged* is the
    reason it was exceeded (empty when it was not). Without it a lock that
    can never be broken is not a test failure — it is a hung session that
    pytest-timeout kills, taking every other test's failure text with it.
    That is how five Windows failures came back with no assertion text.
    """
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
        )
        for i in range(_RACE_PROCS)
    ]
    deadline = time.monotonic() + budget
    wedged = ""
    try:
        for rnd in range(_RACE_ROUNDS):
            prefix = f"ready-{rnd}-"
            while sum(1 for f in os.listdir(bdir) if f.startswith(prefix)) < _RACE_PROCS:
                if time.monotonic() > deadline:
                    # Either a worker died — the exit codes below will say
                    # so — or nobody can make progress at all.
                    wedged = f"the race made no progress past round {rnd} within {budget}s"
                    break
                time.sleep(0.001)
            if wedged:
                break
            # The crashed holder: a lockfile naming an owner that is gone.
            (target.parent / (target.name + ".lock")).write_text(_IMPOSSIBLE_PID + "\n", encoding="utf-8")
            (bdir / f"go-{rnd}").write_text("", encoding="utf-8")
    finally:
        codes = []
        for proc in running:
            try:
                proc.communicate(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:  # pragma: no cover - worker wedged
                proc.kill()
                proc.communicate()
                wedged = wedged or f"a worker had to be killed after {budget}s: it never exited"
            codes.append(proc.returncode)

    counts = tmp_path / f"overlaps_{tag}.txt.counts"
    lines = [ln.split() for ln in counts.read_text(encoding="utf-8").splitlines() if ln.strip()] if counts.exists() else []
    violations = sum(int(parts[1]) for parts in lines)
    breaks = sum(int(parts[2]) for parts in lines)
    return violations, breaks, tuple(codes), evidence.read_text(encoding="utf-8"), wedged


class TestBreakingACrashedHoldersLockHasOneWinner:
    def test_no_two_waiters_break_the_same_lock(self, tmp_path):
        violations, breaks, codes, evidence, wedged = _run_stale_race(tmp_path, tag="fixed")

        # Bounded time, not a timeout: a lock nobody can break has to come
        # back as a named failure here, not as a hung session somebody else
        # has to kill.
        assert not wedged, wedged

        # Positive control: the break path has to have been walked, or a
        # clean result says nothing. Every round plants a lockfile that
        # somebody must break before anyone can enter.
        assert codes == (0,) * _RACE_PROCS, f"a worker failed: {codes}"
        assert breaks >= _RACE_ROUNDS, f"only {breaks} stale breaks over {_RACE_ROUNDS} rounds — the path was not exercised"

        assert violations == 0, f"two processes held the same lock {violations} times:\n{evidence}"


#: The simulation needs ``/proc/self/fd`` to know which files this process
#: holds open. Naming the requirement is the point — a gate that quietly
#: does nothing on a platform is worse than one that says it is not running.
_HAS_PROCFS = os.path.isdir("/proc/self/fd")


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

    def test_the_break_never_unlinks_a_file_it_holds_open(self, tmp_path):
        violations, breaks, codes, evidence, wedged = _run_stale_race(
            tmp_path,
            tag="winsim",
            preamble=_WINDOWS_UNLINK_PREAMBLE,
        )

        assert not wedged, wedged
        # Positive control, twice over: the simulation proves itself active
        # at worker startup (or the worker exits non-zero), and the break
        # path has to have been walked on every round.
        assert codes == (0,) * _RACE_PROCS, f"a worker failed: {codes}"
        assert breaks >= _RACE_ROUNDS, f"only {breaks} stale breaks over {_RACE_ROUNDS} rounds — the path was not exercised"

        assert violations == 0, f"two processes held the same lock {violations} times:\n{evidence}"


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
