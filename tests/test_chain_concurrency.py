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
from dataclasses import dataclass

import pytest

from mind_mem.audit_chain import AuditChain
from mind_mem.evidence_objects import (
    _GENESIS_HASH,
    EvidenceAction,
    EvidenceChain,
    EvidenceChainCompromisedError,
)

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

    mfl.FileLock._unlink_if_ours = _v501_unlink
    mfl.FileLock._stale_identity = _v501_stale

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


class TestStoreLockMutationTwin:
    """The same probe against the v5.0.1 lock, which must come out broken."""

    def test_the_v501_lock_lets_two_processes_in(self, tmp_path):
        # A race needs sampling: the window is real but narrow, so the
        # control is allowed to look more than once before concluding it
        # cannot see the defect. It is never allowed to pass without
        # observing one.
        seen = 0
        for attempt in range(3):
            violations, codes, reported = _run_lock_probe(tmp_path, defeat=True, tag=f"defeat{attempt}")
            assert codes == (0,) * _LOCK_PROCS, f"a control worker failed: {codes}"
            assert reported == _LOCK_PROCS, f"the control did not run: {reported}/{_LOCK_PROCS}"
            seen += violations
            if seen:
                break
        assert seen > 0, "the v5.0.1 lock excluded correctly — this probe cannot see the defect it is here to catch"
