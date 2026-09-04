# Copyright 2026 STARGA, Inc.
"""H3 — chain truncation was invisible, and three ledgers never met.

REPRODUCED on a fresh workspace. Two governed writes (4 chain rows, 4
evidence rows), one served recall, then the two tail rows of
``memory/hash_chain_v2.db`` deleted::

    AFTER   chain=2 evidence=4
    [ok] hash_chain: 2 entries verified
    [ok] evidence_chain: 4 entries verified
    [ok] served_ledger: 2 rows verified
    verify ok: True  exit: 0

and after ``DELETE FROM hash_chain`` outright, ``[ok] hash_chain: 0
entries verified`` — still green. Two things were missing and both are
here:

* the last entry has no successor to bind it and the database held
  nothing outside itself to be compared against, so a contiguous tail
  removal left a chain that walked *perfectly*. Closed by a head sidecar
  (:func:`~mind_mem.hash_chain_v2.verify_head`), copied from the pattern
  ``served_ledger`` already shipped.
* nothing reconciled the three ledgers, even though the pairing keys
  already existed — a close record names its admission's ``entry_id``, a
  served row's ``index_anchor`` is derived from a chain entry's hash.
  Closed by :func:`~mind_mem.cross_ledger.reconcile`.

Every RED below is paired with the GREEN of the same check over the
untouched workspace, and each leg of the reconciliation is fired ALONE —
a three-legged check that only ever goes red with all three broken is one
leg with two decorations.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from mind_mem.cross_ledger import TOLERATED_SHORTFALL, reconcile
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.hash_chain_v2 import GENESIS_HASH, HashChainV2, head_path, read_head, verify_head, write_head
from mind_mem.init_workspace import init
from mind_mem.mm_cli import config_set
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.served_ledger import append_served_run
from mind_mem.storage import get_block_store
from mind_mem.verify_cli import (
    EXIT_CHAIN,
    EXIT_CROSS_LEDGER,
    EXIT_OK,
    VerifyReport,
    check_chain_head_seal,
    check_cross_ledger,
    verify_workspace,
)


@pytest.fixture
def ws(tmp_path) -> str:
    path = str(tmp_path / "ws")
    init(path)
    config_set(os.path.join(path, "mind-mem.json"), "served_ledger", {"enabled": True})
    yield path
    evict_gate(path)


def _governed_write(workspace: str, block_id: str) -> None:
    block = {
        "_id": block_id,
        "Date": "2026-09-03",
        "Status": "pending",
        "Scope": "global",
        "Statement": f"governed statement for {block_id}",
        "Rationale": "governed",
        "Supersedes": "none",
        "Tags": "governed",
        "Sources": "none",
    }
    store = get_block_store(workspace)
    with get_gate(workspace).admit_block(
        "WRITE",
        block_id,
        json.dumps(block, sort_keys=True),
        tier=IngestTier.AUTO_CAPTURE,
        actor="tester",
    ):
        store.write_block(block)


def _serve_one(workspace: str) -> None:
    """One served row anchored to the CHAIN HEAD the run actually observed.

    Not a hand-picked constant: the anchor leg is exactly the claim that a
    served row points at an entry the chain still holds, and a synthetic
    anchor would make the leg untestable in both directions.
    """
    from mind_mem.recall_attestation import _resolve_index_anchor

    ids = ("D-20260903-001",)
    row = append_served_run(
        workspace,
        query_hash=query_hash("why did the rollout land"),
        served_digest=served_set_digest(ids),
        ids=ids,
        pipeline_hash="b" * 64,
        index_anchor=_resolve_index_anchor(workspace),
        scoring_instant="2026-09-03",
    )
    assert row is not None, "positive control: the served ledger must be on or the anchor leg is vacuous"


def _chain_db(workspace: str) -> str:
    return os.path.join(workspace, "memory", "hash_chain_v2.db")


def _rows(workspace: str) -> int:
    conn = sqlite3.connect(_chain_db(workspace))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0])
    finally:
        conn.close()


def _drop_tail(workspace: str, n: int) -> None:
    conn = sqlite3.connect(_chain_db(workspace))
    try:
        conn.execute("DELETE FROM hash_chain WHERE rowid IN (SELECT rowid FROM hash_chain ORDER BY rowid DESC LIMIT ?)", (n,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def written(ws: str) -> str:
    """Two governed writes and one served run. The state the defect was found in."""
    _governed_write(ws, "D-20260903-001")
    _governed_write(ws, "D-20260903-002")
    _serve_one(ws)
    return ws


# ---------------------------------------------------------------------------
# 1. The reproduction, and its control
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_POSITIVE_CONTROL_an_untouched_workspace_reconciles(self, written: str) -> None:
        """Both new rows green, over non-zero counts.

        Without the counts this is the failure shape being fixed: "0
        entries verified" was green too.
        """
        report = verify_workspace(written)
        assert report.checks["chain_head_seal"] is True, report.messages
        assert report.checks["cross_ledger"] is True, report.messages
        details = report.details["cross_ledger"]
        assert details["chain_entries"] >= 4
        assert details["admission_rows"] >= 4
        assert details["served_rows"] == 1
        assert report.details["chain_head_seal"]["sealed"] is True

    def test_dropping_the_tail_is_convicted(self, written: str) -> None:
        before = _rows(written)
        _drop_tail(written, 2)
        assert _rows(written) == before - 2, "positive control: the truncation did not happen"

        report = verify_workspace(written)
        assert report.ok is False
        assert report.checks["hash_chain"] is True, "the link walk still passes — that is the defect"
        assert report.checks["chain_head_seal"] is False
        assert report.checks["cross_ledger"] is False
        assert report.exit_code == EXIT_CHAIN

    def test_emptying_the_chain_is_convicted_by_name(self, written: str) -> None:
        _drop_tail(written, _rows(written))
        assert _rows(written) == 0

        report = verify_workspace(written)
        assert report.ok is False
        assert report.checks["chain_head_seal"] is False
        assert "the rows were removed" in " ".join(report.messages)
        # The row the old verifier printed, still printed, still green — and
        # now no longer the whole story.
        assert report.checks["hash_chain"] is True


# ---------------------------------------------------------------------------
# 2. The head seal
# ---------------------------------------------------------------------------


class TestHeadSeal:
    def test_an_append_seals_the_head(self, tmp_path) -> None:
        db = str(tmp_path / "memory" / "chain.db")
        chain = HashChainV2(db)
        assert read_head(db) is None, "positive control: nothing sealed before the first append"
        entry = chain.append("D-1", "create", "hello")
        assert read_head(db) == entry.entry_hash
        assert verify_head(db).ok is True

    def test_an_import_reseals_the_head(self, tmp_path) -> None:
        """A legitimate write must not be reported as tampering.

        ``import_jsonl`` advances the head. A seal only ``append`` maintained
        would leave the sidecar naming the pre-import tail and convict a
        chain nobody touched.
        """
        source = str(tmp_path / "src" / "chain.db")
        src = HashChainV2(source)
        src.append("D-1", "create", "one")
        src.append("D-2", "create", "two")
        export = str(tmp_path / "export.jsonl")
        src.export_jsonl(export)

        target = str(tmp_path / "dst" / "chain.db")
        assert HashChainV2(target).import_jsonl(export) == 2
        verdict = verify_head(target)
        assert verdict.ok is True and verdict.sealed is True

    def test_a_removed_seal_is_reported_absent_not_passed(self, written: str) -> None:
        """Lenient pass, artifact in ``missing``, strict fail.

        Named rather than hidden: this is the same on-disk state as a chain
        written before 5.0.2, so failing it outright would turn every
        existing workspace red on upgrade. ``--strict`` is what a CI gate
        uses, and it does fail.
        """
        os.remove(head_path(_chain_db(written)))
        lenient = verify_workspace(written)
        assert lenient.checks["chain_head_seal"] is True
        assert lenient.details["chain_head_seal"]["sealed"] is False
        assert "memory/hash_chain_v2.head" in lenient.missing

        strict = verify_workspace(written, strict=True)
        assert strict.checks["chain_head_seal"] is False

    def test_a_blanked_seal_is_an_overwritten_one(self, written: str) -> None:
        """Absent and empty are different facts. Blank fails like any wrong value."""
        with open(head_path(_chain_db(written)), "w", encoding="utf-8") as handle:
            handle.write("")
        report = verify_workspace(written)
        assert report.checks["chain_head_seal"] is False

    def test_the_crash_window_is_admitted(self, tmp_path) -> None:
        """Row committed, seal not yet replaced. Refusing it is a one-way door."""
        db = str(tmp_path / "memory" / "chain.db")
        chain = HashChainV2(db)
        first = chain.append("D-1", "create", "one")
        chain.append("D-2", "create", "two")
        write_head(db, first.entry_hash)  # the seal as a SIGKILL would have left it
        verdict = verify_head(db)
        assert verdict.ok is True
        assert "lags the tail by one row" in verdict.reason

    def test_MUTATION_a_seal_ahead_of_the_chain_is_refused(self, tmp_path) -> None:
        """The pair for the crash window: the tolerance is one-directional.

        A seal naming an entry the chain does not hold is the state a tail
        removal leaves. If this passed, the tolerance above would swallow
        the defect it was written beside.
        """
        db = str(tmp_path / "memory" / "chain.db")
        chain = HashChainV2(db)
        chain.append("D-1", "create", "one")
        write_head(db, "f" * 128)
        assert verify_head(db).ok is False

    def test_an_empty_chain_with_a_seal_is_refused(self, tmp_path) -> None:
        """The one deletion the link walk cannot see: no gap, no broken link."""
        db = str(tmp_path / "memory" / "chain.db")
        chain = HashChainV2(db)
        chain.append("D-1", "create", "one")
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM hash_chain")
        conn.commit()
        conn.close()
        verdict = verify_head(db)
        assert verdict.ok is False
        assert verdict.head == GENESIS_HASH

    def test_a_seal_that_outlived_its_ledger_is_convicted(self, written: str) -> None:
        """Deleting the database must not be quieter than emptying the table.

        ``check_hash_chain`` reports an absent ledger *leniently* — a
        workspace may never have written. A surviving seal is positive
        evidence that one did, so this is fatal in both modes and is a
        different finding from "no ledger present".
        """
        db = _chain_db(written)
        assert read_head(db) is not None, "positive control: there was a seal to survive"
        os.remove(db)
        for sidecar in (f"{db}-wal", f"{db}-shm"):
            if os.path.isfile(sidecar):
                os.remove(sidecar)

        verdict = verify_head(db)
        assert verdict.ok is False
        assert "the chain was removed" in verdict.reason

        report = verify_workspace(written)  # lenient, and still red
        assert report.checks["chain_head_seal"] is False
        assert report.exit_code == EXIT_CHAIN

    def test_MUTATION_removing_the_seal_too_is_the_documented_residual(self, written: str) -> None:
        """The pair. Taking BOTH leaves a directory that never ran, and says so.

        Named rather than papered over: the seal and the ledger both live
        inside the workspace, so removing both is indistinguishable from a
        fresh one. Detecting it needs an anchor kept somewhere else, which
        this module does not own — the same residual
        ``verify_served_chain`` names for its own ledger.
        """
        db = _chain_db(written)
        os.remove(db)
        os.remove(head_path(db))
        for sidecar in (f"{db}-wal", f"{db}-shm"):
            if os.path.isfile(sidecar):
                os.remove(sidecar)
        verdict = verify_head(db)
        assert verdict.ok is True and verdict.sealed is False

    def test_a_fresh_workspace_seals_nothing(self, tmp_path) -> None:
        """No chain and no seal is a real state, and it is not a failure."""
        db = str(tmp_path / "memory" / "chain.db")
        verdict = verify_head(db)
        assert verdict.ok is True and verdict.sealed is False

    def test_verifying_creates_no_seal(self, tmp_path) -> None:
        ws_dir = str(tmp_path / "empty")
        os.makedirs(ws_dir)
        before = sorted(os.listdir(ws_dir))
        verify_workspace(ws_dir)
        assert sorted(os.listdir(ws_dir)) == before

    def test_the_check_owns_the_chain_exit_code(self, written: str) -> None:
        _drop_tail(written, 2)
        report = VerifyReport(workspace=written, ok=True)
        check_chain_head_seal(written, report, strict=False)
        assert report.checks["chain_head_seal"] is False
        assert report.exit_code == EXIT_CHAIN


# ---------------------------------------------------------------------------
# 3. Each reconciliation leg, fired alone
# ---------------------------------------------------------------------------


class TestEachLegFiresAlone:
    def test_leg_1_a_close_records_admission_must_resolve(self, written: str) -> None:
        """Remove ONE entry a close record points at — not the tail.

        Deleting from the middle also breaks the links, so ``hash_chain``
        goes red too; what this pins is that the reconciliation names the
        unresolved admission by id rather than only counting.
        """
        conn = sqlite3.connect(_chain_db(written))
        try:
            entry_id = conn.execute("SELECT entry_id FROM hash_chain ORDER BY rowid LIMIT 1").fetchone()[0]
            conn.execute("DELETE FROM hash_chain WHERE entry_id = ?", (entry_id,))
            conn.commit()
        finally:
            conn.close()

        verdict = reconcile(written)
        assert verdict.ok is False
        assert entry_id in verdict.unresolved_admissions

    def test_leg_2_the_chain_is_never_shorter_than_its_admissions(self, written: str) -> None:
        _drop_tail(written, 2)
        verdict = reconcile(written)
        assert verdict.shortfall == 2
        assert any("admission rows" in reason for reason in verdict.reasons)

    def test_leg_2_tolerates_the_documented_crash_window(self, written: str) -> None:
        """Evidence written, chain append died: a durable, permanent -1 gap.

        Convicting it would make one I/O error a state no workspace can
        leave. The one-entry truncation it costs is what the head seal
        catches — asserted here so the two are not both lenient.
        """
        _drop_tail(written, TOLERATED_SHORTFALL)
        verdict = reconcile(written)
        assert verdict.shortfall == TOLERATED_SHORTFALL
        assert not any("admission rows" in reason for reason in verdict.reasons)
        assert verify_head(_chain_db(written)).ok is False, "the seal must convict what the count leg tolerates"

    def test_leg_3_a_served_rows_anchor_must_resolve(self, written: str) -> None:
        """The served row is intact; the entry it anchored to is gone."""
        assert reconcile(written).unresolved_anchors == (), "positive control: the anchor resolved before the truncation"
        _drop_tail(written, 2)
        verdict = reconcile(written)
        assert verdict.unresolved_anchors == (0,)
        assert any("anchor to a chain entry that is gone" in reason for reason in verdict.reasons)

    def test_a_genesis_anchor_always_resolves(self, ws: str) -> None:
        """A run that observed an empty chain recorded the sentinel, not an entry."""
        from mind_mem.recall_attestation import GENESIS_ANCHOR

        ids = ("D-20260903-001",)
        append_served_run(
            ws,
            query_hash=query_hash("q"),
            served_digest=served_set_digest(ids),
            ids=ids,
            pipeline_hash="b" * 64,
            index_anchor=GENESIS_ANCHOR,
            scoring_instant="2026-09-03",
        )
        assert reconcile(ws).unresolved_anchors == ()


# ---------------------------------------------------------------------------
# 4. Nothing compared must not read as nothing wrong
# ---------------------------------------------------------------------------


class TestNothingComparedIsNotAPass:
    def test_an_empty_workspace_is_unchecked_not_verified(self, tmp_path) -> None:
        ws_dir = str(tmp_path / "empty")
        os.makedirs(ws_dir)
        verdict = reconcile(ws_dir)
        assert verdict.ok is True
        assert verdict.checked is False

        report = verify_workspace(ws_dir)
        assert report.checks["cross_ledger"] is True
        assert "memory/hash_chain_v2.db+evidence_chain.jsonl" in report.missing

        strict = verify_workspace(ws_dir, strict=True)
        assert strict.checks["cross_ledger"] is False

    def test_an_unreadable_chain_is_a_failure_not_an_empty_comparison(self, tmp_path) -> None:
        ws_dir = str(tmp_path / "broken")
        os.makedirs(os.path.join(ws_dir, "memory"))
        with open(os.path.join(ws_dir, "memory", "hash_chain_v2.db"), "w", encoding="utf-8") as handle:
            handle.write("this is not a database")
        verdict = reconcile(ws_dir)
        assert verdict.ok is False
        assert verdict.checked is True

    def test_the_check_owns_its_exit_code(self, written: str) -> None:
        _drop_tail(written, 2)
        report = VerifyReport(workspace=written, ok=True)
        check_cross_ledger(written, report, strict=False)
        assert report.checks["cross_ledger"] is False
        assert report.exit_code == EXIT_CROSS_LEDGER

    def test_MUTATION_the_same_check_is_green_before_the_truncation(self, written: str) -> None:
        report = VerifyReport(workspace=written, ok=True)
        check_cross_ledger(written, report, strict=True)
        assert report.checks["cross_ledger"] is True
        assert report.exit_code == EXIT_OK
