# Copyright 2026 STARGA, Inc.
"""The restore invariant: no snapshot lands over the corpus without an admission.

``write_block`` and ``delete_block`` refuse an ungated caller at the store
seam. ``restore`` did not. It is the third mutation on a ``BlockStore`` and
the most destructive of the three — it withdraws every block written since
the snapshot and reinstates the versions under it — and the only thing that
made it recorded was that every caller happened to go through
``apply_engine.restore_snapshot``.

Measured on 5.0.2 before this file, with the write and delete gates already
closed — a governed write, a snapshot, a second governed write, then
``store.restore(snap)`` called directly with no scope. The reproduction is
:meth:`TestMutationTwin.test_removing_the_seam_check_puts_the_defect_back`
below, which neutralises the check and re-measures it on every run rather
than quoting a number nobody can re-derive::

    restore returned normally
    block D-002 readable before/after: True False
    (evidence, hash_chain) before/after: (4, 4) (4, 4)

A governed block died and neither ledger moved. Positive control in the same
run at the same seam::

    ungated delete_block -> raised UngatedDeleteError
    ungated write_block  -> raised UngatedWriteError

Two halves, and both are needed:

**The seam.** Every ``BlockStore.restore`` implementation calls
:func:`~mind_mem.admission.require_restore_admission` first, so an ungated
restore raises before it reads the snapshot. That is the half a monkeypatch
cannot satisfy at runtime, and the half that makes the rule hold for a
backend nobody has written yet.

**The opener.** ``apply_engine.restore_snapshot`` is the only function in
``src/`` that opens a ``.restore(`` call. The receipt carries no field naming
a snapshot, so the seam cannot check that the open scope recorded *this*
manifest digest; pinning the opener is what closes that gap, because the one
opener passes the same ``snap_dir`` to the store that it hashed into its
record. The scan below is what keeps a second opener from appearing quietly.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Iterator

import _write_path_scan
import pytest
from _write_path_scan import (
    RESTORE_DELEGATES,
    RESTORE_ENFORCEMENT_EXEMPT,
    RESTORE_SEAM_OPENER,
    calls_require_restore_admission,
    iter_source_files,
    scan_restore_calls,
    scan_restore_defs,
)

from mind_mem.admission import (
    BATCH,
    RESTORE_TIER,
    UngatedDeleteError,
    UngatedRestoreError,
    UngatedWriteError,
    current_admission,
)
from mind_mem.apply_engine import RESTORE_VERB, create_snapshot, restore_snapshot
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.storage import get_block_store

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory", "summaries")
EVIDENCE_REL = "memory/evidence_chain.jsonl"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """The zero-config default: blocks of record on the Markdown corpus."""
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    (ws / "memory" / "intel-state.json").write_text("{}\n", encoding="utf-8")
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


def write_governed_block(ws: str, block_id: str, statement: str = "body") -> None:
    """Land one block the honest way: a scope, then the store."""
    gate = get_gate(ws)
    store = get_block_store(ws)
    with gate.admit_block("WRITE", block_id, statement, tier=IngestTier.EXTERNAL_INGEST):
        store.write_block({"_id": block_id, "Statement": statement, "Status": "quarantined", "Date": "2026-09-02"})


def block_is_readable(ws: str, block_id: str) -> bool:
    return get_block_store(ws).get_by_id(block_id) is not None


def evidence_rows(ws: str) -> list[dict]:
    path = os.path.join(ws, EVIDENCE_REL)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def restore_rows(ws: str) -> list[dict]:
    return [r for r in evidence_rows(ws) if (r.get("metadata") or {}).get("action_verb") == RESTORE_VERB]


@pytest.fixture
def snapshot_over_a_live_block(workspace: str) -> tuple[str, str]:
    """``(snap_dir, ws)`` where the workspace holds a block the snapshot does not.

    The state every test below needs: restoring this snapshot WOULD withdraw
    ``D-20260902-002``, so "the block is still readable" is a measurement of
    the refusal and not of a restore that had nothing to do.
    """
    write_governed_block(workspace, "D-20260902-001")
    snap_dir = create_snapshot(workspace, "20260902-190000", files_touched=None)
    write_governed_block(workspace, "D-20260902-002")
    assert block_is_readable(workspace, "D-20260902-002"), "fixture did not land the withdrawable block"
    return snap_dir, workspace


# ---------------------------------------------------------------------------
# Scan E1 — the only opener
# ---------------------------------------------------------------------------


class TestTheOpenerIsPinned:
    def test_every_restore_call_in_src_is_the_opener_or_a_delegate(self) -> None:
        """A second door into a restore fails the build.

        The seam refuses a caller with no scope. It cannot refuse a caller
        that opens the *wrong* scope for the *right* shape — a batch RESTAMP
        receipt naming snapshot A while the store is handed snapshot B — so
        the set of functions allowed to make the call is enumerated here
        instead.
        """
        hits = scan_restore_calls(iter_source_files())
        # Positive control: the scan found the call it is named for. An empty
        # scan satisfies every allowlist ever written.
        callers = {(rel, qualname) for rel, qualname, _line in hits}
        assert RESTORE_SEAM_OPENER in callers, f"the scanner did not find the sanctioned opener at all: {sorted(callers)}"

        rogue = sorted(callers - {RESTORE_SEAM_OPENER} - RESTORE_DELEGATES)
        assert rogue == [], (
            "these functions call a store restore and are neither the sanctioned opener nor a "
            f"store forwarding to the store underneath it: {rogue}. A restore must run inside "
            f"{RESTORE_SEAM_OPENER[1]}, which records the manifest digest, the reinstated ids and "
            "the withdrawn ids."
        )

    def test_each_delegate_is_a_store_forwarding_to_a_store(self) -> None:
        """The allowlist is three wrappers, not three exceptions.

        Every entry in :data:`RESTORE_DELEGATES` is a ``restore`` method whose
        body forwards to another store's ``restore``. Asserting the shape
        rather than the names means an entry cannot be reused later for a
        function that merely wants to skip the opener rule.
        """
        hits = scan_restore_calls(iter_source_files())
        callers = {(rel, qualname) for rel, qualname, _line in hits}
        defined = {(rel, qualname) for rel, qualname, _line, _enforces in scan_restore_defs(iter_source_files())}
        for delegate in sorted(RESTORE_DELEGATES):
            assert delegate in callers, f"{delegate} is allowlisted as a delegate but makes no restore call"
            assert delegate in defined, f"{delegate} is allowlisted as a delegate but is not itself a `def restore`"

    def test_the_opener_actually_opens_a_scope(self) -> None:
        """Named in the allowlist is not the same as gated.

        ``apply_engine.restore_snapshot`` earns its place by opening
        ``admit_batch`` unconditionally. Read from the source, so a caller
        that keeps the name and loses the scope fails here.
        """
        path = os.path.join(_write_path_scan.SRC_ROOT, "apply_engine.py")
        func = _write_path_scan.function_node(_write_path_scan.parse(path), "restore_snapshot")
        assert func is not None, "apply_engine.restore_snapshot is gone; the allowlist above names nothing"
        assert _write_path_scan.opens_admission(func), "the sanctioned opener no longer opens an admission scope"


# ---------------------------------------------------------------------------
# Scan E2 — every implementation enforces
# ---------------------------------------------------------------------------


class TestEveryStoreRestoreEnforces:
    def test_every_restore_implementation_calls_require_restore_admission(self) -> None:
        defs = scan_restore_defs(iter_source_files())
        # Positive control: the scan found implementations at all, and found
        # at least one that enforces.
        assert defs, "the scanner found no `def restore` in src/ — every assertion below is vacuous"
        assert any(enforces for *_rest, enforces in defs), "the scanner cannot see enforcement anywhere; its matcher is broken"

        unenforced = sorted((rel, qualname) for rel, qualname, _line, enforces in defs if not enforces)
        assert unenforced == sorted(RESTORE_ENFORCEMENT_EXEMPT), (
            "these `def restore` implementations do not call require_restore_admission, and are "
            f"not the Protocol stub or a forwarding wrapper: {unenforced}"
        )

    def test_the_three_backends_that_touch_storage_all_enforce(self) -> None:
        """Named, not merely counted.

        The set-difference assertion above passes if an implementation is
        added to the exempt list. These three are the ones that actually
        move bytes, so they are checked by name.
        """
        enforcing = {(rel, qualname) for rel, qualname, _line, enforces in scan_restore_defs(iter_source_files()) if enforces}
        for backend in (
            ("src/mind_mem/block_store.py", "MarkdownBlockStore.restore"),
            ("src/mind_mem/block_store_postgres.py", "PostgresBlockStore.restore"),
            ("src/mind_mem/block_store_encrypted.py", "EncryptedBlockStore.restore"),
        ):
            assert backend in enforcing, f"{backend} does not call require_restore_admission"

    def test_the_enforcement_matcher_can_report_a_missing_call(self, tmp_path: Path) -> None:
        """Negative control on the scanner itself.

        ``assert every implementation enforces`` passes just as well against
        a matcher that answers True for anything. Run it over synthetic
        source that does not make the call and watch it say so.
        """
        rogue = tmp_path / "rogue.py"
        rogue.write_text(
            "class RogueStore:\n    def restore(self, snap_dir):\n        shutil.copytree(snap_dir, self.ws)\n",
            encoding="utf-8",
        )
        tree = _write_path_scan.parse(str(rogue))
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "restore")
        assert calls_require_restore_admission(func) is False

        gated = tmp_path / "gated.py"
        gated.write_text(
            "class GatedStore:\n    def restore(self, snap_dir):\n        require_restore_admission(snap_dir)\n",
            encoding="utf-8",
        )
        tree = _write_path_scan.parse(str(gated))
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "restore")
        assert calls_require_restore_admission(func) is True

    def test_an_import_alone_does_not_count_as_enforcement(self, tmp_path: Path) -> None:
        """ "Imported" is not "wired" — the matcher reads calls, not names."""
        imported = tmp_path / "imported.py"
        imported.write_text(
            "from .admission import require_restore_admission\n\n\n"
            "class S:\n"
            "    def restore(self, snap_dir):\n"
            "        self._inner(snap_dir)\n",
            encoding="utf-8",
        )
        tree = _write_path_scan.parse(str(imported))
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "restore")
        assert calls_require_restore_admission(func) is False


# ---------------------------------------------------------------------------
# The seam at runtime — the reproduced defect, and its positive control
# ---------------------------------------------------------------------------


class TestTheSeamRefusesAnUngatedRestore:
    def test_an_ungated_restore_raises_and_the_block_survives(self, snapshot_over_a_live_block: tuple[str, str]) -> None:
        """R3-02 head-on. Before the seam check this returned normally."""
        snap_dir, ws = snapshot_over_a_live_block
        before = len(evidence_rows(ws))

        with pytest.raises(UngatedRestoreError) as excinfo:
            get_block_store(ws).restore(snap_dir)

        assert "no governance admission is open" in str(excinfo.value)
        assert block_is_readable(ws, "D-20260902-002"), "the refused restore withdrew the block anyway"
        assert len(evidence_rows(ws)) == before, "the refused restore wrote to the ledger"

    def test_the_scoped_restore_still_lands_and_records_itself(self, snapshot_over_a_live_block: tuple[str, str]) -> None:
        """Positive control for every refusal in this class.

        A gate that refuses everything is not a gate. The sanctioned door
        goes through, the post-snapshot block is withdrawn, and exactly one
        RESTORE row names it.
        """
        snap_dir, ws = snapshot_over_a_live_block
        assert restore_rows(ws) == []

        restore_snapshot(ws, snap_dir)

        assert not block_is_readable(ws, "D-20260902-002"), "the sanctioned restore did not withdraw the block"
        assert block_is_readable(ws, "D-20260902-001"), "the sanctioned restore did not reinstate the snapshot"
        rows = restore_rows(ws)
        assert len(rows) == 1, f"expected exactly one RESTORE row, got {len(rows)}"
        assert "D-20260902-002" in rows[0]["metadata"]["withdrawn_block_ids"]

    def test_the_refusal_comes_before_the_snapshot_is_read(self, workspace: str) -> None:
        """Authorisation first, existence second.

        A caller with no scope and a caller naming a snapshot that is not
        there must fail the same way, or the seam is an oracle for what the
        workspace holds. Same ordering rule ``delete_block`` follows.
        """
        missing = os.path.join(workspace, "intelligence", "applied", "does-not-exist")
        assert not os.path.exists(missing)

        with pytest.raises(UngatedRestoreError):
            get_block_store(workspace).restore(missing)

    def test_the_store_log_names_the_authorising_chain_entry(
        self,
        snapshot_over_a_live_block: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The store's own record joins to the chain entry that allowed it."""
        from mind_mem import block_store as bs

        snap_dir, ws = snapshot_over_a_live_block
        events: list[dict] = []
        monkeypatch.setattr(bs._log, "info", lambda event, **kw: events.append({"event": event, **kw}))

        restore_snapshot(ws, snap_dir)

        (summary,) = [e for e in events if e["event"] == "block_store_restore"]
        assert summary["admission"], "the restore log carries no admission id"
        chain_ids = {r.get("metadata", {}).get("admission_entry_id") for r in evidence_rows(ws)}
        chain_ids |= {r.get("entry_id") for r in evidence_rows(ws)}
        assert summary["admission"] in {c for c in chain_ids if c}, (
            f"the store logged admission {summary['admission']!r}, which no evidence row names"
        )


# ---------------------------------------------------------------------------
# A receipt is not transferable INTO a restore
# ---------------------------------------------------------------------------


class TestAReceiptIsNotTransferableIntoARestore:
    def test_a_delete_receipt_does_not_authorise_a_restore(self, snapshot_over_a_live_block: tuple[str, str]) -> None:
        snap_dir, ws = snapshot_over_a_live_block
        gate = get_gate(ws)
        with gate.admit_delete("D-20260902-002", rationale="a delete scope is not a restore scope"):
            # Positive control: the scope really is open and really does
            # authorise its own operation at this same seam.
            assert current_admission() is not None
            with pytest.raises(UngatedRestoreError, match="not a restore"):
                get_block_store(ws).restore(snap_dir)
        assert block_is_readable(ws, "D-20260902-002")

    def test_a_single_block_write_receipt_does_not_authorise_a_restore(self, snapshot_over_a_live_block: tuple[str, str]) -> None:
        snap_dir, ws = snapshot_over_a_live_block
        gate = get_gate(ws)
        with gate.admit_block("WRITE", "D-20260902-003", "body", tier=RESTORE_TIER):
            with pytest.raises(UngatedRestoreError, match="scoped and cannot authorise"):
                get_block_store(ws).restore(snap_dir)
        assert block_is_readable(ws, "D-20260902-002")

    def test_a_proposal_receipt_does_not_authorise_a_restore(self, snapshot_over_a_live_block: tuple[str, str]) -> None:
        """The load-bearing refusal.

        ``apply_engine`` rolls back from inside an open ``admit_proposal``,
        whose receipt authorises every id it is asked about. Without this
        arm that ambient authority would carry a bare ``store.restore()``
        on the one path where a restore is most likely to be reached.
        """
        snap_dir, ws = snapshot_over_a_live_block
        gate = get_gate(ws)
        with gate.admit_proposal("P-20260902-001", "[]", actor="apply_engine") as receipt:
            # Positive control: this receipt genuinely is ambient — it says
            # yes to an id it never named.
            assert receipt.authorizes("an-id-nobody-mentioned")
            with pytest.raises(UngatedRestoreError, match="ambient authority"):
                get_block_store(ws).restore(snap_dir)
        assert block_is_readable(ws, "D-20260902-002")

    def test_a_batch_receipt_on_another_tier_does_not_authorise_a_restore(self, snapshot_over_a_live_block: tuple[str, str]) -> None:
        """A bulk-ingest batch is not a licence to reinstate a snapshot."""
        snap_dir, ws = snapshot_over_a_live_block
        gate = get_gate(ws)
        with gate.admit_batch(
            action="INGEST",
            batch_id="B-20260902-001",
            block_ids=["D-20260902-001", "D-20260902-002"],
            content="[]",
            tier=IngestTier.EXTERNAL_INGEST,
        ):
            with pytest.raises(UngatedRestoreError, match="ingest tier"):
                get_block_store(ws).restore(snap_dir)
        assert block_is_readable(ws, "D-20260902-002")

    def test_the_restore_scope_shape_is_the_one_the_opener_mints(self, snapshot_over_a_live_block: tuple[str, str]) -> None:
        """Positive control for the four refusals above.

        Each of them asserts a receipt is NOT accepted. That is only
        evidence if some receipt IS — and the one that is has to be the
        shape ``apply_engine.restore_snapshot`` actually opens, or the seam
        is enforcing a rule no sanctioned caller satisfies.
        """
        snap_dir, ws = snapshot_over_a_live_block
        gate = get_gate(ws)
        with gate.admit_batch(
            action=RESTORE_VERB,
            batch_id="restore:20260902-190000",
            block_ids=["D-20260902-001", "D-20260902-002"],
            content="[]",
            tier=RESTORE_TIER,
        ) as receipt:
            assert receipt.kind == BATCH
            assert receipt.tier is RESTORE_TIER
            get_block_store(ws).restore(snap_dir)

        assert not block_is_readable(ws, "D-20260902-002"), "the correctly scoped restore did not run"

    def test_a_restore_receipt_does_not_authorise_a_write_or_a_delete(self, workspace: str) -> None:
        """The rule runs both ways, or it is only half a rule.

        A RESTORE scope is a batch on the write side, so it authorises
        writes to the ids it names — that is what makes a restore a restore.
        What it must not do is reach an id it never named.
        """
        write_governed_block(workspace, "D-20260902-001")
        gate = get_gate(workspace)
        store = get_block_store(workspace)
        with gate.admit_batch(
            action=RESTORE_VERB,
            batch_id="restore:20260902-191000",
            block_ids=["D-20260902-001"],
            content="[]",
            tier=RESTORE_TIER,
        ):
            with pytest.raises(UngatedWriteError):
                store.write_block({"_id": "D-20260902-099", "Statement": "unnamed", "Status": "quarantined"})
            with pytest.raises(UngatedDeleteError):
                store.delete_block("D-20260902-001")
        assert block_is_readable(workspace, "D-20260902-001")


# ---------------------------------------------------------------------------
# Mutation twin — a gate never observed failing is not a gate
# ---------------------------------------------------------------------------


class TestMutationTwin:
    def test_removing_the_seam_check_puts_the_defect_back(
        self,
        snapshot_over_a_live_block: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Neutralise ``require_restore_admission``; the 5.0.2 behaviour returns.

        This is the whole file's control. Every refusal above is evidence
        only if the refusals come from the check being asserted — so break
        exactly that call and watch the measured defect reappear: the
        ungated restore returns normally, the governed block is gone, and
        the evidence chain never moved.
        """
        from mind_mem import block_store as bs

        snap_dir, ws = snapshot_over_a_live_block
        before = len(evidence_rows(ws))

        class _NoReceipt:
            entry_id = "mutation-twin"

        monkeypatch.setattr(bs, "require_restore_admission", lambda snap: _NoReceipt())

        get_block_store(ws).restore(snap_dir)

        assert not block_is_readable(ws, "D-20260902-002"), (
            "with the seam check neutralised the ungated restore did NOT run — this file is not measuring that check"
        )
        assert len(evidence_rows(ws)) == before, "an ungated restore wrote a ledger row"
        assert restore_rows(ws) == []
