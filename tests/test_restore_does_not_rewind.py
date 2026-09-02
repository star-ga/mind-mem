# Copyright 2026 STARGA, Inc.
"""The audit chain could be rewound by a restore, and rollback left no record.

Two defects, one shape: the operations that undo the corpus treated the
*record about* the corpus as more corpus.

**GAP-4 — restore rewinds the ledgers.** ``SNAPSHOT_DIRS`` and
``BACKUP_DIRS`` both contain ``memory/``, which holds the append-only
ledgers as well as blocks, so a full apply-engine snapshot and a
``mind-mem-backup`` archive both captured ``memory/hash_chain_v2.db`` and
``memory/evidence_chain.jsonl`` — and put them back on restore. Measured on
5.0.1, fresh workspace, Markdown backend, both doors::

    before snapshot           evidence 1   hash_chain 1
    after one governed write  evidence 2   hash_chain 2   block visible
    AFTER restore             evidence 1   hash_chain 1   block GONE

Nothing anywhere recorded that the block had existed. "Hashes are never
rewritten" was true; "history is never removed" was not, and an audit chain
a restore can rewind is not tamper-evident.

**GAP-6 — rollback minted nothing.** ``apply_engine.rollback`` restored the
files, appended a ``RolledBack:`` line to a Markdown receipt the chain does
not cover, and opened no scope. ``_ACTION_MAP["ROLLBACK"]`` had zero callers
in ``src/``: the verb was classified and never spoken. So the chain went on
asserting that an apply which had just been undone still stood.

The fix is a registry, not a rule to remember.
:data:`~mind_mem.corpus_registry.LEDGER_FILES` says which paths are ledgers;
every snapshot walk filters on it, ``block_store._build_manifest`` *refuses*
a file list naming one (so a walk that forgets cannot produce a rewindable
artifact), the orphan sweep refuses to delete one, and both restore readers
refuse to overwrite one. Then the restore itself runs inside
``admit_batch`` — opened in ``restore_snapshot``, the single function every
apply-engine restore goes through, so a caller cannot restore without
recording it.

Nothing here rewrites an existing hash. Stopping the rewind is not the same
as repairing recorded history, and this change does neither.

Every negative assertion below carries a positive control: a ledger asserted
absent from a manifest is shown to exist on disk at capture time and the
manifest is shown to name the corpus file beside it, because ``assert x not
in manifest`` passes just as well against a walk that never ran.
:class:`TestMutationTwin` breaks each guard and shows the tests going red.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tarfile
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem import backup_restore
from mind_mem import block_store as block_store_mod
from mind_mem.apply_engine import (
    RESTORE_VERB,
    ROLLBACK_VERB,
    create_snapshot,
    restore_snapshot,
    rollback,
    write_receipt,
)
from mind_mem.backup_restore import LEDGER_ARCHIVE_PREFIX, backup_workspace, restore_workspace
from mind_mem.block_store import _build_manifest
from mind_mem.corpus_registry import (
    LEDGER_FILES,
    LEDGER_PATTERNS,
    LedgerCaptureError,
    is_ledger_path,
    iter_ledger_paths,
)
from mind_mem.enums import IngestTier
from mind_mem.evidence_objects import EvidenceAction
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.storage import get_block_store

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory", "summaries")

EVIDENCE_REL = "memory/evidence_chain.jsonl"
HASH_CHAIN_REL = "memory/hash_chain_v2.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """The zero-config default: blocks of record on the Markdown corpus."""
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    (ws / "tasks" / "TASKS.md").write_text("# Tasks\n\n", encoding="utf-8")
    # What ``mind-mem-init`` writes into memory/ (init_workspace.py:92-93).
    # It matters to this file specifically: the rollback orphan sweep only
    # walks a directory the manifest names, so a memory/ holding nothing but
    # the (now-excluded) ledgers would drop out of the sweep entirely and
    # ``test_a_ledger_is_never_swept_as_an_orphan`` would be vacuous. Every
    # shipped workspace has these two, so the fixture has them too.
    (ws / "memory" / "intel-state.json").write_text("{}\n", encoding="utf-8")
    (ws / "memory" / "maint-state.json").write_text("{}\n", encoding="utf-8")
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


def evidence_rows(ws: str) -> list[dict[str, Any]]:
    path = os.path.join(ws, EVIDENCE_REL)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def hash_chain_len(ws: str) -> int:
    path = os.path.join(ws, HASH_CHAIN_REL)
    if not os.path.isfile(path):
        return 0
    con = sqlite3.connect(path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0])
    finally:
        con.close()


def chain_lengths(ws: str) -> tuple[int, int]:
    return len(evidence_rows(ws)), hash_chain_len(ws)


def block_is_readable(ws: str, block_id: str) -> bool:
    return get_block_store(ws).get_by_id(block_id) is not None


def restore_rows(ws: str, verb: str) -> list[dict[str, Any]]:
    """Evidence rows whose raw verb is *verb* (not merely its coarse class)."""
    return [r for r in evidence_rows(ws) if (r.get("metadata") or {}).get("action_verb") == verb]


# ---------------------------------------------------------------------------
# GAP-4 — the measured defect, both doors
# ---------------------------------------------------------------------------


class TestTheChainIsMonotoneAcrossARestore:
    """Snapshot → write → restore must never leave the chain shorter."""

    def test_apply_engine_full_snapshot_restore_never_shortens_the_chain(self, workspace: str) -> None:
        write_governed_block(workspace, "D-20260902-001")
        before = chain_lengths(workspace)
        # Positive control: this workspace's chain really does move when a
        # block lands, so a later "it did not shrink" is a measurement and
        # not an artefact of a chain nothing writes to.
        assert before == (1, 1), f"expected one row per ledger after one governed write, got {before}"

        snap_dir = create_snapshot(workspace, "20260902-120000", files_touched=None)

        write_governed_block(workspace, "D-20260902-002")
        after_write = chain_lengths(workspace)
        assert after_write == (2, 2), f"the post-snapshot write did not reach the ledgers: {after_write}"
        assert block_is_readable(workspace, "D-20260902-002")

        restore_snapshot(workspace, snap_dir)

        after_restore = chain_lengths(workspace)
        assert after_restore[0] >= after_write[0], f"evidence chain went BACKWARDS across a restore: {after_write[0]} → {after_restore[0]}"
        assert after_restore[1] >= after_write[1], f"hash chain went BACKWARDS across a restore: {after_write[1]} → {after_restore[1]}"

    def test_backup_restore_never_shortens_the_chain(self, workspace: str, tmp_path: Path) -> None:
        write_governed_block(workspace, "D-20260902-101")
        before = chain_lengths(workspace)
        assert before == (1, 1), f"positive control: chain not moving on write, got {before}"

        archive = str(tmp_path / "backup.tar.gz")
        backup_workspace(workspace, archive)

        write_governed_block(workspace, "D-20260902-102")
        after_write = chain_lengths(workspace)
        assert after_write == (2, 2)

        restore_workspace(workspace, archive, force=True)

        after_restore = chain_lengths(workspace)
        assert after_restore[0] >= after_write[0], (
            f"evidence chain went BACKWARDS across a backup restore: {after_write[0]} → {after_restore[0]}"
        )
        assert after_restore[1] >= after_write[1], (
            f"hash chain went BACKWARDS across a backup restore: {after_write[1]} → {after_restore[1]}"
        )

    def test_a_block_the_restore_withdrew_is_named_in_the_record(self, workspace: str) -> None:
        """The post-snapshot block dies. Its death has to be legible.

        A restore is allowed to withdraw content — that is what it is for.
        What it may not do is withdraw content invisibly, which is exactly
        what the 5.0.1 behaviour did: the block written after the snapshot
        was gone and no row anywhere named it.
        """
        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-120000", files_touched=None)
        write_governed_block(workspace, "D-20260902-002")
        # Positive control: the block is genuinely in the store before the
        # restore, so "it was withdrawn" is a real event.
        assert block_is_readable(workspace, "D-20260902-002")

        restore_snapshot(workspace, snap_dir)

        assert not block_is_readable(workspace, "D-20260902-002"), "fixture no longer exercises a withdrawal"
        rows = restore_rows(workspace, RESTORE_VERB)
        assert len(rows) == 1, f"expected exactly one RESTORE row, got {len(rows)}"
        meta = rows[0]["metadata"]
        assert "D-20260902-002" in meta["withdrawn_block_ids"], (
            f"the restore withdrew D-20260902-002 and did not name it: {meta['withdrawn_block_ids']}"
        )
        assert "D-20260902-001" not in meta["withdrawn_block_ids"], "the reinstated block is not a withdrawal"
        assert meta["withdrawn_count"] == 1
        assert meta["reinstated_count"] == 1


# ---------------------------------------------------------------------------
# The ledgers are outside the snapshot BY CONSTRUCTION
# ---------------------------------------------------------------------------


class TestLedgersAreStructurallyOutsideTheSnapshot:
    def test_a_full_snapshot_manifest_names_no_ledger(self, workspace: str) -> None:
        write_governed_block(workspace, "D-20260902-001")
        # Positive control 1: the ledgers exist on disk at capture time, so
        # a walk that reached them COULD have captured them.
        assert os.path.isfile(os.path.join(workspace, EVIDENCE_REL))
        assert os.path.isfile(os.path.join(workspace, HASH_CHAIN_REL))

        snap_dir = create_snapshot(workspace, "20260902-130000", files_touched=None)
        manifest = json.loads(Path(snap_dir, "MANIFEST.json").read_text(encoding="utf-8"))["files"]

        # Positive control 2: the walk really ran and really saw memory/ —
        # otherwise "no ledger in the manifest" is true of an empty manifest.
        assert "decisions/DECISIONS.md" in manifest, f"the snapshot captured nothing to speak of: {manifest}"
        ledgers_in_manifest = [f for f in manifest if is_ledger_path(f)]
        assert ledgers_in_manifest == [], f"the manifest names ledgers: {ledgers_in_manifest}"

    def test_the_snapshot_directory_holds_no_ledger_copy(self, workspace: str) -> None:
        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-130001", files_touched=None)

        copied = [
            os.path.relpath(os.path.join(root, name), snap_dir).replace(os.sep, "/")
            for root, _dirs, names in os.walk(snap_dir)
            for name in names
        ]
        # Positive control: something was copied.
        assert "decisions/DECISIONS.md" in copied, f"the snapshot dir is empty: {copied}"
        assert [c for c in copied if is_ledger_path(c)] == []

    def test_build_manifest_refuses_a_file_list_naming_a_ledger(self, tmp_path: Path) -> None:
        """The choke point: a walk that forgets to filter still cannot ship.

        Every snapshot writer filters its own walk, and a filter is a thing
        somebody has to remember. This is the call all of them have to make,
        so forgetting fails loudly at capture time rather than quietly at the
        rollback that follows it.
        """
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir()
        # Positive control: the same call with corpus paths writes a manifest.
        _build_manifest(str(snap_dir), ["decisions/DECISIONS.md"])
        assert (snap_dir / "MANIFEST.json").is_file()

        with pytest.raises(LedgerCaptureError) as excinfo:
            _build_manifest(str(snap_dir), ["decisions/DECISIONS.md", HASH_CHAIN_REL])
        assert HASH_CHAIN_REL in str(excinfo.value)

    def test_a_ledger_is_never_swept_as_an_orphan(self, workspace: str) -> None:
        """The half that makes the other half safe.

        Excluding the ledgers from the manifest is what would otherwise make
        the rollback orphan sweep DELETE them: the sweep removes every file
        under a snapshotted directory the manifest does not name, and
        ``memory/`` is snapshotted. No chain at all is worse than a rewound
        one, so the exclusion has to be stated on both sides.
        """
        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-130002", files_touched=None)

        # A genuine orphan, created after the snapshot, in the same directory
        # as the ledgers. This is the positive control for the sweep itself:
        # if it does not disappear, the sweep never ran and the ledger
        # surviving proves nothing.
        orphan = Path(workspace, "memory", "ORPHAN.md")
        orphan.write_text("# orphan\n", encoding="utf-8")

        restore_snapshot(workspace, snap_dir)

        assert not orphan.exists(), "the orphan sweep did not run — the ledger assertion below would be vacuous"
        assert os.path.isfile(os.path.join(workspace, EVIDENCE_REL)), "the sweep deleted the evidence chain"
        assert os.path.isfile(os.path.join(workspace, HASH_CHAIN_REL)), "the sweep deleted the hash chain"


# ---------------------------------------------------------------------------
# Artifacts written BEFORE this change are the ones an operator reaches for
# ---------------------------------------------------------------------------


class TestPre502ArtifactsCannotRewindEither:
    def test_a_legacy_manifest_naming_a_ledger_is_refused_on_restore(self, workspace: str, tmp_path: Path) -> None:
        write_governed_block(workspace, "D-20260902-001")
        live_evidence = Path(workspace, EVIDENCE_REL).read_bytes()

        # Hand-build the artifact 5.0.1 produced: a manifest naming the
        # ledger, with a STALE copy of it beside a corpus file.
        snap_dir = tmp_path / "legacy-snap"
        (snap_dir / "memory").mkdir(parents=True)
        (snap_dir / "decisions").mkdir(parents=True)
        (snap_dir / "decisions" / "DECISIONS.md").write_text("# Decisions\n\nrestored\n", encoding="utf-8")
        (snap_dir / EVIDENCE_REL).write_text('{"stale": true}\n', encoding="utf-8")
        (snap_dir / "MANIFEST.json").write_text(
            json.dumps({"files": ["decisions/DECISIONS.md", EVIDENCE_REL], "version": 2}), encoding="utf-8"
        )

        get_block_store(workspace).restore(str(snap_dir))

        # Positive control: the non-ledger entry in the SAME manifest was
        # restored, so the restore genuinely ran over this manifest.
        assert "restored" in Path(workspace, "decisions", "DECISIONS.md").read_text(encoding="utf-8")
        after = Path(workspace, EVIDENCE_REL).read_bytes()
        assert after.startswith(live_evidence), "a legacy snapshot rewound the chain"
        assert b'"stale"' not in after, "the stale ledger copy landed in the live chain"

    def test_a_legacy_archive_naming_a_ledger_is_refused_on_restore(self, workspace: str, tmp_path: Path) -> None:
        write_governed_block(workspace, "D-20260902-001")
        live_evidence = Path(workspace, EVIDENCE_REL).read_bytes()

        stale_dir = tmp_path / "stale"
        (stale_dir / "memory").mkdir(parents=True)
        (stale_dir / "decisions").mkdir(parents=True)
        (stale_dir / "decisions" / "DECISIONS.md").write_text("# Decisions\n\nrestored\n", encoding="utf-8")
        (stale_dir / EVIDENCE_REL).write_text('{"stale": true}\n', encoding="utf-8")

        archive = tmp_path / "legacy.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stale_dir / "decisions" / "DECISIONS.md", arcname="decisions/DECISIONS.md")
            tar.add(stale_dir / EVIDENCE_REL, arcname=EVIDENCE_REL)

        result = restore_workspace(workspace, str(archive), force=True)

        assert "restored" in Path(workspace, "decisions", "DECISIONS.md").read_text(encoding="utf-8")
        assert result["refused_ledgers"] == 1, f"the ledger member was not refused: {result}"
        # Prefix, not equality: the restore records ITSELF in this chain, so
        # the file legitimately grew. What it may never do is shrink or be
        # replaced, and the stale marker is how that is checked.
        after = Path(workspace, EVIDENCE_REL).read_bytes()
        assert after.startswith(live_evidence), "a legacy archive rewound the chain"
        assert b'"stale"' not in after, "the stale ledger copy landed in the live chain"


# ---------------------------------------------------------------------------
# The backup still carries the chain — this closes a hole, it does not
# remove a capability
# ---------------------------------------------------------------------------


class TestTheArchiveStillCarriesTheChain:
    def test_ledgers_ride_under_the_quarantine_prefix_not_their_live_path(self, workspace: str, tmp_path: Path) -> None:
        """Relocated, not dropped.

        A backup is how an audit chain survives a lost machine, so excluding
        the ledgers outright would have paid for the rewind with a real
        capability. They travel under a prefix no restore extracts instead:
        the live path is absent from the archive, so no restore can write it.
        """
        write_governed_block(workspace, "D-20260902-001")
        live_evidence = Path(workspace, EVIDENCE_REL).read_bytes()

        archive = str(tmp_path / "backup.tar.gz")
        backup_workspace(workspace, archive)

        with tarfile.open(archive) as tar:
            names = tar.getnames()
            payload = tar.extractfile(f"{LEDGER_ARCHIVE_PREFIX}/{EVIDENCE_REL}")
            assert payload is not None
            with payload:
                archived_bytes = payload.read()

        # Positive control: the archive is a real backup of a real corpus.
        assert "decisions/DECISIONS.md" in names, f"the archive captured no corpus: {names}"
        assert EVIDENCE_REL not in names, "the live ledger path is still in the archive"
        assert HASH_CHAIN_REL not in names, "the live ledger path is still in the archive"
        assert f"{LEDGER_ARCHIVE_PREFIX}/{EVIDENCE_REL}" in names, "the chain was dropped, not relocated"
        assert archived_bytes == live_evidence, "the relocated copy is not the live chain"

    def test_restore_never_extracts_the_quarantine_prefix(self, workspace: str, tmp_path: Path) -> None:
        write_governed_block(workspace, "D-20260902-001")
        archive = str(tmp_path / "backup.tar.gz")
        backup_workspace(workspace, archive)

        result = restore_workspace(workspace, archive, force=True)

        assert result["ledger_archive"] >= 2, f"the prefix members were not seen at all: {result}"
        assert not Path(workspace, LEDGER_ARCHIVE_PREFIX).exists(), "the quarantine prefix was extracted into the workspace"

    def test_a_workspace_holding_only_ledgers_is_still_an_empty_backup(self, tmp_path: Path) -> None:
        """A relocated ledger must not satisfy the mistyped-path guard.

        ``backup_workspace`` refuses an archive that captured nothing,
        because an empty archive is otherwise discovered at restore time.
        Counting a ledger towards "something was captured" would turn that
        guard off for every real workspace.
        """
        ws = tmp_path / "ledger-only"
        (ws / ".mind-mem-ledger").mkdir(parents=True)
        (ws / ".mind-mem-ledger" / "served.jsonl").write_text('{"row": 1}\n', encoding="utf-8")
        # Positive control: there IS a ledger here for the walk to find, and
        # it is outside every BACKUP_DIR so the guard is measuring the ledger
        # rather than the presence of a ``memory/`` directory entry.
        assert iter_ledger_paths(str(ws)) == [".mind-mem-ledger/served.jsonl"]

        with pytest.raises(ValueError, match="captured nothing"):
            backup_workspace(str(ws), str(tmp_path / "empty.tar.gz"))


# ---------------------------------------------------------------------------
# GAP-6 — the restore and the rollback are recorded
# ---------------------------------------------------------------------------


class TestTheRestoreIsRecorded:
    def test_both_verbs_are_classified_onto_an_existing_evidence_action(self) -> None:
        """New facts, no new enum member: a 5.0.1 reader parses these rows.

        ``EvidenceAction`` is the wire vocabulary; adding a member would
        break older readers. Both verbs classify onto ``ROLLBACK`` — content
        withdrawn — and stay distinguishable through the raw verb the chain
        stores beside the class.
        """
        from mind_mem import governance_gate

        assert governance_gate._ACTION_MAP[RESTORE_VERB] is EvidenceAction.ROLLBACK
        assert governance_gate._ACTION_MAP[ROLLBACK_VERB] is EvidenceAction.ROLLBACK
        assert RESTORE_VERB != ROLLBACK_VERB
        # Positive control: the map is a genuine allow-list, not a dict that
        # answers for anything.
        assert "RESTOR" not in governance_gate._ACTION_MAP

    def test_restore_snapshot_writes_one_row_naming_the_manifest_digest(self, workspace: str) -> None:
        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-140000", files_touched=None)
        manifest_bytes = Path(snap_dir, "MANIFEST.json").read_bytes()

        before = len(restore_rows(workspace, RESTORE_VERB))
        assert before == 0, "a RESTORE row exists before any restore ran"

        restore_snapshot(workspace, snap_dir)

        rows = restore_rows(workspace, RESTORE_VERB)
        assert len(rows) == 1, f"expected exactly one RESTORE row, got {len(rows)}"
        meta = rows[0]["metadata"]
        assert rows[0]["action"] == EvidenceAction.ROLLBACK.value
        assert meta["door"] == "apply_engine.restore_snapshot"
        assert meta["snapshot"] == "20260902-140000"
        assert meta["manifest_source"] == "manifest"

        import hashlib

        assert meta["manifest_digest"] == hashlib.sha256(manifest_bytes).hexdigest(), (
            "the record does not name the manifest the restore actually read"
        )

    def test_rollback_writes_a_rollback_row_naming_the_proposal(self, workspace: str) -> None:
        """GAP-6 head-on: the verb that had no caller now has one.

        ``grep -rn 'action="ROLLBACK"' src/`` returned nothing before this
        change — ``_ACTION_MAP`` classified the verb and no door spoke it,
        so the chain kept asserting that an undone apply still stood.
        """
        write_governed_block(workspace, "D-20260902-001")
        receipt_ts = "20260902-150000"
        snap_dir = create_snapshot(workspace, receipt_ts, files_touched=None)
        write_receipt(snap_dir, {"ProposalId": "P-20260902-001", "Ops": []}, receipt_ts, ["seeded"])

        assert restore_rows(workspace, ROLLBACK_VERB) == []

        rollback(workspace, receipt_ts, reason="operator undo of a bad apply")

        rows = restore_rows(workspace, ROLLBACK_VERB)
        assert len(rows) == 1, f"expected exactly one ROLLBACK row, got {len(rows)}"
        meta = rows[0]["metadata"]
        assert meta["door"] == "apply_engine.rollback"
        assert meta["receipt_ts"] == receipt_ts
        assert meta["proposal_id"] == "P-20260902-001", f"the rollback did not name its proposal: {meta}"
        # A rollback is not a restore and the record has to say which.
        assert restore_rows(workspace, RESTORE_VERB) == []

    def test_backup_restore_writes_a_restore_row_naming_the_archive(self, workspace: str, tmp_path: Path) -> None:
        write_governed_block(workspace, "D-20260902-001")
        archive = tmp_path / "backup.tar.gz"
        backup_workspace(workspace, str(archive))

        import hashlib

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert restore_rows(workspace, RESTORE_VERB) == []

        restore_workspace(workspace, str(archive), force=True)

        rows = restore_rows(workspace, RESTORE_VERB)
        assert len(rows) == 1, f"expected exactly one RESTORE row, got {len(rows)}"
        meta = rows[0]["metadata"]
        assert meta["door"] == "backup_restore.restore_workspace"
        assert meta["archive"] == "backup.tar.gz"
        assert meta["archive_sha256"] == digest
        assert meta["force"] is True

    def test_a_restore_inside_an_open_proposal_scope_records_and_restores_the_outer_receipt(self, workspace: str) -> None:
        """The apply engine rolls back from INSIDE its own proposal scope.

        ``_apply_proposal_locked`` calls ``restore_snapshot`` on the op-failure
        and post-check-failure branches, both of which sit inside the open
        ``admit_proposal``. So the new scope nests. Two things have to hold or
        putting the record here breaks the apply path it is meant to cover:
        the nested scope must mint its own row, and the proposal receipt must
        be the current one again afterwards — otherwise the next
        ``write_block`` in the apply raises ``UngatedWriteError``.
        """
        from mind_mem.admission import current_admission

        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-180000", files_touched=None)

        gate = get_gate(workspace)
        store = get_block_store(workspace)
        with gate.admit_proposal("P-20260902-009", "[]", actor="apply_engine") as proposal_receipt:
            # Positive control: the proposal receipt is the open one, so the
            # "restored afterwards" assertion below is about a real change.
            assert current_admission() is proposal_receipt
            restore_snapshot(workspace, snap_dir)
            assert current_admission() is proposal_receipt, "the nested scope did not hand the proposal receipt back"
            # And the outer scope still authorises writes, which is the thing
            # a broken hand-back would actually break.
            store.write_block({"_id": "D-20260902-010", "Statement": "post-rollback", "Status": "active", "Date": "2026-09-02"})

        assert len(restore_rows(workspace, RESTORE_VERB)) == 1
        assert block_is_readable(workspace, "D-20260902-010")

    def test_a_restore_the_gate_refuses_does_not_happen(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail closed: an unrecordable restore is refused, not performed.

        The whole point of putting the scope in ``restore_snapshot`` is that
        recording is not optional. If it were best-effort, a workspace whose
        gate is broken would silently get the 5.0.1 behaviour back.
        """
        from mind_mem import apply_engine
        from mind_mem.admission import GovernanceBypassError

        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-160000", files_touched=None)
        write_governed_block(workspace, "D-20260902-002")
        assert block_is_readable(workspace, "D-20260902-002")

        class RefusingGate:
            def admit_batch(self, **_kwargs: Any) -> Any:
                raise GovernanceBypassError("refused for the test")

        monkeypatch.setattr("mind_mem.governance_gate.get_gate", lambda ws: RefusingGate())

        with pytest.raises(GovernanceBypassError):
            apply_engine.restore_snapshot(workspace, snap_dir)

        assert block_is_readable(workspace, "D-20260902-002"), "the restore ran despite the gate refusing it"


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------


class TestTheRegistryIsLoadBearing:
    def test_the_ledger_rows_name_the_files_the_gate_actually_writes(self, workspace: str) -> None:
        """Bind the registry to the authority, not to a remembered string.

        A registry naming ``memory/hash_chain.db`` would be silently inert:
        every assertion in this file would still pass and every snapshot
        would still capture the real ledger.
        """
        write_governed_block(workspace, "D-20260902-001")
        gate = get_gate(workspace)
        for path in (gate.chain._db_path, gate.evidence._store_path):
            rel = os.path.relpath(str(path), os.path.realpath(workspace)).replace(os.sep, "/")
            assert is_ledger_path(rel), f"the gate writes {rel}, which the registry does not call a ledger"

    def test_the_served_ledger_row_matches_its_module(self) -> None:
        from mind_mem.served_ledger import LEDGER_RELPATH

        assert is_ledger_path(LEDGER_RELPATH), f"served_ledger writes {LEDGER_RELPATH!r}; the registry does not recognise it"

    def test_the_registry_covers_the_sqlite_sidecars(self) -> None:
        """A stale ``.db`` beside a live ``-wal`` is corruption, not a rewind.

        Which reads as tampering to ``mm verify`` — a worse outcome than the
        defect this change removes, so the sidecars are in the table too.
        """
        assert is_ledger_path("memory/hash_chain_v2.db-wal")
        assert is_ledger_path("memory/hash_chain_v2.db-shm")
        assert is_ledger_path("memory/hash_chain_v2.db-journal")
        # Positive control: the pattern is not matching everything in memory/.
        assert not is_ledger_path("memory/INBOX.md")
        assert not is_ledger_path("memory/hash_chain_v2_notes.md")

    def test_no_corpus_file_is_classified_as_a_ledger(self) -> None:
        """The exclusion must not quietly drop blocks out of the snapshot."""
        from mind_mem.corpus_registry import CORPUS_RELPATHS

        misread = [rel for rel in CORPUS_RELPATHS if is_ledger_path(rel)]
        assert misread == [], f"these corpus files would be excluded from every snapshot: {misread}"
        # Positive control: the same predicate does fire on the ledgers.
        assert all(is_ledger_path(rel) for rel in LEDGER_FILES)
        assert LEDGER_PATTERNS, "the pattern table is empty; the sidecar rows above test nothing"


# ---------------------------------------------------------------------------
# Mutation twins — a gate never observed failing is not a gate
# ---------------------------------------------------------------------------


class TestMutationTwin:
    def test_the_rewind_test_depends_on_the_capture_filter(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blind the snapshot walk; the chain rewinds again.

        Both the walk filter and the manifest refusal are disabled, because
        the manifest refusal would otherwise raise instead of letting the
        old behaviour through — and a test that goes red for the wrong
        reason proves nothing about the right one.
        """
        monkeypatch.setattr(block_store_mod, "is_ledger_path", lambda rel: False)
        monkeypatch.setattr(block_store_mod, "assert_ledger_free", lambda paths, *, what: None)

        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-170000", files_touched=None)
        manifest = json.loads(Path(snap_dir, "MANIFEST.json").read_text(encoding="utf-8"))["files"]
        assert EVIDENCE_REL in manifest, "the mutation did not re-open the capture path"

        write_governed_block(workspace, "D-20260902-002")
        after_write = chain_lengths(workspace)
        restore_snapshot(workspace, snap_dir)
        after_restore = chain_lengths(workspace)

        assert after_restore[0] < after_write[0], (
            "with the filter disabled the evidence chain did NOT rewind — the guard under test "
            f"is not what is holding the invariant up ({after_write[0]} → {after_restore[0]})"
        )

    def test_the_orphan_test_depends_on_the_sweep_exemption(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Let the sweep treat a ledger as an orphan; it deletes the chain."""
        monkeypatch.setattr(
            block_store_mod,
            "_is_removable_orphan",
            lambda rel_posix, allowed: rel_posix not in allowed,
        )

        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-170001", files_touched=None)
        assert os.path.isfile(os.path.join(workspace, EVIDENCE_REL))

        restore_snapshot(workspace, snap_dir)

        assert not os.path.isfile(os.path.join(workspace, EVIDENCE_REL)), (
            "with the exemption disabled the sweep did NOT delete the evidence chain — the exemption is not what is keeping it alive"
        )

    def test_the_recorded_restore_test_depends_on_the_scope(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Remove the scope; the restore goes back to minting nothing."""
        from contextlib import nullcontext

        class SilentGate:
            def admit_batch(self, **_kwargs: Any) -> Any:
                return nullcontext(None)

        write_governed_block(workspace, "D-20260902-001")
        snap_dir = create_snapshot(workspace, "20260902-170002", files_touched=None)

        monkeypatch.setattr("mind_mem.governance_gate.get_gate", lambda ws: SilentGate())
        # The store's restore does not need a receipt, so the scope is the
        # only thing that was writing a row.
        get_block_store(workspace).restore(snap_dir)

        assert restore_rows(workspace, RESTORE_VERB) == [], (
            "a RESTORE row appeared with no scope open — this test is not measuring the scope"
        )

    def test_the_legacy_archive_test_depends_on_the_reader_refusal(
        self,
        workspace: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stop refusing ledger members; a pre-5.0.2 archive rewinds again."""
        write_governed_block(workspace, "D-20260902-001")
        live_evidence = Path(workspace, EVIDENCE_REL).read_bytes()

        stale = tmp_path / "stale.jsonl"
        stale.write_text('{"stale": true}\n', encoding="utf-8")
        archive = tmp_path / "legacy.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stale, arcname=EVIDENCE_REL)

        monkeypatch.setattr(backup_restore, "is_ledger_path", lambda rel: False)
        restore_workspace(workspace, str(archive), force=True)

        assert Path(workspace, EVIDENCE_REL).read_bytes() != live_evidence, (
            "with the refusal disabled the archive did NOT overwrite the chain — the refusal is not what is stopping it"
        )
