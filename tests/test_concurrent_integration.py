"""Integration tests for concurrent access and partial failure in mind-mem.

Tests cover:
1. Concurrent apply — two threads applying different proposals simultaneously
2. Partial failure rollback — multi-op proposal where one op fails mid-way
3. WAL crash recovery — simulated crash during write, replay restores state
4. Concurrent recall — multiple threads querying simultaneously
5. File lock contention — mutual exclusion, timeout, stale lock detection
6. Apply during active recall — consistency under mixed read/write
7. Post-check failure rollback — ops succeed but post-checks fail
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
from unittest.mock import patch

# Ensure scripts are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from apply_engine import (
    SNAPSHOT_FILES,
    _list_workspace_files,
    apply_proposal,
    compute_fingerprint,
    create_snapshot,
    restore_snapshot,
)
from backup_restore import WAL
from block_parser import parse_file
from mind_filelock import FileLock, LockTimeout
from init_workspace import init
from recall import recall

# ---------------------------------------------------------------------------
# Helper: Build a minimal workspace with valid blocks and proposals
# ---------------------------------------------------------------------------

def _write_decisions(ws, blocks_text):
    """Write content to decisions/DECISIONS.md."""
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(blocks_text)


def _write_tasks(ws, blocks_text):
    """Write content to tasks/TASKS.md."""
    path = os.path.join(ws, "tasks", "TASKS.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(blocks_text)


def _write_proposal(ws, proposal_text, filename="DECISIONS_PROPOSED.md"):
    """Write a proposal to intelligence/proposed/."""
    path = os.path.join(ws, "intelligence", "proposed", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(proposal_text)


def _build_proposal_block(proposal_id, target_block, ops, risk="low",
                          ptype="edit", evidence=None):
    """Build a valid proposal dict for testing."""
    if evidence is None:
        evidence = ["Test evidence for integration test"]
    proposal = {
        "_id": proposal_id,
        "ProposalId": proposal_id,
        "Type": ptype,
        "TargetBlock": target_block,
        "Risk": risk,
        "Status": "staged",
        "Evidence": evidence,
        "Rollback": ["Restore from snapshot"],
        "Ops": ops,
        "FilesTouched": list({op.get("file", "") for op in ops if op.get("file")}),
    }
    proposal["Fingerprint"] = compute_fingerprint(proposal)
    return proposal


def _proposal_to_markdown(proposal):
    """Serialize a proposal dict to markdown block text."""
    lines = [
        f"[{proposal['ProposalId']}]",
        f"ProposalId: {proposal['ProposalId']}",
        f"Type: {proposal['Type']}",
        f"TargetBlock: {proposal['TargetBlock']}",
        f"Risk: {proposal['Risk']}",
        f"Status: {proposal['Status']}",
        f"Fingerprint: {proposal['Fingerprint']}",
        "Evidence:",
    ]
    for e in proposal.get("Evidence", []):
        lines.append(f"- {e}")
    lines.append("Rollback:")
    for r in proposal.get("Rollback", []):
        lines.append(f"- {r}")
    lines.append("FilesTouched:")
    for ft in proposal.get("FilesTouched", []):
        lines.append(f"- {ft}")
    lines.append("Ops:")
    for op in proposal.get("Ops", []):
        lines.append(f"- op: {op['op']}")
        for k, v in op.items():
            if k == "op":
                continue
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    lines.append(f"    {sk}: {sv}")
            elif isinstance(v, str) and "\n" in v:
                lines.append(f"  {k}: |")
                for patch_line in v.split("\n"):
                    lines.append(f"    {patch_line}")
            else:
                lines.append(f"  {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def _scaffold_workspace(ws):
    """Create a minimal but complete workspace for apply_engine tests.

    Sets up the directory structure, config, and intel-state needed by
    the apply pipeline, without depending on init_workspace templates.
    """
    for d in ["decisions", "tasks", "entities", "memory", "summaries",
              "intelligence/proposed", "intelligence/applied"]:
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    # mind-mem.json
    config = {
        "version": "1.0.0",
        "governance_mode": "enforce",
        "recall": {"backend": "bm25"},
        "proposal_budget": {"per_run": 3, "per_day": 6, "backlog_limit": 30},
    }
    with open(os.path.join(ws, "mind-mem.json"), "w") as f:
        json.dump(config, f)

    # intel-state.json (no last_apply_ts so cooldown check passes)
    with open(os.path.join(ws, "memory", "intel-state.json"), "w") as f:
        json.dump({"governance_mode": "enforce"}, f)

    # Empty files that snapshot expects
    for fname in SNAPSHOT_FILES:
        path = os.path.join(ws, fname)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("")


# ===========================================================================
# 1. Concurrent Apply
# ===========================================================================

class TestConcurrentApply(unittest.TestCase):
    """Two threads trying to apply different proposals simultaneously."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_concurrent_apply_")
        _scaffold_workspace(self.ws)

        # Write a decisions file with a target block
        _write_decisions(self.ws, (
            "# Decisions\n\n"
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL for primary database\n"
            "Status: active\n"
            "Date: 2026-01-01\n"
            "Context: Needed reliable RDBMS\n"
            "\n---\n\n"
            "[D-20260101-002]\n"
            "Statement: Deploy to AWS us-east-1\n"
            "Status: active\n"
            "Date: 2026-01-01\n"
            "Context: Regional proximity to users\n"
            "\n---\n"
        ))

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_concurrent_apply_preserves_workspace_integrity(self):
        """Two threads apply proposals concurrently. At least one must succeed
        and the workspace must be left in a consistent state."""

        # Build two proposals targeting different blocks
        ops_a = [{"op": "update_field", "file": "decisions/DECISIONS.md",
                  "target": "D-20260101-001", "field": "Context",
                  "value": "Updated by thread A"}]
        proposal_a = _build_proposal_block("P-20260201-001", "D-20260101-001", ops_a)

        ops_b = [{"op": "update_field", "file": "decisions/DECISIONS.md",
                  "target": "D-20260101-002", "field": "Context",
                  "value": "Updated by thread B"}]
        proposal_b = _build_proposal_block("P-20260201-002", "D-20260101-002", ops_b)

        # Write both proposals
        _write_proposal(self.ws,
                        _proposal_to_markdown(proposal_a) + "\n---\n\n" +
                        _proposal_to_markdown(proposal_b))

        results = {"a": None, "b": None}
        errors = {"a": None, "b": None}

        def apply_a():
            try:
                # Patch check_preconditions to skip external scripts
                with patch("apply_engine.check_preconditions",
                           return_value=(True, ["validate: PASS (TOTAL 0 issues)"])):
                    results["a"] = apply_proposal(self.ws, "P-20260201-001")
            except Exception as e:
                errors["a"] = e

        def apply_b():
            try:
                with patch("apply_engine.check_preconditions",
                           return_value=(True, ["validate: PASS (TOTAL 0 issues)"])):
                    results["b"] = apply_proposal(self.ws, "P-20260201-002")
            except Exception as e:
                errors["b"] = e

        t1 = threading.Thread(target=apply_a)
        t2 = threading.Thread(target=apply_b)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        # At least one should succeed (no crash)
        a_ok = results["a"] is not None and results["a"][0]
        b_ok = results["b"] is not None and results["b"][0]
        self.assertTrue(a_ok or b_ok,
                        f"Neither thread succeeded. A={results['a']}, B={results['b']}, "
                        f"Errors: A={errors['a']}, B={errors['b']}")

        # Workspace must be parseable (not corrupted)
        dec_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        blocks = parse_file(dec_path)
        self.assertGreaterEqual(len(blocks), 2,
                                "Workspace corrupted: fewer than 2 blocks remain")

        # Both original blocks should still exist
        ids = {b.get("_id") for b in blocks}
        self.assertIn("D-20260101-001", ids)
        self.assertIn("D-20260101-002", ids)


# ===========================================================================
# 2. Partial Failure Rollback
# ===========================================================================

class TestPartialFailureRollback(unittest.TestCase):
    """Multi-op proposal where one op fails mid-way. Verify rollback."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_partial_fail_")
        _scaffold_workspace(self.ws)

        # Write decisions file with one block
        _write_decisions(self.ws, (
            "# Decisions\n\n"
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL for primary database\n"
            "Status: active\n"
            "Date: 2026-01-01\n"
            "Context: Needed reliable RDBMS\n"
            "\n---\n"
        ))

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_partial_failure_rolls_back_all_changes(self):
        """Proposal with 2 ops: first succeeds, second fails.
        Workspace must be restored to pre-apply state."""

        # Read original content for comparison
        dec_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        with open(dec_path) as f:
            original_content = f.read()

        # Op 1: valid update (will succeed)
        # Op 2: targets a non-existent block (will fail)
        ops = [
            {"op": "update_field", "file": "decisions/DECISIONS.md",
             "target": "D-20260101-001", "field": "Context",
             "value": "Modified by partial-fail test"},
            {"op": "update_field", "file": "decisions/DECISIONS.md",
             "target": "D-NONEXISTENT-999", "field": "Status",
             "value": "should-not-exist"},
        ]
        proposal = _build_proposal_block("P-20260202-001", "D-20260101-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        # Apply (mock preconditions to pass)
        with patch("apply_engine.check_preconditions",
                   return_value=(True, ["validate: PASS (TOTAL 0 issues)"])):
            ok, msg = apply_proposal(self.ws, "P-20260202-001")

        # Apply should have failed
        self.assertFalse(ok, f"Expected failure but got success: {msg}")

        # Workspace should be rolled back to original state
        with open(dec_path) as f:
            restored_content = f.read()
        self.assertEqual(original_content, restored_content,
                         "Workspace was not restored to pre-apply state after partial failure")

        # Receipt should exist with rolled_back status
        applied_dir = os.path.join(self.ws, "intelligence", "applied")
        if os.path.isdir(applied_dir):
            receipt_dirs = [d for d in os.listdir(applied_dir)
                           if os.path.isdir(os.path.join(applied_dir, d))]
            found_receipt = False
            for rd in receipt_dirs:
                receipt_path = os.path.join(applied_dir, rd, "APPLY_RECEIPT.md")
                if os.path.isfile(receipt_path):
                    with open(receipt_path) as f:
                        receipt_text = f.read()
                    if "rolled_back" in receipt_text:
                        found_receipt = True
                        break
            self.assertTrue(found_receipt,
                            "No receipt with 'rolled_back' status found after partial failure")

    def test_snapshot_created_before_ops(self):
        """Verify that a snapshot directory is created before ops execute."""

        ops = [
            {"op": "update_field", "file": "decisions/DECISIONS.md",
             "target": "D-NONEXISTENT-999", "field": "Status",
             "value": "fail-immediately"},
        ]
        proposal = _build_proposal_block("P-20260202-002", "D-20260101-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        with patch("apply_engine.check_preconditions",
                   return_value=(True, ["validate: PASS (TOTAL 0 issues)"])):
            ok, msg = apply_proposal(self.ws, "P-20260202-002")

        self.assertFalse(ok)

        # A snapshot directory should exist under intelligence/applied/
        applied_dir = os.path.join(self.ws, "intelligence", "applied")
        self.assertTrue(os.path.isdir(applied_dir))
        snapshot_dirs = [d for d in os.listdir(applied_dir)
                         if os.path.isdir(os.path.join(applied_dir, d))]
        self.assertGreaterEqual(len(snapshot_dirs), 1,
                                "No snapshot directory created before ops executed")

    def test_no_orphan_files_after_rollback(self):
        """If an op creates a file (append_block), rollback should remove it."""

        # First create decisions file, record pre-apply files
        dec_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        _list_workspace_files(self.ws)

        # Op 1: append a block (creates new content -- succeeds)
        # Op 2: fail on non-existent target
        new_block_text = (
            "[D-20260202-099]\n"
            "Statement: Orphan block that should be removed\n"
            "Status: active\n"
            "Date: 2026-02-02\n"
        )
        ops = [
            {"op": "append_block", "file": "decisions/DECISIONS.md",
             "patch": new_block_text},
            {"op": "update_field", "file": "decisions/DECISIONS.md",
             "target": "D-NONEXISTENT-999", "field": "Status",
             "value": "fail"},
        ]
        proposal = _build_proposal_block("P-20260202-003", "D-20260101-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        with patch("apply_engine.check_preconditions",
                   return_value=(True, ["validate: PASS (TOTAL 0 issues)"])):
            ok, msg = apply_proposal(self.ws, "P-20260202-003")

        self.assertFalse(ok)

        # The appended block text should not remain in the file
        with open(dec_path) as f:
            content = f.read()
        self.assertNotIn("Orphan block that should be removed", content,
                         "Orphan content from failed op remains after rollback")


# ===========================================================================
# 3. WAL Crash Recovery
# ===========================================================================

class TestWALCrashRecovery(unittest.TestCase):
    """Simulate a crash during write: begin WAL, modify file, don't commit."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_wal_crash_")
        os.makedirs(os.path.join(self.ws, "decisions"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_wal_replay_restores_pre_write_state(self):
        """Begin WAL, write partial content, don't commit.
        A new WAL instance calling replay() should restore original state."""

        target_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        original_content = (
            "# Decisions\n\n"
            "[D-20260101-001]\n"
            "Statement: Original content before crash\n"
            "Status: active\n"
        )
        with open(target_path, "w") as f:
            f.write(original_content)

        # Begin WAL entry
        wal = WAL(self.ws)
        wal.begin("update", target_path, "new content that simulates crash")

        # Simulate the write happening (as if the process was in the middle of writing)
        with open(target_path, "w") as f:
            f.write("CORRUPTED PARTIAL WRITE -- crash happened here")

        # Verify the WAL entry is pending
        self.assertEqual(wal.pending_count(), 1)

        # Simulate process restart: create a new WAL instance and replay
        wal2 = WAL(self.ws)
        replayed = wal2.replay()

        self.assertEqual(replayed, 1, "Expected exactly 1 WAL entry to be replayed")

        # File should be restored to original content
        with open(target_path) as f:
            restored = f.read()
        self.assertEqual(restored, original_content,
                         "WAL replay did not restore file to pre-write state")

        # No pending entries should remain
        self.assertEqual(wal2.pending_count(), 0,
                         "Pending WAL entries remain after replay")

    def test_wal_committed_entries_are_not_replayed(self):
        """Committed WAL entries should be cleaned up and not replayed."""

        target_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        with open(target_path, "w") as f:
            f.write("original content")

        wal = WAL(self.ws)
        entry_id = wal.begin("update", target_path, "new content")

        # Simulate successful write
        with open(target_path, "w") as f:
            f.write("new content successfully written")

        # Commit the entry
        wal.commit(entry_id)

        # No pending entries
        self.assertEqual(wal.pending_count(), 0)

        # Replay should do nothing
        wal2 = WAL(self.ws)
        replayed = wal2.replay()
        self.assertEqual(replayed, 0, "Committed entry should not be replayed")

        # Content should remain as the new (committed) content
        with open(target_path) as f:
            content = f.read()
        self.assertEqual(content, "new content successfully written")

    def test_wal_new_file_removed_on_rollback(self):
        """If WAL begin has no backup (new file), rollback should remove the file."""

        target_path = os.path.join(self.ws, "decisions", "NEW_FILE.md")
        self.assertFalse(os.path.exists(target_path))

        wal = WAL(self.ws)
        wal.begin("create", target_path, "brand new file content")

        # Simulate the file being created
        with open(target_path, "w") as f:
            f.write("brand new file content")

        # Don't commit -- simulate crash
        wal2 = WAL(self.ws)
        replayed = wal2.replay()
        self.assertEqual(replayed, 1)

        # The new file should have been removed
        self.assertFalse(os.path.exists(target_path),
                         "New file should be removed on WAL rollback")

    def test_multiple_pending_entries_all_replayed(self):
        """Multiple pending WAL entries should all be rolled back on replay."""

        file_a = os.path.join(self.ws, "decisions", "FILE_A.md")
        file_b = os.path.join(self.ws, "decisions", "FILE_B.md")
        with open(file_a, "w") as f:
            f.write("original A")
        with open(file_b, "w") as f:
            f.write("original B")

        wal = WAL(self.ws)
        wal.begin("update", file_a, "new A")
        wal.begin("update", file_b, "new B")

        # Corrupt both files
        with open(file_a, "w") as f:
            f.write("corrupted A")
        with open(file_b, "w") as f:
            f.write("corrupted B")

        self.assertEqual(wal.pending_count(), 2)

        # Replay
        wal2 = WAL(self.ws)
        replayed = wal2.replay()
        self.assertEqual(replayed, 2)

        with open(file_a) as f:
            self.assertEqual(f.read(), "original A")
        with open(file_b) as f:
            self.assertEqual(f.read(), "original B")


# ===========================================================================
# 4. Concurrent Recall
# ===========================================================================

class TestConcurrentRecall(unittest.TestCase):
    """Multiple threads running recall queries on the same workspace."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_concurrent_recall_")
        init(self.ws)

        # Write a decisions file with several distinct blocks
        today = datetime.now().strftime("%Y%m%d")
        _write_decisions(self.ws, (
            "# Decisions\n\n"
            f"[D-{today}-001]\n"
            "Statement: Use PostgreSQL for the primary database\n"
            "Status: active\n"
            "Date: 2026-01-15\n"
            "Context: Evaluated MySQL, PostgreSQL, and SQLite\n"
            "\n---\n\n"
            f"[D-{today}-002]\n"
            "Statement: Deploy authentication service to Kubernetes\n"
            "Status: active\n"
            "Date: 2026-01-15\n"
            "Context: Container orchestration for microservices\n"
            "\n---\n\n"
            f"[D-{today}-003]\n"
            "Statement: Implement rate limiting on public API endpoints\n"
            "Status: active\n"
            "Date: 2026-01-15\n"
            "Context: Protect against abuse and DDoS\n"
            "\n---\n"
        ))

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_concurrent_recall_returns_correct_results(self):
        """Multiple threads querying simultaneously should all get correct results."""

        queries = [
            ("PostgreSQL database", "PostgreSQL"),
            ("authentication Kubernetes", "authentication"),
            ("rate limiting API", "rate"),
        ]

        results = {}
        errors = {}

        def run_recall(idx, query, expected_keyword):
            try:
                r = recall(self.ws, query, limit=5)
                results[idx] = (r, expected_keyword)
            except Exception as e:
                errors[idx] = e

        threads = []
        for i, (query, keyword) in enumerate(queries):
            t = threading.Thread(target=run_recall, args=(i, query, keyword))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # No thread should have errored
        self.assertEqual(len(errors), 0,
                         f"Recall threads raised errors: {errors}")

        # Each thread should return at least one result containing its keyword
        for idx, (r, keyword) in results.items():
            self.assertGreater(len(r), 0,
                               f"Thread {idx} returned no results for its query")
            found = any(keyword.lower() in item.get("excerpt", "").lower()
                        for item in r)
            self.assertTrue(found,
                            f"Thread {idx}: expected keyword '{keyword}' "
                            f"not found in results: {[item.get('excerpt', '')[:50] for item in r]}")

    def test_many_concurrent_recalls_no_corruption(self):
        """10 threads performing recall concurrently should not corrupt state."""

        error_count = [0]
        result_count = [0]
        lock = threading.Lock()

        def run_query(query):
            try:
                r = recall(self.ws, query, limit=3)
                with lock:
                    result_count[0] += len(r)
            except Exception:
                with lock:
                    error_count[0] += 1

        threads = []
        for i in range(10):
            query = ["PostgreSQL", "authentication", "rate limiting",
                     "database", "Kubernetes"][i % 5]
            t = threading.Thread(target=run_query, args=(query,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(error_count[0], 0,
                         f"{error_count[0]} recall threads raised exceptions")
        self.assertGreater(result_count[0], 0,
                           "No results returned across all 10 threads")


# ===========================================================================
# 5. File Lock Contention
# ===========================================================================

class TestFileLockContention(unittest.TestCase):
    """FileLock mutual exclusion, timeout, and stale lock detection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memos_filelock_")
        self.target = os.path.join(self.tmpdir, "target.md")
        with open(self.target, "w") as f:
            f.write("lock target content")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mutual_exclusion(self):
        """Only one thread holds the lock at a time."""

        counter = [0]
        max_concurrent = [0]
        active = [0]
        lock_obj = threading.Lock()

        def critical_section(thread_id):
            fl = FileLock(self.target, timeout=10.0)
            fl.acquire()
            try:
                with lock_obj:
                    active[0] += 1
                    if active[0] > max_concurrent[0]:
                        max_concurrent[0] = active[0]
                    counter[0] += 1

                # Hold the lock briefly to increase contention window
                time.sleep(0.02)

                with lock_obj:
                    active[0] -= 1
            finally:
                fl.release()

        threads = []
        for i in range(5):
            t = threading.Thread(target=critical_section, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(counter[0], 5, "Not all threads completed")
        self.assertEqual(max_concurrent[0], 1,
                         f"Multiple threads held lock simultaneously: max_concurrent={max_concurrent[0]}")

    def test_timeout_raises_lock_timeout(self):
        """A thread waiting longer than timeout should get LockTimeout."""

        # Acquire lock in main thread
        fl1 = FileLock(self.target, timeout=10.0)
        fl1.acquire()

        try:
            # Second lock with very short timeout should fail
            fl2 = FileLock(self.target, timeout=0.2, poll_interval=0.05)
            with self.assertRaises(LockTimeout):
                fl2.acquire()
        finally:
            fl1.release()

    def test_non_blocking_timeout_zero(self):
        """timeout=0 should fail immediately if lock is held."""

        fl1 = FileLock(self.target, timeout=10.0)
        fl1.acquire()

        try:
            fl2 = FileLock(self.target, timeout=0)
            with self.assertRaises(LockTimeout):
                fl2.acquire()
        finally:
            fl1.release()

    def test_stale_lock_detection_dead_pid(self):
        """Lock file with a dead PID should be detected as stale and broken."""

        lock_path = self.target + ".lock"

        # Write a lock file with a PID that does not exist.
        # PID 4194304 (2^22) is very unlikely to exist on any system.
        dead_pid = 4194304
        with open(lock_path, "w") as f:
            f.write(f"{dead_pid}\n")

        # A new FileLock should detect the stale lock and acquire successfully
        fl = FileLock(self.target, timeout=2.0)
        try:
            fl.acquire()  # Should succeed by breaking the stale lock
            acquired = True
        except LockTimeout:
            acquired = False
        finally:
            fl.release()

        self.assertTrue(acquired,
                        "FileLock should detect dead PID and break stale lock")

    def test_context_manager_releases_on_exit(self):
        """Lock released when exiting context manager, even on exception."""

        lock_path = self.target + ".lock"

        try:
            with FileLock(self.target, timeout=2.0):
                self.assertTrue(os.path.exists(lock_path),
                                "Lock file should exist while lock is held")
                raise ValueError("Intentional exception")
        except ValueError:
            pass

        # Lock file should be cleaned up
        self.assertFalse(os.path.exists(lock_path),
                         "Lock file should be removed after context exit")

    def test_reentrant_after_release(self):
        """After releasing, another acquire should succeed immediately."""

        fl = FileLock(self.target, timeout=2.0)
        fl.acquire()
        fl.release()

        fl2 = FileLock(self.target, timeout=0)
        try:
            fl2.acquire()
            acquired = True
        except LockTimeout:
            acquired = False
        finally:
            fl2.release()

        self.assertTrue(acquired,
                        "Should be able to acquire lock after previous holder released")


# ===========================================================================
# 6. Apply During Active Recall
# ===========================================================================

class TestApplyDuringRecall(unittest.TestCase):
    """Recall reads consistent data while apply modifies the workspace."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_apply_recall_")
        _scaffold_workspace(self.ws)

        today = datetime.now().strftime("%Y%m%d")
        _write_decisions(self.ws, (
            "# Decisions\n\n"
            f"[D-{today}-001]\n"
            "Statement: Use Redis for session caching layer\n"
            "Status: active\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            "Context: High-performance in-memory cache for sessions\n"
            "\n---\n\n"
            f"[D-{today}-002]\n"
            "Statement: Use RabbitMQ for message queue infrastructure\n"
            "Status: active\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            "Context: Reliable message broker for async processing\n"
            "\n---\n"
        ))
        self.today = today

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_recall_returns_consistent_results_during_apply(self):
        """Recall should return results (not crash) while apply is running."""

        ops = [{"op": "update_field", "file": "decisions/DECISIONS.md",
                "target": f"D-{self.today}-001", "field": "Context",
                "value": "Updated: now using Redis Cluster with Sentinel"}]
        proposal = _build_proposal_block(
            "P-20260203-001", f"D-{self.today}-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        recall_results = []
        recall_errors = []
        apply_result = [None]
        apply_error = [None]

        start_barrier = threading.Barrier(2, timeout=10)

        def do_recall():
            try:
                start_barrier.wait()
                # Run multiple recalls in quick succession
                for _ in range(5):
                    r = recall(self.ws, "Redis caching session", limit=5)
                    recall_results.append(r)
                    time.sleep(0.01)
            except Exception as e:
                recall_errors.append(e)

        def do_apply():
            try:
                start_barrier.wait()
                with patch("apply_engine.check_preconditions",
                           return_value=(True, ["validate: PASS (TOTAL 0 issues)"])):
                    apply_result[0] = apply_proposal(self.ws, "P-20260203-001")
            except Exception as e:
                apply_error[0] = e

        t_recall = threading.Thread(target=do_recall)
        t_apply = threading.Thread(target=do_apply)
        t_recall.start()
        t_apply.start()
        t_recall.join(timeout=30)
        t_apply.join(timeout=30)

        # Recall should not have crashed
        self.assertEqual(len(recall_errors), 0,
                         f"Recall raised errors during concurrent apply: {recall_errors}")

        # At least some recall invocations should return results
        non_empty = [r for r in recall_results if len(r) > 0]
        self.assertGreater(len(non_empty), 0,
                           "No recall invocation returned results during apply")

        # Every non-empty recall result should have valid structure
        for result_set in non_empty:
            for item in result_set:
                self.assertIn("_id", item)
                self.assertIn("score", item)
                self.assertIn("excerpt", item)


# ===========================================================================
# 7. Post-check Failure Rollback
# ===========================================================================

class TestPostCheckFailureRollback(unittest.TestCase):
    """Ops succeed but post-checks fail. Verify full rollback."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_postcheck_fail_")
        _scaffold_workspace(self.ws)

        _write_decisions(self.ws, (
            "# Decisions\n\n"
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL for primary database\n"
            "Status: active\n"
            "Date: 2026-01-01\n"
            "Context: Needed reliable RDBMS\n"
            "\n---\n"
        ))

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_postcheck_failure_restores_workspace(self):
        """After successful ops, if post-checks fail, workspace is restored."""

        dec_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        with open(dec_path) as f:
            original_content = f.read()

        ops = [{"op": "update_field", "file": "decisions/DECISIONS.md",
                "target": "D-20260101-001", "field": "Context",
                "value": "Updated successfully but post-check will fail"}]
        proposal = _build_proposal_block("P-20260204-001", "D-20260101-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        call_count = [0]

        def mock_preconditions(ws):
            call_count[0] += 1
            if call_count[0] == 1:
                # Pre-check passes
                return True, ["validate: PASS (TOTAL 0 issues)"]
            else:
                # Post-check fails
                return False, ["validate: FAIL (TOTAL 3 issues)"]

        with patch("apply_engine.check_preconditions", side_effect=mock_preconditions):
            ok, msg = apply_proposal(self.ws, "P-20260204-001")

        # Should have failed
        self.assertFalse(ok, f"Expected failure from post-check but got success: {msg}")
        self.assertIn("rolled back", msg.lower(),
                       f"Expected 'rolled back' in message but got: {msg}")

        # Workspace should be restored
        with open(dec_path) as f:
            restored_content = f.read()
        self.assertEqual(original_content, restored_content,
                         "Workspace not restored after post-check failure")

    def test_postcheck_failure_receipt_shows_rolled_back(self):
        """Receipt should contain rolled_back status after post-check failure."""

        ops = [{"op": "update_field", "file": "decisions/DECISIONS.md",
                "target": "D-20260101-001", "field": "Context",
                "value": "Updated but will be rolled back"}]
        proposal = _build_proposal_block("P-20260204-002", "D-20260101-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        call_count = [0]

        def mock_preconditions(ws):
            call_count[0] += 1
            if call_count[0] == 1:
                return True, ["validate: PASS (TOTAL 0 issues)"]
            else:
                return False, ["validate: FAIL (TOTAL 2 issues)"]

        with patch("apply_engine.check_preconditions", side_effect=mock_preconditions):
            ok, msg = apply_proposal(self.ws, "P-20260204-002")

        self.assertFalse(ok)

        # Find the receipt
        applied_dir = os.path.join(self.ws, "intelligence", "applied")
        receipt_found = False
        for rd in os.listdir(applied_dir):
            receipt_path = os.path.join(applied_dir, rd, "APPLY_RECEIPT.md")
            if os.path.isfile(receipt_path):
                with open(receipt_path) as f:
                    receipt_text = f.read()
                if "P-20260204-002" in receipt_text and "rolled_back" in receipt_text:
                    receipt_found = True
                    break

        self.assertTrue(receipt_found,
                        "Receipt with rolled_back status not found for P-20260204-002")

    def test_postcheck_failure_cleans_orphan_files(self):
        """After post-check failure rollback, orphan files from ops are removed."""

        # Record pre-apply file set
        _list_workspace_files(self.ws)

        # Create a proposal that appends a block (creating new content),
        # and then the post-check fails
        new_block = (
            "[D-20260204-099]\n"
            "Statement: Orphan decision created by append_block\n"
            "Status: active\n"
            "Date: 2026-02-04\n"
        )
        ops = [{"op": "append_block", "file": "decisions/DECISIONS.md",
                "patch": new_block}]
        proposal = _build_proposal_block("P-20260204-003", "D-20260101-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        call_count = [0]

        def mock_preconditions(ws):
            call_count[0] += 1
            if call_count[0] == 1:
                return True, ["validate: PASS (TOTAL 0 issues)"]
            else:
                return False, ["validate: FAIL (TOTAL 1 issues)"]

        with patch("apply_engine.check_preconditions", side_effect=mock_preconditions):
            ok, msg = apply_proposal(self.ws, "P-20260204-003")

        self.assertFalse(ok)

        # The appended orphan content should not remain
        dec_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        with open(dec_path) as f:
            content = f.read()
        self.assertNotIn("Orphan decision created by append_block", content,
                         "Orphan content from append_block remains after post-check rollback")

    def test_proposal_marked_rolled_back_after_postcheck_failure(self):
        """The proposal's Status field in the source file should be set to rolled_back."""

        ops = [{"op": "update_field", "file": "decisions/DECISIONS.md",
                "target": "D-20260101-001", "field": "Context",
                "value": "This will be rolled back"}]
        proposal = _build_proposal_block("P-20260204-004", "D-20260101-001", ops)
        _write_proposal(self.ws, _proposal_to_markdown(proposal))

        call_count = [0]

        def mock_preconditions(ws):
            call_count[0] += 1
            if call_count[0] == 1:
                return True, ["validate: PASS (TOTAL 0 issues)"]
            else:
                return False, ["validate: FAIL (TOTAL 1 issues)"]

        with patch("apply_engine.check_preconditions", side_effect=mock_preconditions):
            ok, msg = apply_proposal(self.ws, "P-20260204-004")

        self.assertFalse(ok)

        # Read the proposal source file and check status
        proposal_path = os.path.join(
            self.ws, "intelligence", "proposed", "DECISIONS_PROPOSED.md")
        with open(proposal_path) as f:
            content = f.read()

        # The proposal's status should have been changed to rolled_back
        # (apply_engine._mark_proposal_status is called on post-check failure)
        self.assertIn("Status: rolled_back", content,
                       "Proposal status not updated to rolled_back in source file")


# ===========================================================================
# Additional Edge Case: Snapshot/Restore Fidelity
# ===========================================================================

class TestSnapshotRestoreFidelity(unittest.TestCase):
    """Verify that create_snapshot + restore_snapshot is bit-for-bit faithful."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_snapshot_")
        _scaffold_workspace(self.ws)

        _write_decisions(self.ws, (
            "# Decisions\n\n"
            "[D-20260101-001]\n"
            "Statement: Snapshot fidelity test\n"
            "Status: active\n"
            "Date: 2026-01-01\n"
            "\n---\n"
        ))
        _write_tasks(self.ws, (
            "# Tasks\n\n"
            "[T-20260101-001]\n"
            "Title: Test task for snapshot\n"
            "Status: todo\n"
            "Priority: P1\n"
            "\n---\n"
        ))

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_snapshot_and_restore_preserves_content(self):
        """After snapshot + mutation + restore, content matches original."""

        dec_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        task_path = os.path.join(self.ws, "tasks", "TASKS.md")

        with open(dec_path) as f:
            orig_decisions = f.read()
        with open(task_path) as f:
            orig_tasks = f.read()

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        snap_dir = create_snapshot(self.ws, ts)

        # Mutate the workspace
        with open(dec_path, "w") as f:
            f.write("MUTATED DECISIONS CONTENT")
        with open(task_path, "w") as f:
            f.write("MUTATED TASKS CONTENT")

        # Restore
        restore_snapshot(self.ws, snap_dir)

        with open(dec_path) as f:
            restored_decisions = f.read()
        with open(task_path) as f:
            restored_tasks = f.read()

        self.assertEqual(orig_decisions, restored_decisions,
                         "Decisions file not restored faithfully")
        self.assertEqual(orig_tasks, restored_tasks,
                         "Tasks file not restored faithfully")


if __name__ == "__main__":
    unittest.main()
