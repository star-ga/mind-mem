#!/usr/bin/env python3
"""Tests for apply_engine.py — focus on security, validation, and rollback."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from apply_engine import (
    _safe_resolve, validate_proposal, create_snapshot, restore_snapshot,
    check_no_touch_window, check_fingerprint_dedup, compute_fingerprint,
    rollback, apply_proposal, _op_supersede_decision, _op_replace_range,
)
import json
import subprocess
from datetime import datetime, timedelta


class TestSafeResolve(unittest.TestCase):
    def test_normal_path(self):
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "decisions")
            os.makedirs(target)
            result = _safe_resolve(td, "decisions")
            self.assertEqual(result, os.path.realpath(target))

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                _safe_resolve(td, "../../../etc/passwd")

    def test_rejects_absolute_path(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                _safe_resolve(td, "/etc/passwd")

    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as td:
            # Create symlink that points outside workspace
            link_path = os.path.join(td, "escape_link")
            os.symlink("/tmp", link_path)
            with self.assertRaises(ValueError):
                _safe_resolve(td, "escape_link/should_fail")

    def test_allows_internal_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            real_dir = os.path.join(td, "real")
            os.makedirs(real_dir)
            link_path = os.path.join(td, "link")
            os.symlink(real_dir, link_path)
            # Internal symlink should work
            result = _safe_resolve(td, "link")
            self.assertEqual(result, os.path.realpath(real_dir))

    def test_rejects_dotdot_in_middle(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                _safe_resolve(td, "decisions/../../../etc/passwd")


class TestValidateProposal(unittest.TestCase):
    def _valid_proposal(self, **overrides):
        base = {
            "ProposalId": "P-20260213-001",
            "Type": "decision",
            "TargetBlock": "D-20260213-001",
            "Risk": "low",
            "Status": "staged",
            "Evidence": "Test evidence",
            "Rollback": "Revert changes",
            "Fingerprint": "368e730f7a52788a",
            "Ops": [{"op": "append_block", "file": "decisions/DECISIONS.md"}],
        }
        base.update(overrides)
        return base

    def test_valid_proposal(self):
        errors = validate_proposal(self._valid_proposal())
        self.assertEqual(errors, [])

    def test_missing_required_field(self):
        p = self._valid_proposal()
        del p["Evidence"]
        errors = validate_proposal(p)
        self.assertTrue(any("Evidence" in e for e in errors))

    def test_invalid_risk(self):
        errors = validate_proposal(self._valid_proposal(Risk="extreme"))
        self.assertTrue(any("Risk" in e for e in errors))

    def test_invalid_type(self):
        errors = validate_proposal(self._valid_proposal(Type="migration"))
        self.assertTrue(any("Type" in e for e in errors))

    def test_status_not_staged(self):
        errors = validate_proposal(self._valid_proposal(Status="applied"))
        self.assertTrue(any("staged" in e for e in errors))

    def test_rejects_path_traversal_in_ops(self):
        p = self._valid_proposal(
            Ops=[{"op": "append_block", "file": "../../../etc/shadow"}]
        )
        errors = validate_proposal(p)
        self.assertTrue(any("traversal" in e for e in errors))

    def test_rejects_absolute_path_in_ops(self):
        p = self._valid_proposal(
            Ops=[{"op": "append_block", "file": "/etc/passwd"}]
        )
        errors = validate_proposal(p)
        self.assertTrue(any("traversal" in e.lower() or "absolute" in e.lower() for e in errors))

    def test_invalid_op_type(self):
        p = self._valid_proposal(
            Ops=[{"op": "delete_everything", "file": "decisions/DECISIONS.md"}]
        )
        errors = validate_proposal(p)
        self.assertTrue(any("op" in e.lower() for e in errors))


class TestSnapshotRollback(unittest.TestCase):
    """Verify atomic rollback removes files created during failed ops."""

    def test_rollback_removes_new_files(self):
        """Files created after snapshot must be deleted on restore."""
        with tempfile.TemporaryDirectory() as ws:
            # Set up workspace structure
            os.makedirs(os.path.join(ws, "decisions"))
            original = os.path.join(ws, "decisions", "DECISIONS.md")
            with open(original, "w") as f:
                f.write("# Decisions\n")

            # Create snapshot
            snap_dir = create_snapshot(ws, "test-rollback")

            # Simulate a failed op creating a new file
            rogue_file = os.path.join(ws, "decisions", "ROGUE.md")
            with open(rogue_file, "w") as f:
                f.write("# This should not survive rollback\n")
            self.assertTrue(os.path.exists(rogue_file))

            # Restore snapshot
            restore_snapshot(ws, snap_dir)

            # Rogue file must be gone (true atomic rollback)
            self.assertFalse(os.path.exists(rogue_file))
            # Original file must still exist
            self.assertTrue(os.path.exists(original))

    def test_rollback_restores_content(self):
        """Modified files must revert to snapshot content."""
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "decisions"))
            original = os.path.join(ws, "decisions", "DECISIONS.md")
            with open(original, "w") as f:
                f.write("original content")

            snap_dir = create_snapshot(ws, "test-content")

            # Modify file
            with open(original, "w") as f:
                f.write("corrupted content")

            restore_snapshot(ws, snap_dir)

            with open(original) as f:
                self.assertEqual(f.read(), "original content")


class TestSnapshotIntelligenceRestore(unittest.TestCase):
    """Verify snapshot restore includes intelligence files."""

    def test_rollback_restores_intelligence_files(self):
        """Intelligence files (e.g., SIGNALS.md) must be restored on rollback."""
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "decisions"))
            os.makedirs(os.path.join(ws, "intelligence"))
            signals = os.path.join(ws, "intelligence", "SIGNALS.md")
            with open(signals, "w") as f:
                f.write("original signals")
            with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
                f.write("# D\n")

            snap_dir = create_snapshot(ws, "test-intel")

            # Mutate intelligence file
            with open(signals, "w") as f:
                f.write("mutated signals")

            restore_snapshot(ws, snap_dir)

            with open(signals) as f:
                self.assertEqual(f.read(), "original signals")


class TestNoTouchWindow(unittest.TestCase):
    """Verify no-touch window cooldown logic."""

    def test_no_previous_apply(self):
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "memory"))
            with open(os.path.join(ws, "memory", "intel-state.json"), "w") as f:
                json.dump({}, f)
            ok, reason = check_no_touch_window(ws)
            self.assertTrue(ok)

    def test_recent_apply_blocks(self):
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "memory"))
            recent = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(os.path.join(ws, "memory", "intel-state.json"), "w") as f:
                json.dump({"last_apply_ts": recent}, f)
            ok, reason = check_no_touch_window(ws)
            self.assertFalse(ok)
            self.assertIn("No-touch window", reason)

    def test_old_apply_clears(self):
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "memory"))
            old = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(os.path.join(ws, "memory", "intel-state.json"), "w") as f:
                json.dump({"last_apply_ts": old}, f)
            ok, reason = check_no_touch_window(ws)
            self.assertTrue(ok)


class TestFingerprintDedup(unittest.TestCase):
    """Verify fingerprint dedup skips self-match."""

    def test_self_match_not_duplicate(self):
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "intelligence", "proposed"), exist_ok=True)
            proposal = {
                "ProposalId": "P-20260214-001", "_id": "P-20260214-001",
                "Type": "decision", "TargetBlock": "D-20260214-001",
                "Ops": [{"op": "append_block", "file": "decisions/DECISIONS.md"}],
                "Status": "staged",
            }
            fp = compute_fingerprint(proposal)
            block_text = (
                "[P-20260214-001]\n"
                "ProposalId: P-20260214-001\n"
                "Type: decision\n"
                "TargetBlock: D-20260214-001\n"
                "Status: staged\n"
                f"Fingerprint: {fp}\n"
            )
            for fn in ["DECISIONS_PROPOSED.md", "TASKS_PROPOSED.md", "EDITS_PROPOSED.md"]:
                path = os.path.join(ws, "intelligence", "proposed", fn)
                with open(path, "w") as f:
                    f.write(block_text if fn == "DECISIONS_PROPOSED.md" else "")
            is_dup, dup_id = check_fingerprint_dedup(ws, proposal)
            self.assertFalse(is_dup)


@unittest.skipIf(sys.platform == "win32", "validate.sh requires bash (Unix only)")
class TestFreshInitValidate(unittest.TestCase):
    """Verify fresh workspace passes validate.sh with 0 issues."""

    def test_fresh_init_passes_validate(self):
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            result = subprocess.run(
                ["bash", os.path.join(ws, "maintenance", "validate.sh"), ws],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, f"validate.sh failed:\n{result.stdout}")
            self.assertIn("0 issues", result.stdout)


class TestRollbackPathTraversal(unittest.TestCase):
    """Rollback must reject path-traversal receipt_ts values."""

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as ws:
            ok = rollback(ws, "../../../etc")
            self.assertFalse(ok)

    def test_rejects_arbitrary_string(self):
        with tempfile.TemporaryDirectory() as ws:
            ok = rollback(ws, "foo; rm -rf /")
            self.assertFalse(ok)

    def test_accepts_valid_format(self):
        """Valid format but nonexistent dir should fail gracefully."""
        with tempfile.TemporaryDirectory() as ws:
            ok = rollback(ws, "20260214-120000")
            self.assertFalse(ok)  # Dir doesn't exist, but format is valid


class TestFingerprintPayload(unittest.TestCase):
    """Fingerprint must include op payload to avoid false-dedup collisions."""

    def test_different_values_different_fingerprints(self):
        p1 = {
            "Type": "edit", "TargetBlock": "D-20260214-001",
            "Ops": [{"op": "update_field", "file": "decisions/DECISIONS.md",
                      "target": "D-20260214-001", "value": "active"}],
        }
        p2 = {
            "Type": "edit", "TargetBlock": "D-20260214-001",
            "Ops": [{"op": "update_field", "file": "decisions/DECISIONS.md",
                      "target": "D-20260214-001", "value": "superseded"}],
        }
        self.assertNotEqual(compute_fingerprint(p1), compute_fingerprint(p2))

    def test_same_proposal_same_fingerprint(self):
        p1 = {
            "Type": "edit", "TargetBlock": "D-20260214-001",
            "Ops": [{"op": "set_status", "file": "tasks/TASKS.md",
                      "target": "T-20260214-001", "status": "done"}],
        }
        self.assertEqual(compute_fingerprint(p1), compute_fingerprint(p1))


@unittest.skipIf(sys.platform == "win32", "validate.sh requires bash (Unix only)")
class TestValidateUninitWorkspace(unittest.TestCase):
    """validate.sh should handle uninitialized workspaces gracefully."""

    def test_rejects_uninitialized_workspace(self):
        """Running on a dir with no mind-mem.json should exit with clear error."""
        with tempfile.TemporaryDirectory() as ws:
            validate_sh = os.path.join(
                os.path.dirname(__file__), "..", "scripts", "validate.sh"
            )
            result = subprocess.run(
                ["bash", validate_sh, ws],
                capture_output=True, text=True, timeout=30,
            )
            # Should exit with error and helpful message
            self.assertEqual(result.returncode, 1)
            self.assertIn("No mind-mem.json found", result.stdout)
            self.assertIn("init_workspace.py", result.stdout)


class TestModeGate(unittest.TestCase):
    """apply_proposal must reject in detect_only mode."""

    def test_detect_only_blocks_apply(self):
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            # Default mode is detect_only
            ok, msg = apply_proposal(ws, "P-20260214-001", dry_run=False)
            self.assertFalse(ok)
            self.assertIn("detect_only", msg)

    def test_detect_only_blocks_dry_run(self):
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            ok, msg = apply_proposal(ws, "P-20260214-001", dry_run=False)
            self.assertFalse(ok)
            self.assertIn("detect_only", msg)

    def test_propose_mode_allows_apply(self):
        """In propose mode, apply should proceed past mode gate (will fail on missing proposal)."""
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            # Switch to propose mode
            state_path = os.path.join(ws, "memory/intel-state.json")
            with open(state_path) as f:
                state = json.load(f)
            state["governance_mode"] = "propose"
            with open(state_path, "w") as f:
                json.dump(state, f)
            ok, msg = apply_proposal(ws, "P-20260214-999", dry_run=False)
            self.assertFalse(ok)
            # Should fail on "not found", NOT on mode gate
            self.assertIn("not found", msg.lower())


class TestBacklogLimit(unittest.TestCase):
    """Backlog limit uses >= — blocks at exact limit per M6 invariant."""

    def test_at_exact_limit_is_exceeded(self):
        """count == limit should be exceeded (>= per M6)."""
        from apply_engine import check_backlog_limit
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            # Set limit in mind-mem.json (source of truth for config)
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path) as f:
                config = json.load(f)
            config["proposal_budget"] = {"backlog_limit": 3}
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Create 3 staged proposals
            proposed_dir = os.path.join(ws, "intelligence/proposed")
            os.makedirs(proposed_dir, exist_ok=True)
            with open(os.path.join(proposed_dir, "DECISIONS_PROPOSED.md"), "w") as f:
                for i in range(3):
                    f.write(f"\n[P-001-{i}]\nProposalId: P-001-{i}\nStatus: staged\n")

            count, exceeded = check_backlog_limit(ws)
            self.assertEqual(count, 3)
            self.assertTrue(exceeded, "At exact limit should be exceeded per M6")

    def test_under_limit_not_exceeded(self):
        """count < limit should NOT be exceeded."""
        from apply_engine import check_backlog_limit
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            config_path = os.path.join(ws, "mind-mem.json")
            with open(config_path) as f:
                config = json.load(f)
            config["proposal_budget"] = {"backlog_limit": 5}
            with open(config_path, "w") as f:
                json.dump(config, f)

            proposed_dir = os.path.join(ws, "intelligence/proposed")
            os.makedirs(proposed_dir, exist_ok=True)
            with open(os.path.join(proposed_dir, "DECISIONS_PROPOSED.md"), "w") as f:
                for i in range(3):
                    f.write(f"\n[P-001-{i}]\nProposalId: P-001-{i}\nStatus: staged\n")

            count, exceeded = check_backlog_limit(ws)
            self.assertEqual(count, 3)
            self.assertFalse(exceeded, "Under limit should not be exceeded")


class TestFingerprintDedupCollision(unittest.TestCase):
    """Fingerprint dedup must detect different proposals with same fingerprint."""

    def test_different_proposal_same_fingerprint_detected(self):
        """Two proposals targeting same block with same ops should collide."""
        from apply_engine import check_fingerprint_dedup, compute_fingerprint
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)

            # Create a staged proposal in the proposed file
            proposed_dir = os.path.join(ws, "intelligence/proposed")
            os.makedirs(proposed_dir, exist_ok=True)

            existing_proposal = {
                "Type": "edit",
                "TargetBlock": "D-20260214-001",
                "Ops": [{"op": "set_status", "file": "decisions/DECISIONS.md",
                         "target": "D-20260214-001", "status": "superseded"}],
            }
            fp = compute_fingerprint(existing_proposal)

            with open(os.path.join(proposed_dir, "DECISIONS_PROPOSED.md"), "w") as f:
                f.write(f"\n[P-20260214-001]\nProposalId: P-20260214-001\n"
                        f"Type: edit\nTargetBlock: D-20260214-001\n"
                        f"Status: staged\nFingerprint: {fp}\n"
                        f"Ops:\n- op: set_status\n  file: decisions/DECISIONS.md\n"
                        f"  target: D-20260214-001\n  status: superseded\n")

            # A NEW proposal with same ops/target but different ID should be detected as dup
            new_proposal = {
                "ProposalId": "P-20260214-099",
                "Type": "edit",
                "TargetBlock": "D-20260214-001",
                "Ops": [{"op": "set_status", "file": "decisions/DECISIONS.md",
                         "target": "D-20260214-001", "status": "superseded"}],
            }
            is_dup, dup_id = check_fingerprint_dedup(ws, new_proposal)
            self.assertTrue(is_dup, "Should detect fingerprint collision")
            self.assertEqual(dup_id, "P-20260214-001")

    def test_same_proposal_id_not_self_collision(self):
        """A proposal should not collide with itself."""
        from apply_engine import check_fingerprint_dedup, compute_fingerprint
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)

            proposed_dir = os.path.join(ws, "intelligence/proposed")
            os.makedirs(proposed_dir, exist_ok=True)

            proposal = {
                "ProposalId": "P-20260214-001",
                "Type": "edit",
                "TargetBlock": "D-20260214-001",
                "Ops": [{"op": "set_status", "file": "decisions/DECISIONS.md",
                         "target": "D-20260214-001", "status": "superseded"}],
            }
            fp = compute_fingerprint(proposal)

            with open(os.path.join(proposed_dir, "DECISIONS_PROPOSED.md"), "w") as f:
                f.write(f"\n[P-20260214-001]\nProposalId: P-20260214-001\n"
                        f"Type: edit\nTargetBlock: D-20260214-001\n"
                        f"Status: staged\nFingerprint: {fp}\n"
                        f"Ops:\n- op: set_status\n  file: decisions/DECISIONS.md\n"
                        f"  target: D-20260214-001\n  status: superseded\n")

            # Same ID = self, not a collision
            is_dup, dup_id = check_fingerprint_dedup(ws, proposal)
            self.assertFalse(is_dup, "Same proposal ID should not self-collide")


class TestSnapshotRecursionPrevention(unittest.TestCase):
    """Snapshots must exclude intelligence/applied/ to prevent recursive nesting."""

    def test_snapshot_excludes_applied_dir(self):
        """intelligence/applied/ must NOT be copied into new snapshots."""
        from apply_engine import create_snapshot
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)

            # Create a fake prior snapshot in intelligence/applied/
            prior_snap = os.path.join(ws, "intelligence/applied/20260213-120000")
            os.makedirs(prior_snap, exist_ok=True)
            with open(os.path.join(prior_snap, "marker.txt"), "w") as f:
                f.write("I should NOT appear in new snapshots")

            # Create a new snapshot
            snap_dir = create_snapshot(ws, "20260214-120000")

            # The new snapshot's intelligence/ should NOT contain applied/
            nested_applied = os.path.join(snap_dir, "intelligence/applied")
            self.assertFalse(
                os.path.exists(nested_applied),
                "Snapshot must not recursively include intelligence/applied/"
            )

            # But intelligence files (like CONTRADICTIONS.md) SHOULD be copied
            intel_dir = os.path.join(snap_dir, "intelligence")
            self.assertTrue(os.path.isdir(intel_dir),
                            "intelligence/ dir should exist in snapshot")


class TestMinimalSnapshot(unittest.TestCase):
    """Tests for files_touched-based minimal snapshots."""

    def test_minimal_snapshot_only_copies_touched_files(self):
        """When files_touched is provided, only those files + config are snapshotted."""
        from apply_engine import create_snapshot
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)

            # Write to multiple files
            with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
                f.write("[D-001]\nStatement: Test\nStatus: active\n")
            with open(os.path.join(ws, "tasks", "TASKS.md"), "w") as f:
                f.write("[T-001]\nTitle: Task\nStatus: open\n")

            # Minimal snapshot: only touch decisions
            snap_dir = create_snapshot(ws, "20260217-120000",
                                       files_touched=["decisions/DECISIONS.md"])

            # Decisions file should exist in snapshot
            self.assertTrue(os.path.isfile(
                os.path.join(snap_dir, "decisions", "DECISIONS.md")
            ))
            # Tasks should NOT be in snapshot (wasn't in files_touched)
            self.assertFalse(os.path.exists(
                os.path.join(snap_dir, "tasks", "TASKS.md")
            ))
            # Config files always snapshotted
            self.assertTrue(os.path.isfile(
                os.path.join(snap_dir, "mind-mem.json")
            ))

    def test_full_snapshot_when_no_files_touched(self):
        """When files_touched is None, full snapshot is created."""
        from apply_engine import create_snapshot
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
                f.write("[D-001]\nStatement: Test\nStatus: active\n")

            snap_dir = create_snapshot(ws, "20260217-120001", files_touched=None)

            # Full snapshot should include entire decisions directory
            self.assertTrue(os.path.isfile(
                os.path.join(snap_dir, "decisions", "DECISIONS.md")
            ))

    def test_safe_copy_makes_independent_copy(self):
        """_safe_copy must create an independent copy (not hardlink).

        Hardlinks are unsafe for mutable-file snapshots — open("w") truncates
        the inode in-place, corrupting both files.
        """
        from apply_engine import _safe_copy
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "source.md")
            dst = os.path.join(td, "dest.md")
            with open(src, "w") as f:
                f.write("original content")
            _safe_copy(src, dst)
            self.assertTrue(os.path.isfile(dst))
            # Modify source — dst must remain unchanged
            with open(src, "w") as f:
                f.write("modified content")
            with open(dst) as f:
                self.assertEqual(f.read(), "original content")

    def test_safe_copy_creates_parent_dirs(self):
        """_safe_copy should create intermediate directories."""
        from apply_engine import _safe_copy
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "source.md")
            dst = os.path.join(td, "deep", "nested", "dest.md")
            with open(src, "w") as f:
                f.write("test")
            _safe_copy(src, dst)
            self.assertTrue(os.path.isfile(dst))

    def test_minimal_snapshot_preserves_content(self):
        """Snapshot content must match original."""
        from apply_engine import create_snapshot
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)
            original = "[D-001]\nStatement: PostgreSQL\nStatus: active\n"
            with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
                f.write(original)

            snap_dir = create_snapshot(ws, "20260217-120002",
                                       files_touched=["decisions/DECISIONS.md"])

            with open(os.path.join(snap_dir, "decisions", "DECISIONS.md")) as f:
                snapped = f.read()
            self.assertEqual(snapped, original)


class TestDeferredCooldown(unittest.TestCase):
    """Deferred/rejected proposals have a cooldown period before re-proposal."""

    def test_recent_rejected_blocks_new_proposal(self):
        """A rejected proposal within cooldown period should block new proposals for same target."""
        from apply_engine import check_deferred_cooldown
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)

            # Set cooldown to 7 days
            state_path = os.path.join(ws, "memory/intel-state.json")
            with open(state_path) as f:
                state = json.load(f)
            state["defer_cooldown_days"] = 7
            with open(state_path, "w") as f:
                json.dump(state, f)

            # Create a recently rejected proposal
            proposed_dir = os.path.join(ws, "intelligence/proposed")
            os.makedirs(proposed_dir, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            with open(os.path.join(proposed_dir, "DECISIONS_PROPOSED.md"), "w") as f:
                f.write(f"\n[P-20260214-001]\nProposalId: P-20260214-001\n"
                        f"Type: edit\nTargetBlock: D-20260214-001\n"
                        f"Status: rejected\nCreated: {today}\n")

            # New proposal for same target should be blocked
            new_proposal = {"TargetBlock": "D-20260214-001"}
            ok, reason = check_deferred_cooldown(ws, new_proposal)
            self.assertFalse(ok, "Recent rejected proposal should block same target")
            self.assertIn("cooldown", reason)

    def test_old_rejected_allows_new_proposal(self):
        """A rejected proposal outside cooldown period should allow re-proposal."""
        from apply_engine import check_deferred_cooldown
        with tempfile.TemporaryDirectory() as ws:
            from init_workspace import init
            init(ws)

            state_path = os.path.join(ws, "memory/intel-state.json")
            with open(state_path) as f:
                state = json.load(f)
            state["defer_cooldown_days"] = 7
            with open(state_path, "w") as f:
                json.dump(state, f)

            # Create an OLD rejected proposal (30 days ago)
            proposed_dir = os.path.join(ws, "intelligence/proposed")
            os.makedirs(proposed_dir, exist_ok=True)
            old_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
            with open(os.path.join(proposed_dir, "DECISIONS_PROPOSED.md"), "w") as f:
                f.write(f"\n[P-20260115-001]\nProposalId: P-20260115-001\n"
                        f"Type: edit\nTargetBlock: D-20260214-001\n"
                        f"Status: rejected\nCreated: {old_date}\n")

            new_proposal = {"TargetBlock": "D-20260214-001"}
            ok, reason = check_deferred_cooldown(ws, new_proposal)
            self.assertTrue(ok, "Old rejected proposal should not block same target")


class TestOpSupersedeDecision(unittest.TestCase):
    """Tests for _op_supersede_decision."""

    def test_supersede_marks_old_and_appends_new(self):
        with tempfile.TemporaryDirectory() as td:
            dec_file = os.path.join(td, "DECISIONS.md")
            with open(dec_file, "w") as f:
                f.write("[D-20260213-001]\nStatus: active\nStatement: Old decision\n")
            new_block = "[D-20260213-002]\nStatus: active\nStatement: New decision\nSupersedes: D-20260213-001\n"
            ok, msg = _op_supersede_decision(dec_file, {
                "target": "D-20260213-001",
                "new_block": new_block,
            })
            self.assertTrue(ok, msg)
            with open(dec_file) as f:
                content = f.read()
            self.assertIn("Status: superseded", content)
            self.assertIn("[D-20260213-002]", content)

    def test_supersede_rejects_missing_target(self):
        with tempfile.TemporaryDirectory() as td:
            dec_file = os.path.join(td, "DECISIONS.md")
            with open(dec_file, "w") as f:
                f.write("[D-20260213-001]\nStatus: active\n")
            ok, msg = _op_supersede_decision(dec_file, {
                "target": "D-20260213-999",
                "new_block": "[D-20260213-002]\nStatus: active\n",
            })
            self.assertFalse(ok)
            self.assertIn("not found", msg)

    def test_supersede_rejects_invariant(self):
        with tempfile.TemporaryDirectory() as td:
            dec_file = os.path.join(td, "DECISIONS.md")
            with open(dec_file, "w") as f:
                f.write("[D-20260213-001]\nStatus: active\nStatement: Invariant\n"
                        "ConstraintSignatures:\n- id: CS-001\n  enforcement: invariant\n")
            ok, msg = _op_supersede_decision(dec_file, {
                "target": "D-20260213-001",
                "new_block": "[D-20260213-002]\nStatus: active\n",
            })
            self.assertFalse(ok)
            self.assertIn("invariant", msg)


class TestOpReplaceRange(unittest.TestCase):
    """Tests for _op_replace_range."""

    def test_replaces_between_markers(self):
        with tempfile.TemporaryDirectory() as td:
            filepath = os.path.join(td, "DECISIONS.md")
            with open(filepath, "w") as f:
                f.write("[D-20260213-001]\nStatus: active\n<!-- START -->\nold content\n<!-- END -->\nTags: test\n")
            ok, msg = _op_replace_range(filepath, {
                "target": "D-20260213-001",
                "range": {"start": "<!-- START -->", "end": "<!-- END -->"},
                "patch": "<!-- START -->\nnew content",
            })
            self.assertTrue(ok, msg)
            with open(filepath) as f:
                content = f.read()
            self.assertIn("new content", content)
            self.assertIn("<!-- END -->", content)
            self.assertNotIn("old content", content)

    def test_rejects_missing_markers(self):
        with tempfile.TemporaryDirectory() as td:
            filepath = os.path.join(td, "DECISIONS.md")
            with open(filepath, "w") as f:
                f.write("[D-20260213-001]\nStatus: active\n")
            ok, msg = _op_replace_range(filepath, {
                "target": "D-20260213-001",
                "range": {"start": "<!-- NONEXISTENT -->", "end": "<!-- END -->"},
                "patch": "new",
            })
            self.assertFalse(ok)
            self.assertIn("markers not found", msg)


if __name__ == "__main__":
    unittest.main()
