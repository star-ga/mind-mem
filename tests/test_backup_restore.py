#!/usr/bin/env python3
"""Tests for backup_restore.py — zero external deps (stdlib unittest)."""

import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from backup_restore import WAL, backup_workspace, export_jsonl, restore_workspace


class TestWAL(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_begin_creates_entry(self):
        wal = WAL(self.td)
        entry_id = wal.begin("write", os.path.join(self.td, "test.md"), "new content")
        entry_path = os.path.join(self.td, ".mind-mem-wal", f"{entry_id}.json")
        self.assertTrue(os.path.isfile(entry_path))
        with open(entry_path) as f:
            entry = json.load(f)
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["operation"], "write")

    def test_commit_removes_entry(self):
        wal = WAL(self.td)
        entry_id = wal.begin("write", os.path.join(self.td, "test.md"), "content")
        wal.commit(entry_id)
        entry_path = os.path.join(self.td, ".mind-mem-wal", f"{entry_id}.json")
        self.assertFalse(os.path.isfile(entry_path))

    def test_begin_backs_up_existing_file(self):
        target = os.path.join(self.td, "existing.md")
        with open(target, "w") as f:
            f.write("original content")
        wal = WAL(self.td)
        entry_id = wal.begin("write", target, "new content")
        backup_path = os.path.join(self.td, ".mind-mem-wal", f"{entry_id}.backup")
        self.assertTrue(os.path.isfile(backup_path))
        with open(backup_path) as f:
            self.assertEqual(f.read(), "original content")

    def test_rollback_restores_backup(self):
        target = os.path.join(self.td, "rollback_test.md")
        with open(target, "w") as f:
            f.write("original")
        wal = WAL(self.td)
        entry_id = wal.begin("write", target, "replacement")
        # Simulate the write happening
        with open(target, "w") as f:
            f.write("replacement")
        # Rollback
        result = wal.rollback(entry_id)
        self.assertTrue(result)
        with open(target) as f:
            self.assertEqual(f.read(), "original")

    def test_rollback_removes_new_file(self):
        target = os.path.join(self.td, "new_file.md")
        wal = WAL(self.td)
        entry_id = wal.begin("write", target, "new content")
        # Simulate the write
        with open(target, "w") as f:
            f.write("new content")
        wal.rollback(entry_id)
        self.assertFalse(os.path.isfile(target))

    def test_rollback_nonexistent_entry(self):
        wal = WAL(self.td)
        result = wal.rollback("nonexistent-id")
        self.assertFalse(result)

    def test_replay_rolls_back_pending(self):
        target = os.path.join(self.td, "crash_test.md")
        with open(target, "w") as f:
            f.write("before crash")
        wal = WAL(self.td)
        wal.begin("write", target, "during crash")
        # Simulate crash: file was overwritten but WAL entry still pending
        with open(target, "w") as f:
            f.write("during crash")
        replayed = wal.replay()
        self.assertEqual(replayed, 1)
        with open(target) as f:
            self.assertEqual(f.read(), "before crash")

    def test_replay_skips_committed(self):
        wal = WAL(self.td)
        entry_id = wal.begin("write", os.path.join(self.td, "test.md"), "content")
        wal.commit(entry_id)
        replayed = wal.replay()
        self.assertEqual(replayed, 0)

    def test_pending_count(self):
        wal = WAL(self.td)
        self.assertEqual(wal.pending_count(), 0)
        wal.begin("write", os.path.join(self.td, "a.md"), "a")
        wal.begin("write", os.path.join(self.td, "b.md"), "b")
        self.assertEqual(wal.pending_count(), 2)

    def test_commit_cleans_backup_file(self):
        target = os.path.join(self.td, "cleanup.md")
        with open(target, "w") as f:
            f.write("original")
        wal = WAL(self.td)
        entry_id = wal.begin("write", target, "new")
        backup_path = os.path.join(self.td, ".mind-mem-wal", f"{entry_id}.backup")
        self.assertTrue(os.path.isfile(backup_path))
        wal.commit(entry_id)
        self.assertFalse(os.path.isfile(backup_path))


class TestBackupWorkspace(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        # Create a minimal workspace
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, "tasks"))
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write("[D-20260101-001]\nStatement: Test\nStatus: active\n")
        with open(os.path.join(self.td, "mind-mem.json"), "w") as f:
            json.dump({"version": "0.5.0"}, f)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_creates_tarball(self):
        output = os.path.join(self.td, "backup.tar.gz")
        result = backup_workspace(self.td, output)
        self.assertTrue(os.path.isfile(result))
        self.assertGreater(os.path.getsize(result), 0)

    def test_tarball_contains_decisions(self):
        output = os.path.join(self.td, "backup.tar.gz")
        backup_workspace(self.td, output)
        with tarfile.open(output, "r:gz") as tar:
            names = tar.getnames()
        self.assertIn("decisions/DECISIONS.md", names)

    def test_tarball_contains_config(self):
        output = os.path.join(self.td, "backup.tar.gz")
        backup_workspace(self.td, output)
        with tarfile.open(output, "r:gz") as tar:
            names = tar.getnames()
        self.assertIn("mind-mem.json", names)

    def test_skips_missing_dirs(self):
        output = os.path.join(self.td, "backup.tar.gz")
        backup_workspace(self.td, output)
        with tarfile.open(output, "r:gz") as tar:
            names = tar.getnames()
        # 'entities' dir doesn't exist, should not appear
        entity_files = [n for n in names if n.startswith("entities/")]
        self.assertEqual(len(entity_files), 0)


class TestExportJsonl(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\nStatement: First\nStatus: active\n\n---\n\n"
                "[D-20260101-002]\nStatement: Second\nStatus: active\n"
            )

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_exports_blocks(self):
        output = os.path.join(self.td, "export.jsonl")
        count = export_jsonl(self.td, output)
        self.assertEqual(count, 2)

    def test_jsonl_is_valid(self):
        output = os.path.join(self.td, "export.jsonl")
        export_jsonl(self.td, output)
        with open(output) as f:
            for line in f:
                obj = json.loads(line.strip())
                self.assertIn("_source", obj)
                self.assertEqual(obj["_source"], "decisions")

    def test_empty_workspace(self):
        empty = tempfile.mkdtemp()
        try:
            output = os.path.join(empty, "export.jsonl")
            count = export_jsonl(empty, output)
            self.assertEqual(count, 0)
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class TestRestoreWorkspace(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.restore_td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)
        shutil.rmtree(self.restore_td, ignore_errors=True)

    def _create_backup(self):
        os.makedirs(os.path.join(self.td, "decisions"))
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write("[D-20260101-001]\nStatement: Test\n")
        output = os.path.join(self.td, "backup.tar.gz")
        backup_workspace(self.td, output)
        return output

    def test_restore_creates_files(self):
        backup = self._create_backup()
        result = restore_workspace(self.restore_td, backup)
        self.assertGreater(result["restored"], 0)
        self.assertTrue(os.path.isfile(
            os.path.join(self.restore_td, "decisions", "DECISIONS.md")
        ))

    def test_restore_detects_conflicts(self):
        backup = self._create_backup()
        # Pre-create conflicting file
        os.makedirs(os.path.join(self.restore_td, "decisions"))
        with open(os.path.join(self.restore_td, "decisions", "DECISIONS.md"), "w") as f:
            f.write("existing content")
        result = restore_workspace(self.restore_td, backup, force=False)
        self.assertGreater(result["skipped"], 0)

    def test_restore_force_overwrites(self):
        backup = self._create_backup()
        # Pre-create conflicting file
        os.makedirs(os.path.join(self.restore_td, "decisions"))
        with open(os.path.join(self.restore_td, "decisions", "DECISIONS.md"), "w") as f:
            f.write("existing content")
        result = restore_workspace(self.restore_td, backup, force=True)
        self.assertEqual(result["skipped"], 0)


class TestRestoreSecurity(unittest.TestCase):
    """Security regression tests for tar restore path traversal."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.restore_td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)
        shutil.rmtree(self.restore_td, ignore_errors=True)

    def _make_malicious_tar(self, members):
        """Build a tar.gz with crafted member names."""
        tar_path = os.path.join(self.td, "evil.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for name, content in members:
                import io
                data = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return tar_path

    def test_blocks_dotdot_traversal(self):
        tar_path = self._make_malicious_tar([
            ("../escape.txt", "escaped!"),
            ("good.txt", "safe content"),
        ])
        result = restore_workspace(self.restore_td, tar_path, force=True)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(result["restored"], 1)
        # Escaped file must NOT exist outside workspace
        escaped = os.path.join(self.restore_td, "..", "escape.txt")
        self.assertFalse(os.path.isfile(os.path.abspath(escaped)))
        # Good file should exist
        self.assertTrue(os.path.isfile(os.path.join(self.restore_td, "good.txt")))

    def test_blocks_absolute_path(self):
        tar_path = self._make_malicious_tar([
            ("/tmp/evil_abs.txt", "absolute path attack"),
        ])
        result = restore_workspace(self.restore_td, tar_path, force=True)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(result["restored"], 0)
        self.assertFalse(os.path.isfile("/tmp/evil_abs.txt"))

    def test_blocks_symlink_member(self):
        """Tar with a symlink member should be rejected."""
        tar_path = os.path.join(self.td, "symlink.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add a symlink pointing outside workspace
            info = tarfile.TarInfo(name="link.txt")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        result = restore_workspace(self.restore_td, tar_path, force=True)
        self.assertEqual(result["blocked"], 1)
        self.assertFalse(os.path.exists(os.path.join(self.restore_td, "link.txt")))

    def test_blocks_prefix_trick(self):
        """Member whose resolved path escapes via prefix match."""
        # If workspace is /tmp/ws, a member named "../ws_evil/x" resolves
        # to /tmp/ws_evil/x which starts with /tmp/ws but is NOT inside it.
        tar_path = self._make_malicious_tar([
            ("../ws_evil/payload.txt", "prefix trick!"),
        ])
        result = restore_workspace(self.restore_td, tar_path, force=True)
        self.assertEqual(result["blocked"], 1)

    def test_normal_restore_still_works(self):
        """Ensure security checks don't break legitimate restores."""
        tar_path = self._make_malicious_tar([
            ("decisions/DECISIONS.md", "[D-001]\nStatement: Test\n"),
            ("mind-mem.json", '{"version": "1.0"}'),
        ])
        result = restore_workspace(self.restore_td, tar_path, force=True)
        self.assertEqual(result["blocked"], 0)
        self.assertEqual(result["restored"], 2)


if __name__ == "__main__":
    unittest.main()
