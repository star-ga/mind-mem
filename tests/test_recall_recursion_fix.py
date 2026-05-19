#!/usr/bin/env python3
"""Regression tests for the recall ↔ query_index mutual recursion bug.

Bug: When a workspace is deleted (shutil.rmtree) between two benchmark runs,
the _conn_managers module-level cache in sqlite_index.py retains the old
ConnectionManager.  On the next run, build_index() reuses the stale
ConnectionManager whose write connection points at the deleted file's inode.
The SQLite file is NOT recreated on disk.  query_index() then sees
os.path.isfile(db_path) == False and calls recall() as a fallback.  recall()
sees backend="sqlite" and calls query_index() again.  This produces unbounded
mutual recursion, RecursionError at the default limit, silently swallowed by
a broad "except Exception" in the multi-hop sub-query path.

Fix (sqlite_index._get_conn_manager): evict the cached ConnectionManager when
its DB file no longer exists on disk, so build_index() creates a fresh file.

Defense-in-depth (sqlite_index.query_index): per-thread re-entrancy guard
returns [] instead of entering the recall↔query_index cycle.

Loud-failure fix (_recall_core): except RecursionError must not be caught by
broad "except Exception" handlers in the sub-query and prefetch paths.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Ensure the src layout is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


DECISIONS_CONTENT = """\
[SESSION-abc]
Statement: (user) Alice mentioned she likes hiking and mountain trails.
Date: 2024-03-01
Status: active

---

[SESSION-def]
Statement: (assistant) Bob talked about his job at the tech startup.
Date: 2024-03-02
Status: active
"""


def _write_workspace(tmp: str, question_id: str = "q1", caps_on: bool = True) -> str:
    """Create a minimal workspace under *tmp* and write config."""
    ws = os.path.join(tmp, f"cp_{question_id}")
    d = os.path.join(ws, "decisions")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(DECISIONS_CONTENT)
    cfg: dict = {
        "recall": {
            "backend": "sqlite",
            "knee_cutoff": False,
            "cross_encoder": {"enabled": False, "auto_enable": False},
        }
    }
    if not caps_on:
        cfg["recall"]["dedup"] = {
            "enabled": False,
            "type_cap_enabled": False,
            "source_cap_enabled": False,
            "cosine_enabled": False,
            "best_per_source": False,
        }
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return ws


class TestRecallRecursionFix(unittest.TestCase):
    """Regression tests for the recall↔query_index mutual recursion bug."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _recall(self, ws: str, query: str = "hiking mountain"):
        # Import fresh each call so module-level caches are exercised
        from mind_mem.recall import recall
        return recall(ws, query, limit=10, active_only=False)

    def _build(self, ws: str):
        from mind_mem.sqlite_index import build_index
        build_index(ws, incremental=False)

    def test_no_recursion_after_workspace_delete_and_recreate(self):
        """Core regression: build→recall→rmtree→rebuild→recall must not recurse."""
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(300)  # would blow up before the fix
        try:
            ws = _write_workspace(self.tmp, "q_rerec", caps_on=True)
            self._build(ws)
            results1 = self._recall(ws, "hiking mountain")
            self.assertIsInstance(results1, list, "First recall must return a list")

            # Simulate test-harness teardown
            shutil.rmtree(ws, ignore_errors=True)

            # Second run with same workspace path (same question_id)
            ws2 = _write_workspace(self.tmp, "q_rerec", caps_on=False)
            self.assertEqual(ws, ws2, "workspace paths must match for the bug to trigger")
            self._build(ws2)

            # This call would RecursionError before the fix
            try:
                results2 = self._recall(ws2, "hiking mountain")
            except RecursionError:
                self.fail(
                    "RecursionError raised: the recall↔query_index mutual recursion "
                    "bug is not fixed.  Check _get_conn_manager in sqlite_index.py."
                )
            self.assertIsInstance(results2, list, "Second recall must return a list")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_db_file_exists_after_second_build(self):
        """After workspace delete + rebuild, the DB file must exist on disk."""
        from mind_mem.sqlite_index import _db_path

        ws = _write_workspace(self.tmp, "q_dbfile", caps_on=True)
        self._build(ws)
        db = _db_path(ws)
        self.assertTrue(os.path.isfile(db), "DB must exist after first build")

        shutil.rmtree(ws, ignore_errors=True)
        self.assertFalse(os.path.isfile(db), "DB must be gone after rmtree")

        ws2 = _write_workspace(self.tmp, "q_dbfile", caps_on=False)
        self._build(ws2)
        self.assertTrue(os.path.isfile(db), "DB must exist again after second build")

    def test_reentrant_guard_returns_empty_not_recursive(self):
        """The re-entrancy guard in query_index must return [] not recurse."""
        import sys

        from mind_mem.sqlite_index import _query_index_active, query_index

        ws = _write_workspace(self.tmp, "q_reent", caps_on=True)
        # Do NOT build index — so os.path.isfile(db_path) is False
        # Simulate the re-entrant state
        active = getattr(_query_index_active, "workspaces", None)
        if active is None:
            active = set()
            _query_index_active.workspaces = active
        active.add(ws)
        try:
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(50)
            try:
                result = query_index(ws, "hiking mountain")
                self.assertEqual(result, [], "Re-entrant call must return []")
            except RecursionError:
                self.fail("Re-entrancy guard did not stop the recursion")
            finally:
                sys.setrecursionlimit(old_limit)
        finally:
            active.discard(ws)

    def test_multiple_questions_caps_on_then_off_no_recursion(self):
        """Simulate the full _capfix_probe.py loop: N questions × caps_on, then caps_off."""
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(300)
        questions = [
            ("q001", "hiking mountain trails"),
            ("q002", "job startup technology"),
        ]
        try:
            for caps_on in (True, False):
                for qid, query in questions:
                    ws = _write_workspace(self.tmp, qid, caps_on=caps_on)
                    self._build(ws)
                    try:
                        results = self._recall(ws, query)
                        self.assertIsInstance(results, list)
                    except RecursionError:
                        self.fail(
                            f"RecursionError on qid={qid} caps_on={caps_on}: "
                            "mutual recursion bug is not fixed"
                        )
                    shutil.rmtree(ws, ignore_errors=True)
        finally:
            sys.setrecursionlimit(old_limit)


if __name__ == "__main__":
    unittest.main()
