#!/usr/bin/env python3
"""Tests for namespaces.py — zero external deps (stdlib unittest)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from namespaces import (
    NamespaceManager,
    SharedLedger,
    init_multi_agent_workspace,
    NAMESPACE_DIRS,
)


class TestNamespaceManager(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def _write_acl(self, acl):
        with open(os.path.join(self.td, "mind-mem-acl.json"), "w") as f:
            json.dump(acl, f)

    def test_no_agent_id_has_full_access(self):
        ns = NamespaceManager(self.td, agent_id=None)
        self.assertTrue(ns.can_read("shared/decisions/DECISIONS.md"))
        self.assertTrue(ns.can_write("shared/decisions/DECISIONS.md"))
        self.assertTrue(ns.can_read("agents/coder-1/memory/MEMORY.md"))
        self.assertTrue(ns.can_write("agents/coder-1/memory/MEMORY.md"))

    def test_default_acl_read_only_shared(self):
        ns = NamespaceManager(self.td, agent_id="some-agent")
        self.assertTrue(ns.can_read("shared/decisions/DECISIONS.md"))
        self.assertFalse(ns.can_write("shared/decisions/DECISIONS.md"))

    def test_custom_acl_exact_match(self):
        acl = {
            "default_policy": "read",
            "agents": {
                "coder-1": {
                    "namespaces": ["shared", "agents/coder-1"],
                    "write": ["agents/coder-1"],
                    "read": ["shared", "agents/coder-1"],
                },
                "*": {"namespaces": ["shared"], "write": [], "read": ["shared"]},
            },
        }
        self._write_acl(acl)
        ns = NamespaceManager(self.td, agent_id="coder-1")
        self.assertTrue(ns.can_write("agents/coder-1/decisions/DECISIONS.md"))
        self.assertFalse(ns.can_write("shared/decisions/DECISIONS.md"))
        self.assertTrue(ns.can_read("shared/decisions/DECISIONS.md"))

    def test_custom_acl_pattern_match(self):
        acl = {
            "default_policy": "read",
            "agents": {
                "reviewer-*": {
                    "namespaces": ["shared"],
                    "write": [],
                    "read": ["shared"],
                },
                "*": {"namespaces": ["shared"], "write": [], "read": ["shared"]},
            },
        }
        self._write_acl(acl)
        ns = NamespaceManager(self.td, agent_id="reviewer-42")
        self.assertTrue(ns.can_read("shared/tasks/TASKS.md"))
        self.assertFalse(ns.can_write("shared/tasks/TASKS.md"))

    def test_wildcard_fallback(self):
        ns = NamespaceManager(self.td, agent_id="unknown-agent")
        self.assertTrue(ns.can_read("shared/entities/projects.md"))
        self.assertFalse(ns.can_write("shared/entities/projects.md"))

    def test_init_namespace_creates_dirs(self):
        ns = NamespaceManager(self.td)
        created = ns.init_namespace("shared")
        for d in NAMESPACE_DIRS:
            self.assertTrue(os.path.isdir(os.path.join(self.td, "shared", d)))
        self.assertEqual(len(created), len(NAMESPACE_DIRS))

    def test_init_agent_creates_agent_dirs(self):
        ns = NamespaceManager(self.td)
        ns.init_agent("coder-1")
        self.assertTrue(os.path.isdir(os.path.join(self.td, "agents", "coder-1", "decisions")))
        self.assertTrue(os.path.isdir(os.path.join(self.td, "agents", "coder-1", "memory")))

    def test_list_agents_empty(self):
        ns = NamespaceManager(self.td)
        self.assertEqual(ns.list_agents(), [])

    def test_list_agents_after_init(self):
        ns = NamespaceManager(self.td)
        ns.init_agent("alpha")
        ns.init_agent("beta")
        agents = ns.list_agents()
        self.assertEqual(agents, ["alpha", "beta"])

    def test_get_agent_namespace(self):
        ns = NamespaceManager(self.td, agent_id="coder-1")
        self.assertEqual(ns.get_agent_namespace(), "agents/coder-1")

    def test_get_agent_namespace_none(self):
        ns = NamespaceManager(self.td, agent_id=None)
        self.assertIsNone(ns.get_agent_namespace())

    def test_resolve_corpus_paths(self):
        ns = NamespaceManager(self.td)
        ns.init_namespace("shared")
        # Create a test file
        decisions_path = os.path.join(self.td, "shared", "decisions", "DECISIONS.md")
        with open(decisions_path, "w") as f:
            f.write("[D-20260101-001]\nStatement: Test\n")

        paths = ns.resolve_corpus_paths("decisions/DECISIONS.md")
        self.assertEqual(len(paths), 1)
        # Normalize separators for cross-platform comparison
        normalized = paths[0].replace(os.sep, "/")
        self.assertTrue(normalized.endswith("shared/decisions/DECISIONS.md"))

    def test_path_backslash_normalization(self):
        ns = NamespaceManager(self.td, agent_id=None)
        self.assertTrue(ns.can_read("shared\\decisions\\DECISIONS.md"))

    def test_corrupted_acl_falls_back(self):
        acl_path = os.path.join(self.td, "mind-mem-acl.json")
        with open(acl_path, "w") as f:
            f.write("{bad json")
        ns = NamespaceManager(self.td, agent_id="test")
        # Should fall back to DEFAULT_ACL
        self.assertTrue(ns.can_read("shared/decisions/DECISIONS.md"))


class TestSharedLedger(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        intel_dir = os.path.join(self.td, "shared", "intelligence")
        os.makedirs(intel_dir)
        ledger_path = os.path.join(intel_dir, "LEDGER.md")
        with open(ledger_path, "w") as f:
            f.write("# Shared Fact Ledger\n\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_append_fact(self):
        ledger = SharedLedger(self.td)
        result = ledger.append_fact(
            {"text": "PostgreSQL is the primary database", "confidence": "high", "type": "observation"},
            source_agent="coder-1",
        )
        self.assertTrue(result)
        with open(ledger.ledger_path) as f:
            content = f.read()
        self.assertIn("PostgreSQL is the primary database", content)
        self.assertIn("coder-1", content)

    def test_dedup_prevents_duplicate(self):
        ledger = SharedLedger(self.td)
        ledger.append_fact(
            {"text": "PostgreSQL is the primary database", "confidence": "high"},
            source_agent="coder-1",
        )
        result = ledger.append_fact(
            {"text": "PostgreSQL is the primary database", "confidence": "high"},
            source_agent="coder-2",
        )
        self.assertFalse(result)

    def test_append_fact_no_ledger_dir(self):
        ledger = SharedLedger("/nonexistent/path")
        result = ledger.append_fact({"text": "test"}, source_agent="x")
        self.assertFalse(result)

    def test_get_facts_empty(self):
        ledger = SharedLedger(self.td)
        facts = ledger.get_facts()
        self.assertEqual(facts, [])

    def test_get_facts_after_append(self):
        ledger = SharedLedger(self.td)
        ledger.append_fact(
            {"text": "Fact one", "confidence": "high", "type": "convention"},
            source_agent="agent-1",
        )
        facts = ledger.get_facts()
        self.assertGreaterEqual(len(facts), 1)

    def test_get_facts_filtered_by_status(self):
        ledger = SharedLedger(self.td)
        ledger.append_fact(
            {"text": "Pending fact", "confidence": "medium"},
            source_agent="agent-1",
        )
        pending = ledger.get_facts(status="pending-review")
        approved = ledger.get_facts(status="approved")
        self.assertGreaterEqual(len(pending), 1)
        self.assertEqual(len(approved), 0)


class TestInitMultiAgentWorkspace(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_creates_shared_and_agents(self):
        init_multi_agent_workspace(self.td, agents=["coder-1", "reviewer-1"])
        self.assertTrue(os.path.isdir(os.path.join(self.td, "shared", "decisions")))
        self.assertTrue(os.path.isdir(os.path.join(self.td, "agents", "coder-1", "decisions")))
        self.assertTrue(os.path.isdir(os.path.join(self.td, "agents", "reviewer-1", "decisions")))

    def test_creates_default_acl(self):
        init_multi_agent_workspace(self.td, agents=["coder-1"])
        acl_path = os.path.join(self.td, "mind-mem-acl.json")
        self.assertTrue(os.path.isfile(acl_path))
        with open(acl_path) as f:
            acl = json.load(f)
        self.assertIn("coder-1", acl["agents"])

    def test_creates_shared_ledger(self):
        init_multi_agent_workspace(self.td)
        ledger = os.path.join(self.td, "shared", "intelligence", "LEDGER.md")
        self.assertTrue(os.path.isfile(ledger))

    def test_no_overwrite_existing_acl(self):
        # Pre-create ACL
        acl_path = os.path.join(self.td, "mind-mem-acl.json")
        with open(acl_path, "w") as f:
            json.dump({"custom": True}, f)
        init_multi_agent_workspace(self.td, agents=["x"])
        with open(acl_path) as f:
            acl = json.load(f)
        self.assertTrue(acl.get("custom"))

    def test_no_agents_still_creates_shared(self):
        init_multi_agent_workspace(self.td)
        self.assertTrue(os.path.isdir(os.path.join(self.td, "shared", "decisions")))


if __name__ == "__main__":
    unittest.main()
