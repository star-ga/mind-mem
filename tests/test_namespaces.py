#!/usr/bin/env python3
"""Tests for namespaces.py — zero external deps (stdlib unittest)."""

import json
import os
import shutil
import tempfile
import unittest

from mind_mem.namespaces import (
    NAMESPACE_DIRS,
    InvalidAgentIdError,
    NamespaceManager,
    SharedLedger,
    init_multi_agent_workspace,
)


class TestNamespaceManager(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.td, ignore_errors=True)

    def _write_acl(self, acl):
        with open(os.path.join(self.td, "mind-mem-acl.json"), "w", encoding="utf-8") as f:
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
        with open(decisions_path, "w", encoding="utf-8") as f:
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
        with open(acl_path, "w", encoding="utf-8") as f:
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
        with open(ledger_path, "w", encoding="utf-8") as f:
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
        with open(ledger.ledger_path, encoding="utf-8") as f:
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
        with open(acl_path, encoding="utf-8") as f:
            acl = json.load(f)
        self.assertIn("coder-1", acl["agents"])

    def test_creates_shared_ledger(self):
        init_multi_agent_workspace(self.td)
        ledger = os.path.join(self.td, "shared", "intelligence", "LEDGER.md")
        self.assertTrue(os.path.isfile(ledger))

    def test_no_overwrite_existing_acl(self):
        # Pre-create ACL
        acl_path = os.path.join(self.td, "mind-mem-acl.json")
        with open(acl_path, "w", encoding="utf-8") as f:
            json.dump({"custom": True}, f)
        init_multi_agent_workspace(self.td, agents=["x"])
        with open(acl_path, encoding="utf-8") as f:
            acl = json.load(f)
        self.assertTrue(acl.get("custom"))

    def test_no_agents_still_creates_shared(self):
        init_multi_agent_workspace(self.td)
        self.assertTrue(os.path.isdir(os.path.join(self.td, "shared", "decisions")))


class TestAgentIdValidation(unittest.TestCase):
    """v3.9.x security: agent_id is interpolated into a filesystem path,
    so the constructor rejects path-traversal sequences."""

    def setUp(self) -> None:
        self.td = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.td, ignore_errors=True)

    def test_normal_agent_id_accepted(self) -> None:
        for valid in ("coder-1", "agent_42", "release.bot", "abc", "X"):
            NamespaceManager(self.td, agent_id=valid)  # must not raise

    def test_traversal_rejected(self) -> None:
        for evil in ("../etc", "..", "/etc/passwd", "agent/../bad"):
            with self.assertRaises(InvalidAgentIdError):
                NamespaceManager(self.td, agent_id=evil)

    def test_nul_byte_rejected(self) -> None:
        with self.assertRaises(InvalidAgentIdError):
            NamespaceManager(self.td, agent_id="agent\x00name")

    def test_whitespace_rejected(self) -> None:
        with self.assertRaises(InvalidAgentIdError):
            NamespaceManager(self.td, agent_id="agent name")

    def test_too_long_rejected(self) -> None:
        with self.assertRaises(InvalidAgentIdError):
            NamespaceManager(self.td, agent_id="a" * 65)

    def test_empty_rejected(self) -> None:
        # empty string is *not* the same as None — None means "no agent_id".
        with self.assertRaises(InvalidAgentIdError):
            NamespaceManager(self.td, agent_id="")

    def test_none_still_works(self) -> None:
        ns = NamespaceManager(self.td, agent_id=None)
        self.assertIsNone(ns.agent_id)


class TestAclPathTraversal(unittest.TestCase):
    """The ACL must answer the same question the filesystem will.

    ``can_read``/``can_write`` used to prefix-match the raw string, so
    ``agents/coder-1/../../shared/x`` "started with" the agent's own
    namespace and was granted — while ``os.path.join`` + ``normpath``
    (and ``apply_engine._safe_resolve``) both read it as ``shared/x``.
    Two gates, two predicates: the write was admitted while the run
    printed ``ACL: PASS``.
    """

    def setUp(self) -> None:
        self.td = tempfile.mkdtemp()
        init_multi_agent_workspace(self.td, ["coder-1", "coder-2"])
        self.ns = NamespaceManager(self.td, agent_id="coder-1")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.td, ignore_errors=True)

    def _resolved_rel(self, rel_path: str) -> str | None:
        """What the filesystem makes of the path, relative to the workspace.

        Independent oracle: this is exactly what every consumer of the ACL
        answer does with the string afterwards (``os.path.join`` onto the
        workspace, then normalise). ``None`` = leaves the workspace.
        """
        joined = os.path.normpath(os.path.join(self.td, rel_path.replace("\\", "/")))
        if joined != self.td and not joined.startswith(self.td + os.sep):
            return None
        return os.path.relpath(joined, self.td).replace(os.sep, "/")

    def test_dotdot_into_shared_is_denied(self) -> None:
        """The escape that mattered: shared corpus write via the agent's own ns."""
        escape = "agents/coder-1/../../shared/decisions/DECISIONS.md"
        # Sanity: the direct spelling is denied, and the escape resolves to it.
        self.assertFalse(self.ns.can_write("shared/decisions/DECISIONS.md"))
        self.assertEqual(self._resolved_rel(escape), "shared/decisions/DECISIONS.md")
        self.assertFalse(self.ns.can_write(escape))

    def test_dotdot_into_peer_agent_is_denied(self) -> None:
        escape = "agents/coder-1/../coder-2/decisions/DECISIONS.md"
        self.assertEqual(self._resolved_rel(escape), "agents/coder-2/decisions/DECISIONS.md")
        self.assertFalse(self.ns.can_write(escape))
        self.assertFalse(self.ns.can_read(escape))

    def test_dotdot_out_of_workspace_is_denied(self) -> None:
        escape = "agents/coder-1/../../../../etc/passwd"
        self.assertIsNone(self._resolved_rel(escape))
        self.assertFalse(self.ns.can_write(escape))
        self.assertFalse(self.ns.can_read(escape))

    def test_backslash_dotdot_is_denied(self) -> None:
        """Backslashes fold to forward slashes first, so they cannot smuggle a climb."""
        self.assertFalse(self.ns.can_write("agents\\coder-1\\..\\coder-2\\x.md"))

    def test_absolute_path_is_denied(self) -> None:
        """os.path.join(ws, "/agents/...") discards ws — so it is not the agent's."""
        self.assertIsNone(self._resolved_rel("/agents/coder-1/decisions/D.md"))
        self.assertFalse(self.ns.can_write("/agents/coder-1/decisions/D.md"))
        self.assertFalse(self.ns.can_read("/shared/decisions/DECISIONS.md"))

    def test_nul_byte_is_denied(self) -> None:
        self.assertFalse(self.ns.can_write("agents/coder-1/x\x00.md"))

    def test_workspace_level_access_still_cannot_leave_workspace(self) -> None:
        """agent_id=None means "all of the workspace", not "the whole disk"."""
        ws_ns = NamespaceManager(self.td, agent_id=None)
        self.assertTrue(ws_ns.can_write("agents/coder-1/decisions/D.md"))
        self.assertFalse(ws_ns.can_read("../../etc/passwd"))
        self.assertFalse(ws_ns.can_write("/etc/passwd"))

    def test_benign_dot_segments_still_allowed(self) -> None:
        """Normalisation must not over-reject: these stay inside the namespace."""
        for ok_path in (
            "agents/coder-1/./decisions/D.md",
            "agents/coder-1/tasks/../decisions/D.md",
            "agents/coder-1//decisions/D.md",
            "agents/coder-1/decisions/",
        ):
            with self.subTest(path=ok_path):
                self.assertTrue(self.ns.can_write(ok_path))

    def test_sibling_namespace_prefix_not_confused(self) -> None:
        """A longer sibling name must not match the agent namespace by raw prefix."""
        self.assertFalse(self.ns.can_write("agents/coder-11/decisions/D.md"))

    def test_acl_answer_agrees_with_filesystem_resolution(self) -> None:
        """Property: ACL(path) == ACL(what the filesystem resolves path to)."""
        own = "agents/coder-1"
        for candidate in (
            "agents/coder-1/decisions/D.md",
            "shared/decisions/DECISIONS.md",
            "agents/coder-1/../../shared/decisions/DECISIONS.md",
            "agents/coder-1/../coder-2/decisions/D.md",
            "agents/coder-1/../../../../etc/passwd",
            "agents/coder-1/./decisions/../decisions/D.md",
            "agents\\coder-1\\decisions\\D.md",
            "/agents/coder-1/decisions/D.md",
            "agents/coder-1/decisions/..",
            "..",
            ".",
            "",
        ):
            with self.subTest(path=candidate):
                resolved = self._resolved_rel(candidate)
                expected = resolved is not None and (resolved == own or resolved.startswith(own + "/"))
                self.assertEqual(self.ns.can_write(candidate), expected)

    def test_resolve_corpus_paths_cannot_escape(self) -> None:
        """The same normalisation guards the read-side path expansion.

        ``resolve_corpus_paths`` joins the request onto each accessible
        namespace ("shared/" + rel_path) and keeps the candidate when the
        file exists *and* ``can_read`` allows it — so the traversal has to
        climb out of the namespace **and** the workspace to reach a real
        file. Both halves are asserted here: the escape names an existing
        file, and it is still not returned.
        """
        outside = os.path.join(os.path.dirname(self.td), "outside-decisions.md")
        with open(outside, "w", encoding="utf-8") as f:
            f.write("[D-20260101-001]\nStatement: not yours\n")
        try:
            escape = "../../" + os.path.basename(outside)
            # Precondition: the join really does land on the outside file,
            # otherwise this test would pass for the wrong reason.
            self.assertTrue(os.path.isfile(os.path.join(self.td, "shared", escape)))
            ws_ns = NamespaceManager(self.td, agent_id=None)
            self.assertEqual(ws_ns.resolve_corpus_paths(escape), [])
        finally:
            os.remove(outside)

    def test_unusable_namespace_entry_grants_nothing(self) -> None:
        """An empty, dot, or escaping ACL namespace entry must not become a wildcard."""
        acl_path = os.path.join(self.td, "mind-mem-acl.json")
        with open(acl_path, "w", encoding="utf-8") as f:
            json.dump(
                {"default_policy": "read", "agents": {"weird": {"namespaces": [""], "write": ["", ".", "../.."], "read": [""]}}},
                f,
            )
        ns = NamespaceManager(self.td, agent_id="weird")
        self.assertFalse(ns.can_write("shared/decisions/DECISIONS.md"))
        self.assertFalse(ns.can_write("agents/coder-1/decisions/D.md"))


class TestNamespaceCreationTraversal(unittest.TestCase):
    """Creation entry points must enforce the same agent_id predicate as the
    constructor. ``NamespaceManager(ws)`` with agent_id=None skips the regex
    entirely, so ``init_agent`` / ``init_namespace`` were reachable with an
    arbitrary string that ``os.makedirs`` then followed out of the workspace.
    """

    def setUp(self) -> None:
        # The workspace is a SUBDIRECTORY of an isolated root, so an
        # escape target lands inside the root and is torn down with it —
        # never in the shared system temp dir, where a leftover from an
        # earlier run would make the next one fail (or pass) spuriously.
        self.root = tempfile.mkdtemp()
        self.td = os.path.join(self.root, "ws")
        os.makedirs(self.td)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_init_agent_rejects_path_traversal(self) -> None:
        """init_agent must apply the same agent_id guard as the constructor.

        NamespaceManager(ws) with agent_id=None never runs the regex, so
        init_agent was the one entry point where an unvalidated id reached
        ``os.path.join(workspace, "agents/" + agent_id)`` -> os.makedirs.
        """
        outside = os.path.join(self.root, "escaped-marker")
        ns = NamespaceManager(self.td, agent_id=None)
        for bad in ("../../escaped-marker", "..", "/abs", "a\x00b", "with space", "x" * 65):
            with self.subTest(agent_id=bad):
                with self.assertRaises(InvalidAgentIdError):
                    ns.init_agent(bad)
        self.assertFalse(os.path.exists(outside))
        # And nothing was created under agents/ for the rejected ids.
        self.assertFalse(os.path.isdir(os.path.join(self.td, "agents")))
        # A well-formed id still works.
        self.assertTrue(ns.init_agent("coder-1"))
        self.assertTrue(os.path.isdir(os.path.join(self.td, "agents", "coder-1", "decisions")))

    def test_init_multi_agent_workspace_rejects_traversal_agent_id(self) -> None:
        """The --init CLI path must fail closed rather than write an ACL entry
        keyed on a traversal string and create dirs outside the workspace."""
        escape_target = os.path.join(self.root, "mm-escape-probe")
        rel = os.path.relpath(escape_target, os.path.join(self.td, "agents"))
        with self.assertRaises(InvalidAgentIdError):
            init_multi_agent_workspace(self.td, agents=["good-1", rel])
        self.assertFalse(os.path.exists(escape_target))

    def test_init_namespace_rejects_escaping_namespace(self) -> None:
        """init_namespace itself must not create NAMESPACE_DIRS outside the workspace."""
        ns = NamespaceManager(self.td, agent_id=None)
        with self.assertRaises(ValueError):
            ns.init_namespace("agents/../../mm-escape-probe2")
        self.assertFalse(os.path.exists(os.path.join(self.root, "mm-escape-probe2")))


if __name__ == "__main__":
    unittest.main()


class TestSymlinkContainment(unittest.TestCase):
    """A path the ACL blesses must not resolve outside the workspace.

    Normalising the string closes ``..`` traversal, but that is a LEXICAL
    answer while every downstream operation (open, copy2, makedirs) follows
    symlinks. Measured before the fix: a link created inside an agent's own
    namespace made ``can_write`` return True for a file outside the workspace
    entirely — the ACL and the filesystem answering two different questions
    about the same path, which is the defect class this codebase keeps hitting.
    """

    def setUp(self) -> None:
        self.ws = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        init_multi_agent_workspace(self.ws, ["coder-1"])
        with open(os.path.join(self.outside, "stolen.md"), "w", encoding="utf-8") as fh:
            fh.write("outside the workspace")
        os.symlink(self.outside, os.path.join(self.ws, "agents", "coder-1", "link"))
        self.mgr = NamespaceManager(self.ws, "coder-1")

    def tearDown(self) -> None:
        shutil.rmtree(self.ws, ignore_errors=True)
        shutil.rmtree(self.outside, ignore_errors=True)

    def test_a_symlink_out_of_the_workspace_is_refused(self) -> None:
        self.assertFalse(self.mgr.can_write("agents/coder-1/link/stolen.md"))

    def test_the_agents_own_namespace_still_works(self) -> None:
        self.assertTrue(self.mgr.can_write("agents/coder-1/notes.md"))

    def test_a_not_yet_created_path_is_still_writable(self) -> None:
        """realpath is lexical for missing components; new files must pass."""
        self.assertTrue(self.mgr.can_write("agents/coder-1/deep/new/file.md"))
