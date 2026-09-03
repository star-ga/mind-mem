# Copyright 2026 STARGA, Inc.
"""The auto-capture door lands signals through the gate, or lands nothing.

``capture.append_signals`` is the widest write door in the product —
twelve callers, the ``observe_signal`` MCP tool among them. Until 5.0.2 it
hand-wrote the ``[SIG-…]`` block into ``intelligence/SIGNALS.md`` with the
admission scope held in one arm of a conditional expression::

    _gate = _get_gate(workspace)          # except Exception: return None
    _scope = _gate.admit_batch(...) if _gate is not None else nullcontext()
    with _scope, open(signals_path, "a", encoding="utf-8") as f:

so a gate that could not be constructed produced a no-op scope and the
block was written anyway, with a success returned. Measured on a fresh
workspace with ``memory/hash_chain_v2.db`` replaced by a directory:
``get_gate`` raised ``OperationalError``, ``append_signals`` returned 1,
the signal was in the corpus and both ledgers were at +0.

The static half of the fix lives in ``tests/_write_path_scan`` (an
``ADMIT_OPENER`` in an ``IfExp`` arm no longer counts as opening a scope).
This file is the runtime half: it drives the real door against a real
workspace and asserts the two directions a governed write has to have —
the healthy path mints one block AND one row in each ledger, and a ledger
that cannot open writes nothing at all.

Every negative assertion here carries its positive control in the same
test: the "nothing was written" assertions are made against a workspace
that has just been shown to write when the ledger is healthy, because
``assert nothing_landed`` passes just as well when the door was never
reachable.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest

from mind_mem import capture
from mind_mem.admission import UngatedWriteError
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.enums import INITIAL_STATUS, IngestTier, is_servable
from mind_mem.init_workspace import init

#: The template line ``[SIG-YYYYMMDD-###]`` that ships in a fresh
#: SIGNALS.md. It is prose, not a block, so a count of ``[SIG-`` in the
#: raw file starts at 1 on an empty corpus — hence every assertion below
#: counts BLOCKS through the store rather than substrings in the file.
_TEMPLATE_ID_LINE = "[SIG-YYYYMMDD-###]"


def _signal(text: str, line: int = 1, kind: str = "decision") -> dict:
    return {
        "line": line,
        "type": kind,
        "text": text,
        "pattern": ".*",
        "confidence": "high",
        "priority": "P1",
        "structure": {"subject": "we", "object": "redis", "tags": ["database"]},
    }


class _Workspace(unittest.TestCase):
    """A real initialised workspace, torn down after each test."""

    def setUp(self) -> None:
        self.ws = tempfile.mkdtemp(prefix="mm-capture-gov-")
        self.addCleanup(shutil.rmtree, self.ws, ignore_errors=True)
        init(self.ws)
        self.signals_path = os.path.join(self.ws, "intelligence", "SIGNALS.md")
        self.chain_db = os.path.join(self.ws, "memory", "hash_chain_v2.db")
        self.evidence = os.path.join(self.ws, "memory", "evidence_chain.jsonl")
        # The gate is cached per workspace path; a test that corrupts the
        # ledger must not be served a gate built before the corruption.
        from mind_mem.governance_gate import evict_gate

        self.addCleanup(evict_gate, self.ws)
        evict_gate(self.ws)

    # -- observations ----------------------------------------------------

    def sig_blocks(self) -> list[dict]:
        return [b for b in MarkdownBlockStore(self.ws).get_all(active_only=False) if str(b.get("_id", "")).startswith("SIG-")]

    def chain_rows(self) -> int:
        if not os.path.isfile(self.chain_db):
            return 0
        conn = sqlite3.connect(self.chain_db)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0])
        finally:
            conn.close()

    def evidence_rows(self) -> int:
        if not os.path.isfile(self.evidence):
            return 0
        with open(self.evidence, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def signals_digest(self) -> str:
        with open(self.signals_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()


class TestTheHealthyPathIsGoverned(_Workspace):
    def test_one_signal_mints_one_block_and_one_row_in_each_ledger(self) -> None:
        self.assertEqual(self.sig_blocks(), [], "a fresh workspace already holds SIG blocks; the fixture is wrong")
        chain_before, evidence_before = self.chain_rows(), self.evidence_rows()

        written = capture.append_signals(self.ws, [_signal("We decided to switch to Redis for the cache")], "2026-09-02")

        self.assertEqual(written, 1)
        blocks = self.sig_blocks()
        self.assertEqual(len(blocks), 1, f"expected exactly one SIG block, got {[b.get('_id') for b in blocks]}")
        # "the ledger moved", not "the ledger gained exactly N rows". How many
        # rows one governed write leaves is the GATE's decision and it changed
        # inside this release: ``_run_write_scope`` now records how the scope
        # ended, so a single admit_batch measures 2 hash-chain rows (the
        # admission and the close) where it measured 1 before. Pinning the
        # count here would pin another module's evidence shape and turn every
        # honest addition to it into a failure of this test. The property this
        # test owns is the DISCRIMINATION — a governed write advances both
        # ledgers, and the refused write in
        # TestAGateThatCannotOpenWritesNothing advances neither, asserted
        # there with equality.
        self.assertGreater(self.chain_rows(), chain_before, "the hash chain did not move for a governed write")
        self.assertGreater(self.evidence_rows(), evidence_before, "the evidence chain did not move for a governed write")

    def test_the_block_resolves_by_id_and_carries_the_tier_status(self) -> None:
        capture.append_signals(self.ws, [_signal("We decided to switch to Redis for the cache")], "2026-09-02")

        block = MarkdownBlockStore(self.ws).get_by_id("SIG-20260902-001")
        self.assertIsNotNone(block, "the minted signal does not resolve by id — it is not a block, only text")
        assert block is not None  # nosec B101 — narrowing for the type checker after the assertion above
        expected = INITIAL_STATUS[IngestTier.AUTO_CAPTURE]
        assert expected is not None  # nosec B101 — AUTO_CAPTURE has a status row; pinned below
        self.assertEqual(block.get("Status"), expected.value)
        self.assertFalse(is_servable(block.get("Status")), "a captured signal must not arrive in a state recall will serve")
        self.assertEqual(block.get("Excerpt"), "We decided to switch to Redis for the cache")
        self.assertEqual(block.get("Type"), "auto-capture-decision")

    def test_the_status_comes_from_the_tier_table_not_a_literal(self) -> None:
        """A second definition of the status could only ever drift into a refused write."""
        row = INITIAL_STATUS[IngestTier.AUTO_CAPTURE]
        self.assertIsNotNone(row, "AUTO_CAPTURE lost its INITIAL_STATUS row; captured signals cannot be stamped")
        assert row is not None  # nosec B101 — narrowing after the assertion above
        self.assertEqual(capture._signal_status(), row.value)


class TestAGateThatCannotOpenWritesNothing(_Workspace):
    def test_unopenable_ledger_refuses_the_write_and_leaves_the_file_byte_unchanged(self) -> None:
        """The R2-03 regression, with its positive control in the same test.

        The control runs FIRST and against the same workspace: a signal is
        written while the ledger is healthy, proving this door reaches the
        corpus at all. Only then is the ledger made unopenable. Without
        that half, "nothing was written" would pass on a workspace where
        nothing could ever be written.
        """
        control = capture.append_signals(self.ws, [_signal("We decided to switch to Redis for the cache")], "2026-09-02")
        self.assertEqual(control, 1, "positive control: the healthy door must write, or the negative half proves nothing")
        self.assertEqual(len(self.sig_blocks()), 1)

        # A ledger that cannot be opened: the realistic trigger (an
        # unwritable or clobbered chain db), not a monkeypatched import.
        from mind_mem.governance_gate import evict_gate

        evict_gate(self.ws)
        os.remove(self.chain_db)
        os.makedirs(self.chain_db)

        digest_before = self.signals_digest()
        chain_before, evidence_before = self.chain_rows(), self.evidence_rows()

        with self.assertRaises(sqlite3.OperationalError):
            capture.append_signals(self.ws, [_signal("Need to fix the auth module before Friday", line=2, kind="task")], "2026-09-02")

        self.assertEqual(self.signals_digest(), digest_before, "SIGNALS.md changed while the gate could not be opened")
        self.assertEqual(len(self.sig_blocks()), 1, "a second SIG block landed with no admission")
        self.assertEqual(self.evidence_rows(), evidence_before, "the evidence chain moved for a refused write")
        self.assertEqual(self.chain_rows(), chain_before)

    def test_a_write_with_no_scope_open_is_refused_by_the_store(self) -> None:
        """What the conditional used to substitute: no scope at all.

        Pins the enforcement the fix now relies on. ``append_signals``
        cannot reach a corpus file except through ``write_block``, and
        ``write_block``'s first statement refuses an unadmitted write — so
        a future edit that drops the scope fails loudly here rather than
        appending quietly.
        """
        block = capture._signal_block(_signal("We decided to switch to Redis for the cache"), "SIG-20260902-001", "2026-09-02")
        digest_before = self.signals_digest()

        with self.assertRaises(UngatedWriteError):
            MarkdownBlockStore(self.ws).write_block(block)

        self.assertEqual(self.signals_digest(), digest_before)
        self.assertEqual(self.sig_blocks(), [])


class TestTheDoorHasNoUngatedFallback(unittest.TestCase):
    """The fail-open helpers are gone from the module, not merely unused."""

    def test_get_gate_helper_and_nullcontext_import_are_gone(self) -> None:
        self.assertFalse(
            hasattr(capture, "_get_gate"),
            "capture._get_gate is back; its `except Exception: return None` is what turned a dead gate into a no-op scope",
        )
        self.assertFalse(
            hasattr(capture, "contextlib"),
            "capture imports contextlib again; the nullcontext fallback arm is how the block was written with no receipt",
        )


class TestTheTemplateLineIsNotABlock(_Workspace):
    def test_counting_sig_substrings_in_the_file_overcounts_by_one(self) -> None:
        """Why every assertion above counts blocks, not substrings.

        The shipped SIGNALS.md header documents the id format, so a naive
        ``content.count("[SIG-")`` starts at 1 on an empty corpus. A test
        that measured the door that way would read "1 signal" before the
        door had ever run.
        """
        with open(self.signals_path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        self.assertIn(_TEMPLATE_ID_LINE, raw)
        self.assertEqual(raw.count("[SIG-"), 1)
        self.assertEqual(self.sig_blocks(), [])


if __name__ == "__main__":
    unittest.main()
