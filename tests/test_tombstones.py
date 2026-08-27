#!/usr/bin/env python3
"""Redactable-leaf tombstones — deletion vs tamper-evidence.

Covers the acceptance gate for the feature:

1. the chains still verify end-to-end after a deletion;
2. the content is genuinely unrecoverable from the workspace;
3. the tombstone records actor + reason and is itself chained;
4. recall never returns tombstoned content (fresh *or* stale index);
5. a tombstoned block is distinguishable from a never-existed block;
6. re-deleting is idempotent;

plus the two invariants the house rules demand: the preserved Merkle
leaf keeps the root stable across a redaction, and with the flag OFF
every touched surface is byte-identical to the historical behaviour.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mind_mem import tombstone as tomb  # noqa: E402
from mind_mem.mcp.infra.workspace import use_workspace  # noqa: E402
from mind_mem.merkle_tree import MerkleTree  # noqa: E402
from mind_mem.sqlite_index import _drop_tombstoned, _merge_tombstone_leaves, build_index, merkle_leaves, query_index  # noqa: E402

SENTINEL = "zqxjkvbrpm"  # unique token: if it survives anywhere, redaction failed

_BLOCK_A = f"[D-20260101-001]\nStatement: Use PostgreSQL because {SENTINEL} matters\nStatus: active\nDate: 2026-01-01\n"
_BLOCK_B = "[D-20260102-001]\nStatement: Use Redis for caching\nStatus: active\nDate: 2026-01-02\n"


def _make_workspace(*, tombstones: bool) -> str:
    """Build a minimal Markdown workspace with two decision blocks."""
    ws = tempfile.mkdtemp(prefix="mm-tomb-")
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(_BLOCK_A + "\n---\n\n" + _BLOCK_B)
    config: dict = {"version": "4.0.0"}
    if tombstones:
        config["v4"] = {"redactable_tombstones": {"enabled": True}}
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh)
    build_index(ws, incremental=False)
    return ws


def _delete(ws: str, block_id: str, **kwargs) -> dict:
    """Call the MCP delete tool against *ws* with admin scope."""
    from mind_mem.mcp.tools.memory_ops import delete_memory_item

    os.environ["MIND_MEM_SCOPE"] = "admin"
    with use_workspace(ws):
        return json.loads(delete_memory_item(block_id, **kwargs))


def _get_block(ws: str, block_id: str) -> dict:
    from mind_mem.mcp.tools.memory_ops import get_block

    os.environ["MIND_MEM_SCOPE"] = "admin"
    with use_workspace(ws):
        return json.loads(get_block(block_id))


def _verify_chain(ws: str) -> dict:
    from mind_mem.mcp.tools.audit import verify_chain

    os.environ["MIND_MEM_SCOPE"] = "admin"
    with use_workspace(ws):
        return json.loads(verify_chain())


def _root(ws: str) -> str:
    tree = MerkleTree()
    tree.build(merkle_leaves(ws))
    return tree.root_hash


def _grep_workspace(ws: str, needle: bytes) -> list[str]:
    """Every file under *ws* whose raw bytes contain *needle*."""
    hits: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(ws):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                with open(path, "rb") as fh:
                    if needle in fh.read():
                        hits.append(os.path.relpath(path, ws))
            except OSError:
                continue
    return hits


class TombstoneRedactionTest(unittest.TestCase):
    """The flag-ON path: destroy the content, keep the proof."""

    def setUp(self) -> None:
        self.ws = _make_workspace(tombstones=True)
        self.root_before = _root(self.ws)
        tomb.invalidate_cache()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.ws, ignore_errors=True)

    # -- gate 1 ---------------------------------------------------------

    def test_chain_verifies_end_to_end_after_deletion(self) -> None:
        result = _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        self.assertEqual(result["status"], "tombstoned", result)

        report = _verify_chain(self.ws)
        self.assertTrue(report["valid"], report)
        self.assertTrue(report["hash_chain"]["valid"], report)
        self.assertTrue(report["evidence_chain"]["valid"], report)
        self.assertTrue(report["tombstones"]["valid"], report)
        self.assertEqual(report["tombstones"]["count"], 1)

        from mind_mem.audit_chain import AuditChain
        from mind_mem.verify_cli import verify_workspace

        ok, errors = AuditChain(self.ws).verify()
        self.assertTrue(ok, errors)

        cli_report = verify_workspace(self.ws)
        self.assertTrue(cli_report.ok, cli_report.messages)
        self.assertTrue(cli_report.checks["tombstones"], cli_report.messages)

    def test_merkle_root_unchanged_by_redaction(self) -> None:
        _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        build_index(self.ws, incremental=False)
        self.assertEqual(_root(self.ws), self.root_before)

        # And the preserved leaf still proves inclusion for the dead block.
        tree = MerkleTree()
        tree.build(merkle_leaves(self.ws))
        record = tomb.get_tombstone(self.ws, "D-20260101-001")
        assert record is not None
        proof = tree.get_proof("D-20260101-001")
        self.assertTrue(tree.verify_proof("D-20260101-001", record.leaf_hash, proof, tree.root_hash))

    # -- gate 2 ---------------------------------------------------------

    def test_content_is_unrecoverable_from_the_store(self) -> None:
        _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        self.assertEqual(_grep_workspace(self.ws, SENTINEL.encode()), [])

    def test_pre_existing_plaintext_receipt_is_scrubbed(self) -> None:
        # A recoverable receipt written before tombstones were enabled
        # must not survive the redaction of the same block.
        log = os.path.join(self.ws, "memory", "deleted_blocks.jsonl")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"block_id": "D-20260101-001", "deleted_at": "2026-01-01T00:00:00+00:00", "content": _BLOCK_A}) + "\n")

        _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")

        self.assertEqual(_grep_workspace(self.ws, SENTINEL.encode()), [])
        with open(log, "r", encoding="utf-8") as fh:
            entries = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["content_redacted"])
        self.assertNotIn("content", entries[0])

    # -- gate 3 ---------------------------------------------------------

    def test_tombstone_records_actor_reason_and_is_chained(self) -> None:
        result = _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        record = tomb.get_tombstone(self.ws, "D-20260101-001")
        assert record is not None
        self.assertEqual(record.actor, "auditor")
        self.assertEqual(record.reason, "subject request")
        self.assertEqual(result["tombstone"]["record_hash"], record.record_hash)
        self.assertEqual(record.previous_hash, tomb.GENESIS_HASH)

        # The receipts it names really exist in the chains it claims.
        from mind_mem.audit_chain import AuditChain
        from mind_mem.governance_gate import get_gate

        gate = get_gate(self.ws)
        evidence_ids = {e.evidence_id: e for e in gate.evidence.get_latest(10)}
        self.assertIn(record.evidence_id, evidence_ids)
        self.assertEqual(evidence_ids[record.evidence_id].evidence_hash, record.evidence_hash)
        self.assertEqual(evidence_ids[record.evidence_id].metadata["reason"], "subject request")

        chain_entries = {e.entry_id: e for e in gate.chain.get_block_chain("D-20260101-001")}
        self.assertIn(record.chain_entry_id, chain_entries)
        self.assertEqual(chain_entries[record.chain_entry_id].action, "REDACT")

        audit_entries = {e.seq: e for e in AuditChain(self.ws).entries()}
        self.assertIn(record.audit_seq, audit_entries)
        self.assertEqual(audit_entries[record.audit_seq].entry_hash, record.audit_entry_hash)
        self.assertEqual(audit_entries[record.audit_seq].agent, "auditor")
        self.assertEqual(audit_entries[record.audit_seq].reason, "subject request")

    def test_redaction_requires_a_reason(self) -> None:
        result = _delete(self.ws, "D-20260101-001", actor="auditor")
        self.assertIn("reason is required", result["error"])
        # Nothing was destroyed or chained.
        self.assertFalse(tomb.ledger_exists(self.ws))
        with open(os.path.join(self.ws, "decisions", "DECISIONS.md"), "r", encoding="utf-8") as fh:
            self.assertIn(SENTINEL, fh.read())

    # -- gate 4 ---------------------------------------------------------

    def test_recall_never_returns_tombstoned_content(self) -> None:
        before = query_index(self.ws, "PostgreSQL", limit=10)
        self.assertTrue(any(r.get("_id") == "D-20260101-001" for r in before))

        _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")

        after = query_index(self.ws, "PostgreSQL", limit=10)
        self.assertFalse(any(r.get("_id") == "D-20260101-001" for r in after))
        self.assertFalse(any(SENTINEL in json.dumps(r, default=str) for r in after))

        # A reindex must not resurrect it either (the corpus text is gone).
        build_index(self.ws, incremental=False)
        self.assertFalse(any(r.get("_id") == "D-20260101-001" for r in query_index(self.ws, "PostgreSQL", limit=10)))

    def test_stale_index_rows_are_filtered(self) -> None:
        _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        dead = tomb.tombstoned_ids(self.ws)
        self.assertIn("D-20260101-001", dead)
        results = [{"_id": "D-20260101-001", "score": 1.0}, {"_id": "D-20260102-001", "score": 0.5}]
        kept = _drop_tombstoned(self.ws, results)
        self.assertEqual([r["_id"] for r in kept], ["D-20260102-001"])

    # -- gate 5 ---------------------------------------------------------

    def test_tombstoned_block_is_distinguishable_from_never_existed(self) -> None:
        _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")

        redacted = _get_block(self.ws, "D-20260101-001")
        self.assertFalse(redacted["found"])
        self.assertTrue(redacted["tombstoned"])
        self.assertEqual(redacted["tombstone"]["actor"], "auditor")
        self.assertEqual(redacted["tombstone"]["status"], "tombstoned")
        self.assertNotIn(SENTINEL, json.dumps(redacted))

        never = _get_block(self.ws, "D-19990101-999")
        self.assertFalse(never["found"])
        self.assertNotIn("tombstoned", never)

    # -- gate 6 ---------------------------------------------------------

    def test_re_deleting_is_idempotent(self) -> None:
        first = _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        from mind_mem.governance_gate import get_gate

        gate = get_gate(self.ws)
        chain_len = gate.chain.length
        evidence_len = len(gate.evidence)
        ledger_len = len(tomb.load_tombstones(self.ws))

        second = _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        self.assertEqual(second["status"], "already_tombstoned")
        self.assertEqual(second["tombstone"]["record_hash"], first["tombstone"]["record_hash"])

        self.assertEqual(gate.chain.length, chain_len)
        self.assertEqual(len(gate.evidence), evidence_len)
        self.assertEqual(len(tomb.load_tombstones(self.ws)), ledger_len)
        self.assertTrue(_verify_chain(self.ws)["valid"])

    # -- ledger tamper-evidence -----------------------------------------

    def test_ledger_detects_a_mutated_reason(self) -> None:
        _delete(self.ws, "D-20260101-001", actor="auditor", reason="subject request")
        path = tomb.ledger_path(self.ws)
        with open(path, "r", encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        record["reason"] = "routine cleanup"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        tomb.invalidate_cache(self.ws)

        ok, errors = tomb.verify_ledger(self.ws)
        self.assertFalse(ok)
        self.assertTrue(any("record_hash tampered" in e for e in errors), errors)
        self.assertFalse(_verify_chain(self.ws)["valid"])

    def test_ledger_detects_a_removed_tombstone(self) -> None:
        _delete(self.ws, "D-20260101-001", actor="auditor", reason="first")
        _delete(self.ws, "D-20260102-001", actor="auditor", reason="second")
        path = tomb.ledger_path(self.ws)
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        self.assertEqual(len(lines), 2)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(lines[1])  # drop the first tombstone
        tomb.invalidate_cache(self.ws)

        ok, errors = tomb.verify_ledger(self.ws)
        self.assertFalse(ok)
        self.assertTrue(any("previous_hash mismatch" in e for e in errors), errors)

    def test_block_store_delete_path_also_redacts(self) -> None:
        from mind_mem.block_store import MarkdownBlockStore

        store = MarkdownBlockStore(self.ws)
        self.assertTrue(store.delete_block("D-20260101-001", actor="ops", reason="store path"))
        record = tomb.get_tombstone(self.ws, "D-20260101-001")
        assert record is not None
        self.assertEqual(record.actor, "ops")
        self.assertEqual(_grep_workspace(self.ws, SENTINEL.encode()), [])
        self.assertTrue(_verify_chain(self.ws)["valid"])


class TombstoneFlagOffTest(unittest.TestCase):
    """Flag OFF: every touched surface behaves exactly as before."""

    def setUp(self) -> None:
        self.ws = _make_workspace(tombstones=False)
        tomb.invalidate_cache()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.ws, ignore_errors=True)

    def test_flag_is_off_by_default(self) -> None:
        self.assertFalse(tomb.tombstones_enabled(self.ws))

    def test_delete_keeps_the_legacy_recoverable_behaviour(self) -> None:
        result = _delete(self.ws, "D-20260101-001")
        self.assertEqual(
            result,
            {
                "_schema_version": result["_schema_version"],
                "status": "deleted",
                "block_id": "D-20260101-001",
                "file": "DECISIONS.md",
                "lines_removed": result["lines_removed"],
            },
        )
        self.assertNotIn("tombstone", result)
        self.assertFalse(os.path.exists(tomb.ledger_path(self.ws)))

        with open(os.path.join(self.ws, "memory", "deleted_blocks.jsonl"), "r", encoding="utf-8") as fh:
            entry = json.loads(fh.readline())
        self.assertIn(SENTINEL, entry["content"])  # still recoverable, as before

    def test_get_block_miss_is_unchanged(self) -> None:
        _delete(self.ws, "D-20260101-001")
        missed = _get_block(self.ws, "D-20260101-001")
        self.assertFalse(missed["found"])
        self.assertNotIn("tombstoned", missed)

    def test_verify_chain_envelope_has_no_tombstone_section(self) -> None:
        _delete(self.ws, "D-20260101-001")
        report = _verify_chain(self.ws)
        self.assertNotIn("tombstones", report)

    def test_verify_cli_report_has_no_tombstone_check(self) -> None:
        from mind_mem.verify_cli import verify_workspace

        _delete(self.ws, "D-20260101-001")
        self.assertNotIn("tombstones", verify_workspace(self.ws).checks)

    def test_index_hooks_are_pass_through_without_a_ledger(self) -> None:
        # Byte-identity proof: with no ledger both hooks return the very
        # object they were handed, so the tree and the result set cannot
        # differ from the pre-feature code path.
        leaves = merkle_leaves(self.ws)
        self.assertIs(_merge_tombstone_leaves(self.ws, leaves), leaves)
        results: list[dict] = [{"_id": "D-20260101-001", "score": 1.0}]
        self.assertIs(_drop_tombstoned(self.ws, results), results)

    def test_block_store_delete_keeps_recovery_copy(self) -> None:
        from mind_mem.block_store import MarkdownBlockStore

        self.assertTrue(MarkdownBlockStore(self.ws).delete_block("D-20260101-001"))
        with open(os.path.join(self.ws, "memory", "deleted_blocks.jsonl"), "r", encoding="utf-8") as fh:
            entry = json.loads(fh.readline())
        self.assertIn(SENTINEL, entry["content"])
        self.assertFalse(os.path.exists(tomb.ledger_path(self.ws)))


class TombstoneRecordTest(unittest.TestCase):
    """Pure record-level invariants — no workspace needed."""

    def _fields(self, **overrides) -> dict:
        base = dict(
            block_id="D-20260101-001",
            redacted_at="2026-01-01T00:00:00+00:00",
            actor="auditor",
            reason="subject request",
            leaf_hash="a" * 64,
            leaf_source="index_meta",
            content_sha3_512="b" * 128,
            content_bytes=42,
            source_file="decisions/DECISIONS.md",
            evidence_id="ev-1",
            evidence_hash="c" * 64,
            chain_entry_id="ce-1",
            chain_entry_hash="d" * 128,
            audit_seq=1,
            audit_entry_hash="e" * 64,
        )
        base.update(overrides)
        return base

    def _hash_fields(self, **overrides) -> dict:
        base = self._fields(**overrides)
        base.setdefault("previous_hash", tomb.GENESIS_HASH)
        return base

    def test_record_hash_is_deterministic(self) -> None:
        self.assertEqual(tomb.compute_record_hash(**self._hash_fields()), tomb.compute_record_hash(**self._hash_fields()))

    def test_every_field_is_bound_into_the_hash(self) -> None:
        baseline = tomb.compute_record_hash(**self._hash_fields())
        for field, value in (
            ("block_id", "D-20260101-002"),
            ("actor", "someone-else"),
            ("reason", "routine cleanup"),
            ("leaf_hash", "f" * 64),
            ("leaf_source", "raw_text"),
            ("content_bytes", 43),
            ("audit_seq", 2),
            ("previous_hash", "9" * 64),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(baseline, tomb.compute_record_hash(**self._hash_fields(**{field: value})))

    def test_reason_separator_injection_does_not_collide(self) -> None:
        # The NUL-separated preimage means punctuation in free text
        # cannot be used to forge a matching digest across fields.
        left = tomb.compute_record_hash(**self._hash_fields(actor="a", reason="b|c"))
        right = tomb.compute_record_hash(**self._hash_fields(actor="a|b", reason="c"))
        self.assertNotEqual(left, right)

    def test_append_rejects_a_malformed_record(self) -> None:
        ws = tempfile.mkdtemp(prefix="mm-tomb-rec-")
        try:
            for bad in ({"reason": "  "}, {"actor": ""}, {"leaf_hash": "nothex"}, {"leaf_source": "guessed"}, {"block_id": "lowercase-1"}):
                with self.subTest(bad=bad):
                    with self.assertRaises(tomb.TombstoneError):
                        tomb.append_tombstone(ws, **self._fields(**bad))
            self.assertFalse(os.path.exists(tomb.ledger_path(ws)))
        finally:
            import shutil

            shutil.rmtree(ws, ignore_errors=True)

    def test_merge_leaves_prefers_the_live_leaf(self) -> None:
        live = [("D-2", "live2")]
        dead = [("D-1", "dead1"), ("D-2", "dead2")]
        self.assertEqual(tomb.merge_leaves(live, dead), [("D-1", "dead1"), ("D-2", "live2")])
        self.assertIs(tomb.merge_leaves(live, []), live)


if __name__ == "__main__":
    unittest.main()
