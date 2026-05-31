# Copyright 2026 STARGA, Inc.
"""Tests for the standalone mind-mem-verify CLI (v2.0.0rc1)."""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from mind_mem.verify_cli import (
    EXIT_CHAIN,
    EXIT_EVIDENCE,
    EXIT_GENERIC,
    EXIT_MERKLE,
    EXIT_OK,
    EXIT_SNAPSHOT,
    EXIT_SPEC,
    VerifyReport,
    main,
    verify_workspace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_ws() -> Path:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        yield Path(td)


def _seed_config(ws: Path, payload: dict | None = None) -> Path:
    path = ws / "mind-mem.json"
    path.write_text(json.dumps(payload or {"governance": {}}), encoding="utf-8")
    return path


def _seed_hash_chain(ws: Path, entries: int = 3):
    from mind_mem.hash_chain_v2 import HashChainV2

    (ws / "memory").mkdir(exist_ok=True)
    chain = HashChainV2(str(ws / "memory" / "hash_chain_v2.db"))
    for i in range(entries):
        chain.append(f"blk-{i:03d}", "create", f"payload {i}")
    return chain


def _seed_evidence(ws: Path, entries: int = 2):
    from mind_mem.evidence_objects import EvidenceAction, EvidenceChain

    (ws / "memory").mkdir(exist_ok=True)
    chain = EvidenceChain(store_path=str(ws / "memory" / "evidence_chain.jsonl"))
    for i in range(entries):
        chain.create(
            action=EvidenceAction.APPLY,
            actor="test",
            target_block_id=f"blk-{i:03d}",
            target_file=f"decisions/{i}.md",
            payload=f"payload {i}",
        )
    return chain


# ---------------------------------------------------------------------------
# Empty workspace — all optional checks pass
# ---------------------------------------------------------------------------


class TestEmptyWorkspace:
    def test_fresh_workspace_verifies(self, empty_ws):
        report = verify_workspace(str(empty_ws))
        assert report.ok is True
        assert report.exit_code == EXIT_OK
        # Every optional check should have been recorded, even if only as "no X present".
        assert "hash_chain" in report.checks
        assert "spec_binding" in report.checks
        assert "evidence_chain" in report.checks

    def test_nonexistent_workspace_fails_generic(self):
        report = verify_workspace("/tmp/mind-mem-definitely-missing-9999")
        assert report.ok is False
        assert report.exit_code == EXIT_GENERIC


# ---------------------------------------------------------------------------
# Hash chain
# ---------------------------------------------------------------------------


class TestHashChain:
    def test_clean_chain_verifies(self, empty_ws):
        _seed_hash_chain(empty_ws, entries=5)
        report = verify_workspace(str(empty_ws))
        assert report.checks["hash_chain"] is True
        assert report.ok is True

    def test_tampered_chain_fails(self, empty_ws):
        import sqlite3

        _seed_hash_chain(empty_ws, entries=3)
        # Flip one byte of an entry_hash — deterministic corruption.
        db = sqlite3.connect(str(empty_ws / "memory" / "hash_chain_v2.db"))
        db.execute("UPDATE hash_chain SET entry_hash = REPLACE(entry_hash, SUBSTR(entry_hash,1,1), 'x') WHERE rowid = 2")
        db.commit()
        db.close()

        report = verify_workspace(str(empty_ws))
        assert report.checks["hash_chain"] is False
        assert report.exit_code == EXIT_CHAIN


# ---------------------------------------------------------------------------
# Spec binding
# ---------------------------------------------------------------------------


class TestSpecBinding:
    def test_bound_then_unchanged_verifies(self, empty_ws):
        from mind_mem.spec_binding import SpecBindingManager

        cfg = _seed_config(empty_ws)
        SpecBindingManager(str(cfg)).bind(str(cfg))
        report = verify_workspace(str(empty_ws))
        assert report.checks["spec_binding"] is True

    def test_config_mutation_fails(self, empty_ws):
        from mind_mem.spec_binding import SpecBindingManager

        cfg = _seed_config(empty_ws, {"governance": {"mode": "strict"}})
        SpecBindingManager(str(cfg)).bind(str(cfg))
        # Mutate the config after binding.
        cfg.write_text(json.dumps({"governance": {"mode": "loose"}}), encoding="utf-8")

        report = verify_workspace(str(empty_ws))
        assert report.checks["spec_binding"] is False
        assert report.exit_code == EXIT_SPEC

    def test_corrupt_binding_fails(self, empty_ws):
        from mind_mem.spec_binding import SpecBindingManager

        cfg = _seed_config(empty_ws)
        SpecBindingManager(str(cfg)).bind(str(cfg))
        (empty_ws / ".spec_binding.json").write_text("{ not json", encoding="utf-8")

        report = verify_workspace(str(empty_ws))
        assert report.checks["spec_binding"] is False
        assert report.exit_code == EXIT_SPEC


# ---------------------------------------------------------------------------
# Evidence chain
# ---------------------------------------------------------------------------


class TestEvidenceChain:
    def test_clean_evidence_verifies(self, empty_ws):
        _seed_evidence(empty_ws, entries=3)
        report = verify_workspace(str(empty_ws))
        assert report.checks["evidence_chain"] is True

    def test_tampered_evidence_fails(self, empty_ws):
        _seed_evidence(empty_ws, entries=3)
        jsonl = empty_ws / "memory" / "evidence_chain.jsonl"
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        # Tamper with the actor on entry 2, leaving evidence_hash intact
        # so self-verification breaks.
        doc = json.loads(lines[1])
        doc["actor"] = "tampered"
        lines[1] = json.dumps(doc, separators=(",", ":"))
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        report = verify_workspace(str(empty_ws))
        assert report.checks["evidence_chain"] is False
        assert report.exit_code == EXIT_EVIDENCE


# ---------------------------------------------------------------------------
# Snapshot anchor
# ---------------------------------------------------------------------------


class TestSnapshotAnchor:
    def _make_snapshot(self, ws: Path, chain_head: str | None, merkle_leaves: list | None, merkle_root: str | None) -> str:
        snap = ws / "snapshots" / "snap-001"
        snap.mkdir(parents=True)
        manifest = {}
        if chain_head is not None:
            manifest["chain_head"] = chain_head
        if merkle_leaves is not None:
            manifest["merkle_leaves"] = merkle_leaves
        if merkle_root is not None:
            manifest["merkle_root"] = merkle_root
        (snap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return "snapshots/snap-001"

    def test_chain_head_match(self, empty_ws):
        chain = _seed_hash_chain(empty_ws, entries=3)
        head = chain.get_latest(1)[-1].entry_hash
        rel = self._make_snapshot(empty_ws, chain_head=head, merkle_leaves=None, merkle_root=None)
        report = verify_workspace(str(empty_ws), snapshot=rel)
        assert report.checks["snapshot_anchor"] is True

    def test_chain_head_mismatch(self, empty_ws):
        _seed_hash_chain(empty_ws, entries=3)
        rel = self._make_snapshot(empty_ws, chain_head="f" * 128, merkle_leaves=None, merkle_root=None)
        report = verify_workspace(str(empty_ws), snapshot=rel)
        assert report.checks["snapshot_anchor"] is False
        assert report.exit_code == EXIT_SNAPSHOT

    def test_merkle_root_match(self, empty_ws):
        from mind_mem.merkle_tree import MerkleTree

        leaves = [("blk-001", "aaa111"), ("blk-002", "bbb222")]
        tree = MerkleTree()
        tree.build(leaves)
        leaves_json = [{"block_id": b, "content_hash": h} for b, h in leaves]
        rel = self._make_snapshot(empty_ws, chain_head=None, merkle_leaves=leaves_json, merkle_root=tree.root_hash)
        report = verify_workspace(str(empty_ws), snapshot=rel)
        assert report.checks.get("merkle_root") is True

    def test_merkle_root_mismatch(self, empty_ws):
        leaves_json = [{"block_id": "blk-001", "content_hash": "aaa111"}]
        rel = self._make_snapshot(empty_ws, chain_head=None, merkle_leaves=leaves_json, merkle_root="0" * 128)
        report = verify_workspace(str(empty_ws), snapshot=rel)
        assert report.checks.get("merkle_root") is False
        assert report.exit_code == EXIT_MERKLE


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLIEntryPoint:
    def test_main_exit_zero_on_fresh_ws(self, empty_ws):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main([str(empty_ws)])
        assert code == EXIT_OK
        out = buf.getvalue()
        assert "OK" in out

    def test_main_json_mode(self, empty_ws):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main([str(empty_ws), "--json"])
        payload = json.loads(buf.getvalue())
        assert payload["ok"] is True
        assert payload["exit_code"] == code == EXIT_OK

    def test_main_exit_generic_on_missing_ws(self):
        code = main(["/tmp/mind-mem-missing-9999-x", "--json"])
        assert code == EXIT_GENERIC


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReadOnlyVerifier:
    """Audit regression tests (v2.0.0rc1 3-LLM audit)."""

    def test_read_only_chain_raises_on_append(self, empty_ws):
        from mind_mem.hash_chain_v2 import HashChainV2

        _seed_hash_chain(empty_ws, entries=2)
        ro = HashChainV2.open_readonly(str(empty_ws / "memory" / "hash_chain_v2.db"))
        # Verification path is permitted; mutation is not.
        ok, _ = ro.verify_chain()
        assert ok is True
        with pytest.raises(PermissionError):
            ro.append("blk-new", "create", "payload")

    def test_snapshot_path_traversal_rejected(self, empty_ws):
        # Even if a manifest exists "out there", the CLI must refuse
        # to read it. We don't create one; the rejection should come
        # from the path check, not from manifest-missing.
        report = verify_workspace(str(empty_ws), snapshot="../../etc")
        assert report.checks["snapshot_anchor"] is False
        assert report.exit_code == EXIT_SNAPSHOT

    def test_merkle_one_anchor_without_the_other_fails(self, empty_ws):
        snap = empty_ws / "snapshots" / "snap-001"
        snap.mkdir(parents=True)
        (snap / "manifest.json").write_text(json.dumps({"merkle_root": "f" * 128}), encoding="utf-8")
        report = verify_workspace(str(empty_ws), snapshot="snapshots/snap-001")
        assert report.checks.get("merkle_root") is False
        assert report.exit_code == EXIT_MERKLE

    def test_manifest_bad_encoding_yields_structured_failure(self, empty_ws):
        snap = empty_ws / "snapshots" / "snap-001"
        snap.mkdir(parents=True)
        # Write bytes that aren't valid UTF-8.
        (snap / "manifest.json").write_bytes(b"\xff\xfe{ not utf8 \x00")
        report = verify_workspace(str(empty_ws), snapshot="snapshots/snap-001")
        assert report.checks["snapshot_anchor"] is False
        assert report.exit_code == EXIT_SNAPSHOT


class TestReport:
    def test_as_dict_contains_expected_keys(self):
        r = VerifyReport(workspace="/tmp/x", ok=True)
        r.record("foo", True)
        d = r.as_dict()
        assert set(d.keys()) >= {"workspace", "ok", "checks", "messages", "exit_code"}

    def test_first_failure_sets_exit_but_later_does_not_overwrite(self, empty_ws):
        _seed_hash_chain(empty_ws, entries=3)
        # Tamper hash chain first, then also seed a broken spec binding.
        import sqlite3

        db = sqlite3.connect(str(empty_ws / "memory" / "hash_chain_v2.db"))
        db.execute("UPDATE hash_chain SET entry_hash = REPLACE(entry_hash, SUBSTR(entry_hash,1,1), 'x') WHERE rowid = 1")
        db.commit()
        db.close()

        from mind_mem.spec_binding import SpecBindingManager

        cfg = _seed_config(empty_ws)
        SpecBindingManager(str(cfg)).bind(str(cfg))
        (empty_ws / ".spec_binding.json").write_text("{ broken", encoding="utf-8")

        report = verify_workspace(str(empty_ws))
        # First failure was hash_chain, so exit code sticks at EXIT_CHAIN.
        assert report.exit_code == EXIT_CHAIN
