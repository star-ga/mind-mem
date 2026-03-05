"""Tests for mind-mem hash-chain mutation log (audit_chain.py)."""

import json
import os

import pytest

from mind_mem.audit_chain import (
    _GENESIS_HASH,
    VALID_OPERATIONS,
    AuditChain,
    AuditEntry,
    _payload_hash,
)


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace."""
    return str(tmp_path)


@pytest.fixture
def chain(workspace):
    """Create an AuditChain instance."""
    return AuditChain(workspace)


class TestAuditEntry:
    def test_round_trip(self):
        entry = AuditEntry(
            seq=1,
            timestamp="2026-03-04T00:00:00+00:00",
            operation="create_block",
            target="decisions/DECISIONS.md",
            agent="claude",
            reason="Initial decision",
            payload_hash="abc123",
            prev_hash=_GENESIS_HASH,
            entry_hash="def456",
            fields_changed=["Status", "Priority"],
        )
        d = entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.seq == 1
        assert restored.operation == "create_block"
        assert restored.agent == "claude"
        assert restored.fields_changed == ["Status", "Priority"]

    def test_compute_entry_hash_deterministic(self):
        h1 = AuditEntry.compute_entry_hash(
            1,
            "2026-01-01",
            "create_block",
            "test.md",
            "agent1",
            "reason1",
            "ph1",
            _GENESIS_HASH,
        )
        h2 = AuditEntry.compute_entry_hash(
            1,
            "2026-01-01",
            "create_block",
            "test.md",
            "agent1",
            "reason1",
            "ph1",
            _GENESIS_HASH,
        )
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_different_inputs_different_hash(self):
        h1 = AuditEntry.compute_entry_hash(
            1,
            "2026-01-01",
            "create_block",
            "a.md",
            "",
            "",
            "ph1",
            _GENESIS_HASH,
        )
        h2 = AuditEntry.compute_entry_hash(
            1,
            "2026-01-01",
            "create_block",
            "b.md",
            "",
            "",
            "ph1",
            _GENESIS_HASH,
        )
        assert h1 != h2


class TestPayloadHash:
    def test_none_payload(self):
        assert len(_payload_hash(None)) == 64

    def test_string_payload(self):
        h = _payload_hash("test content")
        assert len(h) == 64

    def test_dict_payload(self):
        h = _payload_hash({"field": "Status", "old": "active", "new": "done"})
        assert len(h) == 64

    def test_deterministic(self):
        d = {"b": 2, "a": 1}
        assert _payload_hash(d) == _payload_hash(d)

    def test_key_order_independent(self):
        # sort_keys=True ensures order independence
        assert _payload_hash({"a": 1, "b": 2}) == _payload_hash({"b": 2, "a": 1})


class TestAuditChain:
    def test_empty_chain_valid(self, chain):
        ok, errors = chain.verify()
        assert ok
        assert errors == []

    def test_append_single(self, chain):
        entry = chain.append(
            "create_block",
            "decisions/DECISIONS.md",
            agent="claude",
            reason="Added new decision",
        )
        assert entry.seq == 1
        assert entry.prev_hash == _GENESIS_HASH
        assert entry.operation == "create_block"
        assert entry.agent == "claude"

    def test_append_chain_linkage(self, chain):
        e1 = chain.append("create_block", "a.md")
        e2 = chain.append("update_field", "a.md")
        e3 = chain.append("set_status", "a.md")

        assert e1.prev_hash == _GENESIS_HASH
        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash
        assert e1.seq == 1
        assert e2.seq == 2
        assert e3.seq == 3

    def test_verify_valid_chain(self, chain):
        chain.append("create_block", "a.md")
        chain.append("update_field", "a.md")
        chain.append("delete_block", "a.md")

        ok, errors = chain.verify()
        assert ok
        assert errors == []

    def test_verify_detects_tamper(self, chain):
        chain.append("create_block", "a.md")
        chain.append("update_field", "a.md")

        # Tamper with the chain file
        chain_path = chain._chain_path
        with open(chain_path, "r") as f:
            lines = f.readlines()
        # Modify an entry's operation
        entry = json.loads(lines[0])
        entry["operation"] = "delete_block"
        lines[0] = json.dumps(entry) + "\n"
        with open(chain_path, "w") as f:
            f.writelines(lines)

        ok, errors = chain.verify()
        assert not ok
        assert len(errors) >= 1

    def test_verify_detects_deleted_entry(self, chain):
        chain.append("create_block", "a.md")
        chain.append("update_field", "a.md")
        chain.append("set_status", "a.md")

        # Delete middle entry
        chain_path = chain._chain_path
        with open(chain_path, "r") as f:
            lines = f.readlines()
        del lines[1]
        with open(chain_path, "w") as f:
            f.writelines(lines)

        ok, errors = chain.verify()
        assert not ok

    def test_invalid_operation_raises(self, chain):
        with pytest.raises(ValueError, match="Invalid operation"):
            chain.append("bad_op", "a.md")

    def test_entries_returns_all(self, chain):
        chain.append("create_block", "a.md")
        chain.append("update_field", "b.md")
        chain.append("set_status", "c.md")

        entries = chain.entries()
        assert len(entries) == 3
        assert entries[0].seq == 1
        assert entries[2].seq == 3

    def test_entries_last_n(self, chain):
        for i in range(10):
            chain.append("create_block", f"f{i}.md")

        last_3 = chain.entries(last_n=3)
        assert len(last_3) == 3
        assert last_3[0].seq == 8
        assert last_3[2].seq == 10

    def test_query_by_target(self, chain):
        chain.append("create_block", "decisions/DECISIONS.md")
        chain.append("create_block", "tasks/TASKS.md")
        chain.append("update_field", "decisions/DECISIONS.md")

        results = chain.query(target="decisions")
        assert len(results) == 2

    def test_query_by_operation(self, chain):
        chain.append("create_block", "a.md")
        chain.append("update_field", "a.md")
        chain.append("create_block", "b.md")

        results = chain.query(operation="create_block")
        assert len(results) == 2

    def test_query_by_agent(self, chain):
        chain.append("create_block", "a.md", agent="claude")
        chain.append("create_block", "b.md", agent="user")
        chain.append("update_field", "a.md", agent="claude")

        results = chain.query(agent="claude")
        assert len(results) == 2

    def test_query_by_field(self, chain):
        chain.append("update_field", "a.md", fields_changed=["Status", "Priority"])
        chain.append("update_field", "a.md", fields_changed=["Description"])

        results = chain.query(field="Status")
        assert len(results) == 1

    def test_entry_count(self, chain):
        assert chain.entry_count() == 0
        chain.append("create_block", "a.md")
        chain.append("create_block", "b.md")
        assert chain.entry_count() == 2

    def test_export(self, chain, workspace):
        chain.append("create_block", "a.md", agent="test")
        chain.append("update_field", "a.md", agent="test")

        export_path = os.path.join(workspace, "audit-export.jsonl")
        count = chain.export(export_path)
        assert count == 2

        with open(export_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 entries
        header = json.loads(lines[0])
        assert header["format"] == "mind-mem-audit-chain"
        assert header["entry_count"] == 2

    def test_fields_changed_stored(self, chain):
        chain.append(
            "update_field",
            "a.md",
            fields_changed=["Status", "Priority", "Scope"],
        )
        entries = chain.entries()
        assert entries[0].fields_changed == ["Status", "Priority", "Scope"]

    def test_payload_hashed_not_stored(self, chain):
        chain.append(
            "create_block",
            "a.md",
            payload={"secret": "my-api-key-12345"},
        )
        # Verify raw payload is NOT in the chain file
        with open(chain._chain_path, "r") as f:
            content = f.read()
        assert "my-api-key-12345" not in content
        assert "secret" not in content

    def test_absolute_path_normalized(self, chain, workspace):
        abs_target = os.path.join(workspace, "decisions", "DECISIONS.md")
        entry = chain.append("create_block", abs_target)
        assert not os.path.isabs(entry.target)
        assert "decisions" in entry.target

    def test_all_valid_operations(self, chain):
        for op in sorted(VALID_OPERATIONS):
            entry = chain.append(op, "test.md")
            assert entry.operation == op

    def test_concurrent_safety(self, chain):
        """Verify sequential appends maintain chain integrity."""
        for i in range(20):
            chain.append("create_block", f"file_{i}.md", agent=f"agent_{i % 3}")

        ok, errors = chain.verify()
        assert ok
        assert errors == []
        assert chain.entry_count() == 20
