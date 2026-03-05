"""Tests for mind-mem per-field mutation audit (field_audit.py)."""


import pytest

from mind_mem.field_audit import FieldAuditor, FieldChange


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def auditor(workspace):
    return FieldAuditor(workspace)


class TestFieldChange:
    def test_to_dict(self):
        fc = FieldChange(
            block_id="D-20260304-001",
            target="decisions/DECISIONS.md",
            field="Status",
            old_value="active",
            new_value="completed",
            agent="claude",
            reason="Task finished",
        )
        d = fc.to_dict()
        assert d["block_id"] == "D-20260304-001"
        assert d["field"] == "Status"
        assert d["old_value"] == "active"
        assert d["new_value"] == "completed"


class TestFieldAuditor:
    def test_record_single_change(self, auditor):
        fc = auditor.record_change(
            "D-20260304-001",
            "decisions/DECISIONS.md",
            "Priority",
            "3",
            "5",
            agent="claude",
            reason="Elevated priority",
        )
        assert fc.block_id == "D-20260304-001"
        assert fc.field == "Priority"
        assert fc.old_value == "3"
        assert fc.new_value == "5"
        assert fc.chain_seq > 0

    def test_field_history(self, auditor):
        # Record multiple changes to same field
        auditor.record_change("D-001", "d.md", "Status", "draft", "active", agent="a")
        auditor.record_change("D-001", "d.md", "Status", "active", "review", agent="b")
        auditor.record_change("D-001", "d.md", "Status", "review", "done", agent="a")

        history = auditor.field_history("D-001", "Status")
        assert len(history) == 3
        # Newest first
        assert history[0].new_value == '"done"' or history[0].new_value == "done"

    def test_field_history_all_fields(self, auditor):
        auditor.record_change("D-001", "d.md", "Status", "draft", "active")
        auditor.record_change("D-001", "d.md", "Priority", "1", "5")
        auditor.record_change("D-001", "d.md", "Scope", None, "global")

        history = auditor.field_history("D-001")
        assert len(history) == 3

    def test_record_block_diff(self, auditor):
        old = {"Status": "active", "Priority": 3, "Scope": "local"}
        new = {"Status": "completed", "Priority": 3, "Scope": "global", "Notes": "Done"}

        changes = auditor.record_block_diff(
            "D-001", "d.md", old, new,
            agent="claude", reason="Bulk update",
        )
        # Status changed, Scope changed, Notes added = 3 changes
        # Priority unchanged = not recorded
        assert len(changes) == 3
        field_names = {c.field for c in changes}
        assert "Status" in field_names
        assert "Scope" in field_names
        assert "Notes" in field_names
        assert "Priority" not in field_names

    def test_record_block_diff_deletion(self, auditor):
        old = {"Status": "active", "Notes": "Important"}
        new = {"Status": "active"}

        changes = auditor.record_block_diff("D-001", "d.md", old, new)
        assert len(changes) == 1
        assert changes[0].field == "Notes"
        assert changes[0].new_value is None

    def test_changes_by_agent(self, auditor):
        auditor.record_change("D-001", "d.md", "A", "1", "2", agent="claude")
        auditor.record_change("D-002", "d.md", "B", "x", "y", agent="user")
        auditor.record_change("D-003", "d.md", "C", "p", "q", agent="claude")

        claude_changes = auditor.changes_by_agent("claude")
        assert len(claude_changes) == 2
        user_changes = auditor.changes_by_agent("user")
        assert len(user_changes) == 1

    def test_change_summary(self, auditor):
        auditor.record_change("D-001", "d.md", "Status", "a", "b", agent="claude")
        auditor.record_change("D-001", "d.md", "Status", "b", "c", agent="user")
        auditor.record_change("D-002", "d.md", "Priority", "1", "2", agent="claude")

        summary = auditor.change_summary()
        assert summary["total_changes"] == 3
        assert summary["by_field"]["Status"] == 2
        assert summary["by_field"]["Priority"] == 1
        assert summary["by_agent"]["claude"] == 2
        assert summary["by_agent"]["user"] == 1

    def test_chain_integration(self, auditor):
        """Verify field changes are linked to audit chain entries."""
        auditor.record_change("D-001", "d.md", "Status", "a", "b")
        auditor.record_change("D-001", "d.md", "Priority", "1", "5")

        # Chain should have 2 entries
        ok, errors = auditor._chain.verify()
        assert ok
        entries = auditor._chain.entries()
        assert len(entries) == 2

    def test_skips_internal_fields(self, auditor):
        old = {"_id": "D-001", "_source": "x", "Status": "a"}
        new = {"_id": "D-001", "_source": "y", "Status": "b"}

        changes = auditor.record_block_diff("D-001", "d.md", old, new)
        assert len(changes) == 1
        assert changes[0].field == "Status"

    def test_new_field_old_is_none(self, auditor):
        fc = auditor.record_change("D-001", "d.md", "NewField", None, "value")
        assert fc.old_value is None
        assert fc.new_value == "value"

    def test_last_n_limit(self, auditor):
        for i in range(20):
            auditor.record_change("D-001", "d.md", "Counter", str(i), str(i + 1))

        history = auditor.field_history("D-001", "Counter", last_n=5)
        assert len(history) == 5
