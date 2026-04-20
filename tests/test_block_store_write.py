"""v3.2.0 §1.4 PR-2 — MarkdownBlockStore.write_block + delete_block tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.block_store import MarkdownBlockStore


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """Workspace with the standard corpus directories."""
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestWriteBlockAppend:
    def test_appends_into_empty_file(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        block = {
            "_id": "D-20260420-001",
            "Statement": "Use SQLite for v3.2.0 default backend",
            "Status": "active",
            "Date": "2026-04-20",
            "Tags": ["storage", "architecture"],
        }
        out_id = store.write_block(block)

        assert out_id == "D-20260420-001"
        target = ws / "decisions" / "DECISIONS.md"
        assert target.is_file()
        content = target.read_text()
        assert "[D-20260420-001]" in content
        assert "Statement: Use SQLite for v3.2.0 default backend" in content
        assert "- storage" in content
        assert content.rstrip().endswith("---")

    def test_appends_after_existing_block(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "D-20260420-001", "Statement": "first", "Status": "active"})
        store.write_block({"_id": "D-20260420-002", "Statement": "second", "Status": "active"})

        content = (ws / "decisions" / "DECISIONS.md").read_text()
        # Both blocks must be present and in the order they were written.
        first_pos = content.index("[D-20260420-001]")
        second_pos = content.index("[D-20260420-002]")
        assert first_pos < second_pos

    def test_targets_correct_file_per_prefix(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "T-20260420-001", "Statement": "task", "Status": "todo"})
        store.write_block({"_id": "PRJ-alpha", "Statement": "project", "Status": "active"})
        assert (ws / "tasks" / "TASKS.md").is_file()
        assert (ws / "entities" / "projects.md").is_file()
        assert not (ws / "decisions" / "DECISIONS.md").exists()


class TestWriteBlockReplace:
    def test_replaces_in_place_preserving_neighbours(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "D-20260420-001", "Statement": "first v1", "Status": "active"})
        store.write_block({"_id": "D-20260420-002", "Statement": "second", "Status": "active"})

        # Now update 001 in place.
        store.write_block({"_id": "D-20260420-001", "Statement": "first v2", "Status": "superseded"})

        content = (ws / "decisions" / "DECISIONS.md").read_text()
        assert "first v2" in content
        assert "first v1" not in content
        # Second block still there in the same relative position.
        assert "[D-20260420-002]" in content
        assert content.index("[D-20260420-001]") < content.index("[D-20260420-002]")
        # Status updated in the replacement.
        assert "Status: superseded" in content
        assert "Status: active\nStatement: first" not in content

    def test_field_rename_and_drop_on_replace(self, ws: Path) -> None:
        """Replacement is authoritative — removed fields disappear."""
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            {
                "_id": "T-20260420-001",
                "Statement": "initial",
                "Status": "todo",
                "Rationale": "to be removed",
            }
        )
        store.write_block(
            {
                "_id": "T-20260420-001",
                "Statement": "initial",
                "Status": "doing",
                # Rationale dropped intentionally.
            }
        )
        content = (ws / "tasks" / "TASKS.md").read_text()
        assert "Rationale" not in content
        assert "Status: doing" in content


class TestWriteBlockSecurity:
    def test_rejects_missing_id(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        with pytest.raises(ValueError, match="_id"):
            store.write_block({"Statement": "no id"})

    def test_rejects_invalid_id_format(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        with pytest.raises(ValueError, match="invalid block id"):
            store.write_block({"_id": "not-a-valid-id", "Statement": "x"})

    def test_rejects_unknown_prefix(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        with pytest.raises(ValueError, match="no canonical file mapping"):
            store.write_block({"_id": "XYZZY-20260420-001", "Statement": "mystery prefix"})

    def test_neutralises_embedded_block_header(self, ws: Path) -> None:
        """A field value containing a newline + ``[ID]`` must not start a new block.

        The serializer replaces the ``\\n[`` bigram with ``\\n `` so
        the opening bracket of any injected header is dropped on
        write. The concrete effect is that the parser sees a line
        starting with whitespace (a continuation) rather than a line
        starting with ``[`` (a new block header).
        """
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            {
                "_id": "D-20260420-001",
                "Statement": "line1\n[D-20260420-999] injected fake header",
                "Status": "active",
            }
        )
        content = (ws / "decisions" / "DECISIONS.md").read_text()
        # Original header form must be absent at column zero.
        assert "\n[D-20260420-999]" not in content
        # The neutralized form is ``\\n D-...]`` — space replaces
        # the dropped ``[``; the rest of the string is preserved so
        # operators can still grep for the original payload.
        assert "\n D-20260420-999] injected fake header" in content
        # Parsing the file back must not produce a second block.
        from mind_mem.block_parser import parse_file

        parsed = parse_file(str(ws / "decisions" / "DECISIONS.md"))
        assert len(parsed) == 1, f"expected 1 block, got {len(parsed)}: {[b.get('_id') for b in parsed]}"
        assert parsed[0]["_id"] == "D-20260420-001"


class TestDeleteBlock:
    def test_removes_block_and_returns_true(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "D-20260420-001", "Statement": "keep", "Status": "active"})
        store.write_block({"_id": "D-20260420-002", "Statement": "remove", "Status": "superseded"})

        assert store.delete_block("D-20260420-002") is True

        content = (ws / "decisions" / "DECISIONS.md").read_text()
        assert "[D-20260420-002]" not in content
        assert "[D-20260420-001]" in content

    def test_missing_block_returns_false(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "D-20260420-001", "Statement": "only", "Status": "active"})
        assert store.delete_block("D-20260420-999") is False

    def test_unknown_prefix_returns_false(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        assert store.delete_block("ZZZ-20260420-001") is False

    def test_records_deletion_to_recovery_journal(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "D-20260420-001", "Statement": "about to be deleted", "Status": "active"})
        store.delete_block("D-20260420-001")

        log_path = ws / "memory" / "deleted_blocks.jsonl"
        assert log_path.is_file()
        entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert len(entries) == 1
        assert entries[0]["block_id"] == "D-20260420-001"
        assert "about to be deleted" in entries[0]["content"]
        assert entries[0]["deleted_at"]  # ISO timestamp present


class TestRoundTrip:
    def test_write_then_read_via_get_by_id(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        original = {
            "_id": "D-20260420-007",
            "Statement": "round-trip test",
            "Status": "active",
            "Date": "2026-04-20",
            "Tags": ["alpha", "beta"],
        }
        store.write_block(original)

        read_back = store.get_by_id("D-20260420-007")
        assert read_back is not None
        assert read_back["_id"] == "D-20260420-007"
        assert read_back["Statement"] == "round-trip test"
        assert read_back["Status"] == "active"
        # Tags round-trip as a list (parser converts "- item" bullets back).
        assert read_back.get("Tags") == ["alpha", "beta"]

    def test_cache_invalidated_so_get_all_sees_new_file(self, ws: Path) -> None:
        store = MarkdownBlockStore(str(ws))
        # Before: no decisions file, so get_all returns nothing.
        assert store.get_all() == []
        store.write_block({"_id": "D-20260420-010", "Statement": "first", "Status": "active"})
        # After: get_all must see the newly-created file.
        all_blocks = store.get_all()
        assert len(all_blocks) == 1
        assert all_blocks[0]["_id"] == "D-20260420-010"
