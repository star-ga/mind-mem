"""v3.2.2 — execute_op routes block-level ops through BlockStore.

Pre-v3.2.2 the seven ``_op_*`` handlers in ``apply_engine.py`` speak
raw ``open()`` / filesystem on corpus Markdown files. On a Postgres
backend they write to the local FS that Postgres never sees, so
``apply_proposal`` succeeds while DB state silently diverges.

These tests pin the v3.2.2 behaviour: when a configured BlockStore
is available, block-level ops (``update_field``, ``append_list_item``,
``set_status``, ``supersede_decision``) route through
``BlockStore.get_by_id`` + ``BlockStore.write_block`` so the write
actually reaches the configured backend.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mind_mem.apply_engine import execute_op
from mind_mem.block_parser import parse_file
from mind_mem.block_store import MarkdownBlockStore


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """Minimal workspace with a decision block in place."""
    (tmp_path / "mind-mem.json").write_text("{}")
    (tmp_path / "decisions").mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "entities").mkdir()
    (tmp_path / "intelligence" / "applied").mkdir(parents=True)
    (tmp_path / "memory").mkdir()
    (tmp_path / "maintenance" / "tracked").mkdir(parents=True)

    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        textwrap.dedent("""\
            [D-20260420-001]
            type: decision
            Status: active
            Statement: Use PostgreSQL.
            Rationale: Scales better than SQLite for the 3.x deployment.
            History:
            - created 2026-04-20 Status: active

            """),
        encoding="utf-8",
    )
    return tmp_path


class TestOpRoutingViaStore:
    """Block-level ops invoke store.write_block on the configured backend."""

    def test_update_field_calls_write_block(self, ws: Path) -> None:
        """``update_field`` on a block routes through BlockStore.write_block.

        A spy store captures the call; the real MarkdownBlockStore
        continues to handle persistence so the file on disk also
        reflects the new Status value.
        """
        store = MarkdownBlockStore(str(ws))
        spy = MagicMock(wraps=store)

        op = {
            "op": "update_field",
            "file": "decisions/DECISIONS.md",
            "target": "D-20260420-001",
            "field": "Status",
            "value": "superseded",
        }
        ok, msg = execute_op(str(ws), op, store=spy)

        assert ok, msg
        # The Markdown backend still writes the file for operators who
        # inspect the on-disk format.
        blocks = parse_file(str(ws / "decisions" / "DECISIONS.md"))
        updated = next(b for b in blocks if b.get("_id") == "D-20260420-001")
        assert updated["Status"] == "superseded"
        # The spy saw the block-level write — not a raw open().
        spy.write_block.assert_called_once()
        written = spy.write_block.call_args[0][0]
        assert written["_id"] == "D-20260420-001"
        assert written["Status"] == "superseded"

    def test_set_status_calls_write_block(self, ws: Path) -> None:
        """``set_status`` updates Status and appends History via store."""
        store = MarkdownBlockStore(str(ws))
        spy = MagicMock(wraps=store)

        op = {
            "op": "set_status",
            "file": "decisions/DECISIONS.md",
            "target": "D-20260420-001",
            "status": "superseded",
            "history": "superseded by D-20260420-002",
        }
        ok, msg = execute_op(str(ws), op, store=spy)

        assert ok, msg
        # Status change + history append → two write_block calls minimum.
        assert spy.write_block.call_count >= 1

    def test_append_list_item_calls_write_block(self, ws: Path) -> None:
        """``append_list_item`` mutates a list field and writes block."""
        store = MarkdownBlockStore(str(ws))
        spy = MagicMock(wraps=store)

        op = {
            "op": "append_list_item",
            "file": "decisions/DECISIONS.md",
            "target": "D-20260420-001",
            "list": "History",
            "item": "reviewed 2026-04-20",
        }
        ok, msg = execute_op(str(ws), op, store=spy)

        assert ok, msg
        spy.write_block.assert_called_once()

    def test_append_block_calls_write_block(self, ws: Path) -> None:
        """``append_block`` parses patch text and writes via BlockStore."""
        store = MarkdownBlockStore(str(ws))
        spy = MagicMock(wraps=store)

        patch_text = textwrap.dedent("""\
            [D-20260420-002]
            type: decision
            Status: active
            Statement: Add a second decision for the test.
            ---
        """)
        op = {
            "op": "append_block",
            "file": "decisions/DECISIONS.md",
            "patch": patch_text,
        }
        ok, msg = execute_op(str(ws), op, store=spy)

        assert ok, msg
        spy.write_block.assert_called_once()
        written = spy.write_block.call_args[0][0]
        assert written["_id"] == "D-20260420-002"

        # The original block is still present on disk.
        blocks = parse_file(str(ws / "decisions" / "DECISIONS.md"))
        ids = {b.get("_id") for b in blocks}
        assert "D-20260420-001" in ids
        assert "D-20260420-002" in ids

    def test_backward_compat_store_none_uses_factory(self, ws: Path) -> None:
        """When no ``store`` argument is given, execute_op resolves via factory.

        Preserves the pre-v3.2.2 signature ``execute_op(ws, op)`` so
        existing callers (intel_apply, apply_proposal) work unchanged
        until they opt in by passing a store.
        """
        op = {
            "op": "update_field",
            "file": "decisions/DECISIONS.md",
            "target": "D-20260420-001",
            "field": "Status",
            "value": "archived",
        }
        ok, msg = execute_op(str(ws), op)
        assert ok, msg

        blocks = parse_file(str(ws / "decisions" / "DECISIONS.md"))
        updated = next(b for b in blocks if b.get("_id") == "D-20260420-001")
        assert updated["Status"] == "archived"
