"""Regression tests for pre-existing code-scanning alerts #181-#189.

Each test pins the exact fix so a revert is immediately visible.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Alert #189 — py/log-injection: log-injection sanitisation in
# federation.THREE_WAY_MERGE audit log (#528)
# ---------------------------------------------------------------------------


@pytest.fixture
def federation_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "fed-on.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": True}}}))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))


class _CapturingHandler(logging.Handler):
    """Stores every LogRecord emitted to it."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_federation_log_strips_crlf_from_agent_id(
    tmp_path: Path, federation_enabled: None
) -> None:
    """CRLF and control characters in agent/block identifiers must be
    stripped before they appear in log extra= fields (prevents log injection
    / log forging via a crafted agent_id or block_id)."""
    from mind_mem.v4 import federation as fed

    handler = _CapturingHandler()
    logger = logging.getLogger("mind_mem.federation")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    td = str(tmp_path)
    # Malicious agent ids containing CR, LF, and a NUL.
    alice = "alice\r\nX-Injected: evil"
    bob = "bob\x00malicious"
    block_id = "B-inject-test\nfake_block: 1"

    fed.record_agent_write(td, block_id, agent_id=alice)
    fed.record_agent_write(td, block_id, agent_id=alice)
    fed.record_agent_write(td, block_id, agent_id=bob)

    report = fed.detect_conflict(td, block_id)
    assert report is not None

    fed.resolve_conflict(
        td,
        block_id,
        strategy=fed.MergeStrategy.THREE_WAY_MERGE,
        merger=lambda r: b"merged",
    )

    logger.removeHandler(handler)

    # Find the three_way_merge_resolved record.
    merge_records = [r for r in handler.records if r.getMessage() == "three_way_merge_resolved"]
    assert merge_records, "three_way_merge_resolved log entry not emitted"

    record = merge_records[0]
    extra_fields = {
        "block_id": getattr(record, "block_id", None),
        "winner_agent": getattr(record, "winner_agent", None),
        "left_agent": getattr(record, "left_agent", None),
        "right_agent": getattr(record, "right_agent", None),
    }

    for field, value in extra_fields.items():
        if value is None:
            continue
        assert "\r" not in value, f"{field!r} contains CR: {value!r}"
        assert "\n" not in value, f"{field!r} contains LF: {value!r}"
        assert "\x00" not in value, f"{field!r} contains NUL: {value!r}"


# ---------------------------------------------------------------------------
# Alert #182 — B608: embedding pipeline IN-clause uses only "?" placeholders
# ---------------------------------------------------------------------------


def test_embedding_pipeline_in_clause_uses_only_placeholders(tmp_path: Path) -> None:
    """derive_embeddings must build the IN-clause from pure '?' marks only.

    Verifies no data is string-interpolated into the SQL query; the
    nosec B608 annotation is correct (false positive)."""
    import sqlite3

    cfg = tmp_path / "ep-on.json"
    cfg.write_text(json.dumps({"v4": {"embedding_pipeline": {"enabled": True}}}))

    import os

    os.environ["MIND_MEM_CONFIG"] = str(cfg)
    try:
        import importlib

        import mind_mem.v4.embedding_pipeline as ep

        importlib.reload(ep)

        db_path = tmp_path / "index.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT)")
            conn.execute("INSERT INTO blocks VALUES ('id-1', 'hello world')")
            conn.execute("INSERT INTO blocks VALUES ('id-2', 'second block')")

        # Craft ids that would be dangerous if interpolated — they contain
        # SQL metacharacters.  With proper '?' binding they are safe.
        dangerous_ids = ["id-1'; DROP TABLE blocks;--", "id-2"]
        result = ep.derive_embeddings(str(tmp_path), dangerous_ids)
        # 'id-2' should be found; the injected fake id won't match any row.
        assert "id-2" in result, "valid id should be found"
        # The blocks table must still exist (no injection succeeded).
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id FROM blocks").fetchall()
        assert len(rows) == 2, "blocks table must be intact (no injection)"
    finally:
        os.environ.pop("MIND_MEM_CONFIG", None)
