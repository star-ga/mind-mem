#!/usr/bin/env python3
"""mind-mem Per-Field Mutation Audit — tracks individual field changes.

Records structured before/after diffs for every field modification,
with attribution (who), justification (why), and integration with
the hash-chain audit ledger.

Usage:
    from .field_audit import FieldAuditor
    auditor = FieldAuditor(workspace)
    auditor.record_change(
        block_id="D-20260304-001",
        target="decisions/DECISIONS.md",
        field="Priority",
        old_value="3",
        new_value="5",
        agent="claude",
        reason="User corrected priority after review",
    )
    history = auditor.field_history("D-20260304-001", "Priority")

Zero external deps — json, os, sqlite3 (all stdlib).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from .audit_chain import AuditChain
from .mind_filelock import FileLock
from .observability import get_logger, metrics

_log = get_logger("field_audit")


def _db_path(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), ".mind-mem-audit", "field_audit.db")


def _connect(workspace: str) -> sqlite3.Connection:
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS field_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id    TEXT NOT NULL,
    target      TEXT NOT NULL,
    field       TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT,
    agent       TEXT DEFAULT '',
    reason      TEXT DEFAULT '',
    chain_seq   INTEGER DEFAULT 0,
    timestamp   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_fc_block ON field_changes(block_id);
CREATE INDEX IF NOT EXISTS idx_fc_field ON field_changes(field);
CREATE INDEX IF NOT EXISTS idx_fc_agent ON field_changes(agent);
CREATE INDEX IF NOT EXISTS idx_fc_ts ON field_changes(timestamp);
CREATE INDEX IF NOT EXISTS idx_fc_block_field ON field_changes(block_id, field);
"""


class FieldChange:
    """Single field-level change record."""

    __slots__ = (
        "id", "block_id", "target", "field",
        "old_value", "new_value", "agent", "reason",
        "chain_seq", "timestamp",
    )

    def __init__(
        self,
        *,
        id: int = 0,
        block_id: str,
        target: str,
        field: str,
        old_value: str | None,
        new_value: str | None,
        agent: str = "",
        reason: str = "",
        chain_seq: int = 0,
        timestamp: str = "",
    ):
        self.id = id
        self.block_id = block_id
        self.target = target
        self.field = field
        self.old_value = old_value
        self.new_value = new_value
        self.agent = agent
        self.reason = reason
        self.chain_seq = chain_seq
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "block_id": self.block_id,
            "target": self.target,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "agent": self.agent,
            "reason": self.reason,
            "chain_seq": self.chain_seq,
            "timestamp": self.timestamp,
        }


class FieldAuditor:
    """Per-field mutation tracker with hash-chain integration."""

    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.realpath(workspace)
        self._chain = AuditChain(workspace)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = _connect(self.workspace)
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def record_change(
        self,
        block_id: str,
        target: str,
        field: str,
        old_value: str | None,
        new_value: str | None,
        *,
        agent: str = "",
        reason: str = "",
    ) -> FieldChange:
        """Record a single field change.

        Also appends to the hash-chain audit ledger.

        Args:
            block_id: The block ID being modified (e.g., "D-20260304-001").
            target: Relative path of the file containing the block.
            field: Field name that changed.
            old_value: Previous value (None for new fields).
            new_value: New value (None for deleted fields).
            agent: Who made the change.
            reason: Why the change was made.

        Returns:
            FieldChange record.
        """
        # Normalize target
        if os.path.isabs(target):
            try:
                target = os.path.relpath(target, self.workspace)
            except ValueError:
                pass

        # Append to hash-chain
        chain_entry = self._chain.append(
            "update_field",
            target,
            agent=agent,
            reason=reason,
            payload={
                "block_id": block_id,
                "field": field,
                "old": old_value,
                "new": new_value,
            },
            fields_changed=[field],
        )

        # Store in SQLite
        conn = _connect(self.workspace)
        try:
            conn.execute(
                "INSERT INTO field_changes "
                "(block_id, target, field, old_value, new_value, agent, reason, chain_seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (block_id, target, field, old_value, new_value, agent, reason, chain_entry.seq),
            )
            conn.commit()
            row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            ts_row = conn.execute(
                "SELECT timestamp FROM field_changes WHERE id = ?", (row_id,)
            ).fetchone()
        finally:
            conn.close()

        change = FieldChange(
            id=row_id,
            block_id=block_id,
            target=target,
            field=field,
            old_value=old_value,
            new_value=new_value,
            agent=agent,
            reason=reason,
            chain_seq=chain_entry.seq,
            timestamp=ts_row["timestamp"] if ts_row else "",
        )

        _log.info(
            "field_change_recorded",
            block_id=block_id,
            field=field,
            agent=agent,
        )
        metrics.inc("field_changes_recorded")
        return change

    def record_block_diff(
        self,
        block_id: str,
        target: str,
        old_block: dict,
        new_block: dict,
        *,
        agent: str = "",
        reason: str = "",
    ) -> list[FieldChange]:
        """Record all field differences between two versions of a block.

        Compares old_block and new_block dicts, recording a FieldChange
        for each field that changed (added, modified, or removed).

        Args:
            block_id: Block ID.
            target: File path.
            old_block: Previous block dict.
            new_block: Updated block dict.
            agent: Who made the changes.
            reason: Why the changes were made.

        Returns:
            List of FieldChange records for each changed field.
        """
        changes = []
        # Skip internal fields
        skip = {"_id", "_source", "_file", "_line", "_raw"}
        all_keys = set(old_block.keys()) | set(new_block.keys())

        for key in sorted(all_keys - skip):
            old_val = old_block.get(key)
            new_val = new_block.get(key)

            old_str = json.dumps(old_val, default=str) if old_val is not None else None
            new_str = json.dumps(new_val, default=str) if new_val is not None else None

            if old_str != new_str:
                change = self.record_change(
                    block_id, target, key,
                    old_value=old_str,
                    new_value=new_str,
                    agent=agent,
                    reason=reason,
                )
                changes.append(change)

        return changes

    def field_history(
        self,
        block_id: str,
        field: str | None = None,
        *,
        last_n: int = 50,
    ) -> list[FieldChange]:
        """Get the change history for a block's field(s).

        Args:
            block_id: Block ID to query.
            field: Specific field name (None for all fields).
            last_n: Maximum number of changes to return.

        Returns:
            List of FieldChange records, newest first.
        """
        conn = _connect(self.workspace)
        try:
            if field:
                rows = conn.execute(
                    "SELECT * FROM field_changes "
                    "WHERE block_id = ? AND field = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (block_id, field, last_n),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM field_changes "
                    "WHERE block_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (block_id, last_n),
                ).fetchall()
        finally:
            conn.close()

        return [
            FieldChange(
                id=row["id"],
                block_id=row["block_id"],
                target=row["target"],
                field=row["field"],
                old_value=row["old_value"],
                new_value=row["new_value"],
                agent=row["agent"],
                reason=row["reason"],
                chain_seq=row["chain_seq"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def changes_by_agent(self, agent: str, *, last_n: int = 50) -> list[FieldChange]:
        """Get all changes made by a specific agent.

        Args:
            agent: Agent identity to query.
            last_n: Maximum number of changes to return.

        Returns:
            List of FieldChange records, newest first.
        """
        conn = _connect(self.workspace)
        try:
            rows = conn.execute(
                "SELECT * FROM field_changes "
                "WHERE agent = ? ORDER BY id DESC LIMIT ?",
                (agent, last_n),
            ).fetchall()
        finally:
            conn.close()

        return [
            FieldChange(
                id=row["id"],
                block_id=row["block_id"],
                target=row["target"],
                field=row["field"],
                old_value=row["old_value"],
                new_value=row["new_value"],
                agent=row["agent"],
                reason=row["reason"],
                chain_seq=row["chain_seq"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def change_summary(self, *, last_n: int = 100) -> dict:
        """Get aggregate statistics on recent field changes.

        Returns:
            Dict with per-field counts, per-agent counts, and total.
        """
        conn = _connect(self.workspace)
        try:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM field_changes"
            ).fetchone()["cnt"]

            field_counts = {}
            for row in conn.execute(
                "SELECT field, COUNT(*) as cnt FROM field_changes "
                "GROUP BY field ORDER BY cnt DESC LIMIT ?",
                (last_n,),
            ):
                field_counts[row["field"]] = row["cnt"]

            agent_counts = {}
            for row in conn.execute(
                "SELECT agent, COUNT(*) as cnt FROM field_changes "
                "WHERE agent != '' GROUP BY agent ORDER BY cnt DESC LIMIT ?",
                (last_n,),
            ):
                agent_counts[row["agent"]] = row["cnt"]

            block_counts = {}
            for row in conn.execute(
                "SELECT block_id, COUNT(*) as cnt FROM field_changes "
                "GROUP BY block_id ORDER BY cnt DESC LIMIT 20",
            ):
                block_counts[row["block_id"]] = row["cnt"]
        finally:
            conn.close()

        return {
            "total_changes": total,
            "by_field": field_counts,
            "by_agent": agent_counts,
            "most_changed_blocks": block_counts,
        }
