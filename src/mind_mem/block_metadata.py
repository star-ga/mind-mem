"""mind-mem A-MEM — auto-evolving block metadata.

Tracks access patterns, evolves keywords, and computes importance scores
for memory blocks. All data stored in SQLite block_meta table.

Uses ConnectionManager (#466) for connection pooling with read/write
separation — reads use thread-local connections, writes use a single
serialized connection, both in WAL mode.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone

from .connection_manager import ConnectionManager
from .observability import get_logger

_log = get_logger("block_metadata")


class BlockMetadataManager:
    """Tracks access patterns, evolves keywords, computes importance.

    Thread-safe: writes are serialized via ConnectionManager.write_lock,
    reads use per-thread connections (#32, #466).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS block_meta (
        id TEXT PRIMARY KEY,
        importance REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0,
        last_accessed TEXT,
        keywords TEXT DEFAULT '',
        connections TEXT DEFAULT ''
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn_mgr = ConnectionManager(db_path)
        self._lock = threading.RLock()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create block_meta table if it doesn't exist."""
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    conn.execute(self.SCHEMA)
                    conn.commit()
            except (sqlite3.Error, ValueError):
                pass  # Graceful degradation if DB unavailable

    def record_access(self, block_ids: list[str], query: str = "") -> None:
        """Update access_count, last_accessed for given blocks.
        Also record co-occurrence for connection tracking."""
        if not block_ids:
            return
        _log.debug("record_access", block_count=len(block_ids))
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    for bid in block_ids:
                        conn.execute(
                            """INSERT INTO block_meta (id, access_count, last_accessed)
                            VALUES (?, 1, ?)
                            ON CONFLICT(id) DO UPDATE SET
                                access_count = access_count + 1,
                                last_accessed = ?""",
                            (bid, now, now),
                        )
                    # Record co-occurrence pairs for connection tracking
                    if len(block_ids) > 1:
                        for i, bid in enumerate(block_ids):
                            others = [b for j, b in enumerate(block_ids) if j != i]
                            row = conn.execute("SELECT connections FROM block_meta WHERE id = ?", (bid,)).fetchone()
                            if row and row[0]:
                                existing = set(json.loads(row[0])) if row[0] else set()
                            else:
                                existing = set()
                            existing.update(others[:10])  # Cap connections
                            conn.execute(
                                "UPDATE block_meta SET connections = ? WHERE id = ?",
                                (json.dumps(list(existing)[:50]), bid),
                            )
                    conn.commit()
            except (sqlite3.Error, json.JSONDecodeError):
                pass  # Graceful degradation

    def update_importance(self, block_id: str, decay_days: int = 30) -> float:
        """Recalculate importance from access frequency + recency + connections.
        Returns importance score in [0.8, 1.5] range."""
        with self._lock:
            try:
                # Read current values
                rconn = self._conn_mgr.get_read_connection()
                row = rconn.execute(
                    "SELECT access_count, last_accessed, connections FROM block_meta WHERE id = ?",
                    (block_id,),
                ).fetchone()
                if not row:
                    return 1.0

                access_count, last_accessed, connections_json = row

                # Log-scaled access frequency
                freq_score = math.log(max(access_count, 0) + 1)

                # Recency: exponential decay
                if last_accessed:
                    try:
                        last_dt = datetime.fromisoformat(last_accessed)
                        now = datetime.now(timezone.utc)
                        days_since = max((now - last_dt).total_seconds() / 86400, 0)
                        recency_score = math.exp(-days_since / max(decay_days, 1))
                    except (ValueError, TypeError):
                        recency_score = 0.5
                else:
                    recency_score = 0.5

                # Connection degree
                connections = json.loads(connections_json) if connections_json else []
                conn_score = math.log(len(connections) + 1)

                # Weighted combination
                raw = 0.4 * freq_score + 0.4 * recency_score + 0.2 * conn_score

                # Clamp to [0.8, 1.5]
                importance = max(0.8, min(1.5, 0.8 + raw * 0.35))
                _log.debug("update_importance", block_id=block_id, importance=round(importance, 3))

                # Update stored importance (write path)
                with self._conn_mgr.write_lock:
                    wconn = self._conn_mgr.get_write_connection()
                    wconn.execute(
                        "UPDATE block_meta SET importance = ? WHERE id = ?",
                        (importance, block_id),
                    )
                    wconn.commit()

                return importance
            except (sqlite3.Error, json.JSONDecodeError, ValueError):
                return 1.0

    def get_importance_boost(self, block_id: str) -> float:
        """Returns [0.8, 1.5] multiplier for reranking."""
        try:
            conn = self._conn_mgr.get_read_connection()
            row = conn.execute("SELECT importance FROM block_meta WHERE id = ?", (block_id,)).fetchone()
            return row[0] if row else 1.0
        except (sqlite3.Error, TypeError):
            return 1.0

    def evolve_keywords(
        self, block_id: str, query_tokens: list[str], block_content: str = "", max_keywords: int = 20
    ) -> None:
        """Add query tokens found in block content to block's keyword set."""
        if not query_tokens:
            return
        with self._lock:
            try:
                # Read current keywords
                rconn = self._conn_mgr.get_read_connection()
                row = rconn.execute("SELECT keywords FROM block_meta WHERE id = ?", (block_id,)).fetchone()

                existing_kw = set()
                if row and row[0]:
                    existing_kw = set(row[0].split(",")) if row[0] else set()

                content_lower = block_content.lower()
                new_kw = set()
                for token in query_tokens:
                    if token.lower() in content_lower:
                        new_kw.add(token.lower())

                combined = existing_kw | new_kw
                # Cap at max_keywords
                kw_list = sorted(combined)[:max_keywords]
                kw_str = ",".join(kw_list)

                # Write updated keywords
                with self._conn_mgr.write_lock:
                    wconn = self._conn_mgr.get_write_connection()
                    wconn.execute(
                        """INSERT INTO block_meta (id, keywords)
                        VALUES (?, ?)
                        ON CONFLICT(id) DO UPDATE SET keywords = ?""",
                        (block_id, kw_str, kw_str),
                    )
                    wconn.commit()
            except (sqlite3.Error, json.JSONDecodeError):
                pass

    def get_co_occurring_blocks(self, block_id: str, limit: int = 5) -> list[str]:
        """Blocks that frequently appear together in results."""
        try:
            conn = self._conn_mgr.get_read_connection()
            row = conn.execute("SELECT connections FROM block_meta WHERE id = ?", (block_id,)).fetchone()
            if row and row[0]:
                connections: list[str] = json.loads(row[0])
                return connections[:limit]
            return []
        except (sqlite3.Error, json.JSONDecodeError):
            return []

    def close(self) -> None:
        """Close the underlying ConnectionManager."""
        self._conn_mgr.close()
