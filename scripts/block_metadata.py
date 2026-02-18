"""mind-mem A-MEM — auto-evolving block metadata.

Tracks access patterns, evolves keywords, and computes importance scores
for memory blocks. All data stored in SQLite block_meta table.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional


class BlockMetadataManager:
    """Tracks access patterns, evolves keywords, computes importance."""

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
        self._ensure_table()

    def _ensure_table(self):
        """Create block_meta table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(self.SCHEMA)
            conn.commit()
            conn.close()
        except Exception:
            pass  # Graceful degradation if DB unavailable

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record_access(self, block_ids: list[str], query: str = ""):
        """Update access_count, last_accessed for given blocks.
        Also record co-occurrence for connection tracking."""
        if not block_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = self._get_conn()
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
                    row = conn.execute(
                        "SELECT connections FROM block_meta WHERE id = ?", (bid,)
                    ).fetchone()
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
            conn.close()
        except Exception:
            pass  # Graceful degradation

    def update_importance(self, block_id: str, decay_days: int = 30) -> float:
        """Recalculate importance from access frequency + recency + connections.
        Returns importance score in [0.8, 1.5] range."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT access_count, last_accessed, connections FROM block_meta WHERE id = ?",
                (block_id,),
            ).fetchone()
            conn.close()
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

            # Update stored importance
            conn2 = self._get_conn()
            conn2.execute(
                "UPDATE block_meta SET importance = ? WHERE id = ?",
                (importance, block_id),
            )
            conn2.commit()
            conn2.close()

            return importance
        except Exception:
            return 1.0

    def get_importance_boost(self, block_id: str) -> float:
        """Returns [0.8, 1.5] multiplier for reranking."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT importance FROM block_meta WHERE id = ?", (block_id,)
            ).fetchone()
            conn.close()
            return row[0] if row else 1.0
        except Exception:
            return 1.0

    def evolve_keywords(self, block_id: str, query_tokens: list[str],
                        block_content: str = "", max_keywords: int = 20):
        """Add query tokens found in block content to block's keyword set."""
        if not query_tokens:
            return
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT keywords FROM block_meta WHERE id = ?", (block_id,)
            ).fetchone()

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

            conn.execute(
                """INSERT INTO block_meta (id, keywords)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET keywords = ?""",
                (block_id, kw_str, kw_str),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_co_occurring_blocks(self, block_id: str, limit: int = 5) -> list[str]:
        """Blocks that frequently appear together in results."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT connections FROM block_meta WHERE id = ?", (block_id,)
            ).fetchone()
            conn.close()
            if row and row[0]:
                connections = json.loads(row[0])
                return connections[:limit]
            return []
        except Exception:
            return []
