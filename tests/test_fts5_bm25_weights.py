"""bm25() weights must align 1:1 with the indexed blocks_fts columns.

Regression: blocks_fts is fts5(block_id, <FTS5_COLUMNS...>) — 8 indexed
columns — but the weight list only covered the 7 FTS5_COLUMNS, so every
field's weight was shifted one column (block_id stole statement's 3.0,
statement got title's 2.5, ...) and all_text fell to the default.
"""

from __future__ import annotations

import sqlite3

from mind_mem.sqlite_index import FTS5_COLUMNS, _bm25_weights


def test_weight_count_matches_fts5_column_count() -> None:
    cols = ", ".join(c for c, _ in FTS5_COLUMNS)
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE VIRTUAL TABLE blocks_fts USING fts5(block_id, {cols})")
    n_cols = len(conn.execute("PRAGMA table_info(blocks_fts)").fetchall())

    weights = _bm25_weights()
    assert len([w for w in weights.split(",") if w.strip()]) == n_cols

    # And bm25() with these weights must run on the real schema.
    vals = ", ".join("'x'" for _ in FTS5_COLUMNS)
    conn.execute(f"INSERT INTO blocks_fts VALUES ('D-1', {vals})")
    rows = conn.execute(f"SELECT bm25(blocks_fts, {weights}) FROM blocks_fts WHERE blocks_fts MATCH 'x'").fetchall()
    assert len(rows) == 1


def test_statement_keeps_its_intended_top_weight() -> None:
    # First weight is block_id (neutral 1.0); the second is statement's 3.0,
    # not title's 2.5 (which is what the shift produced).
    parts = [p.strip() for p in _bm25_weights().split(",")]
    assert parts[0] == "1.0"  # block_id
    assert parts[1] == "3.0"  # statement, undistorted
