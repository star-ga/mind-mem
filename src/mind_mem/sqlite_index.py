#!/usr/bin/env python3
"""Mind Mem SQLite FTS5 Index — incremental lexical indexing. Zero external deps.

Replaces O(corpus) per-query scanning with O(log N) indexed lookup via SQLite
FTS5. Supports incremental updates (only re-indexes changed files), field-weighted
ranking, and deterministic post-processing (recency/status boosts, query type
detection, reranker) matching the existing recall.py pipeline.

Schema:
    blocks(id PK, type, file, line, status, date, speaker, tags, json_blob)
    blocks_fts(Statement, Title, Tags, Description, Context) — FTS5 virtual table
    xref_edges(src, dst) — cross-reference graph
    file_state(path, mtime, size, hash) — incremental rebuild tracking

Usage:
    python3 -m mind_mem.sqlite_index build --workspace .
    python3 -m mind_mem.sqlite_index build --workspace . --incremental
    python3 -m mind_mem.sqlite_index query --workspace . --query "PostgreSQL"
    python3 -m mind_mem.sqlite_index status --workspace .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime

from .block_parser import parse_file
from .connection_manager import ConnectionManager
from .enums import TaskStatus
from .extractor import extract_facts
from .observability import get_logger, metrics
from .recall import (
    _BLOCK_ID_RE,
    _QUERY_TYPE_PARAMS,
    CORPUS_FILES,
    SEARCH_FIELDS,
    _parse_speaker_from_tags,
    date_score,
    detect_query_type,
    expand_months,
    expand_query,
    get_block_type,
    get_excerpt,
    rerank_hits,
    tokenize,
)

_log = get_logger("sqlite_index")

# DB location relative to workspace
DB_REL_PATH = ".mind-mem-index/recall.db"

# FTS5 columns and their weights (order matters for bm25() function)
# bm25() returns negative values; lower = better match
FTS5_COLUMNS = [
    ("statement", 3.0),
    ("title", 2.5),
    ("name", 2.0),
    ("description", 1.2),
    ("tags", 0.8),
    ("context", 0.5),
    ("all_text", 1.0),  # catch-all for other fields
]

# Only allow alphanumeric tokens (plus _ - .) through to FTS5 queries
# to prevent wildcard injection (e.g. "*" matching entire corpus)
_FTS5_SAFE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _db_path(workspace: str) -> str:
    """Return absolute path to the index database."""
    return os.path.join(os.path.abspath(workspace), DB_REL_PATH)


def _connect(workspace: str, readonly: bool = False) -> sqlite3.Connection:
    """Open (or create) the index database.

    For new code, prefer _get_conn_manager() which provides connection
    pooling with read/write separation (#466).
    """
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if readonly:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(path)
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Connection Manager — pooled read/write separation (#466)
# ---------------------------------------------------------------------------

_conn_managers: dict[str, ConnectionManager] = {}
_conn_managers_lock = __import__("threading").Lock()


def _get_conn_manager(workspace: str) -> ConnectionManager:
    """Return a shared ConnectionManager for *workspace*.

    Ensures the index directory exists and caches one manager per db_path.
    """
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _conn_managers_lock:
        mgr = _conn_managers.get(path)
        if mgr is None:
            mgr = ConnectionManager(path)
            _conn_managers[path] = mgr
    return mgr


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blocks (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            file        TEXT NOT NULL,
            line        INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL DEFAULT '',
            date        TEXT NOT NULL DEFAULT '',
            speaker     TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '',
            dia_id      TEXT NOT NULL DEFAULT '',
            parent_id   TEXT NOT NULL DEFAULT '',
            json_blob   TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS xref_edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            PRIMARY KEY (src, dst)
        );

        CREATE TABLE IF NOT EXISTS file_state (
            path    TEXT PRIMARY KEY,
            mtime   REAL NOT NULL,
            size    INTEGER NOT NULL,
            hash    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS block_vectors (
            id        TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            model     TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS block_meta (
            id TEXT PRIMARY KEY,
            importance REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            keywords TEXT DEFAULT '',
            connections TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS index_meta (
            file_path    TEXT NOT NULL,
            block_id     TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            indexed_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (file_path, block_id)
        );
    """)

    # Create standalone FTS5 virtual table (we manage sync ourselves)
    cols = ", ".join(col for col, _ in FTS5_COLUMNS)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts
        USING fts5(block_id, {cols}, tokenize='porter unicode61')
    """)

    # Migration: add parent_id column if missing (existing databases)
    try:
        conn.execute("ALTER TABLE blocks ADD COLUMN parent_id TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Secondary indexes for common query patterns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_parent_id ON blocks(parent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_file ON blocks(file)")

    conn.commit()


# ---------------------------------------------------------------------------
# File State Tracking
# ---------------------------------------------------------------------------


def _file_hash(path: str) -> str:
    """Compute fast hash of file content (first 64KB + size)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(65536))
        h.update(str(os.path.getsize(path)).encode())
    except OSError:
        return ""
    return h.hexdigest()


def _get_changed_files(conn: sqlite3.Connection, workspace: str) -> list[tuple[str, str]]:
    """Return list of (label, rel_path) for corpus files that changed since last index.

    A file is considered changed if:
    - It doesn't exist in file_state table
    - Its mtime or size differs
    - Its hash differs (for mtime-equal but content-changed cases)
    """
    changed = []
    ws = os.path.abspath(workspace)

    for label, rel_path in CORPUS_FILES.items():
        full_path = os.path.join(ws, rel_path)
        if not os.path.isfile(full_path):
            # File doesn't exist — check if it was previously indexed
            row = conn.execute("SELECT path FROM file_state WHERE path = ?", (rel_path,)).fetchone()
            if row:
                changed.append((label, rel_path))  # file was deleted
            continue

        stat = os.stat(full_path)
        row = conn.execute("SELECT mtime, size, hash FROM file_state WHERE path = ?", (rel_path,)).fetchone()

        if row is None:
            changed.append((label, rel_path))
            continue

        if stat.st_mtime != row["mtime"] or stat.st_size != row["size"]:
            changed.append((label, rel_path))
            continue

        # Same mtime+size but verify hash for edge cases
        current_hash = _file_hash(full_path)
        if current_hash != row["hash"]:
            changed.append((label, rel_path))

    return changed


def _update_file_state(conn: sqlite3.Connection, workspace: str, rel_path: str) -> None:
    """Update file_state for a corpus file."""
    ws = os.path.abspath(workspace)
    full_path = os.path.join(ws, rel_path)

    if not os.path.isfile(full_path):
        conn.execute("DELETE FROM file_state WHERE path = ?", (rel_path,))
        return

    stat = os.stat(full_path)
    h = _file_hash(full_path)
    conn.execute(
        "INSERT OR REPLACE INTO file_state (path, mtime, size, hash) VALUES (?, ?, ?, ?)",
        (rel_path, stat.st_mtime, stat.st_size, h),
    )


# ---------------------------------------------------------------------------
# Block-level hashing
# ---------------------------------------------------------------------------


def _compute_block_hash(block: dict) -> str:
    """Compute content hash of a parsed block for change detection.

    Hashes a stable JSON representation of all fields except _line
    (which changes when blocks above shift). Uses SHA-256, stdlib only.
    """
    # Copy without volatile fields
    stable = {k: v for k, v in block.items() if k != "_line"}
    raw = json.dumps(stable, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _extract_fts_fields(block: dict) -> dict:
    """Extract FTS5 column values from a block."""
    return {
        "statement": block.get("Statement", ""),
        "title": block.get("Title", "") or block.get("Name", ""),
        "name": block.get("Name", ""),
        "description": block.get("Description", "") or block.get("Summary", ""),
        "tags": block.get("Tags", "") or block.get("Keywords", ""),
        "context": block.get("Context", "") or block.get("Rationale", ""),
        "all_text": " ".join(str(block.get(f, "")) for f in SEARCH_FIELDS if block.get(f)),
    }


def _extract_xrefs(block: dict, all_block_ids: set) -> list[str]:
    """Extract cross-reference IDs from a block."""
    texts = []
    xref_fields = SEARCH_FIELDS + [
        "Supersedes",
        "SupersededBy",
        "AlignsWith",
        "Dependencies",
        "Next",
        "Sources",
        "Evidence",
        "Rollback",
        "History",
    ]
    for field in xref_fields:
        val = block.get(field, "")
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            texts.extend(str(v) for v in val)

    full_text = " ".join(texts)
    bid = block.get("_id", "")
    refs = []
    for m in _BLOCK_ID_RE.finditer(full_text):
        ref_id = m.group(1)
        if ref_id != bid and ref_id in all_block_ids:
            refs.append(ref_id)
    return refs


def _index_file(
    conn: sqlite3.Connection,
    workspace: str,
    label: str,
    rel_path: str,
    all_block_ids: set,
    force: bool = False,
) -> dict:
    """Index a single corpus file with block-level incremental updates.

    Returns dict with counts: new, modified, deleted, unchanged, total.
    When force=True, skips hash comparison and re-indexes all blocks.
    """
    ws = os.path.abspath(workspace)
    full_path = os.path.join(ws, rel_path)

    counts = {"new": 0, "modified": 0, "deleted": 0, "unchanged": 0, "total": 0}

    # Load existing block hashes for this file
    existing_hashes = {}
    for row in conn.execute(
        "SELECT block_id, content_hash FROM index_meta WHERE file_path = ?",
        (rel_path,),
    ).fetchall():
        existing_hashes[row["block_id"]] = row["content_hash"]

    # Handle deleted file
    if not os.path.isfile(full_path):
        if existing_hashes:
            old_ids = list(existing_hashes.keys())
            _delete_blocks(conn, old_ids, rel_path)
            counts["deleted"] = len(old_ids)
        _update_file_state(conn, workspace, rel_path)
        return counts

    try:
        blocks = parse_file(full_path)
    except (OSError, UnicodeDecodeError, ValueError):
        _update_file_state(conn, workspace, rel_path)
        return counts

    # Build current block map: {block_id: (block_dict, content_hash)}
    current_blocks = {}
    for block in blocks:
        bid = block.get("_id", "")
        if not bid:
            continue
        current_blocks[bid] = (block, _compute_block_hash(block))

    current_ids = set(current_blocks.keys())
    existing_ids = set(existing_hashes.keys())

    # Classify blocks
    new_ids = current_ids - existing_ids
    deleted_ids = existing_ids - current_ids
    common_ids = current_ids & existing_ids

    modified_ids = set()
    unchanged_ids = set()
    for bid in common_ids:
        if force or current_blocks[bid][1] != existing_hashes[bid]:
            modified_ids.add(bid)
        else:
            unchanged_ids.add(bid)

    # Delete removed blocks
    if deleted_ids:
        _delete_blocks(conn, list(deleted_ids), rel_path)

    # Delete modified blocks (will be re-inserted)
    if modified_ids:
        _delete_blocks(conn, list(modified_ids), rel_path)

    # Insert new + modified blocks
    for bid in new_ids | modified_ids:
        block, content_hash = current_blocks[bid]
        _insert_block(conn, block, bid, rel_path, all_block_ids)
        conn.execute(
            """INSERT OR REPLACE INTO index_meta
               (file_path, block_id, content_hash)
               VALUES (?, ?, ?)""",
            (rel_path, bid, content_hash),
        )

    # Update index_meta for unchanged blocks (keep existing entries)
    # No-op — they're already correct in index_meta

    # Clean up index_meta for deleted blocks
    if deleted_ids:
        placeholders = ",".join("?" for _ in deleted_ids)
        conn.execute(
            f"DELETE FROM index_meta WHERE file_path = ? AND block_id IN ({placeholders})",
            [rel_path] + list(deleted_ids),
        )

    counts["new"] = len(new_ids)
    counts["modified"] = len(modified_ids)
    counts["deleted"] = len(deleted_ids)
    counts["unchanged"] = len(unchanged_ids)
    counts["total"] = len(new_ids) + len(modified_ids)

    _update_file_state(conn, workspace, rel_path)
    return counts


def _insert_block(
    conn: sqlite3.Connection,
    block: dict,
    bid: str,
    rel_path: str,
    all_block_ids: set,
) -> None:
    """Insert a single block into blocks, blocks_fts, and xref_edges.

    Also extracts atomic fact cards from the block's Statement field and indexes
    them as sub-blocks (parent_id = bid) for small-to-big retrieval.
    """
    tags_str = block.get("Tags", "")
    speaker = _parse_speaker_from_tags(tags_str)

    conn.execute(
        """INSERT OR REPLACE INTO blocks
           (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            bid,
            get_block_type(bid),
            rel_path,
            block.get("_line", 0),
            block.get("Status", ""),
            block.get("Date", ""),
            speaker,
            tags_str,
            block.get("DiaID", ""),
            "",  # parent_id — empty for top-level blocks
            json.dumps(block, default=str),
        ),
    )

    fts = _extract_fts_fields(block)
    conn.execute(
        """INSERT INTO blocks_fts (block_id, statement, title, name,
           description, tags, context, all_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            bid,
            fts["statement"],
            fts["title"],
            fts["name"],
            fts["description"],
            fts["tags"],
            fts["context"],
            fts["all_text"],
        ),
    )

    refs = _extract_xrefs(block, all_block_ids)
    for ref in refs:
        conn.execute(
            "INSERT OR IGNORE INTO xref_edges (src, dst) VALUES (?, ?)",
            (bid, ref),
        )
        conn.execute(
            "INSERT OR IGNORE INTO xref_edges (src, dst) VALUES (?, ?)",
            (ref, bid),
        )

    # --- Feature 2: Extract and index atomic fact cards as sub-blocks ---
    statement = block.get("Statement", "")
    if statement and len(statement) > 15:
        block_date = block.get("Date", "")
        try:
            facts = extract_facts(statement, speaker=speaker, date=block_date, source_id=bid)
        except (ValueError, TypeError):
            facts = []
        for i, card in enumerate(facts):
            fact_id = f"{bid}::F{i + 1}"
            fact_tags = card.get("type", "FACT")
            if card.get("speaker"):
                fact_tags += f", {card['speaker']}"
            fact_block = {
                "Statement": card["content"],
                "Tags": fact_tags,
                "Date": card.get("date", block_date),
                "Status": block.get("Status", "active"),
                "_id": fact_id,
                "_parent_id": bid,
            }
            conn.execute(
                """INSERT OR REPLACE INTO blocks
                   (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact_id,
                    card.get("type", "FACT"),
                    rel_path,
                    block.get("_line", 0),
                    block.get("Status", "active"),
                    card.get("date", block_date),
                    card.get("speaker", ""),
                    fact_tags,
                    block.get("DiaID", ""),
                    bid,
                    json.dumps(fact_block, default=str),
                ),
            )
            fact_fts = _extract_fts_fields(fact_block)
            conn.execute(
                """INSERT INTO blocks_fts (block_id, statement, title, name,
                   description, tags, context, all_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact_id,
                    fact_fts["statement"],
                    fact_fts["title"],
                    fact_fts["name"],
                    fact_fts["description"],
                    fact_fts["tags"],
                    fact_fts["context"],
                    fact_fts["all_text"],
                ),
            )


def _delete_blocks(
    conn: sqlite3.Connection,
    block_ids: list,
    rel_path: str,
) -> None:
    """Delete blocks from blocks, blocks_fts, and xref_edges.

    Also deletes child fact sub-blocks (parent_id matching any deleted block).
    """
    if not block_ids:
        return
    placeholders = ",".join("?" for _ in block_ids)

    # Find child fact sub-blocks before deleting parents
    child_rows = conn.execute(
        f"SELECT id FROM blocks WHERE parent_id IN ({placeholders})",
        block_ids,
    ).fetchall()
    child_ids = [r["id"] for r in child_rows]

    all_ids = block_ids + child_ids
    all_ph = ",".join("?" for _ in all_ids)

    conn.execute(f"DELETE FROM blocks WHERE id IN ({all_ph})", all_ids)
    conn.execute(f"DELETE FROM blocks_fts WHERE block_id IN ({all_ph})", all_ids)
    conn.execute(
        f"DELETE FROM xref_edges WHERE src IN ({all_ph}) OR dst IN ({all_ph})",
        all_ids + all_ids,
    )


def build_index(workspace: str, incremental: bool = True) -> dict:
    """Build or incrementally update the FTS5 index.

    Uses ConnectionManager (#466) for write connection pooling with
    chunked commits (one commit per file) to reduce lock hold time.

    Args:
        workspace: Workspace root path.
        incremental: If True, only re-index changed files. If False, rebuild all.

    Returns:
        Summary dict with files_checked, files_indexed, blocks_indexed, elapsed_ms.
    """
    ws = os.path.abspath(workspace)
    start = datetime.now()

    mgr = _get_conn_manager(workspace)
    with mgr.write_lock:
        conn = mgr.get_write_connection()
        conn.row_factory = sqlite3.Row
        try:
            _init_schema(conn)

            # Collect all block IDs for xref resolution
            all_block_ids = set()
            for label, rel_path in CORPUS_FILES.items():
                path = os.path.join(ws, rel_path)
                if os.path.isfile(path):
                    try:
                        for b in parse_file(path):
                            bid = b.get("_id", "")
                            if bid:
                                all_block_ids.add(bid)
                    except (OSError, UnicodeDecodeError, ValueError) as e:
                        _log.debug("xref_scan_parse_failed", file=rel_path, error=str(e))

            force = not incremental
            if incremental:
                changed = _get_changed_files(conn, workspace)
            else:
                changed = list(CORPUS_FILES.items())
                # Clear file_state and index_meta for full rebuild
                conn.execute("DELETE FROM file_state")
                conn.execute("DELETE FROM index_meta")

            total_blocks = 0
            total_new = 0
            total_modified = 0
            total_deleted = 0
            total_unchanged = 0
            for label, rel_path in changed:
                counts = _index_file(conn, workspace, label, rel_path, all_block_ids, force=force)
                total_blocks += counts["total"]
                total_new += counts["new"]
                total_modified += counts["modified"]
                total_deleted += counts["deleted"]
                total_unchanged += counts["unchanged"]
                # Chunked commit: commit after each file to reduce lock hold time
                conn.commit()
                _log.info(
                    "indexed_file",
                    file=rel_path,
                    new=counts["new"],
                    modified=counts["modified"],
                    deleted=counts["deleted"],
                    unchanged=counts["unchanged"],
                )

            # Update metadata
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("last_build", datetime.now().isoformat()),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("build_mode", "incremental" if incremental else "full"),
            )

            conn.commit()

            elapsed = (datetime.now() - start).total_seconds() * 1000
            summary = {
                "files_checked": len(CORPUS_FILES),
                "files_indexed": len(changed),
                "blocks_indexed": total_blocks,
                "blocks_new": total_new,
                "blocks_modified": total_modified,
                "blocks_deleted": total_deleted,
                "blocks_unchanged": total_unchanged,
                "elapsed_ms": round(elapsed, 1),
            }

            # Count total blocks in index
            row = conn.execute("SELECT COUNT(*) as cnt FROM blocks").fetchone()
            summary["total_blocks"] = row["cnt"]
        finally:
            # Don't close the manager — it's shared and cached
            pass

    _log.info("build_complete", **summary)
    metrics.inc("index_builds")
    metrics.inc("index_blocks_indexed", total_blocks)
    return summary


# ---------------------------------------------------------------------------
# Fact aggregation — small-to-big retrieval
# ---------------------------------------------------------------------------


def _aggregate_facts_to_parents(
    conn: sqlite3.Connection,
    results: list[dict],
) -> list[dict]:
    """Merge fact sub-block scores into their parent blocks.

    Fact sub-blocks have IDs like "D-20230507-001::F1". This function:
    1. Groups fact sub-blocks by parent ID (everything before "::F")
    2. Boosts parent score by the best fact sub-block score
    3. Removes fact sub-blocks from results (folded into parent)
    4. If a parent is not already in results, fetches it from the DB

    Returns results with fact sub-blocks replaced by boosted parents.
    """
    if not results:
        return results

    # Separate fact sub-blocks and regular blocks
    fact_scores: dict[str, float] = {}  # parent_id -> max fact score
    regular = []
    result_ids = set()

    for r in results:
        rid = r.get("_id", "")
        if "::F" in rid:
            parent_id = rid.split("::F")[0]
            score = r.get("score", 0)
            if score > fact_scores.get(parent_id, 0):
                fact_scores[parent_id] = score
        else:
            regular.append(r)
            result_ids.add(rid)

    if not fact_scores:
        return results

    # Boost parents already in results
    boosted = set()
    for r in regular:
        rid = r.get("_id", "")
        if rid in fact_scores:
            fact_sc = fact_scores[rid]
            parent_sc = r.get("score", 0)
            r["score"] = round(max(parent_sc, fact_sc * 0.8 + parent_sc * 0.2), 4)
            r["_fact_boost"] = True
            boosted.add(rid)

    # Inject parents not in results (fact card matched but parent didn't)
    missing = set(fact_scores.keys()) - result_ids
    if missing:
        placeholders = ",".join("?" for _ in missing)
        rows = conn.execute(
            f"SELECT * FROM blocks WHERE id IN ({placeholders}) AND parent_id = ''",
            list(missing),
        ).fetchall()
        for row in rows:
            block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
            regular.append(
                {
                    "_id": row["id"],
                    "type": row["type"],
                    "score": round(fact_scores[row["id"]] * 0.8, 4),
                    "excerpt": get_excerpt(block_data),
                    "speaker": row["speaker"],
                    "tags": row["tags"],
                    "file": row["file"],
                    "line": row["line"],
                    "status": row["status"],
                    "_fact_boost": True,
                }
            )

    _log.debug(
        "fact_aggregation",
        facts_found=sum(1 for r in results if "::F" in r.get("_id", "")),
        parents_boosted=len(boosted),
        parents_injected=len(missing),
    )
    return regular


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_index(
    workspace: str,
    query: str,
    limit: int = 10,
    active_only: bool = False,
    graph_boost: bool = False,
    retrieve_wide_k: int = 200,
    rerank: bool = True,
    rerank_debug: bool = False,
) -> list[dict]:
    """Query the FTS5 index. Returns ranked results matching recall() format.

    Falls back to filesystem scan (recall.recall()) if index doesn't exist.
    """
    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        _log.info("index_missing_fallback", db=db_path)
        from .recall import recall

        return recall(
            workspace,
            query,
            limit=limit,
            active_only=active_only,
            graph_boost=graph_boost,
            retrieve_wide_k=retrieve_wide_k,
            rerank=rerank,
            rerank_debug=rerank_debug,
        )

    # Initialize calibration manager (optional — graceful degradation)
    _cal_mgr = None
    try:
        from .calibration import CalibrationManager

        _cal_mgr = CalibrationManager(workspace)
    except (ImportError, Exception) as _cal_err:
        _log.debug("calibration_unavailable_in_fts", error=str(_cal_err))

    # Staleness check: warn but don't auto-rebuild (#34)
    if is_stale(workspace):
        _log.info("index_stale", hint="Run 'reindex' tool to update the FTS5 index")
        metrics.inc("index_stale_queries")

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_type = detect_query_type(query)
    qparams: dict = _QUERY_TYPE_PARAMS.get(query_type, _QUERY_TYPE_PARAMS["single-hop"])  # type: ignore[assignment]

    # Month normalization
    query_tokens = expand_months(query, query_tokens)

    expand_mode = qparams.get("expand_query", True)
    if expand_mode:
        mode = expand_mode if isinstance(expand_mode, str) else "full"
        query_tokens = expand_query(query_tokens, mode=mode)

    if qparams.get("graph_boost_override", False):
        graph_boost = True

    mgr = _get_conn_manager(workspace)
    conn = mgr.get_read_connection()
    conn.row_factory = sqlite3.Row

    # Build FTS5 MATCH query from tokens
    # Quote each token to prevent FTS5 operator injection (NOT, AND, NEAR, etc.)
    # Also reject tokens that aren't alphanumeric to prevent wildcard injection (e.g. "*")
    fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in query_tokens if _FTS5_SAFE.match(t))
    if not fts_query:
        return []

    try:
        # FTS5 bm25() returns negative scores (lower = better)
        # We negate to get positive scores (higher = better)
        weights = ", ".join(str(w) for _, w in FTS5_COLUMNS)
        rows = conn.execute(
            f"""SELECT b.*, f.rank as fts_rank,
                       -bm25(blocks_fts, {weights}) as bm25_score
                FROM blocks_fts f
                JOIN blocks b ON b.id = f.block_id
                WHERE blocks_fts MATCH ?
                ORDER BY bm25_score DESC
                LIMIT ?""",
            (fts_query, max(retrieve_wide_k, limit)),
        ).fetchall()
    except sqlite3.OperationalError as e:
        _log.warning(
            "fts_query_error_fallback",
            error=str(e),
            query=fts_query,
            msg="FTS5 query failed, falling back to in-memory BM25 scan",
        )
        # Fallback to filesystem scan — results are still valid but may be slower
        from .recall import recall

        fallback_results = recall(
            workspace,
            query,
            limit=limit,
            active_only=active_only,
            graph_boost=graph_boost,
            retrieve_wide_k=retrieve_wide_k,
            rerank=rerank,
            rerank_debug=rerank_debug,
        )
        for r in fallback_results:
            r["_fallback"] = "bm25_scan"
        return fallback_results

    results = []
    for row in rows:
        if active_only and row["status"] not in ("active", TaskStatus.TODO.value, TaskStatus.DOING.value, "open"):
            continue

        score = row["bm25_score"]

        # Apply same post-processing as recall.py
        # Recency boost
        block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
        recency = date_score(block_data)
        rw = qparams.get("recency_weight", 0.3)
        score *= 1.0 - rw + rw * recency

        # Temporal date boost
        date_boost = qparams.get("date_boost", 1.0)
        if date_boost > 1.0 and row["date"]:
            score *= date_boost

        # Status boost
        if row["status"] == "active":
            score *= 1.2
        elif row["status"] in (TaskStatus.TODO.value, TaskStatus.DOING.value):
            score *= 1.1

        # Priority boost
        priority = block_data.get("Priority", "")
        if priority in ("P0", "P1"):
            score *= 1.1

        # Calibration feedback weight
        if _cal_mgr is not None:
            try:
                cal_weight = _cal_mgr.get_block_weight(row["id"])
                score *= cal_weight
            except Exception:
                pass  # Graceful degradation — skip calibration for this block

        result = {
            "_id": row["id"],
            "type": row["type"],
            "score": round(score, 4),
            "excerpt": get_excerpt(block_data),
            "speaker": row["speaker"],
            "tags": row["tags"],
            "file": row["file"],
            "line": row["line"],
            "status": row["status"],
        }
        if row["dia_id"]:
            result["DiaID"] = row["dia_id"]
        if row["date"]:
            result["Date"] = row["date"]
        results.append(result)

    # --- Feature 2: Aggregate fact sub-blocks to parents (small-to-big) ---
    results = _aggregate_facts_to_parents(conn, results)

    # Graph boost
    if graph_boost and results:
        _apply_graph_boost(conn, results, query_type)

    # Note: read connection is managed by ConnectionManager — not closed here (#466)

    # Sort by score, then by block ID for deterministic tiebreaking
    results.sort(key=lambda r: (-r["score"], r.get("_id", "")))

    # Dedup
    seen_keys = set()
    deduped = []
    for r in results:
        stable_key = (r.get("file", ""), r.get("line", 0))
        if stable_key != ("", 0) and stable_key in seen_keys:
            continue
        if stable_key != ("", 0):
            seen_keys.add(stable_key)
        dia = r.get("DiaID", "")
        if dia:
            rid = r.get("_id", "")
            prefix = "FACT" if rid.startswith("FACT-") else "DIA" if rid.startswith("DIA-") else rid[:4]
            dia_key = (dia, prefix)
            if dia_key in seen_keys:
                continue
            seen_keys.add(dia_key)
        deduped.append(r)

    # Rerank — cap candidates to prevent latency spikes (#9)
    if rerank and len(deduped) > limit:
        rerank_cap = min(len(deduped), 200)
        deduped = rerank_hits(query, deduped[:rerank_cap], debug=rerank_debug)

    top = deduped[:limit]

    _log.info(
        "query_complete",
        query=query,
        query_type=query_type,
        fts_hits=len(rows),
        results=len(top),
        top_score=top[0]["score"] if top else 0,
    )
    metrics.inc("index_queries")
    return top


def _apply_graph_boost(
    conn: sqlite3.Connection,
    results: list[dict],
    query_type: str,
) -> None:
    """Apply cross-reference graph boost to results using xref_edges table."""
    from .recall import GRAPH_BOOST_FACTOR

    score_by_id = {r["_id"]: r["score"] for r in results}
    result_ids = set(score_by_id.keys())

    hop_decays = [GRAPH_BOOST_FACTOR, GRAPH_BOOST_FACTOR * 0.5]
    if query_type == "multi-hop":
        hop_decays.append(GRAPH_BOOST_FACTOR * 0.25)

    neighbor_scores: dict[str, float] = {}

    for hop, decay in enumerate(hop_decays):
        seed_ids = list(result_ids) if hop == 0 else [nid for nid in neighbor_scores if nid not in result_ids]
        if not seed_ids:
            break

        # Sanitize: only allow string IDs matching block ID format
        seed_ids = [sid for sid in seed_ids if isinstance(sid, str) and len(sid) < 100]
        if not seed_ids:
            break

        placeholders = ",".join("?" for _ in seed_ids)
        edges = conn.execute(
            f"SELECT src, dst FROM xref_edges WHERE src IN ({placeholders})",
            seed_ids,
        ).fetchall()

        hop_added = 0
        for edge in edges:
            src, dst = edge["src"], edge["dst"]
            src_score: float = score_by_id.get(src, neighbor_scores.get(src, 0)) or 0.0
            boost = src_score * decay
            if dst not in result_ids:
                if hop_added >= 50:  # Cap neighbors per hop
                    break
                neighbor_scores[dst] = neighbor_scores.get(dst, 0) + boost
                hop_added += 1
            else:
                neighbor_scores[dst] = neighbor_scores.get(dst, 0) + boost * 0.5

    # Apply boosts to existing results
    for r in results:
        if r["_id"] in neighbor_scores:
            r["score"] = round(r["score"] + neighbor_scores[r["_id"]], 4)
            r["via_graph"] = True

    # Add new graph-discovered results
    if neighbor_scores:
        new_ids = [nid for nid in neighbor_scores if nid not in result_ids]
        if new_ids:
            placeholders = ",".join("?" for _ in new_ids)
            rows = conn.execute(
                f"SELECT * FROM blocks WHERE id IN ({placeholders})",
                new_ids,
            ).fetchall()
            for row in rows:
                block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
                results.append(
                    {
                        "_id": row["id"],
                        "type": row["type"],
                        "score": round(neighbor_scores[row["id"]], 4),
                        "excerpt": get_excerpt(block_data),
                        "speaker": row["speaker"],
                        "tags": row["tags"],
                        "file": row["file"],
                        "line": row["line"],
                        "status": row["status"],
                        "via_graph": True,
                    }
                )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def is_stale(workspace: str) -> bool:
    """Check whether any corpus .md files have changed since last index build.

    Returns True if the index doesn't exist or any file mtime differs from
    the recorded state.  This is a lightweight O(files) check that avoids
    hashing -- suitable for a pre-query gate.
    """
    db = _db_path(workspace)
    if not os.path.isfile(db):
        return True
    try:
        conn = _connect(workspace, readonly=True)
        changed = _get_changed_files(conn, workspace)
        conn.close()
        return len(changed) > 0
    except (OSError, sqlite3.OperationalError):
        return True


def merkle_leaves(workspace: str) -> list[tuple[str, str]]:
    """Return (block_id, content_hash) tuples for Merkle tree construction.

    Used by :func:`mcp_server.verify_merkle` and the standalone
    ``mind-mem-verify`` CLI. Leaves are sorted by block_id so two calls
    against the same index produce the same tree (stable root hash).

    Content hashes live on the ``index_meta`` table (``blocks`` itself
    doesn't carry one — it has ``json_blob`` and friends). The join
    below preserves the invariant that a leaf is only emitted when a
    live ``blocks`` row backs the ``index_meta`` content hash.

    Returns an empty list when the FTS index has not yet been built.
    """
    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        return []
    conn = None
    try:
        conn = _connect(workspace, readonly=True)
        _init_schema(conn)
        rows = conn.execute(
            """
            SELECT im.block_id AS block_id, im.content_hash AS content_hash
            FROM index_meta im
            JOIN blocks b ON b.id = im.block_id
            WHERE im.content_hash IS NOT NULL AND im.content_hash != ''
            ORDER BY im.block_id
            """
        ).fetchall()
        # De-dupe when the same block_id appears in multiple files.
        seen: set[str] = set()
        leaves: list[tuple[str, str]] = []
        for r in rows:
            bid = r["block_id"]
            if bid in seen:
                continue
            seen.add(bid)
            leaves.append((bid, r["content_hash"]))
        return leaves
    finally:
        if conn is not None:
            conn.close()


def index_status(workspace: str) -> dict:
    """Return index status: exists, block count, last build time, staleness."""
    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        return {"exists": False, "blocks": 0, "stale_files": len(CORPUS_FILES)}

    conn = None
    try:
        conn = _connect(workspace, readonly=True)
        _init_schema(conn)
        row = conn.execute("SELECT COUNT(*) as cnt FROM blocks").fetchone()
        block_count = row["cnt"]

        last_build = conn.execute("SELECT value FROM meta WHERE key = 'last_build'").fetchone()

        changed = _get_changed_files(conn, workspace)

        return {
            "exists": True,
            "blocks": block_count,
            "last_build": last_build["value"] if last_build else None,
            "stale_files": len(changed),
            "db_size_bytes": os.path.getsize(db_path),
        }
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Mind Mem SQLite FTS5 Index")
    sub = parser.add_subparsers(dest="command")

    bp = sub.add_parser("build", help="Build or update the index")
    bp.add_argument("--workspace", "-w", default=".", help="Workspace path")
    bp.add_argument("--full", action="store_true", help="Full rebuild (not incremental)")

    qp = sub.add_parser("query", help="Query the index")
    qp.add_argument("--workspace", "-w", default=".", help="Workspace path")
    qp.add_argument("--query", "-q", required=True, help="Search query")
    qp.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    qp.add_argument("--active-only", action="store_true")
    qp.add_argument("--graph", action="store_true")
    qp.add_argument("--json", action="store_true")
    qp.add_argument("--no-rerank", action="store_true")
    qp.add_argument("--rerank-debug", action="store_true")

    sp = sub.add_parser("status", help="Show index status")
    sp.add_argument("--workspace", "-w", default=".", help="Workspace path")

    args = parser.parse_args()

    if args.command == "build":
        ws = os.path.abspath(args.workspace)
        result = build_index(ws, incremental=not args.full)
        print("Index build complete:")
        print(f"  Files checked: {result['files_checked']}")
        print(f"  Files indexed: {result['files_indexed']}")
        print(
            f"  Blocks: {result['blocks_new']} new, {result['blocks_modified']} modified, "
            f"{result['blocks_deleted']} deleted, {result['blocks_unchanged']} unchanged"
        )
        print(f"  Total blocks: {result['total_blocks']}")
        print(f"  Elapsed: {result['elapsed_ms']:.0f}ms")

    elif args.command == "query":
        ws = os.path.abspath(args.workspace)
        results = query_index(
            ws,
            args.query,
            limit=args.limit,
            active_only=args.active_only,
            graph_boost=args.graph,
            rerank=not args.no_rerank,
            rerank_debug=args.rerank_debug,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if not results:
                print("No results found.")
            else:
                for r in results:
                    graph_tag = " [graph]" if r.get("via_graph") else ""
                    print(f"[{r['score']:.3f}] {r['_id']} ({r['type']}{graph_tag}) — {r['excerpt'][:80]}")
                    print(f"        {r['file']}:{r['line']}")

    elif args.command == "status":
        ws = os.path.abspath(args.workspace)
        status = index_status(ws)
        if not status["exists"]:
            print("No index found. Run 'build' to create one.")
        else:
            print("Index status:")
            print(f"  Blocks: {status['blocks']}")
            print(f"  Last build: {status['last_build']}")
            print(f"  Stale files: {status['stale_files']}")
            print(f"  DB size: {status['db_size_bytes']} bytes")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
