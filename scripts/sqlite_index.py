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
    python3 scripts/sqlite_index.py build --workspace .
    python3 scripts/sqlite_index.py build --workspace . --incremental
    python3 scripts/sqlite_index.py query --workspace . --query "PostgreSQL"
    python3 scripts/sqlite_index.py status --workspace .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_parser import parse_file
from observability import get_logger, metrics
from recall import (
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


def _db_path(workspace: str) -> str:
    """Return absolute path to the index database."""
    return os.path.join(os.path.abspath(workspace), DB_REL_PATH)


def _connect(workspace: str, readonly: bool = False) -> sqlite3.Connection:
    """Open (or create) the index database."""
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if readonly:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


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
    """)

    # Create standalone FTS5 virtual table (we manage sync ourselves)
    cols = ", ".join(col for col, _ in FTS5_COLUMNS)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts
        USING fts5(block_id, {cols}, tokenize='porter unicode61')
    """)

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
            row = conn.execute(
                "SELECT path FROM file_state WHERE path = ?", (rel_path,)
            ).fetchone()
            if row:
                changed.append((label, rel_path))  # file was deleted
            continue

        stat = os.stat(full_path)
        row = conn.execute(
            "SELECT mtime, size, hash FROM file_state WHERE path = ?",
            (rel_path,)
        ).fetchone()

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
        "all_text": " ".join(
            str(block.get(f, "")) for f in SEARCH_FIELDS if block.get(f)
        ),
    }


def _extract_xrefs(block: dict, all_block_ids: set) -> list[str]:
    """Extract cross-reference IDs from a block."""
    texts = []
    xref_fields = SEARCH_FIELDS + [
        "Supersedes", "SupersededBy", "AlignsWith", "Dependencies",
        "Next", "Sources", "Evidence", "Rollback", "History",
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
) -> int:
    """Index a single corpus file. Returns number of blocks indexed."""
    ws = os.path.abspath(workspace)
    full_path = os.path.join(ws, rel_path)

    # Remove old blocks from this file
    old_ids = [
        row["id"] for row in
        conn.execute("SELECT id FROM blocks WHERE file = ?", (rel_path,)).fetchall()
    ]
    if old_ids:
        placeholders = ",".join("?" for _ in old_ids)
        conn.execute(f"DELETE FROM blocks WHERE id IN ({placeholders})", old_ids)
        conn.execute(f"DELETE FROM blocks_fts WHERE block_id IN ({placeholders})", old_ids)
        conn.execute(
            f"DELETE FROM xref_edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
            old_ids + old_ids,
        )

    if not os.path.isfile(full_path):
        _update_file_state(conn, workspace, rel_path)
        return 0

    try:
        blocks = parse_file(full_path)
    except (OSError, UnicodeDecodeError, ValueError):
        _update_file_state(conn, workspace, rel_path)
        return 0

    count = 0
    for block in blocks:
        bid = block.get("_id", "")
        if not bid:
            continue

        tags_str = block.get("Tags", "")
        speaker = _parse_speaker_from_tags(tags_str)

        # Insert into blocks table
        conn.execute(
            """INSERT OR REPLACE INTO blocks
               (id, type, file, line, status, date, speaker, tags, dia_id, json_blob)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                json.dumps(block, default=str),
            ),
        )

        # Insert into FTS5
        fts = _extract_fts_fields(block)
        conn.execute(
            """INSERT INTO blocks_fts (block_id, statement, title, name,
               description, tags, context, all_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (bid, fts["statement"], fts["title"], fts["name"],
             fts["description"], fts["tags"], fts["context"], fts["all_text"]),
        )

        # Insert xref edges
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

        count += 1

    _update_file_state(conn, workspace, rel_path)
    return count


def build_index(workspace: str, incremental: bool = True) -> dict:
    """Build or incrementally update the FTS5 index.

    Args:
        workspace: Workspace root path.
        incremental: If True, only re-index changed files. If False, rebuild all.

    Returns:
        Summary dict with files_checked, files_indexed, blocks_indexed, elapsed_ms.
    """
    ws = os.path.abspath(workspace)
    start = datetime.now()

    conn = _connect(workspace)
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

    if incremental:
        changed = _get_changed_files(conn, workspace)
    else:
        changed = list(CORPUS_FILES.items())
        # Clear all data for full rebuild
        conn.execute("DELETE FROM blocks")
        conn.execute("DELETE FROM blocks_fts")
        conn.execute("DELETE FROM xref_edges")
        conn.execute("DELETE FROM file_state")

    total_blocks = 0
    for label, rel_path in changed:
        count = _index_file(conn, workspace, label, rel_path, all_block_ids)
        total_blocks += count
        _log.info("indexed_file", file=rel_path, blocks=count)

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
        "elapsed_ms": round(elapsed, 1),
    }

    # Count total blocks in index
    row = conn.execute("SELECT COUNT(*) as cnt FROM blocks").fetchone()
    summary["total_blocks"] = row["cnt"]

    conn.close()

    _log.info("build_complete", **summary)
    metrics.inc("index_builds")
    metrics.inc("index_blocks_indexed", total_blocks)
    return summary


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
        from recall import recall
        return recall(
            workspace, query, limit=limit, active_only=active_only,
            graph_boost=graph_boost, retrieve_wide_k=retrieve_wide_k,
            rerank=rerank, rerank_debug=rerank_debug,
        )

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_type = detect_query_type(query)
    qparams = _QUERY_TYPE_PARAMS.get(query_type, _QUERY_TYPE_PARAMS["single-hop"])

    # Month normalization
    query_tokens = expand_months(query, query_tokens)

    expand_mode = qparams.get("expand_query", True)
    if expand_mode:
        mode = expand_mode if isinstance(expand_mode, str) else "full"
        query_tokens = expand_query(query_tokens, mode=mode)

    if qparams.get("graph_boost_override", False):
        graph_boost = True

    conn = _connect(workspace, readonly=True)

    # Build FTS5 MATCH query from tokens
    # Quote each token to prevent FTS5 operator injection (NOT, AND, NEAR, etc.)
    fts_query = " OR ".join(f'"{t}"' for t in query_tokens)

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
        _log.warning("fts_query_error_fallback", error=str(e), query=fts_query,
                      msg="FTS5 query failed, falling back to in-memory BM25 scan")
        conn.close()
        # Fallback to filesystem scan — results are still valid but may be slower
        from recall import recall
        fallback_results = recall(
            workspace, query, limit=limit, active_only=active_only,
            graph_boost=graph_boost, retrieve_wide_k=retrieve_wide_k,
            rerank=rerank, rerank_debug=rerank_debug,
        )
        for r in fallback_results:
            r["_fallback"] = "bm25_scan"
        return fallback_results

    results = []
    for row in rows:
        if active_only and row["status"] not in ("active", "todo", "doing", "open"):
            continue

        score = row["bm25_score"]

        # Apply same post-processing as recall.py
        # Recency boost
        block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
        recency = date_score(block_data)
        rw = qparams.get("recency_weight", 0.3)
        score *= (1.0 - rw + rw * recency)

        # Temporal date boost
        date_boost = qparams.get("date_boost", 1.0)
        if date_boost > 1.0 and row["date"]:
            score *= date_boost

        # Status boost
        if row["status"] == "active":
            score *= 1.2
        elif row["status"] in ("todo", "doing"):
            score *= 1.1

        # Priority boost
        priority = block_data.get("Priority", "")
        if priority in ("P0", "P1"):
            score *= 1.1

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
        results.append(result)

    # Graph boost
    if graph_boost and results:
        _apply_graph_boost(conn, results, query_type)

    conn.close()

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

    # Rerank
    if rerank and len(deduped) > limit:
        deduped = rerank_hits(query, deduped, debug=rerank_debug)

    top = deduped[:limit]

    _log.info("query_complete", query=query, query_type=query_type,
              fts_hits=len(rows), results=len(top),
              top_score=top[0]["score"] if top else 0)
    metrics.inc("index_queries")
    return top


def _apply_graph_boost(
    conn: sqlite3.Connection,
    results: list[dict],
    query_type: str,
) -> None:
    """Apply cross-reference graph boost to results using xref_edges table."""
    from recall import GRAPH_BOOST_FACTOR

    score_by_id = {r["_id"]: r["score"] for r in results}
    result_ids = set(score_by_id.keys())

    hop_decays = [GRAPH_BOOST_FACTOR, GRAPH_BOOST_FACTOR * 0.5]
    if query_type == "multi-hop":
        hop_decays.append(GRAPH_BOOST_FACTOR * 0.25)

    neighbor_scores = {}

    for hop, decay in enumerate(hop_decays):
        seed_ids = list(result_ids) if hop == 0 else [
            nid for nid in neighbor_scores if nid not in result_ids
        ]
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

        for edge in edges:
            src, dst = edge["src"], edge["dst"]
            src_score = score_by_id.get(src, neighbor_scores.get(src, 0))
            boost = src_score * decay
            if dst not in result_ids:
                neighbor_scores[dst] = neighbor_scores.get(dst, 0) + boost
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
                results.append({
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
                })


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def index_status(workspace: str) -> dict:
    """Return index status: exists, block count, last build time, staleness."""
    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        return {"exists": False, "blocks": 0, "stale_files": len(CORPUS_FILES)}

    conn = _connect(workspace, readonly=True)
    try:
        _init_schema(conn)
        row = conn.execute("SELECT COUNT(*) as cnt FROM blocks").fetchone()
        block_count = row["cnt"]

        last_build = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_build'"
        ).fetchone()

        changed = _get_changed_files(conn, workspace)

        return {
            "exists": True,
            "blocks": block_count,
            "last_build": last_build["value"] if last_build else None,
            "stale_files": len(changed),
            "db_size_bytes": os.path.getsize(db_path),
        }
    finally:
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
        print(f"  Blocks indexed: {result['blocks_indexed']}")
        print(f"  Total blocks: {result['total_blocks']}")
        print(f"  Elapsed: {result['elapsed_ms']:.0f}ms")

    elif args.command == "query":
        ws = os.path.abspath(args.workspace)
        results = query_index(
            ws, args.query, limit=args.limit,
            active_only=args.active_only, graph_boost=args.graph,
            rerank=not args.no_rerank, rerank_debug=args.rerank_debug,
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
