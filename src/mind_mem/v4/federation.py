"""v4 federated cross-agent consistency (Group D).

Round 2 multi-LLM audit (4/4 model consensus 2026-05-10) flagged the
remaining blind spot after round 1's CAS fix landed: **no cross-agent
conflict resolution**. The single-workspace CAS contract works inside
one process; once two agents (or two hosts) write to the same block
independently, last-writer-wins silently and the audit chain says
nothing about the divergence.

This module adds **per-agent version vectors** + an explicit
**conflict log** so divergent writes are detected, recorded, and
resolved by a chosen merge strategy:

    last_writer_wins    Default. Highest agent version wins.
    higher_version      Pick the side with the larger logical clock.
    three_way_merge     Hand the conflict to a caller-supplied
                        merger so the v3 governance layer can route
                        through propose/approve.

Schema additions (lazy on first call, idempotent):

    block_tier_vclock(block_id TEXT, agent_id TEXT, version INTEGER,
                      last_seen_at TEXT, PRIMARY KEY(block_id, agent_id))

    tier_conflict_log(block_id TEXT, detected_at TEXT,
                      left_agent TEXT, left_version INTEGER,
                      right_agent TEXT, right_version INTEGER,
                      resolution TEXT, resolved_to TEXT, resolved_at TEXT)

The reader API exposes:

    get_version_vector(workspace, block_id) -> dict[agent_id, version]
    record_agent_write(workspace, block_id, agent_id) -> int  # new version
    detect_conflict(workspace, block_id) -> ConflictReport | None
    resolve_conflict(workspace, block_id, strategy, *, merger=None)

The version vector is a per-block map; an "agent" is whatever opaque
ID a deployment uses (DID, OAuth client, hostname). No assumption of
shared clock — versions are local-monotonic per agent.

Read-only paths fail-soft (return empty / None on missing schema).
Write paths run inside ``BEGIN IMMEDIATE`` so two threads racing on
the same (block_id, agent_id) pair never produce a torn write.

Feature-flag gated under ``v4.federation``. v3.x callers see no
behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "MergeStrategy",
    "ConflictReport",
    "Resolution",
    "ensure_federation_schema",
    "get_version_vector",
    "record_agent_write",
    "detect_conflict",
    "resolve_conflict",
    "list_conflicts",
]


FLAG: str = "federation"


class MergeStrategy(str, Enum):
    """Conflict resolution policies."""

    LAST_WRITER_WINS = "last_writer_wins"
    HIGHER_VERSION = "higher_version"
    THREE_WAY_MERGE = "three_way_merge"


@dataclass(frozen=True)
class ConflictReport:
    """Detected divergence between two agents on the same block.

    ``left`` and ``right`` are arbitrary labels — the *names* of the
    agents whose logical clocks diverged. The strategy that resolves
    the conflict picks one side or hands the merge off to the
    caller-supplied merger callable.
    """

    block_id: str
    left_agent: str
    left_version: int
    right_agent: str
    right_version: int


@dataclass(frozen=True)
class Resolution:
    """Outcome of :func:`resolve_conflict`."""

    block_id: str
    winner_agent: str
    winner_version: int
    strategy: MergeStrategy
    merged_payload: bytes | None = None
    """Optional caller payload (e.g. merged content from a 3-way merge)."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS block_tier_vclock (
    block_id     TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    version      INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (block_id, agent_id)
);
CREATE INDEX IF NOT EXISTS idx_vclock_block
    ON block_tier_vclock (block_id);

CREATE TABLE IF NOT EXISTS tier_conflict_log (
    rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id       TEXT NOT NULL,
    detected_at    TEXT NOT NULL,
    left_agent     TEXT NOT NULL,
    left_version   INTEGER NOT NULL,
    right_agent    TEXT NOT NULL,
    right_version  INTEGER NOT NULL,
    resolution     TEXT,
    resolved_to    TEXT,
    resolved_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_conflict_block
    ON tier_conflict_log (block_id);
"""


def ensure_federation_schema(workspace: str | Path) -> None:
    """Idempotent. Creates the version-vector + conflict-log tables."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db, timeout=30) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Reader API
# ---------------------------------------------------------------------------


def get_version_vector(workspace: str | Path, block_id: str) -> dict[str, int]:
    """Return the per-agent version map for a block.

    Empty dict for missing schema / unknown block. Each entry is the
    most recent version this agent claimed for this block.
    """
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return {}
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "block_tier_vclock"):
            return {}
        rows = conn.execute(
            "SELECT agent_id, version FROM block_tier_vclock WHERE block_id = ?",
            (block_id,),
        ).fetchall()
    return {agent: int(v) for agent, v in rows}


def list_conflicts(workspace: str | Path, *, limit: int = 100) -> list[ConflictReport]:
    """Return up to ``limit`` outstanding (un-resolved) conflicts.

    Conflicts are detected lazily by :func:`detect_conflict` on read
    paths; calling this function does not scan the workspace —
    it only surfaces what's already been logged.
    """
    require_enabled(FLAG)
    if limit <= 0:
        return []
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return []
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "tier_conflict_log"):
            return []
        rows = conn.execute(
            "SELECT block_id, left_agent, left_version, right_agent, right_version "
            "FROM tier_conflict_log WHERE resolution IS NULL "
            "ORDER BY rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [
        ConflictReport(
            block_id=r[0],
            left_agent=r[1],
            left_version=int(r[2]),
            right_agent=r[3],
            right_version=int(r[4]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def record_agent_write(workspace: str | Path, block_id: str, agent_id: str) -> int:
    """Bump (block_id, agent_id) version atomically; return new version.

    Inside ``BEGIN IMMEDIATE`` so concurrent calls to the same key
    serialise; cross-agent calls run in parallel because the primary
    key is composite and rows don't collide.
    """
    require_enabled(FLAG)
    ensure_federation_schema(workspace)
    db = Path(workspace) / "index.db"
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with sqlite3.connect(db, timeout=10) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT version FROM block_tier_vclock WHERE block_id = ? AND agent_id = ?",
            (block_id, agent_id),
        ).fetchone()
        next_version = (int(row[0]) + 1) if row else 1
        conn.execute(
            "INSERT INTO block_tier_vclock (block_id, agent_id, version, last_seen_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(block_id, agent_id) DO UPDATE SET "
            "version = excluded.version, last_seen_at = excluded.last_seen_at",
            (block_id, agent_id, next_version, now),
        )
        conn.commit()
    return next_version


def detect_conflict(workspace: str | Path, block_id: str) -> ConflictReport | None:
    """Return a :class:`ConflictReport` if two or more agents have
    claimed independently advancing versions, else ``None``.

    The detection rule: if ≥2 agents have versions, surface the pair
    with the largest gap (left = highest, right = next highest). Per
    audit FP-7, every lagging agent (versions below the leader) is
    *also* recorded in ``tier_conflict_log`` so a third or fourth
    diverging agent is never silently invisible — only the top-2 pair
    is returned to the caller for resolution.

    Calling this function lazily logs the detection(s) so subsequent
    :func:`list_conflicts` callers see them.
    """
    require_enabled(FLAG)
    vec = get_version_vector(workspace, block_id)
    if len(vec) < 2:
        return None
    sorted_agents = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)
    left_agent, left_v = sorted_agents[0]
    right_agent, right_v = sorted_agents[1]
    if left_v == right_v:
        # No divergence: same logical clock on both sides → tie, not conflict.
        return None
    report = ConflictReport(
        block_id=block_id,
        left_agent=left_agent,
        left_version=left_v,
        right_agent=right_agent,
        right_version=right_v,
    )
    _log_conflict(workspace, report)
    # FP-7: log every lagging agent so 3+ agent divergence is visible.
    for lag_agent, lag_v in sorted_agents[2:]:
        if lag_v >= left_v:
            continue
        _log_conflict(
            workspace,
            ConflictReport(
                block_id=block_id,
                left_agent=left_agent,
                left_version=left_v,
                right_agent=lag_agent,
                right_version=lag_v,
            ),
        )
    return report


def resolve_conflict(
    workspace: str | Path,
    block_id: str,
    strategy: MergeStrategy | str,
    *,
    merger: Callable[[ConflictReport], bytes] | None = None,
) -> Resolution | None:
    """Apply a merge strategy to the most recent open conflict for ``block_id``.

    Strategies:
        LAST_WRITER_WINS   Pick whichever side wrote most recently (by
                           version vector tip).
        HIGHER_VERSION     Pick the side with the larger logical clock.
        THREE_WAY_MERGE    Call ``merger(report)`` and treat its return
                           as the merged payload; winner_agent is set
                           to a synthetic ``"merge:<left>+<right>"``
                           label for audit.

    Returns ``None`` when no open conflict exists for ``block_id``.
    """
    require_enabled(FLAG)
    if isinstance(strategy, str):
        strategy = MergeStrategy(strategy)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    report = detect_conflict(workspace, block_id)
    if report is None:
        return None
    # FP-3: capture the specific rowid for this exact (left, right, versions)
    # pair so the UPDATE below pins to one row, not "all open conflicts for
    # this block_id" (which would silently falsify the audit metadata of any
    # stale open rows from a different version pair).
    target_rowid = _find_open_conflict_rowid(workspace, block_id, report)
    if target_rowid is None:
        # _log_conflict in detect_conflict should have inserted; if a
        # concurrent resolver beat us, treat as already-resolved.
        return None

    if strategy is MergeStrategy.HIGHER_VERSION:
        # Highest logical-clock wins. `report.left_agent` is already the
        # max-version agent (detect_conflict sorts by version desc).
        winner_agent = report.left_agent
        winner_version = report.left_version
        merged: bytes | None = None
    elif strategy is MergeStrategy.LAST_WRITER_WINS:
        # Latest wall-clock writer wins, by `last_seen_at`. Falls back to
        # higher version when timestamps tie. See audit FP-1: HIGHER_VERSION
        # and LAST_WRITER_WINS used to be implementation-identical; this
        # branch makes them semantically distinct as the public enum implies.
        winner_agent, winner_version = _pick_last_writer(workspace, block_id, report)
        merged = None
    elif strategy is MergeStrategy.THREE_WAY_MERGE:
        if merger is None:
            # FP-4: surface caller error rather than silently returning None
            # (which is indistinguishable from "no conflict to resolve").
            raise ValueError("THREE_WAY_MERGE requires a merger callable; got None")
        winner_agent = f"merge:{report.left_agent}+{report.right_agent}"
        winner_version = max(report.left_version, report.right_version) + 1
        merged = merger(report)
    else:
        raise ValueError(f"unrecognised MergeStrategy: {strategy!r}")

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with sqlite3.connect(db, timeout=10) as conn:
        # FP-3 + FP-8: pin the UPDATE to one specific rowid (so stale
        # open rows for the same block but a different version pair are
        # not collateral-damaged), inside BEGIN IMMEDIATE so two
        # resolvers don't silently overwrite each other.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE tier_conflict_log SET resolution = ?, resolved_to = ?, resolved_at = ? WHERE rowid = ? AND resolution IS NULL",
            (strategy.value, winner_agent, now, target_rowid),
        )
        # Issue #527: persist the resolution to block_tier_vclock so
        # detect_conflict doesn't re-discover the same pair on the next
        # pass. Before this fix, every resolution was a no-op for
        # subsequent conflict detection — winner_version was computed
        # and returned but never written to the vclock table.
        #
        # The resolution must converge BOTH the synthetic winner_agent
        # AND the two original forks (left, right) to winner_version,
        # otherwise the merge agent becomes a third high-version writer
        # and the losers still look like independent forks, so
        # detect_conflict would surface a brand-new conflict pair
        # (winner_agent vs left_agent) on the next pass. By advancing
        # all three rows to winner_version, the vclock reflects the
        # post-merge truth: every party agrees on the merged version.
        for agent in (winner_agent, report.left_agent, report.right_agent):
            conn.execute(
                "INSERT INTO block_tier_vclock (block_id, agent_id, version, last_seen_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(block_id, agent_id) DO UPDATE SET "
                "version = MAX(excluded.version, block_tier_vclock.version), "
                "last_seen_at = excluded.last_seen_at",
                (block_id, agent, winner_version, now),
            )
        conn.commit()

    # Issue #528: audit every THREE_WAY_MERGE with caller-supplied
    # merged_payload. The merger callable can return arbitrary bytes
    # unrelated to either input; we don't validate that here (per the
    # issue, full server-side MergeStrategy is the long-term fix), but
    # we DO emit a structured log with SHA-256 hashes of left, right,
    # and merged payloads so operators can audit anomalies.
    if strategy is MergeStrategy.THREE_WAY_MERGE:
        try:
            import hashlib as _hashlib
            import logging as _logging

            left_bytes = getattr(report, "left_payload", None) or b""
            right_bytes = getattr(report, "right_payload", None) or b""
            merged_bytes = merged if merged is not None else b""
            _logging.getLogger("mind_mem.federation").info(
                "three_way_merge_resolved",
                extra={
                    "block_id": block_id,
                    "winner_agent": winner_agent,
                    "winner_version": winner_version,
                    "left_agent": report.left_agent,
                    "left_version": report.left_version,
                    "left_payload_sha256": _hashlib.sha256(
                        left_bytes if isinstance(left_bytes, (bytes, bytearray)) else str(left_bytes).encode("utf-8")
                    ).hexdigest(),
                    "right_agent": report.right_agent,
                    "right_version": report.right_version,
                    "right_payload_sha256": _hashlib.sha256(
                        right_bytes if isinstance(right_bytes, (bytes, bytearray)) else str(right_bytes).encode("utf-8")
                    ).hexdigest(),
                    "merged_payload_sha256": _hashlib.sha256(
                        merged_bytes if isinstance(merged_bytes, (bytes, bytearray)) else str(merged_bytes).encode("utf-8")
                    ).hexdigest(),
                    "merged_payload_bytes": len(merged_bytes) if isinstance(merged_bytes, (bytes, bytearray)) else 0,
                },
            )
        except Exception:
            # Audit log must never block the merge resolution.
            pass

    return Resolution(
        block_id=block_id,
        winner_agent=winner_agent,
        winner_version=winner_version,
        strategy=strategy,
        merged_payload=merged,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_conflict(workspace: str | Path, report: ConflictReport) -> int | None:
    """Record a conflict for later resolution. Returns the inserted
    ``rowid`` (so callers can pin a subsequent UPDATE to one specific
    row — see FP-3) or ``None`` if the conflict was already logged.

    Idempotent on duplicates via the ``resolution IS NULL`` filter —
    re-detecting the same pair inserts a fresh row only when no open
    row exists. Audit FP-2: SELECT and INSERT now run inside
    ``BEGIN IMMEDIATE`` so two threads racing detect_conflict on the
    same pair cannot both insert (the second waits, sees the first's
    row, and returns without inserting).
    """
    ensure_federation_schema(workspace)
    db = Path(workspace) / "index.db"
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with sqlite3.connect(db, timeout=10) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT rowid FROM tier_conflict_log WHERE block_id = ? AND resolution IS NULL "
            "AND left_agent = ? AND right_agent = ? "
            "AND left_version = ? AND right_version = ?",
            (
                report.block_id,
                report.left_agent,
                report.right_agent,
                report.left_version,
                report.right_version,
            ),
        ).fetchone()
        if existing is not None:
            return int(existing[0])
        cur = conn.execute(
            "INSERT INTO tier_conflict_log "
            "(block_id, detected_at, left_agent, left_version, right_agent, right_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                report.block_id,
                now,
                report.left_agent,
                report.left_version,
                report.right_agent,
                report.right_version,
            ),
        )
        conn.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else None


def _find_open_conflict_rowid(
    workspace: str | Path,
    block_id: str,
    report: ConflictReport,
) -> int | None:
    """Return the rowid of the open `tier_conflict_log` row matching
    exactly this (block_id, left, right, versions) pair, or None.

    Audit FP-3: the per-block ``resolve_conflict`` UPDATE used to close
    every open row for ``block_id`` regardless of the specific version
    pair, silently overwriting the audit metadata of stale rows. This
    helper lets the resolver pin to one row.
    """
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with sqlite3.connect(db, timeout=10) as conn:
        row = conn.execute(
            "SELECT rowid FROM tier_conflict_log "
            "WHERE block_id = ? AND resolution IS NULL "
            "  AND left_agent = ? AND right_agent = ? "
            "  AND left_version = ? AND right_version = ?",
            (
                block_id,
                report.left_agent,
                report.right_agent,
                report.left_version,
                report.right_version,
            ),
        ).fetchone()
    return int(row[0]) if row else None


def _pick_last_writer(
    workspace: str | Path,
    block_id: str,
    report: ConflictReport,
) -> tuple[str, int]:
    """Return (agent, version) of whichever of the two report agents wrote
    most recently by ``last_seen_at``. Ties break to higher version, then
    to left.

    Audit FP-1: LAST_WRITER_WINS was implementation-identical to
    HIGHER_VERSION before this. It now consults the wall-clock column
    that ``record_agent_write`` has been writing all along (schema line
    125, ``block_tier_vclock.last_seen_at``).
    """
    db = Path(workspace) / "index.db"
    with sqlite3.connect(db, timeout=10) as conn:
        rows = conn.execute(
            "SELECT agent_id, version, last_seen_at FROM block_tier_vclock WHERE block_id = ? AND agent_id IN (?, ?)",
            (block_id, report.left_agent, report.right_agent),
        ).fetchall()
    if not rows:
        # No timestamps available — fall back to higher version.
        return (report.left_agent, report.left_version)
    rows_sorted = sorted(rows, key=lambda r: (r[2], int(r[1])), reverse=True)
    agent, version, _ts = rows_sorted[0]
    return (str(agent), int(version))


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
