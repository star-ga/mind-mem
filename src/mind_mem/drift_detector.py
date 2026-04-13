#!/usr/bin/env python3
"""mind-mem Semantic Belief Drift Detection.

Detects when beliefs/facts evolve or contradict over time using
character n-gram similarity (no external embedding dependencies).
Goes beyond keyword overlap by computing Jaccard similarity on
character trigrams for semantic-like matching.

Tracks drift signals with confidence scoring and belief evolution
timelines. Integrates with the existing contradiction detector.

Usage:
    from .drift_detector import DriftDetector
    detector = DriftDetector(workspace)
    signals = detector.scan()
    timeline = detector.belief_timeline("D-20260304-001")

Zero external deps — all stdlib.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime

from .block_parser import parse_file
from .observability import get_logger, metrics

_log = get_logger("drift_detector")


def _trigrams(text: str) -> set[str]:
    """Extract character trigrams from text for similarity computation."""
    text = text.lower().strip()
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _extract_date(block: dict) -> str | None:
    """Extract date from block metadata."""
    for field in ("Date", "Created", "Timestamp"):
        val = block.get(field, "")
        if isinstance(val, str) and re.match(r"\d{4}-\d{2}-\d{2}", val):
            return val[:10]
    bid = block.get("_id", "")
    m = re.match(r"[A-Z]+-(\d{8})-\d{3}", bid)
    if m:
        raw = m.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def _block_text(block: dict) -> str:
    """Extract all text content from a block for comparison."""
    parts = []
    skip = {"_id", "_source", "_file", "_line", "_raw", "Date", "Created", "Timestamp"}
    for key, val in block.items():
        if key in skip or key.startswith("_"):
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if isinstance(v, str))
    return " ".join(parts)


def _db_path(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), ".mind-mem-audit", "drift.db")


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
CREATE TABLE IF NOT EXISTS drift_signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    block_a_id   TEXT NOT NULL,
    block_b_id   TEXT NOT NULL,
    similarity   REAL NOT NULL,
    confidence   REAL NOT NULL,
    drift_type   TEXT NOT NULL,
    description  TEXT DEFAULT '',
    date_a       TEXT DEFAULT '',
    date_b       TEXT DEFAULT '',
    timestamp    TEXT DEFAULT (datetime('now')),
    UNIQUE(block_a_id, block_b_id, drift_type)
);
CREATE INDEX IF NOT EXISTS idx_drift_blocks ON drift_signals(block_a_id, block_b_id);
CREATE INDEX IF NOT EXISTS idx_drift_conf ON drift_signals(confidence);
CREATE INDEX IF NOT EXISTS idx_drift_type ON drift_signals(drift_type);

CREATE TABLE IF NOT EXISTS belief_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    text_preview TEXT DEFAULT '',
    fields_json  TEXT DEFAULT '{}',
    timestamp    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_bs_block ON belief_snapshots(block_id);
"""

# Drift types
DRIFT_SEMANTIC = "semantic"  # Meaning changed (high similarity, different content)
DRIFT_EVOLUTION = "evolution"  # Content updated over time
DRIFT_REVERSAL = "reversal"  # Decision reversed (contradictory modality)
DRIFT_SCOPE = "scope_shift"  # Scope narrowed or widened


class DriftSignal:
    """A detected drift between two blocks."""

    __slots__ = (
        "block_a_id",
        "block_b_id",
        "similarity",
        "confidence",
        "drift_type",
        "description",
        "date_a",
        "date_b",
    )

    def __init__(
        self,
        block_a_id: str,
        block_b_id: str,
        similarity: float,
        confidence: float,
        drift_type: str,
        description: str = "",
        date_a: str = "",
        date_b: str = "",
    ):
        self.block_a_id = block_a_id
        self.block_b_id = block_b_id
        self.similarity = similarity
        self.confidence = confidence
        self.drift_type = drift_type
        self.description = description
        self.date_a = date_a
        self.date_b = date_b

    def to_dict(self) -> dict:
        return {
            "block_a_id": self.block_a_id,
            "block_b_id": self.block_b_id,
            "similarity": round(self.similarity, 4),
            "confidence": round(self.confidence, 4),
            "drift_type": self.drift_type,
            "description": self.description,
            "date_a": self.date_a,
            "date_b": self.date_b,
        }


class DriftDetector:
    """Semantic belief drift detection engine.

    Scans decision blocks for pairs that are semantically similar
    but have evolved over time, detecting potential contradictions
    or belief shifts.
    """

    def __init__(
        self,
        workspace: str,
        *,
        similarity_threshold: float = 0.35,
        high_confidence_threshold: float = 0.6,
    ) -> None:
        self.workspace = os.path.realpath(workspace)
        self.similarity_threshold = similarity_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = _connect(self.workspace)
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _load_blocks(self) -> list[dict]:
        """Load all decision blocks from the workspace."""
        blocks = []
        decisions_path = os.path.join(self.workspace, "decisions", "DECISIONS.md")
        if os.path.isfile(decisions_path):
            try:
                blocks.extend(parse_file(decisions_path))
            except (OSError, ValueError):
                pass
        return blocks

    @staticmethod
    def _parse_constraint_sigs(raw) -> list[dict]:
        """Parse ConstraintSignatures from block — may be string or list."""
        if isinstance(raw, list):
            return [s for s in raw if isinstance(s, dict)]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [s for s in parsed if isinstance(s, dict)]
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    def _detect_modality_conflict(self, block_a: dict, block_b: dict) -> bool:
        """Check if blocks have conflicting modalities (must vs must_not)."""
        conflict_pairs = {
            frozenset({"must", "must_not"}),
            frozenset({"should", "should_not"}),
        }
        mods_a = set()
        mods_b = set()
        for sig in self._parse_constraint_sigs(block_a.get("ConstraintSignatures", [])):
            mod = sig.get("modality", "")
            if mod:
                mods_a.add(mod.lower())
        for sig in self._parse_constraint_sigs(block_b.get("ConstraintSignatures", [])):
            mod = sig.get("modality", "")
            if mod:
                mods_b.add(mod.lower())

        for mod_a in mods_a:
            for mod_b in mods_b:
                if frozenset({mod_a, mod_b}) in conflict_pairs:
                    return True
        return False

    def _compute_drift_confidence(
        self,
        similarity: float,
        has_modality_conflict: bool,
        date_diff_days: int | None,
    ) -> float:
        """Compute confidence score for a drift signal.

        Higher when:
        - Similarity is in the "suspicious" range (0.4-0.8)
        - Modality conflicts exist
        - Blocks are close in time (recent drift)
        """
        # Base confidence from similarity being in drift range
        if similarity > 0.85:
            # Very similar — likely a duplicate, not drift
            base = 0.3
        elif similarity > 0.6:
            # Moderate similarity — strong drift signal
            base = 0.8
        elif similarity > 0.4:
            # Lower similarity — possible drift
            base = 0.5
        else:
            base = 0.2

        # Modality conflict boosts confidence significantly
        if has_modality_conflict:
            base = min(base + 0.3, 1.0)

        # Recent drift (within 30 days) is more confident
        if date_diff_days is not None and date_diff_days < 30:
            base = min(base + 0.1, 1.0)

        return base

    def scan(self) -> list[DriftSignal]:
        """Scan all blocks for semantic drift signals.

        Returns:
            List of DriftSignal objects sorted by confidence (highest first).
        """
        blocks = self._load_blocks()
        if len(blocks) < 2:
            return []

        # Pre-compute trigrams for all blocks
        block_data: list[tuple[dict, str, set[str], str | None]] = []
        for block in blocks:
            bid = block.get("_id", "")
            if not bid:
                continue
            text = _block_text(block)
            tris = _trigrams(text)
            date = _extract_date(block)
            block_data.append((block, text, tris, date))

        signals: list[DriftSignal] = []
        seen_pairs: set[frozenset[str]] = set()

        for i, (block_a, text_a, tris_a, date_a) in enumerate(block_data):
            for j, (block_b, text_b, tris_b, date_b) in enumerate(block_data[i + 1 :], i + 1):
                id_a = block_a.get("_id", "")
                id_b = block_b.get("_id", "")
                pair = frozenset({id_a, id_b})
                if pair in seen_pairs:
                    continue

                sim = _jaccard(tris_a, tris_b)
                if sim < self.similarity_threshold:
                    continue

                seen_pairs.add(pair)

                # Determine drift type
                has_conflict = self._detect_modality_conflict(block_a, block_b)

                # Compute date difference
                date_diff = None
                if date_a and date_b:
                    try:
                        d_a = datetime.strptime(date_a, "%Y-%m-%d")
                        d_b = datetime.strptime(date_b, "%Y-%m-%d")
                        date_diff = abs((d_a - d_b).days)
                    except ValueError:
                        pass

                confidence = self._compute_drift_confidence(sim, has_conflict, date_diff)

                if has_conflict:
                    drift_type = DRIFT_REVERSAL
                    desc = f"Modality conflict between {id_a} and {id_b}"
                elif sim > 0.7 and text_a != text_b:
                    drift_type = DRIFT_EVOLUTION
                    desc = f"Content evolved: {id_a} → {id_b} (similarity: {sim:.2f})"
                else:
                    drift_type = DRIFT_SEMANTIC
                    desc = f"Semantic drift between {id_a} and {id_b} (similarity: {sim:.2f})"

                signals.append(
                    DriftSignal(
                        block_a_id=id_a,
                        block_b_id=id_b,
                        similarity=sim,
                        confidence=confidence,
                        drift_type=drift_type,
                        description=desc,
                        date_a=date_a or "",
                        date_b=date_b or "",
                    )
                )

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)

        # Persist signals
        self._store_signals(signals)

        _log.info("drift_scan_complete", signals=len(signals), blocks=len(block_data))
        metrics.inc("drift_scans")
        metrics.inc("drift_signals_found", len(signals))
        return signals

    def _store_signals(self, signals: list[DriftSignal]) -> None:
        """Persist drift signals to SQLite."""
        if not signals:
            return
        conn = _connect(self.workspace)
        try:
            conn.execute("BEGIN")
            for sig in signals:
                # INSERT OR REPLACE via the UNIQUE(block_a, block_b, drift_type)
                # constraint keeps the most recent similarity/confidence
                # observation per drift pair instead of accumulating
                # duplicate rows on every scan.
                conn.execute(
                    "INSERT INTO drift_signals "
                    "(block_a_id, block_b_id, similarity, confidence, "
                    "drift_type, description, date_a, date_b) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(block_a_id, block_b_id, drift_type) DO UPDATE SET "
                    "similarity = excluded.similarity, "
                    "confidence = excluded.confidence, "
                    "description = excluded.description, "
                    "date_a = excluded.date_a, "
                    "date_b = excluded.date_b, "
                    "timestamp = datetime('now')",
                    (
                        sig.block_a_id,
                        sig.block_b_id,
                        sig.similarity,
                        sig.confidence,
                        sig.drift_type,
                        sig.description,
                        sig.date_a,
                        sig.date_b,
                    ),
                )
            conn.commit()
        except Exception as e:
            _log.debug("drift_store_failed", error=str(e))
        finally:
            conn.close()

    def snapshot_belief(self, block_id: str, block: dict) -> None:
        """Take a snapshot of a block's current state for timeline tracking.

        Call this before mutations to record the pre-change state.
        """
        import hashlib

        text = _block_text(block)
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        fields = {k: v for k, v in block.items() if not k.startswith("_")}

        conn = _connect(self.workspace)
        try:
            conn.execute(
                "INSERT INTO belief_snapshots (block_id, content_hash, text_preview, fields_json) VALUES (?, ?, ?, ?)",
                (block_id, content_hash, text[:200], json.dumps(fields, default=str)),
            )
            conn.commit()
        finally:
            conn.close()

    def belief_timeline(self, block_id: str) -> list[dict]:
        """Get the evolution timeline of a block's beliefs.

        Returns snapshots in chronological order showing how the
        block's content has changed over time.
        """
        conn = _connect(self.workspace)
        try:
            rows = conn.execute(
                "SELECT * FROM belief_snapshots WHERE block_id = ? ORDER BY timestamp ASC",
                (block_id,),
            ).fetchall()
        finally:
            conn.close()

        timeline = []
        prev_hash = None
        for row in rows:
            entry = {
                "block_id": row["block_id"],
                "content_hash": row["content_hash"],
                "text_preview": row["text_preview"],
                "timestamp": row["timestamp"],
                "changed": row["content_hash"] != prev_hash if prev_hash else True,
            }
            try:
                entry["fields"] = json.loads(row["fields_json"])
            except (json.JSONDecodeError, TypeError):
                entry["fields"] = {}
            prev_hash = row["content_hash"]
            timeline.append(entry)

        return timeline

    def recent_signals(
        self,
        *,
        min_confidence: float = 0.0,
        drift_type: str | None = None,
        last_n: int = 50,
    ) -> list[dict]:
        """Query recent drift signals with optional filters.

        Args:
            min_confidence: Minimum confidence threshold.
            drift_type: Filter by drift type.
            last_n: Maximum number of signals.

        Returns:
            List of signal dicts, newest first.
        """
        conn = _connect(self.workspace)
        try:
            query = "SELECT * FROM drift_signals WHERE confidence >= ?"
            params: list = [min_confidence]

            if drift_type:
                query += " AND drift_type = ?"
                params.append(drift_type)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(last_n)

            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        return [dict(row) for row in rows]
