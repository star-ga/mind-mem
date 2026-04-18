# Copyright 2026 STARGA, Inc.
"""Knowledge graph layer (v2.2.0) — SQLite-backed triple store.

Every memory block exists in a small relational graph. A triple
``(subject, predicate, object)`` captures a typed relationship, backed
by an SQLite adjacency table. The default backend requires no external
services; Neo4j / FalkorDB drop-ins stay on the roadmap.

Design highlights:

- Subjects and objects are canonicalised through an entity registry so
  alias chains (``"STARGA"`` ↔ ``"STARGA Inc"``) resolve to a single id.
- Predicates are typed (:class:`Predicate`) to avoid free-form string
  drift; callers wishing to register custom predicates use
  :func:`Predicate.register`.
- Each edge carries provenance: the `source_block_id`, a
  `confidence` in [0, 1], and optional `valid_from` / `valid_until`
  timestamps so downstream retrieval can weight or filter by freshness.

Pure Python, stdlib-only. Concurrency-safe.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Predicate registry
# ---------------------------------------------------------------------------


class Predicate(str, Enum):
    """Typed predicates for the triple store.

    Inherits ``str`` so values serialise cleanly into MCP / JSONL
    envelopes. Callers may register additional predicates at runtime
    via :meth:`register` — they join the enum set under the same value
    convention (lowercase snake_case).
    """

    AUTHORED_BY = "authored_by"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    PART_OF = "part_of"
    MENTIONED_IN = "mentioned_in"
    RELATED_TO = "related_to"

    @classmethod
    def from_str(cls, name: str) -> "Predicate":
        """Hyphen-tolerant, case-insensitive lookup."""
        normalised = name.strip().lower().replace("-", "_")
        for pred in cls:
            if pred.value == normalised:
                return pred
        valid = ", ".join(p.value for p in cls)
        raise ValueError(f"Unknown predicate: {name!r}. Valid: {valid}")


# ---------------------------------------------------------------------------
# Entity registry — canonical entity resolution
# ---------------------------------------------------------------------------


def _canonicalise(name: str) -> str:
    """Lowercased + whitespace-collapsed canonical form for an entity name."""
    return " ".join(name.strip().lower().split())


def _parse_iso8601(value: str) -> datetime:
    """Parse ``value`` as an ISO 8601 UTC timestamp.

    Accepts both ``Z`` and explicit ``+00:00`` suffixes. Raises
    :class:`ValueError` on malformed input so the caller can refuse
    the write instead of corrupting the store.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"timestamp must be a non-empty ISO 8601 string, got {value!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class EntityRegistry:
    """Map surface forms (aliases) to canonical entity ids.

    The registry is intentionally lightweight — no fuzzy matching, no
    embeddings. It handles the common coreference case where the same
    entity appears under varied capitalisation or abbreviation.

    Backed by two SQLite tables:

        entities(id PRIMARY KEY, canonical TEXT)
        aliases(alias PRIMARY KEY, entity_id TEXT)

    The ``alias`` column is the canonicalised surface form so lookups
    never depend on the caller's casing.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entities (
        id         TEXT PRIMARY KEY,
        canonical  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS aliases (
        alias      TEXT PRIMARY KEY,
        entity_id  TEXT NOT NULL,
        FOREIGN KEY (entity_id) REFERENCES entities(id)
    );
    CREATE INDEX IF NOT EXISTS idx_aliases_entity ON aliases(entity_id);
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = conn
        self._lock = lock

    def resolve(self, surface: str) -> str:
        """Return the canonical entity id for *surface*, creating it if new."""
        canon = _canonicalise(surface)
        if not canon:
            raise ValueError("entity name must not be empty")
        with self._lock:
            row = self._conn.execute("SELECT entity_id FROM aliases WHERE alias = ?", (canon,)).fetchone()
            if row is not None:
                return row["entity_id"]
            # New entity — id derived from the canonical form. Surface
            # strings that canonicalise identically are intentionally
            # merged (that's the whole point of the registry).
            entity_id = canon
            self._conn.execute(
                "INSERT OR IGNORE INTO entities(id, canonical) VALUES (?, ?)",
                (entity_id, canon),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases(alias, entity_id) VALUES (?, ?)",
                (canon, entity_id),
            )
            self._conn.commit()
            return entity_id

    def add_alias(self, alias: str, entity_id: str) -> None:
        """Bind *alias* → *entity_id*. Idempotent.

        Raises ValueError when the target id is unknown.
        """
        with self._lock:
            exists = self._conn.execute("SELECT 1 FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if exists is None:
                raise ValueError(f"unknown entity_id: {entity_id!r}")
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases(alias, entity_id) VALUES (?, ?)",
                (_canonicalise(alias), entity_id),
            )
            self._conn.commit()

    def aliases_for(self, entity_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT alias FROM aliases WHERE entity_id = ? ORDER BY alias",
                (entity_id,),
            ).fetchall()
        return [r["alias"] for r in rows]


# ---------------------------------------------------------------------------
# Edge value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """Immutable representation of a triple + provenance."""

    subject: str
    predicate: Predicate
    object: str
    source_block_id: str
    confidence: float = 1.0
    valid_from: Optional[str] = None  # ISO8601; None = "always"
    valid_until: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
            "source_block_id": self.source_block_id,
            "confidence": round(self.confidence, 6),
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# KnowledgeGraph — the public surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphStats:
    """Snapshot of graph size for observability."""

    entities: int
    edges: int
    predicates: dict[str, int]
    orphan_entities: int  # entities with no incoming or outgoing edges

    def as_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "edges": self.edges,
            "predicates": self.predicates,
            "orphan_entities": self.orphan_entities,
        }


class KnowledgeGraph:
    """SQLite-backed triple store with entity registry + N-hop traversal.

    Args:
        db_path: Absolute path to the SQLite database. The directory
            is created if absent.
    """

    SCHEMA = (
        EntityRegistry.SCHEMA
        + """
    CREATE TABLE IF NOT EXISTS edges (
        subject          TEXT NOT NULL,
        predicate        TEXT NOT NULL,
        object           TEXT NOT NULL,
        source_block_id  TEXT NOT NULL,
        confidence       REAL NOT NULL DEFAULT 1.0,
        valid_from       TEXT,
        valid_until      TEXT,
        metadata         TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (subject, predicate, object, source_block_id)
    );
    CREATE INDEX IF NOT EXISTS idx_edges_subject ON edges(subject);
    CREATE INDEX IF NOT EXISTS idx_edges_object ON edges(object);
    CREATE INDEX IF NOT EXISTS idx_edges_predicate ON edges(predicate);
    """
    )

    def __init__(self, db_path: str) -> None:
        self._db_path = os.path.realpath(db_path)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self._db_path,
            timeout=30.0,
            isolation_level="DEFERRED",
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        with self._lock:
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()
        self.entities = EntityRegistry(self._conn, self._lock)

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def add_edge(
        self,
        subject: str,
        predicate: "Predicate | str",
        object: str,
        *,
        source_block_id: str,
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Edge:
        """Resolve both ends via the registry, then insert the edge.

        Duplicate edges with the same ``(subject, predicate, object,
        source_block_id)`` tuple collapse to a single row; the ``INSERT
        OR IGNORE`` ensures repeated ingestion is idempotent.
        """
        import json as _json

        pred = predicate if isinstance(predicate, Predicate) else Predicate.from_str(predicate)
        s_id = self.entities.resolve(subject)
        o_id = self.entities.resolve(object)
        if not source_block_id or not source_block_id.strip():
            raise ValueError("source_block_id is required for provenance")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")
        # Validate timestamps up front so malformed ISO 8601 is caught
        # at write time, not silently compared as a raw string at read.
        if valid_from is not None:
            _parse_iso8601(valid_from)
        if valid_until is not None:
            _parse_iso8601(valid_until)
        if valid_from is not None and valid_until is not None:
            if _parse_iso8601(valid_until) < _parse_iso8601(valid_from):
                raise ValueError("valid_until must be >= valid_from")

        edge = Edge(
            subject=s_id,
            predicate=pred,
            object=o_id,
            source_block_id=source_block_id.strip(),
            confidence=float(confidence),
            valid_from=valid_from,
            valid_until=valid_until,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO edges
                    (subject, predicate, object, source_block_id,
                     confidence, valid_from, valid_until, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.subject,
                    edge.predicate.value,
                    edge.object,
                    edge.source_block_id,
                    edge.confidence,
                    edge.valid_from,
                    edge.valid_until,
                    # default=str handles datetime / set / other non-
                    # JSON-native metadata values gracefully instead of
                    # crashing mid-insert.
                    _json.dumps(
                        edge.metadata,
                        separators=(",", ":"),
                        default=str,
                        sort_keys=True,
                    ),
                ),
            )
            self._conn.commit()
        return edge

    def edges_from(
        self,
        subject: str,
        *,
        predicate: "Predicate | str | None" = None,
        include_expired: bool = False,
    ) -> list[Edge]:
        """Return outgoing edges from *subject*, optionally filtered."""
        s_id = self.entities.resolve(subject)
        return self._query_edges(
            subject=s_id,
            predicate=predicate,
            include_expired=include_expired,
        )

    def edges_to(
        self,
        object: str,
        *,
        predicate: "Predicate | str | None" = None,
        include_expired: bool = False,
    ) -> list[Edge]:
        """Return incoming edges to *object*."""
        o_id = self.entities.resolve(object)
        return self._query_edges(
            object_=o_id,
            predicate=predicate,
            include_expired=include_expired,
        )

    def _query_edges(
        self,
        *,
        subject: Optional[str] = None,
        object_: Optional[str] = None,
        predicate: "Predicate | str | None" = None,
        include_expired: bool = False,
    ) -> list[Edge]:
        import json as _json

        clauses: list[str] = []
        params: list[Any] = []
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if object_ is not None:
            clauses.append("object = ?")
            params.append(object_)
        if predicate is not None:
            pred = predicate if isinstance(predicate, Predicate) else Predicate.from_str(predicate)
            clauses.append("predicate = ?")
            params.append(pred.value)
        # We load the raw rows and apply the temporal filter in Python
        # because SQLite's text comparison on ISO 8601 strings breaks
        # on fractional seconds (`12:34:56.999Z` < `12:34:56Z` by ASCII
        # ordering). Parsing to datetime keeps the comparison correct.
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT subject, predicate, object, source_block_id, confidence, "
            "valid_from, valid_until, metadata FROM edges "
            f"{where} ORDER BY confidence DESC, subject, predicate, object"
        )
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        if not include_expired:
            now_dt = datetime.now(timezone.utc)
            filtered: list[Any] = []
            for row in rows:
                vu = row["valid_until"]
                if vu is None:
                    filtered.append(row)
                    continue
                try:
                    if _parse_iso8601(vu) >= now_dt:
                        filtered.append(row)
                except ValueError:
                    # Edge with a malformed valid_until is treated as
                    # expired so corrupt data cannot mask stale
                    # information.
                    continue
            rows = filtered
        out: list[Edge] = []
        for row in rows:
            try:
                meta = _json.loads(row["metadata"] or "{}")
            except (ValueError, TypeError):
                meta = {}
            out.append(
                Edge(
                    subject=row["subject"],
                    predicate=Predicate.from_str(row["predicate"]),
                    object=row["object"],
                    source_block_id=row["source_block_id"],
                    confidence=float(row["confidence"]),
                    valid_from=row["valid_from"],
                    valid_until=row["valid_until"],
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return out

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        entity: str,
        *,
        depth: int = 1,
        predicate: "Predicate | str | None" = None,
        direction: str = "outgoing",
        include_expired: bool = False,
        max_results: int = 256,
    ) -> list[dict[str, Any]]:
        """Breadth-first N-hop expansion from *entity*.

        Args:
            entity: Starting entity (any alias — it's resolved).
            depth: Maximum hops (1 = direct neighbours). Capped at 8
                so hostile callers can't blow up CPU on huge graphs.
            predicate: Optional predicate filter. `None` traverses all.
            direction: ``"outgoing"`` follows subject→object, ``"incoming"``
                follows object→subject, ``"both"`` does both.
            include_expired: Include edges past their ``valid_until``.
            max_results: Cap on returned neighbours (stops the traversal
                once reached).

        Returns:
            List of dicts: ``{entity, hop, path}`` where ``path`` is the
            sequence of intermediate entities.
        """
        if depth < 1:
            return []
        depth = min(int(depth), 8)
        direction = direction.lower()
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction must be 'outgoing', 'incoming', or 'both'")

        start = self.entities.resolve(entity)
        seen: set[str] = {start}
        queue: deque[tuple[str, int, tuple[str, ...]]] = deque()
        queue.append((start, 0, (start,)))
        out: list[dict[str, Any]] = []

        while queue and len(out) < max_results:
            node, hop, path = queue.popleft()
            if hop >= depth:
                continue
            neighbours: list[tuple[str, str]] = []  # (next_node, predicate)
            if direction in {"outgoing", "both"}:
                for e in self._query_edges(
                    subject=node,
                    predicate=predicate,
                    include_expired=include_expired,
                ):
                    neighbours.append((e.object, e.predicate.value))
            if direction in {"incoming", "both"}:
                for e in self._query_edges(
                    object_=node,
                    predicate=predicate,
                    include_expired=include_expired,
                ):
                    neighbours.append((e.subject, e.predicate.value))
            for nxt, pred in neighbours:
                if nxt in seen:
                    continue
                seen.add(nxt)
                new_path = path + (nxt,)
                out.append({"entity": nxt, "hop": hop + 1, "path": list(new_path), "predicate": pred})
                if len(out) >= max_results:
                    break
                queue.append((nxt, hop + 1, new_path))

        return out

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> GraphStats:
        with self._lock:
            entity_count = self._conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"]
            edge_count = self._conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
            rows = self._conn.execute("SELECT predicate, COUNT(*) AS c FROM edges GROUP BY predicate").fetchall()
            pred_counts = {r["predicate"]: r["c"] for r in rows}
            # Orphans: entities that do not appear as subject or object.
            orphan_row = self._conn.execute(
                """
                SELECT COUNT(*) AS c FROM entities e
                WHERE NOT EXISTS (
                    SELECT 1 FROM edges WHERE subject = e.id OR object = e.id
                )
                """
            ).fetchone()
        return GraphStats(
            entities=entity_count,
            edges=edge_count,
            predicates=pred_counts,
            orphan_entities=orphan_row["c"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "KnowledgeGraph":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = [
    "Predicate",
    "Edge",
    "EntityRegistry",
    "KnowledgeGraph",
    "GraphStats",
]
