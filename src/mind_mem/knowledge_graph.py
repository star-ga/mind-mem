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

import hashlib
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
    SUPPORTS = "supports"
    REFINES = "refines"
    DERIVED_FROM = "derived_from"

    @classmethod
    def from_str(cls, name: str) -> "Predicate":
        """Hyphen-tolerant, case-insensitive lookup. Honors runtime-registered predicates.

        Legacy predicate spellings recorded before the typed-edge layer
        existed (:data:`_PREDICATE_ALIASES` — e.g. the singular
        ``"contradiction"`` or the older ``"contraindicates"``) are folded
        onto their canonical typed relation so historical data keeps
        resolving. This is the back-compat half of subsuming the existing
        contradiction-edge work onto :attr:`CONTRADICTS`.
        """
        normalised = name.strip().lower().replace("-", "_")
        normalised = _PREDICATE_ALIASES.get(normalised, normalised)
        for pred in cls:
            if pred.value == normalised:
                return pred
        if normalised in _RUNTIME_PREDICATES:
            return _RUNTIME_PREDICATES[normalised]  # type: ignore[return-value]
        valid_builtin = [p.value for p in cls]
        valid_runtime = list(_RUNTIME_PREDICATES.keys())
        valid = ", ".join(valid_builtin + valid_runtime)
        raise ValueError(f"Unknown predicate: {name!r}. Valid: {valid}")

    @classmethod
    def register(cls, name: str) -> "Predicate":
        """Register a new predicate at runtime.

        Returns a stable enum-like sentinel under the same lowercase
        snake_case value convention as the built-in predicates. Idempotent —
        re-registering an existing value (built-in or runtime) returns
        the same instance.

        The sentinel inherits ``str`` so it serialises identically to
        built-in predicates into MCP / JSONL envelopes. ``from_str`` looks
        runtime-registered values up alongside the closed enum members.
        """
        normalised = name.strip().lower().replace("-", "_")
        for pred in cls:
            if pred.value == normalised:
                return pred
        if normalised in _RUNTIME_PREDICATES:
            return _RUNTIME_PREDICATES[normalised]  # type: ignore[return-value]
        sentinel = str.__new__(_RuntimePredicate, normalised)
        sentinel._name_ = normalised.upper()
        sentinel._value_ = normalised
        _RUNTIME_PREDICATES[normalised] = sentinel
        return sentinel  # type: ignore[return-value]


class _RuntimePredicate(str):
    """str-subclass sentinel for runtime-registered predicates.

    Quacks like a closed ``Predicate`` enum member in the contexts where
    the rest of the module touches predicates (``.value`` / equality on
    the string form / SQLite TEXT serialisation). Not a true enum member,
    but indistinguishable for the call sites that read ``predicate.value``
    or compare to a string predicate field.
    """

    __slots__ = ("_name_", "_value_")
    # Slot type annotations so mypy treats the attributes as declared
    # instance state instead of "no attribute" on dynamic assignment.
    _name_: str
    _value_: str

    @property
    def name(self) -> str:
        return self._name_

    @property
    def value(self) -> str:
        return self._value_

    def __repr__(self) -> str:
        return f"<Predicate.{self._name_}: {self._value_!r}>"


_RUNTIME_PREDICATES: dict[str, "_RuntimePredicate"] = {}

#: Legacy / synonymous predicate spellings folded onto a canonical typed
#: relation by :meth:`Predicate.from_str`. Keys are already
#: hyphen-normalised (``-`` → ``_``, lowercased). The contradiction entries
#: subsume the pre-typed-edge contradiction-edge work onto
#: :attr:`Predicate.CONTRADICTS` without a data migration: any row or caller
#: still using the old spelling resolves to the one canonical predicate.
_PREDICATE_ALIASES: dict[str, str] = {
    "contradiction": "contradicts",
    "contradicted_by": "contradicts",
    "contraindicates": "contradicts",
    "refined_by": "refines",
    "supported_by": "supports",
    "superseded_by": "supersedes",
}

#: The five first-class typed relations of the edge layer (roadmap §a).
#: Recall can weight edges per relation rather than fusing a flat edge set.
TYPED_RELATIONS: frozenset[Predicate] = frozenset(
    {
        Predicate.SUPPORTS,
        Predicate.CONTRADICTS,
        Predicate.REFINES,
        Predicate.SUPERSEDES,
        Predicate.DERIVED_FROM,
    }
)

#: Lex-sorted string values of the typed relations — a deterministic order
#: for iteration / bucketing (no clock, no set-iteration nondeterminism).
TYPED_RELATION_VALUES: tuple[str, ...] = tuple(sorted(p.value for p in TYPED_RELATIONS))


def is_typed_relation(predicate: "Predicate | str") -> bool:
    """True iff *predicate* is one of the five first-class typed relations."""
    try:
        pred = predicate if isinstance(predicate, Predicate) else Predicate.from_str(predicate)
    except ValueError:
        return False
    return pred.value in {p.value for p in TYPED_RELATIONS}


# ---------------------------------------------------------------------------
# Entity registry — canonical entity resolution
# ---------------------------------------------------------------------------


def _canonicalise(name: str) -> str:
    """Lowercased + whitespace-collapsed canonical form for an entity name."""
    return " ".join(name.strip().lower().split())


def _decode_observations(raw: Optional[str]) -> list[str]:
    """Parse the ``entities.observations`` JSON column into a list of facts.

    Tolerant of NULL / malformed content (returns ``[]``) so a corrupt row
    can never crash a read. Non-string members are dropped; the result is
    lex-sorted for a deterministic view regardless of stored order.
    """
    import json as _json

    if not raw:
        return []
    try:
        data = _json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return sorted(str(x) for x in data if isinstance(x, str))


def default_db_path(workspace: str) -> str:
    """Canonical on-disk location of the workspace knowledge graph.

    Mirrors the MCP layer's ``_kg_path`` helper so library callers
    (ingest, recall fusion, CLI) and the MCP tools always address the
    same database file.
    """
    return os.path.join(workspace, "memory", "knowledge_graph.db")


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
                return str(row["entity_id"])
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

    def lookup(self, surface: str) -> Optional[str]:
        """Read-only alias resolution: return the entity id for
        *surface* or ``None`` when unknown.

        Unlike :meth:`resolve`, this never creates an entity — safe to
        call from read paths (recall fusion) where minting registry
        rows as a side effect of a query would pollute the graph.
        """
        canon = _canonicalise(surface)
        if not canon:
            return None
        with self._lock:
            row = self._conn.execute("SELECT entity_id FROM aliases WHERE alias = ?", (canon,)).fetchone()
        return None if row is None else str(row["entity_id"])

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

    # ------------------------------------------------------------------
    # Observations — accreted per-entity facts (roadmap §b, flag-gated)
    # ------------------------------------------------------------------

    def migrate_observations(self) -> bool:
        """Idempotently add the ``entities.observations`` column.

        ``observations`` is a JSON list of accreted facts about an entity,
        so entity-centric multi-hop questions that never mention a block id
        have somewhere to aggregate. Guarded by a column-exists check, so
        re-running on an already-migrated database is a pure no-op.

        Returns ``True`` when the column was added by this call, ``False``
        when it already existed (no-op). Deterministic — no clock / rand.
        """
        with self._lock:
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(entities)").fetchall()}
            if "observations" in cols:
                return False
            self._conn.execute("ALTER TABLE entities ADD COLUMN observations TEXT NOT NULL DEFAULT '[]'")
            self._conn.commit()
        return True

    def _require_observations(self) -> None:
        """Gate observation access behind the v4 ``entity_observations`` flag
        and ensure the column exists (lazy, idempotent migration)."""
        from .v4 import feature_flags

        feature_flags.require_enabled("entity_observations")
        self.migrate_observations()

    def add_observation(self, surface: str, fact: str) -> list[str]:
        """Append *fact* to *surface*'s entity observation list.

        Facts are deduplicated and stored in lexicographic order so the
        serialised column is a deterministic function of the fact *set* —
        re-adding the same fact, or adding facts in a different order,
        yields byte-identical storage. Returns the full accreted list.

        Requires the ``entity_observations`` v4 flag.
        """
        import json as _json

        text = fact.strip()
        if not text:
            raise ValueError("observation fact must be a non-empty string")
        self._require_observations()
        entity_id = self.resolve(surface)
        with self._lock:
            row = self._conn.execute("SELECT observations FROM entities WHERE id = ?", (entity_id,)).fetchone()
            current = _decode_observations(row["observations"] if row is not None else "[]")
            merged = sorted(set(current) | {text})
            self._conn.execute(
                "UPDATE entities SET observations = ? WHERE id = ?",
                (_json.dumps(merged, separators=(",", ":"), ensure_ascii=False), entity_id),
            )
            self._conn.commit()
        return merged

    def observations(self, surface: str) -> list[str]:
        """Return the lex-ordered accreted fact list for *surface*.

        Read-only: never mints a registry row. Unknown entities and
        never-observed entities both return ``[]``. Requires the
        ``entity_observations`` v4 flag.
        """
        self._require_observations()
        entity_id = self.lookup(surface)
        if entity_id is None:
            return []
        with self._lock:
            row = self._conn.execute("SELECT observations FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return []
        return _decode_observations(row["observations"])


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
# Edge proposal — HITL-gated typed-edge write (wedge guardrail, roadmap §a)
# ---------------------------------------------------------------------------

#: Proposal lifecycle states. A staged proposal has NOT touched the
#: source-of-truth ``edges`` table; only ``approve_edge`` commits.
PROPOSAL_STAGED = "staged"
PROPOSAL_APPLIED = "applied"
PROPOSAL_REJECTED = "rejected"

#: Provenance markers stamped onto an edge's ``metadata.origin`` so the write
#: path is auditable after the fact. ``approve_edge`` (the HITL propose→approve
#: flow) stamps :data:`EDGE_ORIGIN_HITL_APPROVED`; the direct admin-scoped MCP
#: ``graph_add_edge`` tool stamps :data:`EDGE_ORIGIN_DIRECT_ADMIN`. A read of an
#: edge's origin thus tells an operator whether it went through review or an
#: admin bypass.
EDGE_ORIGIN_HITL_APPROVED = "hitl_approved"
EDGE_ORIGIN_DIRECT_ADMIN = "direct_admin"


def _edge_proposal_id(subject: str, predicate_value: str, object_: str, source_block_id: str) -> str:
    """Deterministic id for an edge proposal.

    Derived by SHA-256 over a canonical, NUL-separated preimage of the
    edge tuple — no clock, no counter, no randomness — so proposing the
    same edge twice collapses onto one id (idempotent staging) and the id
    is reproducible across processes and substrates.
    """
    preimage = "\x00".join(
        [
            "mm-edge-proposal-v1",
            _canonicalise(subject),
            predicate_value,
            _canonicalise(object_),
            source_block_id.strip(),
        ]
    ).encode("utf-8")
    return "EP-" + hashlib.sha256(preimage).hexdigest()[:16]


@dataclass(frozen=True)
class EdgeProposal:
    """A staged, human-review-gated typed edge.

    The source-of-truth graph never self-modifies: an ``EdgeProposal`` sits
    in a separate ``edge_proposals`` table until :meth:`KnowledgeGraph.approve_edge`
    commits it. Entity resolution is deferred to approval time so staging a
    proposal touches neither the ``entities`` nor the ``edges`` table.
    """

    proposal_id: str
    subject: str
    predicate: Predicate
    object: str
    source_block_id: str
    status: str = PROPOSAL_STAGED
    confidence: float = 1.0
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
            "source_block_id": self.source_block_id,
            "status": self.status,
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
    CREATE TABLE IF NOT EXISTS edge_proposals (
        proposal_id      TEXT PRIMARY KEY,
        subject          TEXT NOT NULL,
        predicate        TEXT NOT NULL,
        object           TEXT NOT NULL,
        source_block_id  TEXT NOT NULL,
        status           TEXT NOT NULL DEFAULT 'staged',
        confidence       REAL NOT NULL DEFAULT 1.0,
        valid_from       TEXT,
        valid_until      TEXT,
        metadata         TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_edge_proposals_status ON edge_proposals(status);
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

    # ------------------------------------------------------------------
    # HITL-gated typed-edge proposals (wedge guardrail, roadmap §a)
    # ------------------------------------------------------------------

    def propose_edge(
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
    ) -> EdgeProposal:
        """Stage a typed edge for human review — the ONLY sanctioned write
        path for the typed-edge layer.

        The source-of-truth graph never self-modifies: this inserts a row
        into ``edge_proposals`` and touches neither ``entities`` nor
        ``edges``. A later :meth:`approve_edge` is the sole committer.
        Restaging the same edge tuple is idempotent (deterministic id,
        ``INSERT OR IGNORE``) and never clobbers an already-decided
        proposal's status.

        All validation (predicate, provenance, confidence, ISO-8601
        timestamps) runs here so a malformed proposal is refused up front
        rather than at approval.
        """
        import json as _json

        pred = predicate if isinstance(predicate, Predicate) else Predicate.from_str(predicate)
        if not source_block_id or not source_block_id.strip():
            raise ValueError("source_block_id is required for provenance")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")
        if valid_from is not None:
            _parse_iso8601(valid_from)
        if valid_until is not None:
            _parse_iso8601(valid_until)
        if valid_from is not None and valid_until is not None:
            if _parse_iso8601(valid_until) < _parse_iso8601(valid_from):
                raise ValueError("valid_until must be >= valid_from")

        meta = dict(metadata or {})
        pid = _edge_proposal_id(subject, pred.value, object, source_block_id)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO edge_proposals
                    (proposal_id, subject, predicate, object, source_block_id,
                     status, confidence, valid_from, valid_until, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    subject,
                    pred.value,
                    object,
                    source_block_id.strip(),
                    PROPOSAL_STAGED,
                    float(confidence),
                    valid_from,
                    valid_until,
                    _json.dumps(meta, separators=(",", ":"), default=str, sort_keys=True),
                ),
            )
            self._conn.commit()
        got = self.get_edge_proposal(pid)
        assert got is not None  # nosec B101 — just inserted-or-ignored above
        return got

    def get_edge_proposal(self, proposal_id: str) -> Optional[EdgeProposal]:
        """Return the staged/decided proposal by id, or ``None`` if unknown."""
        import json as _json

        with self._lock:
            row = self._conn.execute(
                "SELECT proposal_id, subject, predicate, object, source_block_id, "
                "status, confidence, valid_from, valid_until, metadata "
                "FROM edge_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            meta = _json.loads(row["metadata"] or "{}")
        except (ValueError, TypeError):
            meta = {}
        return EdgeProposal(
            proposal_id=row["proposal_id"],
            subject=row["subject"],
            predicate=Predicate.from_str(row["predicate"]),
            object=row["object"],
            source_block_id=row["source_block_id"],
            status=row["status"],
            confidence=float(row["confidence"]),
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            metadata=meta if isinstance(meta, dict) else {},
        )

    def list_edge_proposals(self, *, status: Optional[str] = None) -> list[EdgeProposal]:
        """List proposals, optionally filtered by status, in deterministic
        (``proposal_id`` lex) order."""
        import json as _json

        clauses = ""
        params: tuple[Any, ...] = ()
        if status is not None:
            clauses = "WHERE status = ?"
            params = (status,)
        sql = (
            "SELECT proposal_id, subject, predicate, object, source_block_id, "
            "status, confidence, valid_from, valid_until, metadata "
            f"FROM edge_proposals {clauses} ORDER BY proposal_id"  # nosec B608 — `clauses` is a fixed literal; status passes as a bind param
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: list[EdgeProposal] = []
        for row in rows:
            try:
                meta = _json.loads(row["metadata"] or "{}")
            except (ValueError, TypeError):
                meta = {}
            out.append(
                EdgeProposal(
                    proposal_id=row["proposal_id"],
                    subject=row["subject"],
                    predicate=Predicate.from_str(row["predicate"]),
                    object=row["object"],
                    source_block_id=row["source_block_id"],
                    status=row["status"],
                    confidence=float(row["confidence"]),
                    valid_from=row["valid_from"],
                    valid_until=row["valid_until"],
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return out

    def approve_edge(self, proposal_id: str) -> Edge:
        """Commit a staged proposal to the source-of-truth graph.

        This is the sole path that turns a proposal into a real edge:
        it resolves both endpoints and delegates to :meth:`add_edge`
        (idempotent), then marks the proposal ``applied``. Approving an
        already-applied proposal re-commits idempotently. A ``rejected``
        proposal cannot be approved.

        Provenance: the committed edge is stamped ``metadata.origin =
        "hitl_approved"`` so an operator-approved edge is distinguishable
        from one written through the direct admin-scoped :meth:`add_edge`
        path (which stamps ``"direct_admin"``). This closes the audit gap
        where an approved edge and a directly-added edge were byte-identical.
        """
        prop = self.get_edge_proposal(proposal_id)
        if prop is None:
            raise KeyError(f"unknown edge proposal: {proposal_id!r}")
        if prop.status == PROPOSAL_REJECTED:
            raise ValueError(f"cannot approve a rejected proposal: {proposal_id!r}")
        edge = self.add_edge(
            prop.subject,
            prop.predicate,
            prop.object,
            source_block_id=prop.source_block_id,
            confidence=prop.confidence,
            valid_from=prop.valid_from,
            valid_until=prop.valid_until,
            metadata={**prop.metadata, "origin": EDGE_ORIGIN_HITL_APPROVED},
        )
        with self._lock:
            self._conn.execute(
                "UPDATE edge_proposals SET status = ? WHERE proposal_id = ?",
                (PROPOSAL_APPLIED, proposal_id),
            )
            self._conn.commit()
        return edge

    def reject_edge(self, proposal_id: str) -> EdgeProposal:
        """Mark a staged proposal ``rejected``. No edge is ever written.

        Rejecting an already-applied proposal is refused (the edge is
        already committed — rejecting the proposal would misrepresent the
        graph). Rejecting an already-rejected proposal is idempotent.
        """
        prop = self.get_edge_proposal(proposal_id)
        if prop is None:
            raise KeyError(f"unknown edge proposal: {proposal_id!r}")
        if prop.status == PROPOSAL_APPLIED:
            raise ValueError(f"cannot reject an already-applied proposal: {proposal_id!r}")
        with self._lock:
            self._conn.execute(
                "UPDATE edge_proposals SET status = ? WHERE proposal_id = ?",
                (PROPOSAL_REJECTED, proposal_id),
            )
            self._conn.commit()
        updated = self.get_edge_proposal(proposal_id)
        assert updated is not None  # nosec B101 — row existed above, only status changed
        return updated

    # ------------------------------------------------------------------
    # Relationship-aware view (roadmap §a — weight per relation, not flat)
    # ------------------------------------------------------------------

    def relations_of(
        self,
        entity: str,
        *,
        direction: str = "outgoing",
        include_expired: bool = False,
    ) -> dict[str, list[Edge]]:
        """Bucket an entity's typed edges by relation for relationship-aware
        recall.

        Returns a dict keyed by every typed-relation value (lex order,
        always all five keys present — empty lists when absent) so a recall
        fuser can weight ``supports`` differently from ``contradicts``
        instead of collapsing them into one flat neighbour set. ``direction``
        is ``"outgoing"`` (subject→object), ``"incoming"`` (object→subject),
        or ``"both"``.
        """
        direction = direction.lower()
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction must be 'outgoing', 'incoming', or 'both'")
        buckets: dict[str, list[Edge]] = {value: [] for value in TYPED_RELATION_VALUES}
        for value in TYPED_RELATION_VALUES:
            pred = Predicate.from_str(value)
            collected: list[Edge] = []
            if direction in {"outgoing", "both"}:
                collected.extend(self.edges_from(entity, predicate=pred, include_expired=include_expired))
            if direction in {"incoming", "both"}:
                collected.extend(self.edges_to(entity, predicate=pred, include_expired=include_expired))
            collected.sort(key=lambda e: (-e.confidence, e.subject, e.object, e.source_block_id))
            buckets[value] = collected
        return buckets

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
            f"{where} ORDER BY confidence DESC, subject, predicate, object"  # nosec B608 — clauses are built from a fixed set of column names ("subject", "object", "predicate"); all values pass as bind params via `params`
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
    "EdgeProposal",
    "EntityRegistry",
    "KnowledgeGraph",
    "GraphStats",
    "TYPED_RELATIONS",
    "TYPED_RELATION_VALUES",
    "is_typed_relation",
    "default_db_path",
    "PROPOSAL_STAGED",
    "PROPOSAL_APPLIED",
    "PROPOSAL_REJECTED",
    "EDGE_ORIGIN_HITL_APPROVED",
    "EDGE_ORIGIN_DIRECT_ADMIN",
]
