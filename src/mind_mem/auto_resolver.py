#!/usr/bin/env python3
"""mind-mem Automatic Contradiction Resolution Suggestions.

Extends the conflict_resolver with:
- Auto-generated resolution proposals with confidence scoring
- Side-effect analysis (downstream impacts)
- User preference learning (track which strategies humans prefer)
- Batch resolution with constraint checking

Usage:
    from .auto_resolver import AutoResolver
    resolver = AutoResolver(workspace)
    suggestions = resolver.suggest_resolutions()
    resolver.record_preference("timestamp_priority", domain="security")

Zero external deps — json, os, sqlite3 (all stdlib).
"""

from __future__ import annotations

import json
import os
import sqlite3

from .audit_chain import AuditChain
from .causal_graph import CausalGraph
from .conflict_resolver import ResolutionStrategy, resolve_contradictions
from .observability import get_logger, metrics

_log = get_logger("auto_resolver")


def _db_path(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), ".mind-mem-audit", "resolver.db")


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
CREATE TABLE IF NOT EXISTS resolution_preferences (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy    TEXT NOT NULL,
    domain      TEXT DEFAULT '',
    chosen      INTEGER DEFAULT 0,
    rejected    INTEGER DEFAULT 0,
    timestamp   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rp_strategy ON resolution_preferences(strategy);
CREATE INDEX IF NOT EXISTS idx_rp_domain ON resolution_preferences(domain);

CREATE TABLE IF NOT EXISTS resolution_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contradiction_id TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    accepted        INTEGER DEFAULT 0,
    side_effects    TEXT DEFAULT '[]',
    timestamp       TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rh_contra ON resolution_history(contradiction_id);
"""


class ResolutionSuggestion:
    """A suggested resolution for a contradiction."""

    __slots__ = (
        "contradiction_id",
        "block_a",
        "block_b",
        "strategy",
        "confidence_score",
        "winner_id",
        "loser_id",
        "rationale",
        "side_effects",
        "preferred_score",
    )

    def __init__(
        self,
        contradiction_id: str,
        block_a: str,
        block_b: str,
        strategy: str,
        confidence_score: float,
        winner_id: str | None,
        loser_id: str | None,
        rationale: str,
        side_effects: list[str] | None = None,
        preferred_score: float = 0.0,
    ):
        self.contradiction_id = contradiction_id
        self.block_a = block_a
        self.block_b = block_b
        self.strategy = strategy
        self.confidence_score = confidence_score
        self.winner_id = winner_id
        self.loser_id = loser_id
        self.rationale = rationale
        self.side_effects = side_effects or []
        self.preferred_score = preferred_score

    def to_dict(self) -> dict:
        return {
            "contradiction_id": self.contradiction_id,
            "block_a": self.block_a,
            "block_b": self.block_b,
            "strategy": self.strategy,
            "confidence_score": round(self.confidence_score, 3),
            "winner_id": self.winner_id,
            "loser_id": self.loser_id,
            "rationale": self.rationale,
            "side_effects": self.side_effects,
            "preferred_score": round(self.preferred_score, 3),
        }


class AutoResolver:
    """Automatic contradiction resolution with learning."""

    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.realpath(workspace)
        self._graph = CausalGraph(workspace)
        self._chain = AuditChain(workspace)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = _connect(self.workspace)
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _confidence_to_score(self, confidence_str: str) -> float:
        """Convert string confidence to numeric score."""
        mapping = {"high": 0.85, "medium": 0.6, "low": 0.3}
        return mapping.get(confidence_str, 0.5)

    def _get_preference_boost(self, strategy: str, domain: str = "") -> float:
        """Get preference boost for a strategy based on historical choices.

        Returns a boost factor [0, 0.15] based on how often this strategy
        has been chosen vs rejected in the past.
        """
        conn = _connect(self.workspace)
        try:
            # Check domain-specific preference
            if domain:
                row = conn.execute(
                    "SELECT SUM(chosen) as c, SUM(rejected) as r "
                    "FROM resolution_preferences WHERE strategy = ? AND domain = ?",
                    (strategy, domain),
                ).fetchone()
                if row and row["c"]:
                    total = (row["c"] or 0) + (row["r"] or 0)
                    if total > 0:
                        return float(0.15 * (row["c"] / total))

            # Fallback to global preference
            row = conn.execute(
                "SELECT SUM(chosen) as c, SUM(rejected) as r FROM resolution_preferences WHERE strategy = ?",
                (strategy,),
            ).fetchone()
            if row and row["c"]:
                total = (row["c"] or 0) + (row["r"] or 0)
                if total > 0:
                    return float(0.1 * (row["c"] / total))
        finally:
            conn.close()

        return 0.0

    def _analyze_side_effects(self, block_id: str) -> list[str]:
        """Analyze downstream impacts if a block is superseded.

        Uses the causal graph to find dependent blocks.
        """
        effects = []
        dependents = self._graph.dependents(block_id)
        for dep in dependents:
            effects.append(f"Block {dep.source_id} depends on {block_id} via {dep.edge_type} — may need review")
        return effects

    def suggest_resolutions(self) -> list[ResolutionSuggestion]:
        """Generate resolution suggestions for all contradictions.

        Returns suggestions sorted by confidence score (highest first).
        """
        resolutions = resolve_contradictions(self.workspace)
        suggestions = []

        for res in resolutions:
            base_score = self._confidence_to_score(res.get("confidence", "low"))
            strategy = res.get("strategy", ResolutionStrategy.MANUAL)

            # Add preference boost
            pref_boost = self._get_preference_boost(strategy)
            confidence_score = min(base_score + pref_boost, 1.0)

            # Analyze side effects for the loser block
            side_effects = []
            if res.get("loser_id"):
                side_effects = self._analyze_side_effects(res["loser_id"])

            # Reduce confidence if there are side effects
            if side_effects:
                confidence_score = max(confidence_score - 0.1 * len(side_effects), 0.1)

            suggestions.append(
                ResolutionSuggestion(
                    contradiction_id=res.get("contradiction_id", "?"),
                    block_a=res.get("block_a", ""),
                    block_b=res.get("block_b", ""),
                    strategy=strategy,
                    confidence_score=confidence_score,
                    winner_id=res.get("winner_id"),
                    loser_id=res.get("loser_id"),
                    rationale=res.get("rationale", ""),
                    side_effects=side_effects,
                    preferred_score=pref_boost,
                )
            )

        suggestions.sort(key=lambda s: s.confidence_score, reverse=True)

        _log.info("auto_resolution_suggestions", count=len(suggestions))
        metrics.inc("auto_resolutions_suggested", len(suggestions))
        return suggestions

    def record_preference(
        self,
        strategy: str,
        *,
        domain: str = "",
        chosen: bool = True,
    ) -> None:
        """Record a user preference for a resolution strategy.

        Args:
            strategy: The strategy that was chosen/rejected.
            domain: Optional domain context (e.g., "security", "api").
            chosen: True if user accepted, False if rejected.
        """
        conn = _connect(self.workspace)
        try:
            # Check if exists
            row = conn.execute(
                "SELECT id FROM resolution_preferences WHERE strategy = ? AND domain = ?",
                (strategy, domain),
            ).fetchone()

            if row:
                if chosen:
                    sql = (
                        "UPDATE resolution_preferences SET chosen = chosen + 1,"
                        " timestamp = datetime('now') WHERE id = ?"
                    )
                    conn.execute(sql, (row["id"],))
                else:
                    sql = (
                        "UPDATE resolution_preferences SET rejected = rejected + 1,"
                        " timestamp = datetime('now') WHERE id = ?"
                    )
                    conn.execute(sql, (row["id"],))
            else:
                conn.execute(
                    "INSERT INTO resolution_preferences (strategy, domain, chosen, rejected) VALUES (?, ?, ?, ?)",
                    (strategy, domain, 1 if chosen else 0, 0 if chosen else 1),
                )
            conn.commit()
        finally:
            conn.close()

        _log.info("preference_recorded", strategy=strategy, domain=domain, chosen=chosen)

    def accept_suggestion(self, suggestion: ResolutionSuggestion) -> None:
        """Record that a suggestion was accepted.

        Logs to resolution history and records preference.
        """
        conn = _connect(self.workspace)
        try:
            conn.execute(
                "INSERT INTO resolution_history "
                "(contradiction_id, strategy, confidence, accepted, side_effects) "
                "VALUES (?, ?, ?, 1, ?)",
                (
                    suggestion.contradiction_id,
                    suggestion.strategy,
                    suggestion.confidence_score,
                    json.dumps(suggestion.side_effects),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        self.record_preference(suggestion.strategy, chosen=True)

        # Log to audit chain
        self._chain.append(
            "apply_proposal",
            f"resolution/{suggestion.contradiction_id}",
            agent="auto_resolver",
            reason=suggestion.rationale,
            payload={
                "strategy": suggestion.strategy,
                "winner": suggestion.winner_id,
                "loser": suggestion.loser_id,
            },
        )

    def reject_suggestion(self, suggestion: ResolutionSuggestion, *, reason: str = "") -> None:
        """Record that a suggestion was rejected."""
        conn = _connect(self.workspace)
        try:
            conn.execute(
                "INSERT INTO resolution_history "
                "(contradiction_id, strategy, confidence, accepted, side_effects) "
                "VALUES (?, ?, ?, 0, ?)",
                (
                    suggestion.contradiction_id,
                    suggestion.strategy,
                    suggestion.confidence_score,
                    json.dumps(suggestion.side_effects),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        self.record_preference(suggestion.strategy, chosen=False)

    def preference_summary(self) -> dict:
        """Get summary of learned resolution preferences."""
        conn = _connect(self.workspace)
        try:
            rows = conn.execute(
                "SELECT strategy, domain, SUM(chosen) as total_chosen, "
                "SUM(rejected) as total_rejected "
                "FROM resolution_preferences "
                "GROUP BY strategy, domain"
            ).fetchall()
        finally:
            conn.close()

        summary = {}
        for row in rows:
            key = row["strategy"]
            if row["domain"]:
                key += f":{row['domain']}"
            total = (row["total_chosen"] or 0) + (row["total_rejected"] or 0)
            summary[key] = {
                "chosen": row["total_chosen"] or 0,
                "rejected": row["total_rejected"] or 0,
                "acceptance_rate": round((row["total_chosen"] or 0) / total, 3) if total > 0 else 0,
            }

        return summary
