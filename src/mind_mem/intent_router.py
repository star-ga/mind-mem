"""mind-mem Intent Router — 9-type adaptive query intent classification.

Classifies queries into intent types with confidence scoring.
Each intent maps to retrieval parameters (expansion mode, rerank weights, graph depth).
Regex-based classification with adaptive confidence adjustment from historical
query performance feedback (#470).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .observability import get_logger

_log = get_logger("intent_router")


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: str
    confidence: float
    sub_intents: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


# Intent definitions with retrieval parameter overrides
INTENT_CONFIG = {
    "WHY": {
        "expansion": "rm3",
        "graph_depth": 2,
        "rerank": "causal",
        "patterns": [
            r"\bwhy\b",
            r"\breason\b",
            r"\bcause\b",
            r"\bexplain\b",
            r"\bhow come\b",
            r"\bwhat made\b",
        ],
    },
    "WHEN": {
        "expansion": "date",
        "graph_depth": 1,
        "rerank": "temporal",
        "patterns": [
            r"\bwhen\b",
            r"\bdate\b",
            r"\btime\b",
            r"\byear\b",
            r"\bmonth\b",
            r"\blast\b.*\b(week|month|year)\b",
            r"\d{4}-\d{2}-\d{2}",
            r"\bin \d{4}\b",
        ],
    },
    "ENTITY": {
        "expansion": "entity",
        "graph_depth": 1,
        "rerank": "exact",
        "patterns": [
            r"\bwho is\b",
            r"\bwho was\b",
            r"\bwhat is\b",
            r"\bwhich\b",
            r"\bname of\b",
            r"\btell me about\b",
        ],
    },
    "WHAT": {
        "expansion": "rm3",
        "graph_depth": 1,
        "rerank": "default",
        "patterns": [
            r"\bwhat\b",
            r"\bdescribe\b",
            r"\bdefine\b",
        ],
    },
    "HOW": {
        "expansion": "rm3",
        "graph_depth": 2,
        "rerank": "procedural",
        "patterns": [
            r"\bhow\b(?!\s+come)",
            r"\bsteps?\b",
            r"\bprocess\b",
            r"\bprocedure\b",
            r"\bmethod\b",
            r"\binstructions?\b",
        ],
    },
    "LIST": {
        "expansion": "broad",
        "graph_depth": 0,
        "rerank": "diversity",
        "patterns": [
            r"\blist\b",
            r"\ball\b.*\b(of|the)\b",
            r"\bwhich\b.*\ball\b",
            r"\bevery\b",
            r"\benumerate\b",
            r"\bshow me all\b",
        ],
    },
    "VERIFY": {
        "expansion": "exact",
        "graph_depth": 1,
        "rerank": "evidence",
        "patterns": [
            r"\bis it true\b",
            r"\bdid\b.*\b(ever|really)\b",
            r"\bconfirm\b",
            r"\bverify\b",
            r"\btrue that\b",
            r"\bcorrect that\b",
            r"\bis that right\b",
        ],
    },
    "COMPARE": {
        "expansion": "multi",
        "graph_depth": 2,
        "rerank": "balanced",
        "patterns": [
            r"\bcompare\b",
            r"\bdifference\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bbetter\b.*\bor\b",
            r"\bwhich\b.*\bor\b",
            r"\bcontrast\b",
            r"\bsimilar\b",
        ],
    },
    "TRACE": {
        "expansion": "chain",
        "graph_depth": 3,
        "rerank": "temporal",
        "patterns": [
            r"\btrace\b",
            r"\bsequence\b",
            r"\bhistory of\b",
            r"\bevolution\b",
            r"\bprogress\b",
            r"\btimeline\b",
            r"\bover time\b",
            r"\bchanged?\b.*\bover\b",
        ],
    },
}


class IntentRouter:
    """Classifies queries into 9 intent types with adaptive confidence scoring.

    Regex-based classification is augmented by historical performance feedback.
    When enough samples exist (>= ``_MIN_SAMPLES``), per-intent success ratios
    are used to scale confidence — poorly-performing intents get down-weighted
    while well-performing intents retain full confidence.
    """

    _MIN_SAMPLES = 5  # minimum feedback samples before adaptation kicks in

    def __init__(self, workspace: str | None = None):
        self._compiled: dict[str, list[re.Pattern[str]]] = {}
        for intent, cfg in INTENT_CONFIG.items():
            patterns = cfg["patterns"]
            if not isinstance(patterns, list):
                raise RuntimeError(f"invariant violated: INTENT_CONFIG[{intent!r}]['patterns'] is not a list")
            self._compiled[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]

        # Adaptive routing state (#470)
        self._intent_stats: dict[str, dict[str, int]] = {}
        self._adaptation_weights: dict[str, float] = {}
        self._workspace: str | None = workspace

        # Load persisted stats if workspace is set
        if workspace:
            self._load_stats()

    # --- Adaptive routing: feedback + persistence (#470) ---

    def record_feedback(
        self,
        query: str,
        intent: str,
        result_count: int,
        avg_score: float,
        user_selected: int = 0,
    ) -> None:
        """Record query outcome for adaptive confidence adjustment.

        Args:
            query: Original query string (logged but not stored long-term).
            intent: The intent that was classified for this query.
            result_count: Number of results returned.
            avg_score: Average BM25/reranked score of returned results.
            user_selected: Number of results the user actually used (0 = unknown).
        """
        stats = self._intent_stats.setdefault(intent, {"good": 0, "poor": 0, "total": 0})
        stats["total"] += 1

        # A query outcome is "good" when results exist and have reasonable scores
        if result_count > 0 and avg_score > 0.3:
            stats["good"] += 1
        else:
            stats["poor"] += 1

        # Recalculate adaptation weight once we have enough samples
        total = stats["total"]
        if total >= self._MIN_SAMPLES:
            ratio = stats["good"] / total
            # Range: 0.5 (all poor) to 1.0 (all good)
            self._adaptation_weights[intent] = 0.5 + 0.5 * ratio

        _log.debug(
            "feedback_recorded",
            intent=intent,
            result_count=result_count,
            avg_score=round(avg_score, 3),
            total=total,
            weight=self._adaptation_weights.get(intent),
        )

        # Persist to disk
        if self._workspace:
            self._save_stats()

    def _stats_path(self) -> str | None:
        """Return path to persisted stats JSON, or None if no workspace."""
        if not self._workspace:
            return None
        return os.path.join(self._workspace, "memory", "intent_router_stats.json")

    def _load_stats(self) -> None:
        """Load persisted intent stats and adaptation weights from disk."""
        path = self._stats_path()
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._intent_stats = data.get("intent_stats", {})
            self._adaptation_weights = data.get("adaptation_weights", {})
            _log.debug(
                "stats_loaded",
                path=path,
                intents=len(self._intent_stats),
                adapted=len(self._adaptation_weights),
            )
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("stats_load_failed", path=path, error=str(exc))

    def _save_stats(self) -> None:
        """Persist intent stats and adaptation weights to disk."""
        path = self._stats_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "intent_stats": self._intent_stats,
                "adaptation_weights": self._adaptation_weights,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            _log.debug("stats_saved", path=path)
        except OSError as exc:
            _log.warning("stats_save_failed", path=path, error=str(exc))

    # --- Classification ---

    def classify(self, query: str) -> IntentResult:
        """Classify query intent with adaptive confidence.

        Returns IntentResult with intent type, confidence, sub-intents,
        and retrieval parameter overrides.  When historical feedback data
        exists (>= 5 samples for the winning intent), the confidence is
        scaled by the adaptation weight (range 0.5--1.0).
        """
        _log.debug("classify", query=query[:80] if query else "")
        if not query or not query.strip():
            return IntentResult(
                intent="WHAT",
                confidence=0.0,
                params=INTENT_CONFIG["WHAT"].copy(),
            )

        scores = {}
        for intent, patterns in self._compiled.items():
            match_count = sum(1 for p in patterns if p.search(query))
            if match_count > 0:
                # Confidence = matched patterns / total patterns, boosted for more matches
                total = len(patterns)
                scores[intent] = match_count / total

        if not scores:
            # Default to WHAT with low confidence
            return IntentResult(
                intent="WHAT",
                confidence=0.1,
                params=_get_params("WHAT"),
            )

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = ranked[0]
        sub_intents = [r[0] for r in ranked[1:] if r[1] > 0.2]

        # Confidence: primary score normalized
        confidence = min(primary[1] * 2, 1.0)  # Scale up, cap at 1.0

        # Adaptive confidence adjustment (#470): scale by historical success ratio
        intent_name = primary[0]
        if intent_name in self._adaptation_weights:
            weight = self._adaptation_weights[intent_name]
            confidence *= weight
            _log.debug(
                "adaptive_confidence",
                intent=intent_name,
                weight=round(weight, 3),
                adjusted=round(confidence, 3),
            )

        return IntentResult(
            intent=intent_name,
            confidence=round(confidence, 3),
            sub_intents=sub_intents,
            params=_get_params(intent_name),
        )

    def classify_with_fallback(self, query: str) -> IntentResult:
        """Classify with fallback to legacy detect_query_type()."""
        result = self.classify(query)
        if result.confidence >= 0.3:
            return result

        # Fall back to legacy detection if confidence is low
        try:
            from .recall import detect_query_type

            legacy = detect_query_type(query)
            # Map legacy types to our intents
            legacy_map = {
                "temporal": "WHEN",
                "multi_hop": "TRACE",
                "adversarial": "VERIFY",
                "single_hop": "WHAT",
            }
            mapped = legacy_map.get(legacy, "WHAT")
            return IntentResult(
                intent=mapped,
                confidence=0.5,
                sub_intents=result.sub_intents,
                params=_get_params(mapped),
            )
        except (ImportError, AttributeError):
            return result


def _get_params(intent: str) -> dict:
    """Get retrieval parameters for an intent type."""
    cfg = INTENT_CONFIG.get(intent, INTENT_CONFIG["WHAT"])
    return {
        "expansion": cfg["expansion"],
        "graph_depth": cfg["graph_depth"],
        "rerank": cfg["rerank"],
    }


# Singleton for convenience
_router = None


def get_router(workspace: str | None = None) -> IntentRouter:
    """Get or create singleton IntentRouter.

    Args:
        workspace: Optional workspace path for persistence of adaptive stats.
            If provided and the singleton was created without one, updates it.
    """
    global _router
    if _router is None:
        _router = IntentRouter(workspace=workspace)
    elif workspace and not _router._workspace:
        # Upgrade existing singleton with workspace for persistence
        _router._workspace = workspace
        _router._load_stats()
    return _router
