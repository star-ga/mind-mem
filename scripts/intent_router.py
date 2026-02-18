"""mind-mem Intent Router — 9-type query intent classification.

Classifies queries into intent types with confidence scoring.
Each intent maps to retrieval parameters (expansion mode, rerank weights, graph depth).
Deterministic, regex-based — no model required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


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
            r'\bwhy\b', r'\breason\b', r'\bcause\b', r'\bexplain\b',
            r'\bhow come\b', r'\bwhat made\b',
        ],
    },
    "WHEN": {
        "expansion": "date",
        "graph_depth": 1,
        "rerank": "temporal",
        "patterns": [
            r'\bwhen\b', r'\bdate\b', r'\btime\b', r'\byear\b',
            r'\bmonth\b', r'\blast\b.*\b(week|month|year)\b',
            r'\d{4}-\d{2}-\d{2}', r'\bin \d{4}\b',
        ],
    },
    "ENTITY": {
        "expansion": "entity",
        "graph_depth": 1,
        "rerank": "exact",
        "patterns": [
            r'\bwho is\b', r'\bwho was\b', r'\bwhat is\b', r'\bwhich\b',
            r'\bname of\b', r'\btell me about\b',
        ],
    },
    "WHAT": {
        "expansion": "rm3",
        "graph_depth": 1,
        "rerank": "default",
        "patterns": [
            r'\bwhat\b', r'\bdescribe\b', r'\bdefine\b',
        ],
    },
    "HOW": {
        "expansion": "rm3",
        "graph_depth": 2,
        "rerank": "procedural",
        "patterns": [
            r'\bhow\b(?!\s+come)', r'\bsteps?\b', r'\bprocess\b',
            r'\bprocedure\b', r'\bmethod\b', r'\binstructions?\b',
        ],
    },
    "LIST": {
        "expansion": "broad",
        "graph_depth": 0,
        "rerank": "diversity",
        "patterns": [
            r'\blist\b', r'\ball\b.*\b(of|the)\b', r'\bwhich\b.*\ball\b',
            r'\bevery\b', r'\benumerate\b', r'\bshow me all\b',
        ],
    },
    "VERIFY": {
        "expansion": "exact",
        "graph_depth": 1,
        "rerank": "evidence",
        "patterns": [
            r'\bis it true\b', r'\bdid\b.*\b(ever|really)\b',
            r'\bconfirm\b', r'\bverify\b', r'\btrue that\b',
            r'\bcorrect that\b', r'\bis that right\b',
        ],
    },
    "COMPARE": {
        "expansion": "multi",
        "graph_depth": 2,
        "rerank": "balanced",
        "patterns": [
            r'\bcompare\b', r'\bdifference\b', r'\bvs\.?\b',
            r'\bversus\b', r'\bbetter\b.*\bor\b', r'\bwhich\b.*\bor\b',
            r'\bcontrast\b', r'\bsimilar\b',
        ],
    },
    "TRACE": {
        "expansion": "chain",
        "graph_depth": 3,
        "rerank": "temporal",
        "patterns": [
            r'\btrace\b', r'\bsequence\b', r'\bhistory of\b',
            r'\bevolution\b', r'\bprogress\b', r'\btimeline\b',
            r'\bover time\b', r'\bchanged?\b.*\bover\b',
        ],
    },
}


class IntentRouter:
    """Classifies queries into 9 intent types with confidence scoring."""

    def __init__(self):
        self._compiled = {}
        for intent, cfg in INTENT_CONFIG.items():
            self._compiled[intent] = [re.compile(p, re.IGNORECASE) for p in cfg["patterns"]]

    def classify(self, query: str) -> IntentResult:
        """Classify query intent with confidence.

        Returns IntentResult with intent type, confidence, sub-intents,
        and retrieval parameter overrides.
        """
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

        return IntentResult(
            intent=primary[0],
            confidence=round(confidence, 3),
            sub_intents=sub_intents,
            params=_get_params(primary[0]),
        )

    def classify_with_fallback(self, query: str) -> IntentResult:
        """Classify with fallback to legacy detect_query_type()."""
        result = self.classify(query)
        if result.confidence >= 0.3:
            return result

        # Fall back to legacy detection if confidence is low
        try:
            from recall import detect_query_type
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

def get_router() -> IntentRouter:
    """Get or create singleton IntentRouter."""
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router
