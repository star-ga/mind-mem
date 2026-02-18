"""mind-mem Optional Cross-Encoder Reranker.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (80MB, CPU-friendly).
Entirely optional — disabled by default. Requires sentence-transformers.
"""
from __future__ import annotations

_CE_MODEL = None
_CE_AVAILABLE = None


def _check_available() -> bool:
    global _CE_AVAILABLE
    if _CE_AVAILABLE is not None:
        return _CE_AVAILABLE
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        _CE_AVAILABLE = True
    except ImportError:
        _CE_AVAILABLE = False
    return _CE_AVAILABLE


class CrossEncoderReranker:
    """CPU-friendly cross-encoder reranker."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if not _check_available():
            raise ImportError("sentence-transformers required for cross-encoder")
        from sentence_transformers import CrossEncoder
        global _CE_MODEL
        if _CE_MODEL is None:
            _CE_MODEL = CrossEncoder(model)
        self._model = _CE_MODEL

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10,
               blend_weight: float = 0.6) -> list[dict]:
        """Score with cross-encoder, blend with original scores.

        Final score = blend_weight * CE_score + (1 - blend_weight) * original_score
        """
        if not candidates:
            return []

        # Prepare pairs
        texts = [c.get("content", c.get("text", "")) for c in candidates]
        pairs = [(query, t) for t in texts]

        # Score
        ce_scores = self._model.predict(pairs)

        # Normalize CE scores to [0, 1]
        min_s, max_s = min(ce_scores), max(ce_scores)
        range_s = max_s - min_s if max_s > min_s else 1.0
        ce_norm = [(s - min_s) / range_s for s in ce_scores]

        # Normalize original scores to [0, 1]
        orig_scores = [c.get("score", 0) for c in candidates]
        min_o = min(orig_scores) if orig_scores else 0
        max_o = max(orig_scores) if orig_scores else 1
        range_o = max_o - min_o if max_o > min_o else 1.0
        orig_norm = [(s - min_o) / range_o for s in orig_scores]

        # Blend
        results = []
        for i, c in enumerate(candidates):
            item = c.copy()
            item["ce_score"] = float(ce_scores[i])
            item["score"] = blend_weight * ce_norm[i] + (1 - blend_weight) * orig_norm[i]
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def is_available() -> bool:
        """Check if cross-encoder model is loadable."""
        return _check_available()
