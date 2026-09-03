"""mind-mem Optional Cross-Encoder Reranker.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (80MB, CPU-friendly).
Entirely optional — disabled by default. Requires sentence-transformers.
"""

from __future__ import annotations

import threading
from typing import Any

#: Loaded cross-encoders, keyed by ``(model, device)``.  The key is the
#: whole point: a single-slot cache silently hands the first caller's model
#: to every later one, so a benchmark comparing two models measures one.
_CE_MODELS: dict[tuple[str, str], Any] = {}
_CE_LOAD_LOCK = threading.Lock()
_CE_AVAILABLE = None

#: Bumped once per real model construction. The cache above is a performance
#: claim -- "a full benchmark run loads the weights once, not once per query"
#: -- and a performance claim needs a number rather than an argument, so the
#: reload-count regression test reads this counter instead of inspecting code.
_CE_LOADS = 0


def ce_load_count() -> int:
    """How many times cross-encoder weights were loaded in this process."""
    return _CE_LOADS


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

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        """Load (or reuse) the requested cross-encoder.

        Models are cached per ``(model, device)`` for the life of the
        process, so asking for a different model or device really loads
        it instead of returning whatever the first caller loaded. The
        pair that was actually loaded is recorded on :attr:`model_name`
        and :attr:`device` — scores attributed to a model must be
        traceable to the model that produced them.
        """
        if not _check_available():
            raise ImportError("sentence-transformers required for cross-encoder")
        from sentence_transformers import CrossEncoder

        self.model_name = model
        self.device = device
        global _CE_LOADS
        key = (model, device)
        cached = _CE_MODELS.get(key)
        if cached is None:
            with _CE_LOAD_LOCK:
                # Re-check: another thread may have loaded it while we waited.
                cached = _CE_MODELS.get(key)
                if cached is None:
                    cached = CrossEncoder(model, device=device)
                    _CE_MODELS[key] = cached
                    _CE_LOADS += 1
        self._model = cached

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
        blend_weight: float = 0.6,
        batch_size: int = 32,
    ) -> list[dict]:
        """Score with cross-encoder, blend with original scores.

        Final score = blend_weight * CE_score + (1 - blend_weight) * original_score

        Args:
            batch_size: Number of (query, candidate) pairs per predict() call.
                        Prevents OOM on large candidate sets (default: 32).
        """
        if not candidates:
            return []

        # Prepare pairs
        texts = [c.get("content", c.get("text", "")) for c in candidates]
        pairs = [(query, t) for t in texts]

        # Score in batches to avoid OOM on large candidate sets
        ce_scores = self._model.predict(pairs, batch_size=batch_size)

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
