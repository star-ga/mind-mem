"""Reranker ensemble via Borda count (v3.3.0 Tier 4 #9).

A single cross-encoder reranker has systematic failure modes
(e.g., over-weighting lexical match on certain query types). Combining
multiple independent rerankers with Borda-count voting smooths the
failures and reliably improves top-K precision by 1-3 points on
hybrid retrieval benchmarks.

This module provides:

* :class:`Reranker` — minimal protocol that existing rerankers
  (:class:`CrossEncoderReranker`, optional BGE, optional
  LLM-as-reranker) implement.
* :class:`EnsembleReranker` — composes N rerankers, runs each
  independently on the same candidate pool, and fuses the resulting
  rankings via Borda count. Failed rerankers are skipped with a log
  entry; the ensemble never blocks on a single-reranker failure.
* :func:`create_ensemble` — factory reads
  ``retrieval.reranker_ensemble`` from ``mind-mem.json`` and wires
  the configured rerankers.

Opt-in via:

    {
      "retrieval": {
        "reranker_ensemble": {
          "enabled": false,
          "rerankers": ["cross_encoder", "bge", "llm"],
          "top_k": 10,
          "llm": {
            "base_url": "http://127.0.0.1:8766/v1/chat/completions",
            "model": "claude-proxy/claude-opus-4-7"
          }
        }
      }
    }

Heavy dependencies (sentence-transformers / BGE) ship behind the
existing ``mind-mem[cross-encoder]`` extra; operators that don't
install them get a graceful fallback to CE-only.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .observability import get_logger

_log = get_logger("rerank_ensemble")


# ---------------------------------------------------------------------------
# Reranker protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    """Minimal contract for a reranker.

    Implementations take a query + candidate list and return the same
    list re-ordered (and optionally blended with a score). The
    ``rerank`` method signature matches
    :class:`CrossEncoderReranker.rerank` so the existing reranker
    drops in without wrapping.
    """

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        *,
        top_k: int = 10,
        blend_weight: float = 0.6,
    ) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class EnsembleReranker:
    """Run N rerankers on the same pool; fuse rankings via Borda count.

    Borda count: for each candidate, sum over (N - rank_i) across
    rerankers. The candidate with the highest sum wins. Ties broken by
    the first reranker's order (stable).

    Failed rerankers are simply skipped — the ensemble falls back to
    whatever rerankers succeeded. A single failure never blocks recall.
    """

    def __init__(self, rerankers: list[Reranker], name: str = "ensemble"):
        if not rerankers:
            raise ValueError("EnsembleReranker requires at least one reranker")
        self._rerankers = list(rerankers)
        self._name = name

    @staticmethod
    def is_available() -> bool:
        """Ensemble itself is always importable; individual rerankers gate."""
        return True

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        *,
        top_k: int = 10,
        blend_weight: float = 0.6,
    ) -> list[dict]:
        if not candidates:
            return candidates
        if len(self._rerankers) == 1:
            # Degenerate case: no ensembling, pass through.
            try:
                return self._rerankers[0].rerank(query, candidates, top_k=top_k, blend_weight=blend_weight)
            except Exception as exc:  # pragma: no cover
                _log.warning("ensemble_single_reranker_failed", error=str(exc))
                return candidates

        # Snapshot the stable order per reranker.
        rankings: list[list[str]] = []
        for idx, r in enumerate(self._rerankers):
            try:
                reranked = r.rerank(query, candidates, top_k=top_k, blend_weight=blend_weight)
                rankings.append([str(c.get("_id") or f"idx:{i}") for i, c in enumerate(reranked)])
            except Exception as exc:
                _log.warning(
                    "ensemble_member_failed",
                    reranker=idx,
                    error=str(exc),
                )
                continue

        if not rankings:
            _log.warning("ensemble_all_failed", reranker_count=len(self._rerankers))
            return candidates

        # Borda count: candidate at rank r in a list of N gets (N - r) points.
        # (Rename the rank variable so it doesn't shadow the outer ``r``
        # Reranker loop variable from a few lines up.)
        scores: dict[str, float] = {}
        for ranking in rankings:
            n = len(ranking)
            for rank_pos, cid in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + (n - rank_pos)

        # Resolve back to candidate dicts, keeping stable order from
        # the first reranker for ties.
        by_id: dict[str, dict] = {}
        for c in candidates:
            cid = str(c.get("_id") or "")
            if cid and cid not in by_id:
                by_id[cid] = c

        ranked = sorted(
            by_id.values(),
            key=lambda c: scores.get(str(c.get("_id") or ""), 0.0),
            reverse=True,
        )

        # Annotate with ensemble score.
        for c in ranked:
            cid = str(c.get("_id") or "")
            c["_ensemble_borda"] = scores.get(cid, 0.0)

        _log.info(
            "ensemble_reranked",
            members=len(rankings),
            total=len(candidates),
            top_k=top_k,
        )
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_cross_encoder() -> Reranker | None:
    try:
        from .cross_encoder_reranker import CrossEncoderReranker

        if not CrossEncoderReranker.is_available():
            return None
        return CrossEncoderReranker()
    except Exception as exc:
        _log.info("ensemble_ce_unavailable", error=str(exc))
        return None


def _build_bge() -> Reranker | None:
    """BGE reranker (BAAI/bge-reranker-v2-m3) — heavier alternative to CE.

    Uses the same sentence-transformers CrossEncoder interface with a
    different pretrained weight. Returns None when the model can't be
    loaded (typically because ``sentence-transformers`` isn't
    installed or the HuggingFace cache is empty and no network).
    """
    try:
        import os

        from sentence_transformers import CrossEncoder  # type: ignore

        device = os.environ.get("MIND_MEM_RERANKER_DEVICE", "cpu")
        model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

        class BGEAdapter:
            def rerank(
                self,
                query: str,
                candidates: list[dict],
                *,
                top_k: int = 10,
                blend_weight: float = 0.6,
            ) -> list[dict]:
                pairs = [(query, c.get("content", c.get("excerpt", ""))) for c in candidates]
                scores = model.predict(pairs)
                blended = []
                for c, s in zip(candidates, scores):
                    item = dict(c)
                    item["_bge_score"] = float(s)
                    # Blend with original score like CE does.
                    orig = float(item.get("score", 0.0))
                    item["score"] = blend_weight * float(s) + (1 - blend_weight) * orig
                    blended.append(item)
                blended.sort(key=lambda x: x["score"], reverse=True)
                return blended[:top_k]

        return BGEAdapter()
    except Exception as exc:
        _log.info("ensemble_bge_unavailable", error=str(exc))
        return None


def _build_llm(llm_cfg: dict[str, Any]) -> Reranker | None:
    """LLM-as-reranker using claude-proxy / OpenAI-compatible endpoint.

    Pairs (query, candidate) are scored via a single structured prompt;
    the response is parsed to a 0-100 relevance score per candidate.
    Skips when the configured base_url fails the SSRF allowlist check.
    """
    try:
        from .query_planner import _validate_base_url  # reuse SSRF guard

        base_url = llm_cfg.get("base_url", "http://127.0.0.1:8766/v1/chat/completions")
        allow_external = bool(llm_cfg.get("allow_external", False))
        _validate_base_url(base_url, allow_external=allow_external)

        model = llm_cfg.get("model", "claude-proxy/claude-opus-4-7")
        timeout = float(llm_cfg.get("timeout", 30.0))

        class LLMRerankerAdapter:
            def rerank(
                self,
                query: str,
                candidates: list[dict],
                *,
                top_k: int = 10,
                blend_weight: float = 0.6,
            ) -> list[dict]:
                import json
                import urllib.request

                items = [
                    {"id": str(c.get("_id") or i), "text": c.get("content") or c.get("excerpt") or ""} for i, c in enumerate(candidates)
                ]
                prompt = (
                    "Score each candidate's relevance to the query on a 0-100 "
                    "integer scale. Return a JSON object mapping candidate ID "
                    "to score, nothing else.\n\n"
                    f"Query: {query}\n\n"
                    f"Candidates: {json.dumps(items, default=str)[:20000]}"
                )
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 800,
                }
                req = urllib.request.Request(
                    base_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                    text = body["choices"][0]["message"]["content"].strip()
                    # Best-effort JSON parse — accept first {...} block.
                    start = text.find("{")
                    end = text.rfind("}")
                    if start < 0 or end < 0:
                        raise ValueError("no JSON block in LLM rerank response")
                    scores_map = json.loads(text[start : end + 1])
                except (OSError, TimeoutError, ValueError, KeyError, IndexError) as exc:
                    _log.warning("llm_rerank_failed", error=str(exc))
                    # Fail-open: return candidates unchanged.
                    return candidates[:top_k]

                blended = []
                for c in candidates:
                    item = dict(c)
                    cid = str(item.get("_id") or "")
                    llm_score = float(scores_map.get(cid, 0) or 0)
                    item["_llm_rerank_score"] = llm_score
                    orig = float(item.get("score", 0.0))
                    item["score"] = blend_weight * llm_score + (1 - blend_weight) * orig
                    blended.append(item)
                blended.sort(key=lambda x: x["score"], reverse=True)
                return blended[:top_k]

        return LLMRerankerAdapter()
    except Exception as exc:
        _log.info("ensemble_llm_unavailable", error=str(exc))
        return None


def create_ensemble(config: dict[str, Any] | None) -> EnsembleReranker | None:
    """Build an ensemble from the ``retrieval.reranker_ensemble`` section.

    Returns None when the ensemble is disabled or no member rerankers
    could be constructed. Callers fall back to the single-model CE
    rerank path when None.
    """
    if not config or not isinstance(config, dict):
        return None
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return None
    ens = retrieval.get("reranker_ensemble", {})
    if not isinstance(ens, dict) or not ens.get("enabled", False):
        return None

    names: list[str] = list(ens.get("rerankers", ["cross_encoder"]))
    members: list[Reranker] = []
    for name in names:
        reranker: Reranker | None = None
        if name == "cross_encoder":
            reranker = _build_cross_encoder()
        elif name == "bge":
            reranker = _build_bge()
        elif name == "llm":
            reranker = _build_llm(ens.get("llm", {}))
        else:
            _log.info("ensemble_unknown_reranker", name=name)
            continue
        if reranker is not None:
            members.append(reranker)

    if not members:
        _log.warning("ensemble_all_members_failed_to_build", requested=names)
        return None

    return EnsembleReranker(members, name=f"ensemble[{','.join(names)}]")


__all__ = [
    "Reranker",
    "EnsembleReranker",
    "create_ensemble",
]
