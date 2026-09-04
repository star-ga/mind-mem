#!/usr/bin/env python3
"""mind-mem Hybrid Recall -- BM25 + Vector + RRF fusion.

Orchestrates BM25 (via recall.py or sqlite_index.py) and vector search
(via recall_vector.py) in parallel, then fuses rankings using Reciprocal
Rank Fusion (RRF).  Falls back gracefully to BM25-only when vector
dependencies (sentence-transformers) are not installed.

Configuration (mind-mem.json):
    {
      "recall": {
        "backend": "hybrid",
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "vector_model": "all-MiniLM-L6-v2",
        "vector_enabled": false,
        "rerank_depth": 50
      }
    }

``rerank_depth`` is how many fused candidates the cross-encoder is allowed
to see. It defaults to ``min(50, 5 * limit)`` and is capped at
``MAX_RERANK_CANDIDATES`` (200) -- the same ceiling the scan/sqlite path
has always used. It costs nothing unless the cross-encoder actually runs.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from contextlib import AbstractContextManager, nullcontext
from datetime import date
from typing import Any, Mapping

from . import vector_inertness
from ._recall_constants import MAX_RERANK_CANDIDATES
from .admissibility import admit_corpus, admit_leg, is_admissible_status, live_statuses, with_live_statuses, workspace_release_ids
from .enums import Leg
from .observability import get_logger, metrics, timed
from .retrieval_trace import current_trace, is_trace_enabled
from .retrieval_trace import step as _record_step
from .retrieval_trace import trace as _open_trace
from .scoring_instant import as_utc_datetime, resolve_scoring_instant

_log = get_logger("hybrid_recall")

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_NUMERIC_POSITIVE_KEYS = ("bm25_weight", "vector_weight", "rrf_k")


def validate_recall_config(cfg: dict[str, Any]) -> list[str]:
    """Validate recall config section. Returns list of error strings (empty = valid).

    Checks that bm25_weight, vector_weight, and rrf_k are numeric and positive.
    """
    errors: list[str] = []
    for key in _NUMERIC_POSITIVE_KEYS:
        if key not in cfg:
            continue
        val = cfg[key]
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric, got {type(val).__name__}: {val!r}")
            continue
        if numeric <= 0:
            errors.append(f"{key} must be positive, got {numeric}")
    return errors


#: Ceiling on the default rerank depth. ``min(50, 5 * limit)`` keeps the
#: cross-encoder's per-query cost bounded (measured: ~1.25 s at depth 50 on
#: a 12-core i7-5930K) while giving it enough candidates that reranking can
#: change WHICH blocks are served, not merely their order.
DEFAULT_RERANK_DEPTH = 50


def resolve_rerank_depth(config: dict[str, Any] | None, limit: int) -> int:
    """How many fused candidates the reranker may see for this request.

    Reranking that runs over exactly the ``limit`` blocks already chosen for
    the response cannot change recall@k by construction -- every block it
    could promote is already being served, so the only thing the model's
    latency buys is a permutation. The hybrid path did exactly that: it
    sliced ``fused[:limit]`` and handed the slice to the cross-encoder. The
    scan/sqlite path never had this defect; it reranks over up to
    ``MAX_RERANK_CANDIDATES``.

    Resolution order: ``recall.rerank_depth`` when set, else
    ``min(DEFAULT_RERANK_DEPTH, 5 * limit)``. The result is capped at
    ``MAX_RERANK_CANDIDATES`` and then floored at ``limit`` -- the floor is
    applied LAST and deliberately outranks the cap, because a depth below
    the response size would hand the reranker fewer candidates than the
    caller asked to be served and silently truncate the response. For
    ``limit > MAX_RERANK_CANDIDATES`` the depth therefore equals ``limit``:
    no recall gain (the pool is the response), but no narrowing either.
    """
    limit = max(1, int(limit))
    default = min(DEFAULT_RERANK_DEPTH, 5 * limit)
    raw: Any = default
    if isinstance(config, dict) and config.get("rerank_depth") is not None:
        raw = config.get("rerank_depth")
    try:
        depth = int(raw)
    except (TypeError, ValueError):
        _log.warning("rerank_depth_invalid", value=repr(raw), fallback=default)
        depth = default
    if depth <= 0:
        _log.warning("rerank_depth_non_positive", value=depth, fallback=default)
        depth = default
    return max(limit, min(depth, MAX_RERANK_CANDIDATES))


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    ranked_lists: list[list[dict]],
    weights: list[float],
    k: int = 60,
    id_key: str = "_id",
    source_names: list[str] | None = None,
) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    For each document, RRF score = sum_i( weight_i / (k + rank_i) )
    where rank_i is the 1-based rank of the document in list i.

    Each fused hit is annotated with ``fusion_sources`` — a
    ``{arm_name: rank_1based}`` map recording WHICH input lists it came from
    and at what rank, e.g. ``{"bm25": 3, "vector": 1}``. Arm names come from
    ``source_names`` (positional, aligned to ``ranked_lists``); when absent
    the list index is used. This is the machine-readable signal that
    separates a genuinely FUSED hit (names >=2 arms) from a single-arm hit
    sitting on the 1/(k+1) noise floor (names exactly one arm) — the exact
    silent failure this annotation exists to expose.

    Audit R-2 — weight semantics: ``weights`` are RAW multipliers
    applied to each list's contribution. The output ``rrf_score``
    therefore scales linearly with the weight magnitudes — passing
    ``[1.0, 1.0]`` and ``[0.5, 0.5]`` produce identical RANKINGS but
    different absolute scores. Callers comparing absolute scores
    across requests must keep weights stable; callers comparing only
    RANKINGS are unaffected.

    Audit R-3: if only ONE non-empty list is present (the others
    are empty), ``hybrid_single_list_degenerate`` is incremented on
    the metrics registry so dashboards can flag silent BM25-only or
    vector-only fallbacks.

    Audit R-1: when two lists report the same document with different
    metadata, the entry whose ``Date`` field is the most recent is
    retained rather than blindly preferring the first-seen copy.

    Args:
        ranked_lists: List of ranked result lists. Each result is a dict
            that must contain ``id_key`` for dedup.
        weights: Per-list raw weight multipliers (same length as
            ranked_lists). NOT normalized — see R-2 above.
        k: RRF smoothing constant (default 60). Higher values dampen the
            advantage of top-ranked documents.
        id_key: Dict key used to identify unique documents.

    Returns:
        Fused list sorted by descending RRF score. Each item is a copy of
        the freshest (by Date metadata) dict for that ID, with
        ``rrf_score`` and ``fusion`` fields injected.
    """
    if not ranked_lists:
        return []

    # Audit R-3: count non-empty source lists; if only one survived,
    # emit a metric so dashboards can flag degenerate fusion.
    non_empty = sum(1 for r in ranked_lists if r)
    if non_empty <= 1 and len(ranked_lists) > 1:
        try:
            from .observability import metrics as _metrics

            _metrics.inc("hybrid_single_list_degenerate")
        except Exception:  # nosec B110 — optional observability metric; import or inc failure is non-fatal
            pass

    scores: dict[str, float] = {}
    block_data: dict[str, dict] = {}
    # Per-arm 1-based ranks per fused id, e.g. {"bm25": 3, "vector": 1}.
    fusion_sources: dict[str, dict[str, int]] = {}
    # Per-arm RAW score per fused id, e.g. {"bm25": 8.31, "vector": 0.74}.
    # Captured HERE, inside the per-list loop, because the fused item below
    # is a copy of exactly ONE leg's dict -- the other leg's raw value is
    # destroyed by that choice and is unrecoverable afterwards. Recording it
    # per (id, arm) as the lists are walked is the only point at which both
    # legs' values are still in scope.
    leg_raw: dict[str, dict[str, float]] = {}

    for list_idx, results in enumerate(ranked_lists):
        w = weights[list_idx] if list_idx < len(weights) else 1.0
        arm = str(source_names[list_idx]) if source_names is not None and list_idx < len(source_names) else str(list_idx)
        for rank_0, item in enumerate(results):
            bid = _get_block_id(item, id_key)
            scores[bid] = scores.get(bid, 0.0) + w / (k + rank_0 + 1)
            fusion_sources.setdefault(bid, {})[arm] = rank_0 + 1
            try:
                leg_raw.setdefault(bid, {})[arm] = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                leg_raw.setdefault(bid, {})[arm] = 0.0
            existing = block_data.get(bid)
            if existing is None:
                block_data[bid] = item
                continue
            # Audit R-1: prefer the dict whose Date metadata is more
            # recent. Date comes from frontmatter and is typically
            # ISO-8601. We compare as strings (ISO ordering is correct
            # lexicographically) and fall back to first-seen on ties
            # or when either side is missing.
            new_date = item.get("Date") or item.get("date")
            old_date = existing.get("Date") or existing.get("date")
            if isinstance(new_date, str) and isinstance(old_date, str):
                if new_date > old_date:
                    block_data[bid] = item
            elif new_date and not old_date:
                block_data[bid] = item

    # Total-order tie-break (score, block_id): equal fused scores are common
    # (RRF sums are coarse w/(k+rank)), and without the block_id secondary key
    # ties fall back to dict-insertion = BM25-vs-vector arrival order =
    # non-reproducible recall. Matches the (score, _id) discipline used
    # throughout _recall_core / _recall_reranking, so the fused order is a pure
    # function of the input ranked-list multiset.
    sorted_ids = sorted(scores, key=lambda x: (scores[x], x), reverse=True)
    fused = []
    for bid in sorted_ids:
        item = block_data[bid].copy()
        item["rrf_score"] = round(scores[bid], 6)
        # ONE SCORE CONTRACT: ``score`` is the sort key at every stage exit.
        # Before this, fusion wrote ``rrf_score`` and left ``score`` holding
        # whichever leg's raw value happened to survive the dict copy above --
        # an unbounded BM25F number on some hits and a [0,1] cosine on others,
        # in the SAME column. Every consumer that reads ``score`` (the
        # cross-encoder's min-max normalisation, session_boost's re-sort,
        # dedup's chunk-winner, _explain) was reading a mixed-scale column.
        # The fusion stage sets the scale it produced, and the raw leg values
        # survive beside it in ``leg_scores`` rather than inside it.
        item["score"] = item["rrf_score"]
        item["fusion"] = "rrf"
        item["fusion_sources"] = dict(fusion_sources.get(bid, {}))
        item["leg_scores"] = dict(leg_raw.get(bid, {}))
        fused.append(item)

    return fused


def _get_block_id(item: dict, id_key: str) -> str:
    """Extract a stable block identifier from a result dict.

    Audit R-4: emit a ``rrf_fallback_id_used`` warning + metric when
    the file:line fallback path is taken, so silent ID collisions
    (two distinct blocks at the same file:line) show up in logs
    rather than producing wrong merge results.
    """
    bid = item.get(id_key)
    if bid:
        return str(bid)
    for alt in ("id", "block_id", "_id"):
        val = item.get(alt)
        if val:
            return str(val)
    fallback = f"{item.get('file', '?')}:{item.get('line', 0)}"
    try:
        from .observability import get_logger as _get_logger
        from .observability import metrics as _metrics

        _get_logger("mind_mem.hybrid_recall").warning(
            "rrf_fallback_id_used",
            fallback_id=fallback,
            advice="result dict lacks _id / id / block_id; collisions may merge distinct blocks",
        )
        _metrics.inc("rrf_fallback_id_used")
    except Exception:  # nosec B110 — best-effort warning + metric; fallback id is always returned regardless
        pass
    return fallback


# ---------------------------------------------------------------------------
# HybridBackend
# ---------------------------------------------------------------------------


class VectorLegError(RuntimeError):
    """The vector leg could not run (import/embed/store failure).

    Raised by :meth:`HybridBackend._vector_search` so its sole caller can
    tell a *failed* vector leg (degrade to BM25, mark it) apart from a
    vector leg that ran fine and simply matched nothing. ``reason`` is a
    short machine-readable tag surfaced in the ``degraded`` marker.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"vector leg unavailable: {reason}")
        self.reason = reason


class BM25LegError(RuntimeError):
    """The BM25 (lexical) arm was STRUCTURALLY unavailable.

    Raised by :meth:`HybridBackend._bm25_search` only under
    ``recall.strict_hybrid=true`` when the lexical index is empty/missing
    while the store has blocks — the exact structural failure that
    otherwise collapses hybrid fusion to the 1/(k+1) single-arm noise
    floor. ``reason`` is the machine-readable tag surfaced in the
    ``degraded`` marker (mirrors :class:`VectorLegError`).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"bm25 leg unavailable: {reason}")
        self.reason = reason


def _fts_row_count(db_path: str) -> int | None:
    """Row count of the recall.db ``blocks_fts`` table (read-only, no DDL).

    Returns ``0`` when the table exists but is empty, and ``None`` when the
    DB file is absent/unreadable or the FTS table/schema is missing — both
    of which count as a structurally-empty lexical index. Opens a
    short-lived ``mode=ro`` connection and never mutates the store.
    """
    import sqlite3

    if not os.path.isfile(db_path):
        return None
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute("SELECT count(*) FROM blocks_fts").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:  # nosec B110 — best-effort close of a read-only probe conn
                pass


#: A degradation marker. Values are NOT all strings: the inertness marker
#: carries the measured spread and floor (floats) and the sample size (int),
#: because a verdict shipped without its evidence is an assertion, not a
#: report. Widened from ``dict[str, str]`` when the vector-inertness gauge
#: landed.
LegMarker = dict[str, object]


def _merge_leg_markers(*markers: LegMarker | None) -> LegMarker | None:
    """Union >=1 ``{leg, reason}`` degradation markers into one (or ``None``).

    Legs and reasons are deduped, sorted, and comma-joined — the same shape
    :func:`_union_degraded` produces for the multi-query path — so a fused
    result can carry BOTH a degraded bm25 arm and a degraded vector arm at
    once (``"bm25,vector"`` / ``"index_empty,unavailable"``) without either
    masking the other. Returns ``None`` when no marker is present.
    """
    present = [m for m in markers if m]
    if not present:
        return None
    if len(present) == 1:
        # A single marker keeps its own extra keys. Some carry EVIDENCE, not
        # just a label -- the inertness marker ships the measured spread, the
        # floor it failed, and the sample size, so a caller can argue with the
        # verdict instead of taking "inert" on trust. Those keys are dropped
        # below only when several legs are merged, because a spread from one
        # leg attached to a union of two would be a lie about which leg it
        # described.
        return dict(present[0])
    legs = sorted({str(m.get("leg", "")).strip() for m in present if m.get("leg")})
    reasons = sorted({str(m.get("reason", "")).strip() for m in present if m.get("reason")})
    out: LegMarker = {}
    if legs:
        out["leg"] = ",".join(legs)
    if reasons:
        out["reason"] = ",".join(reasons)
    return out or None


class RecallResults(list):
    """A ranked result list that can carry a ``degraded`` marker.

    Backward-compatible on purpose: it *is* a ``list``, so every existing
    caller that iterates / slices / indexes the results is unaffected.
    Degradation-aware callers read ``.degraded`` — ``None`` when the full
    requested pipeline ran, otherwise ``{"leg": ..., "reason": ...}``. The
    whole point (silent degradation is the bug): a caller can finally tell
    that a "hybrid" recall actually served BM25-only because the vector leg
    was unavailable / timed out / failed.

    ``.trace`` carries the per-feature attribution summary
    (:meth:`~mind_mem.retrieval_trace.RetrievalTrace.summary`) for the run that
    produced this list — which of the conditional retrieval features actually
    fired, how long each took, and how many hits it added. ``None`` unless
    ``recall.retrieval.trace_attribution`` is on, so the default path carries
    exactly what it carried before.

    ``.attestation`` generalises ``.degraded``: it is the full per-run recall
    attestation (which legs ran, the effective config hash, the index anchor,
    plus the degraded marker folded in) when a caller has derived one. It is a
    runtime artifact only — populated on the response, never written to the
    store. See :mod:`mind_mem.recall_attestation`. Default ``None`` so callers
    that never derive one are unaffected.
    """

    degraded: LegMarker | None = None
    attestation: Any | None = None
    trace: dict[str, Any] | None = None


def _as_results(items: list[dict], degraded: LegMarker | None = None) -> RecallResults:
    rr = RecallResults(items)
    rr.degraded = degraded
    return rr


def _union_degraded(
    markers: list[LegMarker | None],
    total: int,
) -> LegMarker | None:
    """Combine per-variant degradation markers for a multi-query recall.

    Multi-query expansion / decomposition fans a recall out across query
    variants and RRF-fuses the per-variant sub-results. Each sub-result
    carries its own ``degraded`` marker (or ``None``). If ANY variant
    degraded — its vector leg was unavailable / timed out / failed — the
    fused result is not the full hybrid fusion the ``hybrid`` label
    implies, so the combined result must carry a ``degraded`` marker too;
    otherwise the single-query fix (419bee5) would be silently lost on the
    multi-query paths. Legs and reasons are unioned (deduped, sorted,
    comma-joined) and the degraded/total variant counts recorded so the
    degradation stays loud and machine-readable.

    Returns ``None`` when no variant degraded.
    """
    present = [m for m in markers if m]
    if not present:
        return None
    legs = sorted({str(m.get("leg", "vector")) for m in present})
    reasons = sorted({str(m.get("reason", "unknown")) for m in present})
    return {
        "leg": ",".join(legs),
        "reason": ",".join(reasons),
        "variants_degraded": str(len(present)),
        "variants_total": str(total),
    }


# ---------------------------------------------------------------------------
# Per-feature attribution (retrieval.trace_attribution, default OFF)
# ---------------------------------------------------------------------------


def _step(feature: str, **metadata: Any) -> AbstractContextManager[dict[str, Any]]:
    """:func:`~mind_mem.retrieval_trace.step` when a trace is open, else a no-op.

    ``step`` unconditionally reads the monotonic clock twice and emits a debug
    log line. With ``retrieval.trace_attribution`` off no trace is ever opened,
    and this collapses to one ContextVar read plus a :class:`nullcontext` — so
    the default path pays no timer and emits no extra log record. The yielded
    dict is fresh per call either way, so a caller's ``added_count`` /
    ``top_score_delta`` writes never leak between features.
    """
    if current_trace() is None:
        return nullcontext({"added_count": 0, "top_score_delta": 0.0})
    return _record_step(feature, **metadata)


def _top_score(items: list[dict]) -> float:
    """Score of the head hit, for a step's ``top_score_delta``.

    Reads whichever score field the stage in question ranks on
    (``score`` post-expansion, ``rrf_score`` straight out of fusion). Never
    raises: attribution is observation, and a malformed hit must not be able
    to fail a recall.
    """
    if not items:
        return 0.0
    head = items[0]
    if not isinstance(head, dict):  # pragma: no cover - defensive
        return 0.0
    for key in ("score", "rrf_score", "_score"):
        if key in head:
            try:
                return float(head[key] or 0.0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return 0.0
    return 0.0


class HybridBackend:
    """Orchestrates BM25 and vector search with RRF fusion.

    When vector search is unavailable (no sentence-transformers or
    ``vector_enabled`` is False), transparently falls back to BM25-only.

    Supports optional multi-query expansion: when ``query_expansion.enabled``
    is True in config, generates alternative query phrasings and fuses
    results across all variants for improved recall.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}

        # Validate numeric fields; fall back to defaults on bad values
        errors = validate_recall_config(cfg)
        if errors:
            _log.warning(
                "hybrid_config_validation_failed",
                errors=errors,
                fallback="bm25_only",
            )
            # Reset bad values to defaults so constructor doesn't raise
            for key in _NUMERIC_POSITIVE_KEYS:
                if any(key in e for e in errors):
                    cfg.pop(key, None)

        self.rrf_k: int = int(cfg.get("rrf_k", 60))
        self.bm25_weight: float = float(cfg.get("bm25_weight", 1.0))
        self.vector_weight: float = float(cfg.get("vector_weight", 1.0))
        self.vector_enabled: bool = bool(cfg.get("vector_enabled", False))
        self.vector_model: str = cfg.get("vector_model", "all-MiniLM-L6-v2")
        self._config = cfg
        self._config_errors: list[str] = errors

        # Strict-hybrid knob (default false). When true, a structurally-empty
        # arm (an unpopulated/missing BM25 index while the store has blocks)
        # RAISES (BM25LegError) instead of silently degrading to the 1/(k+1)
        # single-arm RRF noise floor.
        self._strict_hybrid: bool = bool(cfg.get("strict_hybrid", False))

        # Auto-build the FTS5 index on first recall when it is absent
        # (``recall.auto_build_index``, default FALSE). This completes a
        # promise ``docs/performance-tuning.md`` has always made and nothing
        # ever kept — but building the index moves a default workspace from
        # the BM25F scan to FTS5 bm25(), and those RANK DIFFERENTLY, so it
        # stays opt-in until the paired scorecard shows parity. See
        # ``sqlite_index.ensure_index``.
        self._auto_build_index: bool = bool(cfg.get("auto_build_index", False))

        # Query expansion config (opt-in: adds ~3x query latency when enabled)
        qe_cfg = cfg.get("query_expansion", {})
        if not isinstance(qe_cfg, dict):
            qe_cfg = {}
        self._query_expansion_enabled: bool = bool(qe_cfg.get("enabled", False))
        self._query_expansion_config: dict[str, Any] = qe_cfg

        # Probe vector availability once at init. When the operator
        # explicitly enabled vector recall (recall.vector_enabled=true) but
        # the backend is not importable / not serviceable, FAIL LOUD
        # (warning + metric) instead of silently degrading to BM25 — a
        # misconfigured or missing embedder must be visible, never invisible
        # (#139: silent BM25-only fallback).
        if self.vector_enabled:
            self._vector_available = self._check_vector()
            if not self._vector_available:
                _log.warning(
                    "hybrid_vector_requested_but_unavailable",
                    hint=(
                        "recall.vector_enabled=true but the vector backend is not "
                        "importable; recall is degraded to BM25-only. Install the "
                        "vector extras and verify the embedder is reachable."
                    ),
                    fallback="bm25_only",
                )
                metrics.inc("hybrid_vector_requested_but_unavailable")
        else:
            self._vector_available = False

        _log.info(
            "hybrid_backend_init",
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
            vector_available=self._vector_available,
            query_expansion=self._query_expansion_enabled,
            strict_hybrid=self._strict_hybrid,
        )

    # -- capability probing ------------------------------------------------

    def _detect_query_type(self, query: str) -> str | None:
        """Classify *query*, or ``None`` when the detector is unavailable.

        One home for the try/except so ``_search``'s per-request memo and
        the reranker predicate cannot drift into two different answers for
        the same string.
        """
        try:
            from ._recall_detection import detect_query_type

            return detect_query_type(query)
        except Exception as exc:  # pragma: no cover — defensive
            _log.debug("query_type_detect_skipped", error=str(exc))
            return None

    def _check_vector(self) -> bool:
        """Return True if recall_vector + sentence-transformers are importable."""
        try:
            from . import recall_vector  # noqa: F401

            return True
        except ImportError:
            _log.info("vector_backend_unavailable", reason="import failed")
            return False

    @property
    def vector_available(self) -> bool:
        return self._vector_available

    def _vector_deadline_seconds(self) -> float:
        """Wall-clock bound (seconds) on the parallel vector leg.

        Guarantees recall degrades to BM25-only fusion rather than
        blocking when the embedder or vector store stalls. Defaults to a
        margin above the per-request embed timeout; override via
        ``recall.vector_deadline_seconds``. Clamped to a sane range.
        """
        default = 14.0
        try:
            val = float(self._config.get("vector_deadline_seconds", default))
        except (TypeError, ValueError):
            return default
        if val <= 0:
            return default
        return max(1.0, min(val, 120.0))

    # -- search entry point -------------------------------------------------

    def search(
        self,
        query: str,
        workspace: str,
        limit: int = 10,
        active_only: bool = False,
        graph_boost: bool = False,
        retrieve_wide_k: int = 200,
        rerank: bool = True,
        rerank_depth: int | None = None,
        _skip_auto_features: bool = False,
        scoring_instant: date | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Public recall entry point; see :meth:`_search` for the pipeline.

        This wrapper does exactly one thing on top of :meth:`_search`: when
        ``recall.retrieval.trace_attribution`` is on it opens a
        :func:`~mind_mem.retrieval_trace.trace` for the call, so every
        conditional feature that fires downstream records what it contributed,
        and hangs the resulting summary off the returned
        :class:`RecallResults` as ``.trace``.

        With the flag off — the default — the trace is never opened, no
        feature records a step (see :func:`_step`), and this returns the exact
        object :meth:`_search` returned. A trace already open on this context
        is NOT nested over: the multi-query paths recurse back through
        ``search`` per variant, and the steps those variants record belong to
        the one outer trace the caller opened.

        The trace is observation only. Nothing it measures is read back by any
        scoring, ranking or fusion step, so turning it on cannot move a
        result. It carries wall-clock latencies, which is precisely why it is
        opt-in: recall's ranked output stays a pure function of (corpus,
        config, scoring_instant), while the attribution envelope beside it
        does not.
        """
        if current_trace() is not None or not is_trace_enabled(self._config):
            return self._search(
                query,
                workspace,
                limit=limit,
                active_only=active_only,
                graph_boost=graph_boost,
                retrieve_wide_k=retrieve_wide_k,
                rerank=rerank,
                rerank_depth=rerank_depth,
                _skip_auto_features=_skip_auto_features,
                scoring_instant=scoring_instant,
                **kwargs,
            )
        with _open_trace(query) as tracer:
            out = self._search(
                query,
                workspace,
                limit=limit,
                active_only=active_only,
                graph_boost=graph_boost,
                retrieve_wide_k=retrieve_wide_k,
                rerank=rerank,
                rerank_depth=rerank_depth,
                _skip_auto_features=_skip_auto_features,
                scoring_instant=scoring_instant,
                **kwargs,
            )
            if not isinstance(out, RecallResults):
                # The empty-query short-circuit returns a plain list.
                out = _as_results(list(out))
            out.trace = tracer.summary()
            return out

    def _search(
        self,
        query: str,
        workspace: str,
        limit: int = 10,
        active_only: bool = False,
        graph_boost: bool = False,
        retrieve_wide_k: int = 200,
        rerank: bool = True,
        rerank_depth: int | None = None,
        _skip_auto_features: bool = False,
        scoring_instant: date | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Run BM25 and (optionally) vector search, fuse via RRF.

        When vector search is unavailable the method returns BM25
        results directly (no fusion overhead).

        Args:
            query: Search query string.
            workspace: Workspace root path.
            limit: Maximum results to return.
            active_only: Only return active blocks.
            graph_boost: Enable cross-reference graph boosting (BM25).
            retrieve_wide_k: Candidate pool size per backend.
            rerank: Enable BM25 reranker (passed through).
            scoring_instant: UTC date every recency term scores against —
                the BM25 leg's ramp and calibration window, the vector leg's
                recency boost, and the half-life decay. ``None`` resolves to
                today in UTC. The fusion itself (RRF) reads no clock; see
                :mod:`mind_mem.scoring_instant`.
            **kwargs: Forwarded to underlying backends.

        Returns:
            Ranked list of result dicts.
        """
        if not query or not query.strip():
            return []

        # Audit R-6: detect_query_type is called from the expansion
        # path, the decomposition path, and the cross-encoder path on
        # every search() invocation. The detector itself is regex-only
        # and cheap, but the import + dispatch shows up in profiles
        # when called 3× per request. Memoize per request so the same
        # query string is classified once.
        _qt_cache: dict[str, str] = {}

        def _qt() -> str | None:
            if query in _qt_cache:
                return _qt_cache[query]
            value = self._detect_query_type(query)
            if value is None:
                return None
            _qt_cache[query] = value
            return value

        # --- Multi-query expansion ---
        # When enabled, expand the query into alternative phrasings and
        # run a search for each variant.  Results are fused via RRF so
        # documents matching multiple phrasings rank higher.
        # v3.3.0 Tier 2 #4: auto-enable on multi-hop/temporal queries
        # even when operator hasn't flipped ``query_expansion.enabled``,
        # unless ``query_expansion.auto_enable: false`` is set.
        # Thread-safety: _search_expanded recurses into search() with
        # _skip_auto_features=True to avoid re-entering expansion /
        # decomposition loops. Previous version mutated
        # ``self._query_expansion_enabled`` which races between
        # concurrent requests (python-reviewer 2026-04-20).
        if _skip_auto_features:
            expansion_active = False
        else:
            expansion_active = self._query_expansion_enabled
        if not expansion_active and self._query_expansion_config.get("auto_enable", True):
            qt = _qt()
            if qt in ("multi-hop", "temporal"):
                expansion_active = True
                _log.info(
                    "query_expansion_auto_enabled",
                    query_type=qt,
                    reason="v3.3.0_tier2_ambiguous_query",
                )
        if expansion_active:
            try:
                from .query_expansion import expand_queries

                expanded = expand_queries(
                    query,
                    config=self._query_expansion_config,
                )
                if len(expanded) > 1:
                    _log.info(
                        "multi_query_expansion",
                        original=query,
                        variants=len(expanded),
                    )
                    metrics.inc("query_expansion_used")
                    return self._search_expanded(
                        queries=expanded,
                        workspace=workspace,
                        limit=limit,
                        active_only=active_only,
                        graph_boost=graph_boost,
                        retrieve_wide_k=retrieve_wide_k,
                        rerank=rerank,
                        **kwargs,
                    )
            except Exception as exc:
                _log.warning(
                    "query_expansion_failed",
                    error=str(exc),
                    fallback="single_query",
                )

        # v3.3.0 Tier 1 #1 — query decomposition for multi-hop queries.
        # Split compound questions ("A after B") into independent
        # sub-queries, run retrieval on each, RRF-fuse. Same opt-out
        # shape as expansion: ``retrieval.query_decomposition.auto_enable
        # = false`` to skip.
        decomp_cfg = self._config.get("retrieval", {}).get("query_decomposition", {})
        if not isinstance(decomp_cfg, dict):
            decomp_cfg = {}
        decomp_active = False if _skip_auto_features else bool(decomp_cfg.get("enabled", False))
        if not decomp_active and decomp_cfg.get("auto_enable", True):
            if _qt() == "multi-hop":
                decomp_active = True
                _log.info(
                    "query_decomposition_auto_enabled",
                    reason="v3.3.0_tier1_multi_hop",
                )
        if decomp_active:
            try:
                from .query_planner import decompose_query

                decomposed = decompose_query(
                    query,
                    config=self._config,
                    max_subqueries=int(decomp_cfg.get("max_subqueries", 4)),
                )
                if len(decomposed) > 1:
                    _log.info(
                        "multi_query_decomposition",
                        original=query,
                        sub_queries=len(decomposed),
                    )
                    metrics.inc("query_decomposition_used")
                    return self._search_expanded(
                        queries=decomposed,
                        workspace=workspace,
                        limit=limit,
                        active_only=active_only,
                        graph_boost=graph_boost,
                        retrieve_wide_k=retrieve_wide_k,
                        rerank=rerank,
                        **kwargs,
                    )
            except Exception as exc:
                _log.warning(
                    "query_decomposition_failed",
                    error=str(exc),
                    fallback="single_query",
                )

        # Postgres workspaces fuse BM25 + pgvector SERVER-SIDE: the local
        # "BM25" leg here (recall -> PostgresRecallBackend.search) is itself
        # the store's ``hybrid_search`` (BM25 + pgvector RRF, labeled
        # ``hybrid_pgvector`` / ``bm25_fallback``). Running HybridBackend's
        # OWN second local vector leg on top would (a) double-count the
        # vector contribution and (b) drive the provider=postgres path in
        # ``search_batch`` (audit 1a). So for postgres we take the single-leg
        # local path — which is already server-side hybrid — and let the
        # cross-encoder rerank below still apply, instead of fusing twice.
        pg_server_side = isinstance(self._config, dict) and self._config.get("provider") == "postgres"

        # How deep the reranker may look, and whether one will run at all.
        # Both are resolved HERE, before any leg is dispatched, because the
        # legs decide how many candidates to fetch and a depth the legs never
        # filled is a depth that does not exist. Widening is conditional on a
        # reranker actually running: fetching more candidates changes which
        # documents RRF sees, so doing it unconditionally would move the
        # fused order of requests that never asked for a cross-encoder.
        _depth = rerank_depth if rerank_depth is not None else resolve_rerank_depth(self._config, limit)
        _ce_active = self._cross_encoder_active(_qt())
        _leg_k = max(retrieve_wide_k, _depth) if _ce_active else retrieve_wide_k

        with timed("hybrid_search"):
            if not self._vector_available or pg_server_side:
                _log.info("hybrid_bm25_only", query=query, pg_server_side=pg_server_side)
                # ``max(limit, _depth)`` only when a reranker will run — see
                # the comment above ``_leg_k``. With no reranker this is
                # ``limit``, the value it has always been.
                results = self._bm25_search(
                    query,
                    workspace,
                    limit=max(limit, _depth) if _ce_active else limit,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=_leg_k,
                    rerank=rerank,
                    scoring_instant=scoring_instant,
                    **kwargs,
                )
                # Capture the bm25 leg's structural-degradation marker BEFORE
                # rerank (which returns a plain list and drops ``.degraded``).
                bm25_degraded: LegMarker | None = getattr(results, "degraded", None)
                metrics.inc("hybrid_searches_bm25_only")
                results = self._admit(results, workspace, leg=Leg.BM25, overrides=live_statuses(workspace))
                # v3.3.0 Tier 2: cross-encoder rerank also applies to
                # BM25-only deployments (previously only post-fusion).
                results = self._maybe_cross_encoder_rerank(
                    query,
                    results,
                    limit,
                    rerank_depth=_depth,
                    ce_active=_ce_active,
                )
                # Mark degradation ONLY when the operator asked for vector
                # recall but the backend was unavailable at init — a plain
                # BM25 config (vector never requested) is not "degraded",
                # and a postgres server-side hybrid is its own leg.
                vector_degraded: LegMarker | None = None
                if self.vector_enabled and not self._vector_available and not pg_server_side:
                    vector_degraded = {"leg": "vector", "reason": "unavailable"}
                return _as_results(results, _merge_leg_markers(bm25_degraded, vector_degraded))

            # Run BM25 + vector in parallel
            _log.info("hybrid_parallel_search", query=query)
            bm25_results: list[dict] = []
            vec_results: list[dict] = []

            # Manual pool lifecycle — NOT ``with ThreadPoolExecutor(...) as
            # pool``. The context manager's __exit__ calls shutdown(wait=True),
            # which re-joins a still-running vector worker even after the
            # deadline below fired — so ``timeout=`` bounded the RESULT wait
            # but NOT the wall-clock of an unbounded leg (e.g. a provider=local
            # sentence-transformers embed that hangs on model download). We
            # therefore shut the pool down with wait=False + cancel_futures so
            # recall returns at the deadline and abandons the leaked worker
            # instead of blocking on it (audit finding 4). cancel_futures drops
            # any not-yet-started task; an already-running embed thread cannot
            # be force-killed, but it no longer holds up the response.
            # Records why the vector leg did not contribute, if it didn't.
            # None => the full two-leg fusion ran as requested. No type
            # annotation here: both names are already bound (unannotated) in
            # the mutually-exclusive BM25-only branch above, so an annotated
            # re-declaration trips mypy [no-redef]; the union type is inferred
            # from the branch assignments.
            vector_degraded = None
            bm25_degraded = None
            pool = ThreadPoolExecutor(max_workers=2)
            try:
                bm25_future: Future = pool.submit(
                    self._bm25_search,
                    query,
                    workspace,
                    limit=_leg_k,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=_leg_k,
                    rerank=False,  # defer reranking to post-fusion
                    scoring_instant=scoring_instant,
                    **kwargs,
                )
                vec_future: Future = pool.submit(
                    self._vector_search,
                    query,
                    workspace,
                    limit=_leg_k,
                    active_only=active_only,
                    scoring_instant=scoring_instant,
                )
                bm25_results = bm25_future.result()
                # A structurally-empty BM25 arm (index unbuilt while the store
                # has blocks) is now marked LOUD by _bm25_search — fold it into
                # the fused result's degraded marker so a single healthy leg
                # (the 1/(k+1) noise floor) can never masquerade as "hybrid".
                bm25_degraded = getattr(bm25_results, "degraded", None)
                # Hard bound on the vector leg: if the embedder is cold /
                # slow / down, degrade to BM25-only fusion instead of
                # blocking the whole recall request (the vector work
                # includes an embedding HTTP round-trip, itself bounded).
                try:
                    vec_results = vec_future.result(timeout=self._vector_deadline_seconds())
                except _FutureTimeout:
                    _log.warning(
                        "hybrid_vector_leg_timeout",
                        deadline=self._vector_deadline_seconds(),
                        fallback="bm25_only",
                    )
                    vec_future.cancel()
                    vec_results = []
                    vector_degraded = {"leg": "vector", "reason": "deadline_exceeded"}
                except VectorLegError as exc:
                    # Embed/store/import failure inside the vector leg:
                    # degrade to BM25-only fusion and MARK it so a caller
                    # can tell (previously this was swallowed to [] and the
                    # "hybrid" label silently lied).
                    _log.warning(
                        "hybrid_vector_leg_failed",
                        reason=exc.reason,
                        fallback="bm25_only",
                    )
                    vec_results = []
                    vector_degraded = {"leg": "vector", "reason": exc.reason}
            finally:
                # wait=False so a hung vector leg cannot re-block the response
                # here (the whole point of the deadline above).
                pool.shutdown(wait=False, cancel_futures=True)

            # Admissibility runs HERE, before fusion, and not once after it.
            # RRF scores an item at ``sum_leg 1/(k + rank_leg(i))``; drop a
            # withheld item after fusion and every admitted item below it
            # carries a worse rank than it should, so the presence of
            # withheld content stays observable through its neighbours'
            # ranks. Filtering each leg's candidates closes that channel.
            # Resolved once and shared by both legs: the staleness check
            # opens the index, so paying it per leg would double it.
            _live = live_statuses(workspace)
            bm25_results = self._admit(bm25_results, workspace, leg=Leg.BM25, overrides=_live)
            vec_results = self._admit(vec_results, workspace, leg=Leg.VECTOR, overrides=_live)

            _log.info(
                "hybrid_results_pre_fusion",
                bm25_count=len(bm25_results),
                vector_count=len(vec_results),
            )

            # Honesty gauge: a vector leg whose blocks are mutually
            # indistinguishable contributes a ranking that is noise, and RRF
            # would blend that noise into BM25's real signal while the answer
            # kept the "hybrid" label. Measured: a near-duplicate corpus sits
            # at an inter-block cosine spread of 0.002 and ranks at chance,
            # against 0.108 for prose. Drop the leg rather than fake the
            # fusion -- and SAY so, both in the degraded marker and in
            # retrieval_diagnostics.
            #
            # Only overrides when the gauge is confident; it abstains on a
            # sample too small to judge, because silently disabling a working
            # leg is the worse error.
            vector_weight = self.vector_weight
            if vec_results and vector_degraded is None:
                _inert = vector_inertness.inertness_for(workspace)
                if _inert.inert:
                    vector_weight = 0.0
                    vector_degraded = {
                        "leg": "vector",
                        "reason": "inert",
                        "spread": _inert.spread,
                        "floor": _inert.floor,
                        "sampled": _inert.sampled,
                        "detail": _inert.reason,
                    }

            fused = rrf_fuse(
                ranked_lists=[bm25_results, vec_results],
                weights=[self.bm25_weight, vector_weight],
                k=self.rrf_k,
                source_names=["bm25", "vector"],
            )

            metrics.inc("hybrid_searches_fused")

            # Cross-encoder reranking (post-fusion) — v3.3.0 Tier 2
            # extracted into a helper so the BM25-only early-return path
            # also benefits from auto-enable on multi-hop/temporal queries.
            #
            # The FULL fused pool goes in, not ``fused[:limit]``. Slicing
            # first handed the reranker the same blocks that were already
            # going to be served, so it could only permute them — the model's
            # latency bought reordering and exactly zero recall. The helper
            # takes ``rerank_depth`` candidates and returns at most ``limit``,
            # so with no reranker running this is still ``fused[:limit]``.
            result = self._maybe_cross_encoder_rerank(
                query,
                fused,
                limit,
                rerank_depth=_depth,
                ce_active=_ce_active,
            )

            # v3.3.0 Tier 1 #2 + Tier 3 #8 — multi-hop graph expansion +
            # entity prefetch. Corpus is loaded once and shared across
            # both helpers (was O(2N) disk reads, now O(N); architect +
            # python-reviewer 2026-04-20).
            corpus = self._load_corpus_if_needed(query, workspace)
            result = self._maybe_graph_expand(query, workspace, result, corpus=corpus)
            result = self._maybe_kg_expand(query, workspace, result, corpus=corpus)
            result = self._maybe_entity_prefetch(query, workspace, result, corpus=corpus)

            # v3.3.0 Tier 2 #5 — session-boundary preservation.
            result = self._maybe_session_boost(result)

            # v3.3.0 — temporal half-life decay (opt-in hot-path).
            result = self._maybe_temporal_decay(result, scoring_instant=scoring_instant)

            # v3.3.0 — probabilistic truth_score annotation.
            result = self._maybe_truth_score(result)

            # Per-actor trust scores (opt-in, default OFF). Runs AFTER
            # truth_score so it can reuse that annotation instead of
            # recomputing it.
            result = self._maybe_trust_scores(result, workspace, scoring_instant=scoring_instant)

            # Enforce the caller's limit AFTER expansions — previous code
            # truncated before the graph/entity expansions appended
            # blocks, so the final list could exceed ``limit``. Dedup
            # runs next, then we slice to the requested size.
            # (python-reviewer 2026-04-20)

            # 4-layer dedup filter (post-fusion, post-rerank)
            dedup_cfg = self._config.get("dedup")
            if dedup_cfg is None or (isinstance(dedup_cfg, dict) and dedup_cfg.get("enabled", True)):
                try:
                    from .dedup import DedupConfig, deduplicate_results

                    dc = DedupConfig(dedup_cfg if isinstance(dedup_cfg, dict) else None)
                    result = deduplicate_results(result, config=dc)
                except Exception as e:
                    _log.warning("hybrid_dedup_failed", error=str(e))

            # Final slice so callers never receive more than they asked for.
            result = result[:limit]

            _log.info(
                "hybrid_search_complete",
                query=query,
                results=len(result),
                top_rrf=result[0].get("rrf_score", 0) if result else 0,
                degraded=bool(vector_degraded or bm25_degraded),
            )
            return _as_results(result, _merge_leg_markers(bm25_degraded, vector_degraded))

    # -- multi-query expansion search ----------------------------------------

    def _search_expanded(
        self,
        queries: list[str],
        workspace: str,
        limit: int = 10,
        active_only: bool = False,
        graph_boost: bool = False,
        retrieve_wide_k: int = 200,
        rerank: bool = True,
        **kwargs: Any,
    ) -> list[dict]:
        """Search with multiple query variants and fuse results via RRF.

        Each query variant is searched independently using the standard
        single-query pipeline.  Results from all variants are then fused
        using RRF with equal weights, ensuring documents that match
        multiple phrasings rank higher.

        Args:
            queries: List of query variant strings (original + expansions).
            workspace: Workspace root path.
            limit: Maximum results to return.
            active_only: Only return active blocks.
            graph_boost: Enable cross-reference graph boosting.
            retrieve_wide_k: Candidate pool size per backend per query.
            rerank: Enable BM25 reranker.
            **kwargs: Forwarded to underlying search.

        Returns:
            RRF-fused :class:`RecallResults`. Carries a ``degraded``
            marker (the union of the per-variant markers) whenever any
            variant's sub-result was degraded, so a caller can tell the
            fused ``hybrid`` result actually served BM25-only for one or
            more variants.
        """
        # Pass _skip_auto_features=True so the recursion into search()
        # doesn't re-trigger expansion/decomposition. Previous code
        # mutated self._query_expansion_enabled which raced under
        # concurrent calls (python-reviewer 2026-04-20 → commit
        # b31e862 follow-up).
        #
        # Audit R-5: each variant is an independent BM25 + vector
        # search. Sequential dispatch dominates wall-clock latency
        # when expansion is enabled (default 3 variants → 3× latency).
        # Use a ThreadPoolExecutor to fan out — the SQLite backend is
        # WAL-mode and safe for concurrent reads, and the vector
        # backend is read-only at query time. Single-query callers
        # take the in-line path to avoid pool-spinup overhead.
        per_query_results: list[list[dict]] = []
        if len(queries) <= 1:
            for q in queries:
                per_query_results.append(
                    self.search(
                        q,
                        workspace,
                        limit=retrieve_wide_k,
                        active_only=active_only,
                        graph_boost=graph_boost,
                        retrieve_wide_k=retrieve_wide_k,
                        rerank=rerank,
                        _skip_auto_features=True,
                        **kwargs,
                    )
                )
        else:
            from concurrent.futures import ThreadPoolExecutor

            # deferred: an open attribution trace does NOT reach these workers —
            # a ThreadPoolExecutor task runs in a fresh context, so each variant
            # opens (and discards) its own trace and the outer summary lists no
            # steps for the multi-variant expansion path. The in-line branch
            # above traces correctly. Upgrade path: hand the active
            # RetrievalTrace to the worker explicitly (a re-entrant activation
            # helper on retrieval_trace) rather than copying the context, since
            # one Context cannot be entered by two threads at once.
            def _one(q: str) -> list[dict]:
                return self.search(
                    q,
                    workspace,
                    limit=retrieve_wide_k,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=retrieve_wide_k,
                    rerank=rerank,
                    _skip_auto_features=True,
                    **kwargs,
                )

            max_workers = min(len(queries), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                per_query_results = list(ex.map(_one, queries))

        if not per_query_results:
            return _as_results([])

        # Aggregate per-variant degradation BEFORE fusion strips the
        # ``.degraded`` markers: each sub-result is a RecallResults from
        # the single-query path, so a vector leg that was unavailable /
        # timed out / failed for any variant is recorded on that variant.
        # Union them so the fused result carries the degradation too —
        # without this, the single-query degraded marker (419bee5) would
        # be silently dropped on the expansion / decomposition paths.
        variant_markers = [getattr(r, "degraded", None) for r in per_query_results]
        combined_degraded: LegMarker | None = _union_degraded(variant_markers, total=len(per_query_results))

        # Fuse all query variant results with equal weights
        weights = [1.0] * len(per_query_results)
        fused = rrf_fuse(
            ranked_lists=per_query_results,
            weights=weights,
            k=self.rrf_k,
        )

        if combined_degraded is not None:
            _log.warning(
                "multi_query_recall_degraded",
                query_variants=len(queries),
                variants_degraded=combined_degraded["variants_degraded"],
                leg=combined_degraded["leg"],
                reason=combined_degraded["reason"],
                fallback="bm25_only",
            )

        _log.info(
            "multi_query_fusion_complete",
            query_variants=len(queries),
            total_fused=len(fused),
            limit=limit,
            degraded=bool(combined_degraded),
        )

        return _as_results(fused[:limit], combined_degraded)

    def _maybe_session_boost(self, results: list[dict]) -> list[dict]:
        """Apply session-boundary preservation (v3.3.0 Tier 2 #5)."""
        if not results:
            return results
        try:
            from .session_boost import (
                apply_session_boost,
                is_session_boost_enabled,
                resolve_session_boost_config,
            )

            if not is_session_boost_enabled(self._config, results):
                return results
            params = resolve_session_boost_config(self._config)
            with _step("session_boost") as rec:
                boosted = apply_session_boost(results, **params)
                rec["added_count"] = len(boosted) - len(results)
                rec["top_score_delta"] = _top_score(boosted) - _top_score(results)
            return boosted
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("session_boost_failed", error=str(exc))
            return results

    def _maybe_truth_score(self, results: list[dict]) -> list[dict]:
        """Annotate results with probabilistic truth_score (v3.3.0)."""
        if not results:
            return results
        try:
            from .truth_score import annotate_results, is_truth_score_enabled

            if not is_truth_score_enabled(self._config):
                return results
            # Contradiction graph is passed through when the caller
            # supplies one via config; otherwise just Status/age/access
            # signals feed the score.
            with _step("truth_score") as rec:
                annotated = annotate_results(results)
                rec["added_count"] = len(annotated) - len(results)
                rec["top_score_delta"] = _top_score(annotated) - _top_score(results)
            return annotated
        except Exception as exc:  # pragma: no cover
            _log.warning("truth_score_failed", error=str(exc))
            return results

    def _maybe_trust_scores(
        self,
        results: list[dict],
        workspace: str | None = None,
        *,
        scoring_instant: date | None = None,
    ) -> list[dict]:
        """Annotate hits with the gate's provenance class; re-rank when opted in.

        Gated on ``retrieval.trust_scores.enabled`` (default false). With
        the gate off this returns the *same list object* it was given —
        no added fields, no reordering, byte-identical output.

        deferred: only the fused hybrid path is wired; the BM25-only
        early-return path (same as ``_maybe_truth_score``) is not —
        upgrade path: call this helper there too once the BM25-only
        branch also carries provenance-annotated hits.
        """
        if not results:
            return results
        try:
            from .trust_scores import apply_trust_scores, is_trust_scores_enabled

            if not is_trust_scores_enabled(self._config):
                return results
            with _step("trust_scores") as rec:
                scored = apply_trust_scores(
                    results,
                    config=self._config,
                    workspace=workspace,
                    scoring_instant=scoring_instant,
                )
                rec["added_count"] = len(scored) - len(results)
                rec["top_score_delta"] = _top_score(scored) - _top_score(results)
            return scored
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("trust_scores_failed", error=str(exc))
            return results

    def _maybe_temporal_decay(self, results: list[dict], *, scoring_instant: date | None = None) -> list[dict]:
        """Apply half-life decay to every result's score (v3.3.0 Tier 1 #3).

        Opt-in via ``retrieval.temporal_decay_hot_path`` — the raw
        function is always available (Tier 1 #3) but the hot-path
        wiring is gated because it changes ranking. With the gate
        off, the function stays a standalone helper callers invoke
        explicitly.
        """
        if not results:
            return results
        cfg = self._config.get("retrieval", {}) if isinstance(self._config, dict) else {}
        if not isinstance(cfg, dict) or not cfg.get("temporal_decay_hot_path", False):
            return results
        try:
            from ._recall_scoring import _resolve_half_life_days, temporal_decay_score

            half_life = _resolve_half_life_days(self._config)
            _decay_moment = as_utc_datetime(resolve_scoring_instant(scoring_instant))
            # Audit R-10: copy-on-write so we don't mutate dicts the
            # caller still holds a reference to. Two upstream paths
            # (cross-encoder rerank, session boost) reuse the input
            # list, and in-place score mutation corrupted their views
            # when temporal_decay_hot_path was enabled mid-request.
            with _step("temporal_decay", half_life_days=half_life) as rec:
                decayed: list[dict] = []
                for r in results:
                    mult = temporal_decay_score(r, half_life_days=half_life, now=_decay_moment)
                    current = float(r.get("score", 0.0) or 0.0)
                    copy = dict(r)
                    copy["score"] = current * mult
                    copy["_temporal_decay"] = round(mult, 4)
                    decayed.append(copy)
                decayed.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
                rec["added_count"] = len(decayed) - len(results)
                rec["top_score_delta"] = _top_score(decayed) - _top_score(results)
            _log.info("temporal_decay_applied", count=len(decayed), half_life_days=half_life)
            return decayed
        except Exception as exc:  # pragma: no cover
            _log.warning("temporal_decay_failed", error=str(exc))
        return results

    def _admit(
        self,
        hits: list[dict],
        workspace: str,
        *,
        leg: Leg,
        overrides: Mapping[str, str] | None = None,
    ) -> list[dict]:
        """Withhold everything recall may not serve from one leg's candidates.

        Returns the input object untouched when every candidate is
        admissible — the common case — so an unwithheld workspace pays
        nothing, not even the release lookup.

        *overrides* carries the live statuses resolved once per request
        (:func:`~mind_mem.admissibility.live_statuses`) and is applied
        BEFORE the all-admissible fast path, not after: an index-cached
        ``active`` that the corpus has since flipped to ``quarantined``
        would otherwise take the fast path and be served. Empty whenever
        the index is current, which is what keeps that ordering free.
        """
        if overrides is None:
            overrides = live_statuses(workspace)
        hits = with_live_statuses(hits, overrides)
        if all(is_admissible_status(hit.get("status")) for hit in hits):
            return hits
        try:
            releases = workspace_release_ids(workspace)
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("release_lookup_failed", error=str(exc))
            releases = frozenset()
        return admit_leg(hits, status_key="status", releases=releases, leg=leg.value)

    def _load_corpus_if_needed(self, query: str, workspace: str) -> list[dict] | None:
        """Return the workspace block corpus — shared by graph + entity
        helpers so we load once per request rather than twice.

        Returns None when no feature needs the corpus — avoids paying
        the disk cost at all when both auto-enables are off.
        """
        try:
            from .entity_prefetch import is_entity_prefetch_enabled
            from .graph_recall import is_graph_expand_enabled
            from .kg_fusion import is_kg_fusion_enabled
        except ImportError:  # pragma: no cover
            return None
        if not (
            is_graph_expand_enabled(self._config, query) or is_entity_prefetch_enabled(self._config) or is_kg_fusion_enabled(self._config)
        ):
            return None
        try:
            from .block_parser import parse_file
            from .block_store import MarkdownBlockStore

            store = MarkdownBlockStore(workspace)
            blocks: list[dict] = []
            for path in store.list_blocks():
                try:
                    blocks.extend(parse_file(path))
                except Exception as exc:  # pragma: no cover
                    _log.debug("corpus_block_parse_skipped", error=str(exc))
                    continue
            # The graph / KG / prefetch legs resolve neighbour ids straight
            # out of this list, so filtering it here is what makes a withheld
            # block unresolvable to all three at once. Each of them filters
            # again on its own corpus argument — this is not redundancy, it
            # is the same rule enforced at the point of use so a caller that
            # loads its own corpus cannot bypass it.
            return admit_corpus(blocks)
        except Exception as exc:  # pragma: no cover
            _log.warning("corpus_load_failed", error=str(exc))
            return None

    def _maybe_entity_prefetch(
        self,
        query: str,
        workspace: str,
        results: list[dict],
        *,
        corpus: list[dict] | None = None,
    ) -> list[dict]:
        """Inject entity-graph prefetched blocks (v3.3.0 Tier 3 #8).

        When the query mentions a Person/Project/Tool/Incident, fetch
        the entity block + 1-hop neighbourhood and merge into the
        result set. ``corpus`` — when passed — skips a workspace reload
        (shared with graph_expand). Fails open on any error.
        """
        try:
            from .entity_prefetch import (
                is_entity_prefetch_enabled,
                prefetch_entity_blocks,
                resolve_entity_prefetch_config,
            )

            if not is_entity_prefetch_enabled(self._config):
                return results
            params = resolve_entity_prefetch_config(self._config)
            with _step("entity_prefetch", max_hops=params["max_hops"]) as rec:
                prefetched = prefetch_entity_blocks(
                    query,
                    workspace,
                    max_entities=params["max_entities"],
                    max_hops=params["max_hops"],
                    entity_score=params["entity_score"],
                    corpus=corpus,
                )
                if not prefetched:
                    return results
                # Merge: keep original order, append prefetched blocks that
                # aren't already in the result set. Downstream dedup catches
                # any ID collisions.
                seen_ids = {str(r.get("_id")) for r in results if r.get("_id")}
                merged = list(results)
                for b in prefetched:
                    bid = str(b.get("_id") or "")
                    if not bid or bid in seen_ids:
                        continue
                    seen_ids.add(bid)
                    merged.append(b)
                rec["added_count"] = len(merged) - len(results)
                rec["top_score_delta"] = _top_score(merged) - _top_score(results)
            if len(merged) > len(results):
                _log.info(
                    "entity_prefetch_merged",
                    seeds=len(results),
                    added=len(merged) - len(results),
                )
            return merged
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("entity_prefetch_failed", error=str(exc))
            return results

    def _maybe_graph_expand(
        self,
        query: str,
        workspace: str,
        results: list[dict],
        *,
        corpus: list[dict] | None = None,
    ) -> list[dict]:
        """Append graph-walked blocks when enabled (v3.3.0 Tier 1 #2).

        ``corpus`` — when provided — skips a workspace reload (shared
        with entity_prefetch). Fails open on any error so recall
        never blocks on graph issues.
        """
        if not results:
            return results
        try:
            from .graph_recall import (
                graph_expand,
                is_graph_expand_enabled,
                resolve_graph_config,
            )

            if not is_graph_expand_enabled(self._config, query):
                return results
            if corpus is not None:
                all_blocks = corpus
            else:
                # Legacy path: caller didn't pre-load the corpus.
                from .block_parser import parse_file
                from .block_store import MarkdownBlockStore

                store = MarkdownBlockStore(workspace)
                all_blocks = []
                for path in store.list_blocks():
                    try:
                        all_blocks.extend(parse_file(path))
                    except Exception as exc:  # pragma: no cover
                        _log.debug("graph_expand_block_parse_skipped", error=str(exc))
                        continue
            params = resolve_graph_config(self._config)
            with _step("graph_expand", max_hops=params["max_hops"]) as rec:
                expanded = graph_expand(results, all_blocks, **params)
                rec["added_count"] = len(expanded) - len(results)
                rec["top_score_delta"] = _top_score(expanded) - _top_score(results)
            if len(expanded) > len(results):
                _log.info(
                    "graph_expand_applied",
                    seeds=len(results),
                    final=len(expanded),
                    max_hops=params["max_hops"],
                )
            return expanded
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("graph_expand_failed", error=str(exc))
            return results

    def _maybe_kg_expand(
        self,
        query: str,
        workspace: str,
        results: list[dict],
        *,
        corpus: list[dict] | None = None,
    ) -> list[dict]:
        """Append typed-knowledge-graph blocks when enabled.

        Gated behind ``retrieval.kg_fusion.enabled`` (default false)
        so existing recall replays byte-identical until the operator
        opts in. Read-only against the graph — query terms resolve via
        ``EntityRegistry.lookup`` and never mint entities. Fails open
        on any error so recall never blocks on graph issues.
        """
        if not results:
            return results
        try:
            from .kg_fusion import (
                is_kg_fusion_enabled,
                kg_expand,
                resolve_kg_fusion_config,
            )
            from .knowledge_graph import KnowledgeGraph, default_db_path

            if not is_kg_fusion_enabled(self._config):
                return results
            db_path = default_db_path(workspace)
            if not os.path.isfile(db_path):
                return results
            if corpus is not None:
                all_blocks = corpus
            else:
                from .block_parser import parse_file
                from .block_store import MarkdownBlockStore

                store = MarkdownBlockStore(workspace)
                all_blocks = []
                for path in store.list_blocks():
                    try:
                        all_blocks.extend(parse_file(path))
                    except Exception as exc:  # pragma: no cover
                        _log.debug("kg_expand_block_parse_skipped", error=str(exc))
                        continue
            params = resolve_kg_fusion_config(self._config)
            with _step("kg_expand", max_hops=params["max_hops"]) as rec:
                with KnowledgeGraph(db_path) as kg:
                    expanded = kg_expand(results, all_blocks, kg, query, **params)
                rec["added_count"] = len(expanded) - len(results)
                rec["top_score_delta"] = _top_score(expanded) - _top_score(results)
            if len(expanded) > len(results):
                _log.info(
                    "kg_fusion_applied",
                    seeds=len(results),
                    final=len(expanded),
                    max_hops=params["max_hops"],
                )
            return expanded
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("kg_expand_failed", error=str(exc))
            return results

    def _cross_encoder_config(self) -> dict[str, Any]:
        """The ``cross_encoder`` config section, or ``{}`` when malformed."""
        ce_cfg = self._config.get("cross_encoder", {}) if isinstance(self._config, dict) else {}
        return ce_cfg if isinstance(ce_cfg, dict) else {}

    def _cross_encoder_active(self, query_type: str | None) -> bool:
        """Will a reranker actually run for this request?

        Extracted from :meth:`_maybe_cross_encoder_rerank` because the answer
        is needed BEFORE the retrieval legs are dispatched: the legs must
        fetch at least ``rerank_depth`` candidates for the depth to mean
        anything, and widening them when no reranker will run would move the
        fused order of a request that never asked for a cross-encoder.

        ``query_type`` is passed in already computed. ``_search`` memoizes the
        detector for the whole request, so this predicate adds no second
        regex pass -- it removes one, since the old inline copy classified the
        query a second time on every search.

        Order matters. The config questions are dict lookups and are asked
        first; the availability probe is asked LAST because its first call
        imports ``sentence_transformers``. A deployment that has not enabled
        the cross-encoder never reaches it.
        """
        ce_cfg = self._cross_encoder_config()
        auto = bool(ce_cfg.get("auto_enable", True))
        auto_fired = False
        if bool(ce_cfg.get("enabled", False)):
            pass
        elif auto and query_type in ("multi-hop", "temporal"):
            auto_fired = True
        else:
            return False
        # Configured on, but is there a reranker to run? Answering this here
        # is what keeps a box without ``sentence-transformers`` from widening
        # its retrieval legs for a reranker that will never run.
        #
        # The ensemble is settled by READING its enabled flag, not by calling
        # ``create_ensemble``: that factory CONSTRUCTS its members and logs on
        # an unknown name or an all-members-failed build. A probe that decides
        # "no" must leave no trace and do no work, so it must not be the thing
        # that builds the feature it is asking about. An ensemble may hold
        # members needing no local weights, so its flag alone answers yes.
        if self._reranker_ensemble_enabled():
            available = True
        else:
            try:
                from .cross_encoder_reranker import CrossEncoderReranker

                available = bool(CrossEncoderReranker.is_available())
            except Exception as exc:  # pragma: no cover — defensive
                _log.debug("ce_availability_probe_failed", error=str(exc))
                available = False
        # Announced only once a reranker will really run. Logging
        # "auto enabled" and then finding nothing to run announces a feature
        # the request never got, and an operator reading that line would be
        # looking for rerank latency that was never spent.
        if auto_fired and available:
            _log.info(
                "cross_encoder_auto_enabled",
                query_type=query_type,
                reason="v3.3.0_tier2_ambiguous_query",
            )
        return available

    def _reranker_ensemble_enabled(self) -> bool:
        """``retrieval.reranker_ensemble.enabled``, read without building it."""
        retrieval = self._config.get("retrieval") if isinstance(self._config, dict) else None
        if not isinstance(retrieval, dict):
            return False
        ens = retrieval.get("reranker_ensemble")
        return bool(ens.get("enabled", False)) if isinstance(ens, dict) else False

    def _maybe_cross_encoder_rerank(
        self,
        query: str,
        result: list[dict],
        limit: int,
        *,
        rerank_depth: int | None = None,
        ce_active: bool | None = None,
    ) -> list[dict]:
        """Rerank the candidate pool, then cut it to ``limit``.

        v3.3.0 Tier 2 #6 — auto-enables on multi-hop / temporal queries
        (per detect_query_type) even when ``cross_encoder.enabled`` is
        false, unless operator sets ``cross_encoder.auto_enable: false``.

        **This method now owns the ``[:limit]`` cut**, and that is the whole
        point of the change. Callers hand it the FULL candidate pool instead
        of a pre-sliced ``pool[:limit]``; it reranks over ``rerank_depth`` of
        them and returns the reranker's own top-k. Slicing to ``limit``
        first — which is what the hybrid path did — meant every block the
        model could promote was already in the response, so reranking could
        not change recall@k by construction and bought reordering only.

        When no reranker runs, the return is ``result[:limit]``: exactly the
        list the caller used to compute for itself, unchanged and in the same
        order. Nothing widened the legs in that case either, so a
        reranker-off deployment sees the same candidates as before.

        Returns ``result[:limit]`` on any reranker failure, never a wider
        list than the caller asked for.

        deferred: this stage records NO attribution step, unlike the seven
        ``_maybe_*`` hooks above it — it rebinds ``result`` across two
        independent early-return paths (ensemble, then single-model CE), so a
        single honest ``_step`` around it needs the reranker body extracted
        into its own method first. Upgrade path: extract
        ``_apply_reranker(query, result, limit, ce_cfg)`` and wrap the one
        call, rather than sprinkling three partial steps.
        """
        if not result:
            return result
        ce_cfg = self._cross_encoder_config()
        if ce_active is None:
            ce_active = self._cross_encoder_active(self._detect_query_type(query))
        if not ce_active:
            return result[:limit]
        if rerank_depth is None:
            rerank_depth = resolve_rerank_depth(self._config, limit)
        candidates = result[:rerank_depth]
        # ``None`` until a reranker actually produced an ordering. The
        # distinction is load-bearing: on a failure we must hand back
        # ``limit`` blocks, not the ``rerank_depth`` candidates we fetched
        # for a reranker that never ran.
        reranked: list[dict] | None = None
        # v3.3.0 Tier 4 #9 — prefer reranker_ensemble when configured,
        # fall back to single-model CE. The ensemble's single-member
        # degenerate case is also the same as CE alone, so wiring this
        # in doesn't regress existing CE-only deployments.
        try:
            from .rerank_ensemble import create_ensemble

            ensemble = create_ensemble(self._config)
            if ensemble is not None:
                for r in candidates:
                    if "content" not in r:
                        r["content"] = r.get("excerpt", "")
                reranked = ensemble.rerank(
                    query,
                    candidates,
                    top_k=ce_cfg.get("top_k", limit),
                    blend_weight=ce_cfg.get("blend_weight", 0.6),
                )
                _log.info("reranker_ensemble_applied", candidates=len(candidates), depth=rerank_depth)
        except Exception as exc:
            _log.warning("reranker_ensemble_failed", error=str(exc))
            reranked = None
        if reranked is None:
            try:
                from .cross_encoder_reranker import CrossEncoderReranker

                if CrossEncoderReranker.is_available():
                    ce = CrossEncoderReranker()
                    for r in candidates:
                        if "content" not in r:
                            r["content"] = r.get("excerpt", "")
                    reranked = ce.rerank(
                        query,
                        candidates,
                        top_k=ce_cfg.get("top_k", limit),
                        blend_weight=ce_cfg.get("blend_weight", 0.6),
                    )
                    _log.info(
                        "cross_encoder_rerank",
                        candidates=len(candidates),
                        depth=rerank_depth,
                        blend_weight=ce_cfg.get("blend_weight", 0.6),
                    )
            except ImportError as ie:
                _log.warning("cross_encoder_import_failed", error=str(ie))
            except Exception as e:
                _log.warning("cross_encoder_unavailable", error=str(e))
        if reranked is None:
            return candidates[:limit]
        return reranked

    # -- backend wrappers ---------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        workspace: str,
        limit: int = 200,
        *,
        scoring_instant: date | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """BM25 leg with a STRUCTURAL empty-arm guard.

        Runs the lexical search (:meth:`_bm25_search_raw`); when it yields
        ZERO hits, distinguishes a legitimate zero-match (the FTS index is
        populated, this query simply matched nothing — passes through
        silently) from a STRUCTURAL failure (the lexical index is
        empty/missing while the store has blocks). The latter is the exact
        silent bug that collapses hybrid fusion to the 1/(k+1) single-arm
        noise floor, so it is marked LOUD via the same ``degraded`` plumbing
        the vector leg uses (:attr:`RecallResults.degraded`) — asymmetric no
        longer: an empty BM25 arm is now as visible as a failed vector arm.

        A marker the RAW leg already raised is carried through rather than
        replaced. The scan leg reports ``corpus_truncated`` when a workspace
        exceeds ``MAX_BLOCKS_PER_QUERY`` and only an arbitrary prefix was
        scored — a NON-EMPTY degraded result, i.e. exactly the case the
        early ``if results`` return used to drop on the floor. Passing it up
        is what stops a truncated answer from being indistinguishable from a
        complete one at the surface.

        Under ``recall.strict_hybrid=true`` the structural failure RAISES
        (:class:`BM25LegError`) instead of degrading.
        """
        # scoring_instant is a NAMED parameter of this function, so it does
        # NOT ride in **kwargs -- forwarding only **kwargs silently dropped
        # it and the BM25 leg re-resolved to today, which is precisely the
        # pass-through the clock guard exists to catch.
        results = self._bm25_search_raw(query, workspace, limit=limit, scoring_instant=scoring_instant, **kwargs)
        raw_marker: LegMarker | None = getattr(results, "degraded", None)
        if results:
            return _as_results(results, raw_marker)

        marker = self._bm25_empty_arm_marker(workspace)
        if marker is None:
            return _as_results(results, raw_marker)
        if self._strict_hybrid:
            raise BM25LegError(str(marker["reason"]))
        metrics.inc("hybrid_bm25_leg_index_empty")
        _log.warning(
            "hybrid_bm25_leg_degraded",
            leg=marker["leg"],
            reason=marker["reason"],
            recall_db=marker.get("recall_db", ""),
            fts_rows=marker.get("fts_rows", "0"),
            advice="BM25 index empty/missing while store has blocks — run reindex / `mm doctor --rebuild-cache`",
        )
        return _as_results(results, _merge_leg_markers(raw_marker, {"leg": marker["leg"], "reason": marker["reason"]}))

    def _bm25_search_raw(
        self,
        query: str,
        workspace: str,
        limit: int = 200,
        *,
        scoring_instant: date | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """BM25 search via the existing recall engine.

        Tries sqlite_index first (O(log N)), then falls back to recall.py
        (O(corpus)).

        ``scoring_instant`` is named explicitly rather than left to ride in
        ``**kwargs``: an implicit pass-through is exactly how the recency layer
        came to read a hidden clock in the first place, and a named parameter is
        what lets a reader — and the call-site guard in
        ``tests/test_recall_clock_guard.py`` — see the seam at every hop.
        """
        try:
            from .sqlite_index import _db_path, ensure_index, query_index

            db = _db_path(workspace)
            # Flag first, deliberately: with ``auto_build_index`` off this is
            # one attribute test and NOTHING else — no stat, no config read,
            # no import cost beyond the one already taken above. ``stat`` is
            # only reached by the ``os.path.isfile`` that was always here.
            if self._auto_build_index:
                ensure_index(workspace, enabled=True)
            if os.path.isfile(db):
                return query_index(workspace, query, limit=limit, scoring_instant=scoring_instant, **kwargs)
        except ImportError:
            _log.debug("sqlite_index_not_available")
        except Exception as exc:
            _log.warning("sqlite_index_fallback", error=str(exc))

        try:
            from .recall import recall

            return recall(workspace, query, limit=limit, scoring_instant=scoring_instant, **kwargs)
        except Exception as exc:
            _log.error("bm25_search_failed", error=str(exc))
            return []

    def _bm25_empty_arm_marker(self, workspace: str) -> LegMarker | None:
        """Return a ``bm25`` degradation marker IFF the lexical index is
        STRUCTURALLY empty while the store has blocks; else ``None``.

        Only invoked on the zero-hit path. Resolves the absolute recall.db
        path, counts its FTS rows, and — only when that FTS is empty/missing
        — checks whether the configured store actually has blocks. This
        ordering keeps the (heavier) store probe OFF the common legitimate
        zero-match path: a populated FTS short-circuits to ``None`` cheaply.

        The returned marker carries ``recall_db`` + ``fts_rows`` for the
        degrade log's visibility; the caller strips it to ``{leg, reason}``
        for the marker that flows into fusion / the recall attestation.
        """
        # The Postgres server-side hybrid provides its OWN lexical arm
        # (PostgresBlockStore.hybrid_search); the sqlite FTS is not its
        # source, so an empty sqlite FTS there is not a degradation.
        if isinstance(self._config, dict) and self._config.get("provider") == "postgres":
            return None
        try:
            from .sqlite_index import _db_path

            db = _db_path(workspace)
        except Exception:  # pragma: no cover — defensive
            return None
        fts_rows = _fts_row_count(db)  # None => file/table/schema missing
        if fts_rows and fts_rows > 0:
            # Populated FTS + zero hits = legitimate zero-match; stay silent.
            _log.debug("hybrid_bm25_index_resolved", recall_db=db, fts_rows=fts_rows)
            return None
        if not self._store_has_blocks(workspace):
            # Empty FTS AND empty store = a fresh workspace; not a failure.
            return None
        return {
            "leg": "bm25",
            "reason": "index_empty",
            "recall_db": db,
            "fts_rows": str(fts_rows if fts_rows is not None else 0),
        }

    def _store_has_blocks(self, workspace: str) -> bool:
        """Best-effort: does the CONFIGURED store hold >=1 active block?

        Routes through the backend-aware ``iter_active_blocks`` (``config=None``
        auto-loads the workspace's mind-mem.json, so a Postgres store's blocks
        are counted rather than the empty local Markdown corpus). Returns
        ``False`` on any failure so an unavailable store never manufactures a
        false ``index_empty`` degradation on a genuinely fresh workspace. Only
        reached on the empty-FTS path, so its cost stays off the hot path.
        """
        try:
            from .storage import iter_active_blocks

            return bool(iter_active_blocks(workspace, config=None))
        except Exception as exc:  # pragma: no cover — defensive
            _log.debug("bm25_store_probe_failed", error=str(exc))
            return False

    def _vector_search(
        self,
        query: str,
        workspace: str,
        limit: int = 200,
        active_only: bool = False,
        *,
        scoring_instant: date | None = None,
    ) -> list[dict]:
        """Vector search via recall_vector.search_batch (for RRF) or .search."""
        try:
            from . import recall_vector

            # Prefer search_batch (returns all results for RRF)
            if hasattr(recall_vector, "search_batch"):
                return list(
                    recall_vector.search_batch(
                        workspace,
                        query,
                        limit=limit,
                        active_only=active_only,
                        config=self._config,
                        scoring_instant=scoring_instant,
                    )
                )

            # Fallback: VectorBackend.search
            backend = recall_vector.VectorBackend(self._config)
            return list(backend.search(workspace, query, limit=limit, active_only=active_only, scoring_instant=scoring_instant))
        except ImportError as exc:
            _log.warning("vector_search_import_failed")
            raise VectorLegError("import_failed") from exc
        except Exception as exc:
            _log.error("vector_search_failed", error=str(exc))
            raise VectorLegError("error") from exc

    # -- factory ------------------------------------------------------------

    @staticmethod
    def from_config(config: dict[str, Any]) -> "HybridBackend":
        """Create HybridBackend from a full mind-mem.json config dict.

        Validates that ``config`` contains a ``recall`` section.  When
        the section is missing or not a dict, logs a warning and falls
        back to BM25-only defaults.
        """
        recall_cfg = config.get("recall")
        if recall_cfg is None:
            _log.warning(
                "hybrid_config_missing_recall_section",
                hint="Expected 'recall' key in config. Using BM25-only defaults.",
            )
            recall_cfg = {}
        elif not isinstance(recall_cfg, dict):
            _log.warning(
                "hybrid_config_recall_not_dict",
                type=type(recall_cfg).__name__,
                hint="'recall' must be a dict. Using BM25-only defaults.",
            )
            recall_cfg = {}
        return HybridBackend(config=recall_cfg)
