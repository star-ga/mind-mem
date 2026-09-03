# Copyright 2026 STARGA, Inc.
"""The anticipation cache — a local bundle store keyed on the governed-ledger head.

Group J asked for a client-side cache: the idle ``prefetch`` /
``speculative_prefetch`` tools assemble bundles of likely-relevant blocks, a
recall consults those bundles locally before paying a round-trip to the store,
and a cheap BM25 front plus the :mod:`~mind_mem.novel_term_gate` decides
local-vs-source. This module is that consumer. Two things about it are
deliberate and load-bearing.

Not to be confused with its two same-named neighbours, which do different jobs:
the ``prefetch`` MCP tool and :func:`mind_mem.recall.prefetch_context` *produce*
a bundle from conversation signals (scored by the ``mind/prefetch.mind``
kernel), and :mod:`mind_mem.speculative_prefetch` predicts a likely next block.
This module is the store those producers fill and the consumer that reads it.

**Keyed on the ledger head, never on a time-to-live.**
The obvious cache expires an entry after N seconds. That is wrong here, and not
by a little. mind-mem sells *replay*: an attested recall names the corpus state
it ran against — ``index_anchor``, the head of the governed hash chain — and
re-running at that anchor reproduces the answer. A TTL cache breaks that on its
own schedule: a governed write moves the head, the cached entry does not notice,
and for the rest of its window the surface serves content the ledger has already
superseded while the attestation stamps the *new* anchor. The run then names a
corpus state it did not read.

The existing recall cache (:mod:`mind_mem.recall_cache`) papered over this with
explicit invalidation calls in the governance MCP tools — which is the
"filtered by whoever remembered" shape: a write that lands through the CLI, the
HTTP transport, the apply engine or federation replication never calls them, and
the stale entry is served anyway.

So every bundle here records the head it was assembled at, and a lookup at a
different head does not find it. There is no invalidation call to forget,
because a superseded generation is **unresolvable by construction**. Every
admitted write and every admitted delete appends to that ledger, so any of them
retires the whole generation with it.

**Deterministic, LLM-free, clock-free.**
Bundle selection is BM25 over the cached text using the recall tokenizer and the
shared :func:`~mind_mem._recall_scoring.bm25_idf` / ``bm25f_score_terms``
arithmetic — the same vocabulary and the same formula the store-side scorer
uses, so a locally-served answer is not scored by a second, divergent ranker.
The serve/fall-through decision is :func:`mind_mem.novel_term_gate.evaluate_stems`,
a pure function of ``(query, cached stems)``. Nothing here reads a clock, draws
a random number, opens a network connection or calls a model. Thresholds arrive
as a validated frozen config; the cache never reweights itself.

**Cost of the off path.** The flag is read from a config dict the caller has
*already* loaded (:func:`anticipation_enabled`) — no file read, no parse, no
syscall, no log line when the answer is no, and no per-hit tokenization. The one
thing the recall path pays unconditionally is the generation key itself: one
read-only open of the governed ledger per recall, which is the same read
``_apply_attestation`` already performs on every request. See :func:`chain_head`
for why that read is not memoized on a file stat.

deferred: the co-retrieval graph is fed but not yet *consulted*. The roadmap
item asks for it in both directions — a recall's served ids now reach
:class:`~mind_mem.speculative_prefetch.PrefetchPredictor` through
:func:`observe_served`, which is what took ``prefetch observations`` off zero,
but nothing yet takes ``predict()``'s answer and pulls those blocks into a
bundle ahead of the query that will want them. Stubbed because warming means
*reading blocks by id from the governed store*, and that read has to go through
the admission filter on the store-owning path rather than be re-implemented
here — a cache that fetched its own blocks would be a second read door.
Upgrade path: have the warmer call the same governed by-id read the recall
legs use, record the result through :meth:`AnticipationCache.record` at the
current head, and gate it on idle rather than on the request path. Nothing in
this module changes; it gains one more producer.
"""

from __future__ import annotations

import os
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence

from ._recall_constants import FIELD_WEIGHTS, SEARCH_FIELDS
from ._recall_scoring import bm25_idf, bm25f_score_terms, compute_weighted_tf
from ._recall_tokenization import tokenize
from .novel_term_gate import DEFAULT_CONFIG as GATE_DEFAULT_CONFIG
from .novel_term_gate import NovelTermGateConfig, NovelTermVerdict, evaluate_stems
from .recall_attestation import GENESIS_ANCHOR, _resolve_index_anchor

__all__ = [
    "DEFAULT_MAX_BUNDLES",
    "AnticipationCache",
    "AnticipationDecision",
    "Bundle",
    "REASON_COLD",
    "REASON_NO_MATCH",
    "anticipation_config",
    "anticipation_enabled",
    "chain_head",
    "get_cache",
    "reset_cache",
]

#: Bundles retained per head generation. Each bundle holds at most a recall
#: page of blocks, so a few hundred is a small number of megabytes and the
#: generation is discarded wholesale the moment the chain head moves.
DEFAULT_MAX_BUNDLES: Final = 128

#: No bundle has been recorded at the current head yet.
REASON_COLD: Final = "cache_cold"

#: Bundles exist at this head but none shares a single stem with the query.
REASON_NO_MATCH: Final = "no_lexical_overlap"

#: Marker written onto every hit served from here, so a reader of the envelope
#: or the retrieval log can always tell a locally-answered recall from one that
#: went to the store. An anticipation-served run answers from a *bundle*, not
#: from the corpus, so it must never be mistaken for the attested path.
RETRIEVAL_SOURCE: Final = "anticipation_cache"


# ---------------------------------------------------------------------------
# Chain head — the generation key
# ---------------------------------------------------------------------------


def chain_head(workspace: str) -> str:
    """The governed-ledger head for *workspace* — this cache's generation key.

    Delegates to :func:`mind_mem.recall_attestation._resolve_index_anchor`, the
    very function that produces the ``index_anchor`` an attestation binds. Using
    it rather than a second reader is the point: the cache generation and the
    attested corpus state can never become two different opinions of "which
    corpus is this", and when the ledger moves — as it did in 5.0.2, from the
    field-audit sidecar to ``memory/hash_chain_v2.db`` — this follows without a
    second edit.

    **Resolved fresh on every call, deliberately.** The obvious optimisation is
    to memoize the head and re-read the ledger only when an ``os.stat`` of its
    file has moved. Stated separately, because one of these is measured and the
    other is not:

    *Measured, on this tree:* the ledger file's ``(size, mtime, inode)`` does
    move on every append. Three consecutive appends each changed it, so a
    stat-keyed memo would be correct today.

    *Inferred, and why the memo is still not taken:* it would be correct for a
    reason that lives in another module and is not part of its contract.
    :class:`~mind_mem.hash_chain_v2.HashChainV2` runs the database in **WAL**
    mode and opens and closes a connection per append; SQLite checkpoints WAL
    when the last connection closes, and that checkpoint is what touches the
    main file. Hold a connection open across appends — an ordinary performance
    change to make in that module — and the frames stay in the ``-wal`` sidecar
    with the main file untouched. The memo would then report a stale head, the
    stale head would keep a retired generation alive, and this cache would serve
    content the ledger has superseded. No test here would fail; the defect would
    surface as wrong answers.

    So the read is paid. A freshness check whose correctness depends on another
    module's connection lifetime is not a freshness check, and the failure it
    guards against is the one thing this module exists to make impossible.

    The read is one read-only SQLite open per recall, and the recall path
    already pays exactly that: ``_apply_attestation`` resolves the same anchor
    through the same function on every request. Handing that value in instead
    of resolving it twice is a one-line change in the attestation module and is
    left to the lane that owns it.
    """
    if not workspace:
        return GENESIS_ANCHOR
    try:
        return _resolve_index_anchor(workspace)
    except Exception:  # pragma: no cover — the resolver already degrades internally
        return GENESIS_ANCHOR


# ---------------------------------------------------------------------------
# Config — read from an already-loaded dict, never from disk
# ---------------------------------------------------------------------------


def anticipation_enabled(config: Mapping[str, Any] | None) -> bool:
    """Is ``cache.anticipation.enabled`` on in an already-loaded config?

    A pure function of the dict handed in. The caller on the recall path has
    loaded ``mind-mem.json`` already, so asking this question costs no syscall,
    no parse and no log line — the off path is indistinguishable from a build
    that never had the feature, which is the point of shipping it gated.

    Anything malformed reads as off. Default is off: an anticipation-served
    recall answers from a local bundle rather than the corpus, so it is a
    latency mode an operator opts into, not the default retrieval path.
    """
    if not isinstance(config, Mapping):
        return False
    cache_cfg = config.get("cache")
    if not isinstance(cache_cfg, Mapping):
        return False
    section = cache_cfg.get("anticipation")
    if not isinstance(section, Mapping):
        return False
    return section.get("enabled") is True


def anticipation_config(config: Mapping[str, Any] | None) -> NovelTermGateConfig:
    """Gate thresholds from ``cache.anticipation``, validated.

    Versioned config, not autonomous reweighting: the two knobs are read from
    the workspace config and validated by :class:`NovelTermGateConfig` itself,
    so an out-of-range value raises at the boundary instead of quietly changing
    the gate. A malformed or absent section yields the shipped defaults.
    """
    if not isinstance(config, Mapping):
        return GATE_DEFAULT_CONFIG
    cache_cfg = config.get("cache")
    if not isinstance(cache_cfg, Mapping):
        return GATE_DEFAULT_CONFIG
    section = cache_cfg.get("anticipation")
    if not isinstance(section, Mapping):
        return GATE_DEFAULT_CONFIG
    threshold = section.get("novel_ratio_threshold", GATE_DEFAULT_CONFIG.novel_ratio_threshold)
    floor = section.get("min_corpus_stems", GATE_DEFAULT_CONFIG.min_corpus_stems)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        threshold = GATE_DEFAULT_CONFIG.novel_ratio_threshold
    if not isinstance(floor, int) or isinstance(floor, bool):
        floor = GATE_DEFAULT_CONFIG.min_corpus_stems
    return NovelTermGateConfig(novel_ratio_threshold=float(threshold), min_corpus_stems=int(floor))


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


#: Presentation keys on a *served* recall hit, mapped to the canonical field
#: whose text they carry. A hit arrives here in one of two shapes: a raw block
#: (canonical ``Statement`` / ``Tags`` / …), or the summary shape the MCP recall
#: envelope serialises (``excerpt`` / ``tags`` / ``speaker``). The alias is only
#: consulted when the canonical field is absent, so a dict carrying both is
#: never counted twice.
#:
#: ``excerpt`` is the first 300 characters of the highest-priority text field
#: (:func:`mind_mem._recall_detection.get_excerpt`), so a bundle built from
#: served envelopes indexes *less* text than the store holds. That is a real
#: limitation and it fails in the safe direction: a query whose terms live past
#: the excerpt reads as novel to the gate and falls through to the store.
_PRESENTATION_ALIASES: Final[dict[str, str]] = {
    "excerpt": "Statement",
    "tags": "Tags",
    "speaker": "Name",
}


def _hit_field_tokens(hit: Mapping[str, Any]) -> dict[str, list[str]]:
    """Tokenize one hit into the same weighted fields the store-side scorer uses."""

    def _text(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value)
        return str(value)

    tokens: dict[str, list[str]] = {}
    for field_name in SEARCH_FIELDS:
        value = hit.get(field_name)
        if value in (None, ""):
            continue
        toks = tokenize(_text(value))
        if toks:
            tokens[field_name] = toks
    for alias, canonical in _PRESENTATION_ALIASES.items():
        if canonical in tokens:
            continue
        value = hit.get(alias)
        if value in (None, ""):
            continue
        toks = tokenize(_text(value))
        if toks:
            tokens[canonical] = toks
    return tokens


@dataclass(frozen=True)
class _Document:
    """One cached block, pre-tokenized so a lookup does no re-parsing."""

    block_id: str
    hit: Mapping[str, Any]
    weighted_tf: Counter
    wdl: float
    stems: frozenset[str]


@dataclass(frozen=True)
class Bundle:
    """A set of blocks assembled at one chain head.

    ``head`` is the generation. A bundle is only ever consulted while the
    workspace's chain head still equals it; the moment a governed write moves
    the head, this bundle is unreachable and is dropped on the next touch.
    """

    head: str
    origin: str
    documents: tuple[_Document, ...] = field(default_factory=tuple)

    @property
    def stems(self) -> frozenset[str]:
        out: set[str] = set()
        for doc in self.documents:
            out |= doc.stems
        return frozenset(out)


@dataclass(frozen=True)
class AnticipationDecision:
    """What the cache decided, and the numbers it decided from.

    Logged verbatim: a reader can recompute the outcome from ``verdict`` plus
    the bundle counts without replaying the query.
    """

    served: tuple[dict[str, Any], ...]
    serve_from_cache: bool
    reason: str
    head: str
    bundle_count: int
    document_count: int
    verdict: NovelTermVerdict | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "serve_from_cache": self.serve_from_cache,
            "reason": self.reason,
            "head": self.head,
            "bundles": self.bundle_count,
            "documents": self.document_count,
        }
        if self.verdict is not None:
            out["novel_ratio"] = round(self.verdict.novel_ratio, 6)
            out["query_stems"] = self.verdict.query_stem_count
            out["corpus_stems"] = self.verdict.corpus_stem_count
        return out


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class AnticipationCache:
    """Process-local bundle store, partitioned by chain head.

    Entirely in memory and entirely per-process: this is a *client-side* cache,
    and pushing it to a shared transport would put one agent's locally-answered
    recall in another agent's window. Redis remains the push transport for the
    federation, per the roadmap item — this layer does not add a second one.
    """

    def __init__(self, max_bundles: int = DEFAULT_MAX_BUNDLES) -> None:
        self._max = max(1, int(max_bundles))
        self._lock = threading.RLock()
        # workspace -> (head, OrderedDict[bundle_key, Bundle]) — one live
        # generation per workspace. A head change replaces the mapping rather
        # than pruning it, so a superseded generation cannot be half-retired.
        self._generations: dict[str, tuple[str, "OrderedDict[str, Bundle]"]] = {}
        self._hits = 0
        self._misses = 0
        self._retired = 0

    # -- producer ----------------------------------------------------------

    def record(
        self,
        workspace: str,
        origin: str,
        hits: Sequence[Mapping[str, Any]],
        *,
        head: str | None = None,
    ) -> Bundle | None:
        """Store *hits* as a bundle at the workspace's current chain head.

        *origin* names the producer (``"recall"``, ``"prefetch"``, …) and is the
        bundle key: a producer replaces its own previous bundle rather than
        accumulating one per query, which keeps the generation bounded without
        needing an age to evict on.

        Returns the stored bundle, or ``None`` when there was nothing usable to
        store. Never raises: a cache that can fail a recall is worse than no
        cache.
        """
        if not workspace or not hits:
            return None
        resolved_head = chain_head(workspace) if head is None else head
        documents: list[_Document] = []
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            field_tokens = _hit_field_tokens(hit)
            if not field_tokens:
                continue
            weighted_tf, wdl = compute_weighted_tf(field_tokens, FIELD_WEIGHTS)
            documents.append(
                _Document(
                    block_id=str(hit.get("_id", "") or ""),
                    hit=dict(hit),
                    weighted_tf=weighted_tf,
                    wdl=wdl,
                    stems=frozenset(weighted_tf),
                )
            )
        if not documents:
            return None
        bundle = Bundle(head=resolved_head, origin=str(origin), documents=tuple(documents))
        with self._lock:
            generation = self._generation_for(workspace, resolved_head)
            generation[bundle.origin] = bundle
            generation.move_to_end(bundle.origin)
            while len(generation) > self._max:
                generation.popitem(last=False)
        return bundle

    # -- consumer ----------------------------------------------------------

    def lookup(
        self,
        workspace: str,
        query: str,
        *,
        limit: int = 10,
        gate_config: NovelTermGateConfig = GATE_DEFAULT_CONFIG,
        head: str | None = None,
    ) -> AnticipationDecision:
        """Decide whether *query* can be answered from the local bundles.

        The bundles consulted are only those recorded at the workspace's
        *current* chain head. A governed write that moved the head has already
        retired every bundle behind it, so this can never serve content the
        chain has superseded — the property a TTL cannot provide.

        On a serve, the returned hits are re-ranked **for this query** with the
        shared BM25 arithmetic. The cache answers the question asked; it never
        replays another query's ranked answer under a new query's name.
        """
        resolved_head = chain_head(workspace) if head is None else head
        with self._lock:
            generation = self._generation_for(workspace, resolved_head)
            bundles = tuple(generation.values())
        documents = tuple(doc for bundle in bundles for doc in bundle.documents)
        if not documents:
            self._count(miss=True)
            return AnticipationDecision(
                served=(),
                serve_from_cache=False,
                reason=REASON_COLD,
                head=resolved_head,
                bundle_count=len(bundles),
                document_count=0,
            )

        corpus_stems: set[str] = set()
        for doc in documents:
            corpus_stems |= doc.stems
        verdict = evaluate_stems(query, corpus_stems, gate_config)
        if not verdict.serve_from_cache:
            self._count(miss=True)
            return AnticipationDecision(
                served=(),
                serve_from_cache=False,
                reason=verdict.reason,
                head=resolved_head,
                bundle_count=len(bundles),
                document_count=len(documents),
                verdict=verdict,
            )

        ranked = self._rank(query, documents, limit=limit)
        if not ranked:
            self._count(miss=True)
            return AnticipationDecision(
                served=(),
                serve_from_cache=False,
                reason=REASON_NO_MATCH,
                head=resolved_head,
                bundle_count=len(bundles),
                document_count=len(documents),
                verdict=verdict,
            )
        self._count(miss=False)
        return AnticipationDecision(
            served=ranked,
            serve_from_cache=True,
            reason=verdict.reason,
            head=resolved_head,
            bundle_count=len(bundles),
            document_count=len(documents),
            verdict=verdict,
        )

    def _count(self, *, miss: bool) -> None:
        """Record one outcome. Under the lock, because two threads losing an
        increment turns the hit rate into a number nobody can act on."""
        with self._lock:
            if miss:
                self._misses += 1
            else:
                self._hits += 1

    # -- ranking -----------------------------------------------------------

    @staticmethod
    def _rank(query: str, documents: Sequence[_Document], *, limit: int) -> tuple[dict[str, Any], ...]:
        """BM25 the cached documents against *query*; highest first.

        Uses the shared IDF and BM25F helpers rather than a private formula, so
        a locally-served ranking is the same arithmetic the store-side scorer
        would have applied to the same text. Ties break on block id so the
        order is total and reproducible rather than dependent on dict ordering.
        """
        query_terms = tokenize(query)
        if not query_terms:
            return ()
        n_docs = len(documents)
        avg_wdl = sum(doc.wdl for doc in documents) / n_docs
        if avg_wdl <= 0:
            return ()
        idf_cache: dict[str, float] = {}
        for term in set(query_terms):
            df = sum(1 for doc in documents if term in doc.weighted_tf)
            idf_cache[term] = bm25_idf(n_docs, df)
        scored: list[tuple[float, str, _Document]] = []
        for doc in documents:
            score = bm25f_score_terms(query_terms, doc.weighted_tf, doc.wdl, idf_cache, avg_wdl)
            if score > 0:
                scored.append((score, doc.block_id, doc))
        if not scored:
            return ()
        scored.sort(key=lambda row: (-row[0], row[1]))
        out: list[dict[str, Any]] = []
        for score, _block_id, doc in scored[: max(1, int(limit))]:
            hit = dict(doc.hit)
            hit["_score"] = round(score, 6)
            hit["_retrieval_source"] = RETRIEVAL_SOURCE
            out.append(hit)
        return tuple(out)

    # -- generations -------------------------------------------------------

    def _generation_for(self, workspace: str, head: str) -> "OrderedDict[str, Bundle]":
        """The live bundle map for (*workspace*, *head*), retiring any older one.

        Called under the lock. When the recorded head differs from *head* the
        whole map is replaced: a generation is retired wholesale, never
        selectively, so there is no path on which one stale bundle survives a
        head move because a filter forgot about it.
        """
        key = os.path.abspath(workspace) if workspace else ""
        existing = self._generations.get(key)
        if existing is not None and existing[0] == head:
            return existing[1]
        if existing is not None:
            self._retired += len(existing[1])
        fresh: "OrderedDict[str, Bundle]" = OrderedDict()
        self._generations[key] = (head, fresh)
        return fresh

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Counters an operator or a diagnostic can read. No clock, no I/O.

        Every value is sampled under one lock hold, so the numbers describe a
        single moment rather than a smear across concurrent lookups.
        """
        with self._lock:
            bundles = sum(len(gen) for _head, gen in self._generations.values())
            documents = sum(len(b.documents) for _h, gen in self._generations.values() for b in gen.values())
            hits, misses, retired = self._hits, self._misses, self._retired
            workspaces = len(self._generations)
        total = hits + misses
        return {
            "workspaces": workspaces,
            "bundles": bundles,
            "documents": documents,
            "hits": hits,
            "misses": misses,
            "retired_bundles": retired,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    def clear(self) -> None:
        with self._lock:
            self._generations.clear()
            self._hits = 0
            self._misses = 0
            self._retired = 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: AnticipationCache | None = None


def get_cache() -> AnticipationCache:
    """The process-wide anticipation cache."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = AnticipationCache()
        return _singleton


def reset_cache() -> None:
    """Drop the singleton. Test-only hook."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def observe_served(query: str, hits: Iterable[Mapping[str, Any]]) -> None:
    """Feed the co-retrieval predictor with what a recall actually served.

    The roadmap item's other half: the idle prefetch machinery has been
    starving because nothing ever told it what a query resolved to
    (``prefetch observations = 0``). Every recall that reaches this cache now
    reports its served ids, which is what makes
    :meth:`~mind_mem.speculative_prefetch.PrefetchPredictor.predict` able to
    name a next-block at all.

    Process-local and governance-free — the predictor is a first-order Markov
    table over block ids in memory, not a write to the corpus. Never raises:
    an observation is telemetry, and telemetry must not be able to fail a
    recall.
    """
    block_ids = [str(h.get("_id", "") or "") for h in hits if isinstance(h, Mapping)]
    block_ids = [b for b in block_ids if b]
    if not block_ids:
        return
    try:
        from .speculative_prefetch import get_default_predictor

        get_default_predictor().observe(query, block_ids)
    except Exception:  # pragma: no cover — telemetry must never break recall
        return
