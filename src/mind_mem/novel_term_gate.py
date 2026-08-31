# Copyright 2026 STARGA, Inc.
"""Novel-term gate — the local-vs-source confidence signal (Group J).

The client-side anticipation cache serves likely-relevant context from a
local bundle instead of paying a round-trip to the store. That is only
safe while the local corpus can plausibly answer the query. This module
is that check, and nothing else: given a query and the cached context,
decide whether the local hit may be served or whether the caller must
fall through to the governed store.

The rule, per the roadmap item:

* Compute the **novel-stem ratio** — the fraction of the query's distinct
  stems that do not appear in the cached corpus.
* Suppress the local hit when that ratio **exceeds** the configured
  threshold (default 0.45). "Exceeds" is strict: a ratio exactly at the
  threshold is served.
* Apply the ratio only once the cached corpus holds at least
  ``min_corpus_stems`` distinct stems (default 200). Below that floor the
  ratio is dominated by whatever happened to land in a cold cache, so the
  caller falls through to the store instead of trusting it.

Two degenerate inputs resolve in the safe direction — fall through, never
serve blind: a query with no usable stems, and a corpus below the floor.
Suppressing a good local hit costs one round-trip; serving a bad one puts
wrong context in the window.

Wedge guardrail (load-bearing): this is a **pure function** of
``(query, cached-context)``. No clock, no randomness, no I/O, no network,
no model call, stdlib only. The same inputs produce the same verdict on
every substrate, so the decision is reproducible from the retrieval log.
Thresholds arrive as an explicit frozen config — the gate never reweights
itself.

Stems come from :func:`mind_mem._recall_tokenization.tokenize`, the same
tokenizer the recall/BM25 path uses. That is deliberate: the gate must
judge novelty in the same vocabulary the cache is indexed under, or it
would be reasoning about stems the cache does not have.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import AbstractSet, Final, Iterable, Union

from ._recall_tokenization import tokenize

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_MIN_CORPUS_STEMS",
    "DEFAULT_NOVEL_RATIO_THRESHOLD",
    "REASON_CORPUS_BELOW_FLOOR",
    "REASON_KNOWN_TERMS",
    "REASON_NOVEL_RATIO_EXCEEDED",
    "REASON_NO_QUERY_STEMS",
    "NovelTermGateConfig",
    "NovelTermVerdict",
    "corpus_stems",
    "evaluate",
    "evaluate_stems",
]

#: Suppress the local hit above this share of novel query stems.
DEFAULT_NOVEL_RATIO_THRESHOLD: Final = 0.45

#: Distinct cached stems required before the ratio is trusted at all.
DEFAULT_MIN_CORPUS_STEMS: Final = 200

REASON_KNOWN_TERMS: Final = "known_terms"
REASON_NOVEL_RATIO_EXCEEDED: Final = "novel_ratio_exceeded"
REASON_CORPUS_BELOW_FLOOR: Final = "corpus_below_floor"
REASON_NO_QUERY_STEMS: Final = "no_query_stems"

#: Cached context as raw text: one document, or several.
CachedContext = Union[str, Iterable[str]]


@dataclass(frozen=True)
class NovelTermGateConfig:
    """Versioned knobs. Validated here so a bad value cannot reach the gate."""

    novel_ratio_threshold: float = DEFAULT_NOVEL_RATIO_THRESHOLD
    min_corpus_stems: int = DEFAULT_MIN_CORPUS_STEMS

    def __post_init__(self) -> None:
        if not 0.0 <= self.novel_ratio_threshold <= 1.0:
            raise ValueError(f"novel_ratio_threshold must be within [0.0, 1.0], got {self.novel_ratio_threshold!r}")
        if self.min_corpus_stems < 0:
            raise ValueError(f"min_corpus_stems must be >= 0, got {self.min_corpus_stems!r}")


DEFAULT_CONFIG: Final = NovelTermGateConfig()


@dataclass(frozen=True)
class NovelTermVerdict:
    """One gate decision, with the numbers it was made from.

    Carries enough detail to be logged verbatim: a reader can recompute
    the verdict from ``novel_ratio``, ``corpus_stem_count`` and the config
    without re-running the query.
    """

    serve_from_cache: bool
    reason: str
    novel_ratio: float
    query_stem_count: int
    corpus_stem_count: int
    novel_stems: tuple[str, ...]


def _stems(text: str) -> list[str]:
    """Stems of one text, normalised so spelling variants agree.

    NFC normalisation is applied first because the shared tokenizer keeps
    only ``[a-z0-9_]`` runs: decomposed ``cafe`` + combining acute and
    precomposed ``café`` would otherwise stem to two different tokens, and
    the same word would read as novel purely because of how it was
    encoded.

    deferred: non-ASCII letters are dropped entirely by the shared recall
    tokenizer, so a non-Latin-script query yields no stems and fails safe
    (falls through to the store) rather than being judged. Upgrade path:
    widen the token pattern in ``_recall_tokenization.tokenize`` to
    Unicode word characters — it must change there, for the recall index
    and this gate together, never here alone.
    """
    return tokenize(unicodedata.normalize("NFC", text))


def corpus_stems(cached_context: CachedContext) -> frozenset[str]:
    """Distinct stems held by the cached context.

    Accepts one document or an iterable of documents; a bare ``str`` is
    treated as a single document, not as an iterable of characters.
    """
    documents = (cached_context,) if isinstance(cached_context, str) else cached_context
    return frozenset(stem for document in documents for stem in _stems(document))


def evaluate_stems(
    query: str,
    corpus: AbstractSet[str],
    config: NovelTermGateConfig = DEFAULT_CONFIG,
) -> NovelTermVerdict:
    """Decide on a pre-computed corpus stem set.

    The hot-path entry point: a cache consumer builds the stem set once
    per bundle and reuses it across queries.
    """
    query_set = frozenset(_stems(query))
    novel = tuple(sorted(query_set - corpus))
    # An unjudgeable query is maximally novel — the reported ratio agrees
    # with the fall-through verdict instead of contradicting it.
    ratio = len(novel) / len(query_set) if query_set else 1.0

    if not query_set:
        reason, serve = REASON_NO_QUERY_STEMS, False
    elif len(corpus) < config.min_corpus_stems:
        reason, serve = REASON_CORPUS_BELOW_FLOOR, False
    elif ratio > config.novel_ratio_threshold:
        reason, serve = REASON_NOVEL_RATIO_EXCEEDED, False
    else:
        reason, serve = REASON_KNOWN_TERMS, True

    return NovelTermVerdict(
        serve_from_cache=serve,
        reason=reason,
        novel_ratio=ratio,
        query_stem_count=len(query_set),
        corpus_stem_count=len(corpus),
        novel_stems=novel,
    )


def evaluate(
    query: str,
    cached_context: CachedContext,
    config: NovelTermGateConfig = DEFAULT_CONFIG,
) -> NovelTermVerdict:
    """Decide whether ``query`` may be served from ``cached_context``."""
    return evaluate_stems(query, corpus_stems(cached_context), config)
