# Copyright 2026 STARGA, Inc.
"""Tests for the novel-term gate (Group J, client-side anticipation cache).

The gate is a pure function of ``(query, cached-context)``: no clock, no
randomness, no I/O. Every test here is therefore an exact-value assertion,
not a tolerance — a heuristic whose decision cannot be pinned down is not a
gate.
"""

from __future__ import annotations

import unicodedata

import pytest

from mind_mem.novel_term_gate import (
    DEFAULT_MIN_CORPUS_STEMS,
    DEFAULT_NOVEL_RATIO_THRESHOLD,
    REASON_CORPUS_BELOW_FLOOR,
    REASON_KNOWN_TERMS,
    REASON_NO_QUERY_STEMS,
    REASON_NOVEL_RATIO_EXCEEDED,
    NovelTermGateConfig,
    corpus_stems,
    evaluate,
    evaluate_stems,
)

# A corpus comfortably above the default stem floor, so the floor is never
# the reason a ratio-focused test passes or fails.
WARM_CORPUS = " ".join(f"alpha{i}" for i in range(300))


def test_defaults_match_the_specified_thresholds():
    """Roadmap Group J: ratio threshold ~0.45 once the corpus has >=200 stems."""
    assert DEFAULT_NOVEL_RATIO_THRESHOLD == 0.45
    assert DEFAULT_MIN_CORPUS_STEMS == 200
    assert NovelTermGateConfig().novel_ratio_threshold == DEFAULT_NOVEL_RATIO_THRESHOLD
    assert NovelTermGateConfig().min_corpus_stems == DEFAULT_MIN_CORPUS_STEMS


def test_corpus_stems_counts_distinct_stems_from_one_string_or_many():
    """A bare string is one document, not an iterable of characters."""
    assert corpus_stems("alpha0 alpha1 alpha0") == frozenset({"alpha0", "alpha1"})
    assert corpus_stems(["alpha0", "alpha1"]) == corpus_stems("alpha0 alpha1")
    assert corpus_stems("") == frozenset()


# ---------------------------------------------------------------------------
# The two headline cases
# ---------------------------------------------------------------------------


def test_entirely_known_query_is_served_from_cache():
    verdict = evaluate("alpha7 alpha42 alpha255", WARM_CORPUS)

    assert verdict.serve_from_cache is True
    assert verdict.reason == REASON_KNOWN_TERMS
    assert verdict.novel_ratio == 0.0
    assert verdict.novel_stems == ()
    assert verdict.query_stem_count == 3
    assert verdict.corpus_stem_count == 300


def test_entirely_novel_query_is_suppressed():
    verdict = evaluate("zeta0 zeta1 zeta2", WARM_CORPUS)

    assert verdict.serve_from_cache is False
    assert verdict.reason == REASON_NOVEL_RATIO_EXCEEDED
    assert verdict.novel_ratio == 1.0
    assert verdict.novel_stems == ("zeta0", "zeta1", "zeta2")


# ---------------------------------------------------------------------------
# Threshold semantics: "exceeds" is strict
# ---------------------------------------------------------------------------


def _query(known: int, novel: int) -> str:
    """A query with exactly ``known`` cached stems and ``novel`` uncached ones."""
    return " ".join([f"alpha{i}" for i in range(known)] + [f"zeta{i}" for i in range(novel)])


def test_ratio_exactly_at_threshold_is_allowed():
    """0.45 does not *exceed* 0.45 — the boundary hit is served."""
    verdict = evaluate(_query(known=11, novel=9), WARM_CORPUS)

    assert verdict.query_stem_count == 20
    assert verdict.novel_ratio == 0.45
    assert verdict.novel_ratio == DEFAULT_NOVEL_RATIO_THRESHOLD
    assert verdict.serve_from_cache is True
    assert verdict.reason == REASON_KNOWN_TERMS


def test_one_stem_past_the_threshold_is_suppressed():
    verdict = evaluate(_query(known=10, novel=10), WARM_CORPUS)

    assert verdict.novel_ratio == 0.5
    assert verdict.serve_from_cache is False
    assert verdict.reason == REASON_NOVEL_RATIO_EXCEEDED


def test_threshold_is_configurable_and_still_strict():
    config = NovelTermGateConfig(novel_ratio_threshold=0.5)

    at_threshold = evaluate(_query(known=10, novel=10), WARM_CORPUS, config)
    past_threshold = evaluate(_query(known=9, novel=11), WARM_CORPUS, config)

    assert at_threshold.novel_ratio == 0.5
    assert at_threshold.serve_from_cache is True
    assert past_threshold.novel_ratio == 0.55
    assert past_threshold.serve_from_cache is False


# ---------------------------------------------------------------------------
# Corpus-size floor: a cold cache is never trusted
# ---------------------------------------------------------------------------


def test_corpus_below_the_floor_falls_through_even_for_a_fully_known_query():
    cold = " ".join(f"alpha{i}" for i in range(DEFAULT_MIN_CORPUS_STEMS - 1))

    verdict = evaluate("alpha7 alpha42", cold)

    assert verdict.corpus_stem_count == DEFAULT_MIN_CORPUS_STEMS - 1
    assert verdict.novel_ratio == 0.0
    assert verdict.serve_from_cache is False
    assert verdict.reason == REASON_CORPUS_BELOW_FLOOR


def test_corpus_exactly_at_the_floor_is_trusted():
    """The floor reads '>= N stems', so N itself is warm enough."""
    warm = " ".join(f"alpha{i}" for i in range(DEFAULT_MIN_CORPUS_STEMS))

    verdict = evaluate("alpha7 alpha42", warm)

    assert verdict.corpus_stem_count == DEFAULT_MIN_CORPUS_STEMS
    assert verdict.serve_from_cache is True
    assert verdict.reason == REASON_KNOWN_TERMS


def test_empty_corpus_falls_through():
    verdict = evaluate("alpha7", "")

    assert verdict.corpus_stem_count == 0
    assert verdict.serve_from_cache is False
    assert verdict.reason == REASON_CORPUS_BELOW_FLOOR


# ---------------------------------------------------------------------------
# Degenerate queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\t\n ", "!!! ??? ---", "a i"])
def test_query_with_no_usable_stems_falls_through(query):
    """No stems means no evidence the local cache can answer — fail safe."""
    verdict = evaluate(query, WARM_CORPUS)

    assert verdict.query_stem_count == 0
    assert verdict.novel_ratio == 1.0
    assert verdict.serve_from_cache is False
    assert verdict.reason == REASON_NO_QUERY_STEMS


def test_no_query_stems_outranks_the_corpus_floor():
    """Precedence is fixed and asserted, not incidental."""
    verdict = evaluate("   ", "")

    assert verdict.reason == REASON_NO_QUERY_STEMS


def test_repeated_stems_are_counted_once():
    """The ratio is over distinct stems, so a repeated term cannot skew it."""
    verdict = evaluate("zeta0 zeta0 zeta0 alpha1", WARM_CORPUS)

    assert verdict.query_stem_count == 2
    assert verdict.novel_ratio == 0.5


# ---------------------------------------------------------------------------
# Unicode
# ---------------------------------------------------------------------------


def test_unicode_query_and_corpus_agree_across_normalisation_forms():
    """The same word decomposed differently must not read as a novel term."""
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert nfc != nfd  # guard: the fixture really is testing normalisation

    corpus = WARM_CORPUS + " " + nfc
    from_nfc = evaluate(nfc, corpus)
    from_nfd = evaluate(nfd, corpus)

    assert from_nfc.novel_ratio == 0.0
    assert from_nfc.serve_from_cache is True
    assert from_nfd == from_nfc


def test_unicode_only_query_falls_through():
    """Non-Latin script yields no ASCII stems — fail safe, never a blind hit."""
    verdict = evaluate("Привет мир", WARM_CORPUS)

    assert verdict.query_stem_count == 0
    assert verdict.serve_from_cache is False
    assert verdict.reason == REASON_NO_QUERY_STEMS


def test_mixed_script_query_is_decided_by_its_ascii_stems():
    verdict = evaluate("Привет alpha7 zeta9", WARM_CORPUS)

    assert verdict.query_stem_count == 2
    assert verdict.novel_stems == ("zeta9",)
    assert verdict.novel_ratio == 0.5
    assert verdict.serve_from_cache is False


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_gate_is_a_pure_function_of_query_and_context():
    first = evaluate("alpha1 zeta1", WARM_CORPUS)
    second = evaluate("alpha1 zeta1", WARM_CORPUS)

    assert first == second
    assert evaluate_stems("alpha1 zeta1", corpus_stems(WARM_CORPUS)) == first


def test_verdict_and_config_are_immutable():
    verdict = evaluate("alpha1", WARM_CORPUS)

    with pytest.raises(Exception):
        verdict.serve_from_cache = False  # type: ignore[misc]
    with pytest.raises(Exception):
        NovelTermGateConfig().min_corpus_stems = 1  # type: ignore[misc]


def test_invalid_config_is_refused_at_construction():
    with pytest.raises(ValueError):
        NovelTermGateConfig(novel_ratio_threshold=1.5)
    with pytest.raises(ValueError):
        NovelTermGateConfig(novel_ratio_threshold=-0.1)
    with pytest.raises(ValueError):
        NovelTermGateConfig(min_corpus_stems=-1)
