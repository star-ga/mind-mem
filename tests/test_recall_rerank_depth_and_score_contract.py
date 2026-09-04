"""One score contract (F1) + rerank depth (F2).

Two defects are pinned here, both of which made a configured feature buy
nothing:

F1 -- ``rrf_fuse`` wrote ``rrf_score`` and never wrote ``score``, so the
fused ``score`` column held whichever leg's raw value survived the dict
copy: an unbounded BM25F number on some hits, a ``[0, 1]`` cosine on
others. Every consumer that reads ``score`` was reading a mixed-scale
column -- including the cross-encoder, which min-maxes it, so a
vector-sourced hit's original weight collapsed to ~0 and ``blend_weight``
was a fiction. The ensemble members did not normalise at all.

F2 -- the hybrid path sliced ``fused[:limit]`` BEFORE the cross-encoder
ran. Every block the model could promote was already being served, so
reranking could not change recall@k by construction: the latency bought a
permutation and nothing else.

Every guard added for these has a test here that shows it FAILING when the
guard is removed, not merely one that passes while it is present.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from mind_mem.cross_encoder_reranker import normalize_scores
from mind_mem.hybrid_recall import (
    DEFAULT_RERANK_DEPTH,
    HybridBackend,
    resolve_rerank_depth,
    rrf_fuse,
)

# ---------------------------------------------------------------------------
# F1 -- the fusion stage owns the scale
# ---------------------------------------------------------------------------


class TestFusionOwnsTheScale:
    def test_fused_score_is_the_rrf_score(self):
        """``score`` is the sort key at every stage exit, fusion included."""
        bm25 = [{"_id": "A", "score": 11.4}, {"_id": "B", "score": 9.1}]
        vec = [{"_id": "B", "score": 0.81}, {"_id": "C", "score": 0.62}]

        fused = rrf_fuse([bm25, vec], weights=[1.0, 1.0], source_names=["bm25", "vector"])

        for hit in fused:
            assert hit["score"] == hit["rrf_score"], f"{hit['_id']} score != rrf_score"

    def test_fused_list_is_sorted_by_score(self):
        bm25 = [{"_id": "A", "score": 11.4}, {"_id": "B", "score": 9.1}]
        vec = [{"_id": "B", "score": 0.81}, {"_id": "C", "score": 0.62}]

        fused = rrf_fuse([bm25, vec], weights=[1.0, 1.0], source_names=["bm25", "vector"])

        scores = [h["score"] for h in fused]
        assert scores == sorted(scores, reverse=True)

    def test_both_legs_raw_scores_survive_fusion(self):
        """The losing leg's raw value is preserved, not destroyed by the copy.

        The fused item is a copy of exactly ONE leg's dict, so before
        ``leg_scores`` the other leg's raw score was unrecoverable. ``B`` is
        in both legs with two very different raw values; both must survive.
        """
        bm25 = [{"_id": "A", "score": 11.4}, {"_id": "B", "score": 9.1}]
        vec = [{"_id": "B", "score": 0.81}, {"_id": "C", "score": 0.62}]

        fused = rrf_fuse([bm25, vec], weights=[1.0, 1.0], source_names=["bm25", "vector"])
        by_id = {h["_id"]: h for h in fused}

        assert by_id["B"]["leg_scores"] == {"bm25": 9.1, "vector": 0.81}
        assert by_id["A"]["leg_scores"] == {"bm25": 11.4}
        assert by_id["C"]["leg_scores"] == {"vector": 0.62}

    def test_leg_scores_are_not_the_fused_score(self):
        """Positive control for the scale mix the whole change is about.

        A BM25F score of 11.4 and an RRF score of ~0.016 are three orders of
        magnitude apart. If ``leg_scores`` were populated from the fused
        column instead of from the legs, this passes silently -- so assert
        the separation explicitly.
        """
        bm25 = [{"_id": "A", "score": 11.4}]
        vec = [{"_id": "A", "score": 0.81}]

        fused = rrf_fuse([bm25, vec], weights=[1.0, 1.0], source_names=["bm25", "vector"])

        assert fused[0]["score"] < 1.0, "fused score should be an RRF score"
        assert fused[0]["leg_scores"]["bm25"] == 11.4
        assert fused[0]["leg_scores"]["bm25"] > 100 * fused[0]["score"]

    def test_missing_leg_score_does_not_raise(self):
        fused = rrf_fuse([[{"_id": "A"}], [{"_id": "A", "score": None}]], weights=[1.0, 1.0], source_names=["bm25", "vector"])
        assert fused[0]["leg_scores"] == {"bm25": 0.0, "vector": 0.0}


# ---------------------------------------------------------------------------
# F1 -- adapters normalise their own output
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_min_max_to_unit_interval(self):
        assert normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]

    def test_degenerate_input_is_all_zero_not_a_divide_by_zero(self):
        assert normalize_scores([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]

    def test_empty(self):
        assert normalize_scores([]) == []

    def test_numpy_vector_does_not_raise(self):
        """The single-model reranker hands this a numpy array from ``predict``.

        ``if not values`` raises ValueError on an ndarray of more than one
        element, so this is the regression test for the truth-test, not a
        style preference.
        """
        np = pytest.importorskip("numpy")
        out = normalize_scores(np.array([-2.5, 3.0, 11.0], dtype=np.float32))
        assert [round(float(v), 6) for v in out] == [0.0, 0.407407, 1.0]


class _FakeSTModel:
    """Stand-in for a sentence-transformers CrossEncoder."""

    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs, **kwargs):
        return list(self._scores)


def _bge_adapter(monkeypatch, scores):
    """Build the real BGE adapter over a fake model."""
    import mind_mem.rerank_ensemble as re_mod

    monkeypatch.setattr(re_mod, "_load_bge", lambda model, device: _FakeSTModel(scores))
    adapter = re_mod._build_bge()
    assert adapter is not None
    return adapter


class TestBlendWeightIsARealConvexWeight:
    """``blend_weight`` must actually weigh two comparable columns.

    The BGE member blended a raw logit against an incoming score, and the
    LLM member blended a 0-100 integer against one. On an RRF column
    (~0.016) either raw column dominates at every weight an operator can
    set, which is what makes the knob a fiction. The check is the same for
    both: at ``blend_weight=0`` the reranker must not move the incoming
    order at all, and at ``blend_weight=1`` it must impose its own -- which
    is only true once both columns are normalised.
    """

    #: Incoming order C > B > A on an RRF-scale column; the reranker's own
    #: opinion is the exact reverse.
    CANDIDATES = [
        {"_id": "C", "score": 0.0181, "content": "c"},
        {"_id": "B", "score": 0.0166, "content": "b"},
        {"_id": "A", "score": 0.0152, "content": "a"},
    ]
    REVERSED_LOGITS = [-8.0, 0.5, 9.25]  # A wins, C loses

    def test_bge_weight_zero_preserves_incoming_order(self, monkeypatch):
        adapter = _bge_adapter(monkeypatch, self.REVERSED_LOGITS)
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.0)
        assert [h["_id"] for h in out] == ["C", "B", "A"]

    def test_bge_weight_one_imposes_reranker_order(self, monkeypatch):
        adapter = _bge_adapter(monkeypatch, self.REVERSED_LOGITS)
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=1.0)
        assert [h["_id"] for h in out] == ["A", "B", "C"]

    def test_bge_keeps_the_raw_logit(self, monkeypatch):
        """Normalising must not cost the operator the raw number."""
        adapter = _bge_adapter(monkeypatch, self.REVERSED_LOGITS)
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)
        assert sorted(h["_bge_score"] for h in out) == sorted(self.REVERSED_LOGITS)

    def test_bge_blended_score_stays_in_unit_interval(self, monkeypatch):
        adapter = _bge_adapter(monkeypatch, self.REVERSED_LOGITS)
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)
        for h in out:
            assert 0.0 <= h["score"] <= 1.0, h

    def test_bge_output_is_invariant_to_the_logit_scale(self, monkeypatch):
        """The sharpest statement of "the adapter normalises its OWN output".

        Multiply every logit by 1000. A normalised member is unmoved: the
        column carries a ranking, not a magnitude. An un-normalised member
        blends the magnitude straight into the sum, so scaling it changes
        both the scores and, against any real RRF column, the order.
        """
        small = _bge_adapter(monkeypatch, [-8.0, 0.5, 9.25])
        out_small = small.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)
        big = _bge_adapter(monkeypatch, [-8000.0, 500.0, 9250.0])
        out_big = big.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)

        assert [h["_id"] for h in out_small] == [h["_id"] for h in out_big]
        assert [round(h["score"], 9) for h in out_small] == [round(h["score"], 9) for h in out_big]

    def test_bge_empty_candidates(self, monkeypatch):
        adapter = _bge_adapter(monkeypatch, [])
        assert adapter.rerank("q", [], top_k=10) == []


class TestLLMMemberNormalisation:
    """Same contract for the 0-100 LLM judge."""

    CANDIDATES = [
        {"_id": "C", "score": 0.0181, "content": "c"},
        {"_id": "B", "score": 0.0166, "content": "b"},
        {"_id": "A", "score": 0.0152, "content": "a"},
    ]

    @staticmethod
    def _adapter(monkeypatch, verdict):
        import json
        import urllib.request

        import mind_mem.rerank_ensemble as re_mod

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                body = {"choices": [{"message": {"content": json.dumps(verdict)}}]}
                return json.dumps(body).encode("utf-8")

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp())
        adapter = re_mod._build_llm({"base_url": "http://127.0.0.1:9/v1/chat/completions", "model": "m"})
        assert adapter is not None
        return adapter

    def test_weight_zero_preserves_incoming_order(self, monkeypatch):
        adapter = self._adapter(monkeypatch, {"A": 100, "B": 50, "C": 0})
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.0)
        assert [h["_id"] for h in out] == ["C", "B", "A"]

    def test_weight_one_imposes_judge_order(self, monkeypatch):
        adapter = self._adapter(monkeypatch, {"A": 100, "B": 50, "C": 0})
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=1.0)
        assert [h["_id"] for h in out] == ["A", "B", "C"]

    def test_keeps_the_raw_0_100_value(self, monkeypatch):
        adapter = self._adapter(monkeypatch, {"A": 100, "B": 50, "C": 0})
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)
        assert {h["_id"]: h["_llm_rerank_score"] for h in out} == {"A": 100.0, "B": 50.0, "C": 0.0}

    def test_blended_score_stays_in_unit_interval(self, monkeypatch):
        adapter = self._adapter(monkeypatch, {"A": 100, "B": 50, "C": 0})
        out = adapter.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)
        for h in out:
            assert 0.0 <= h["score"] <= 1.0, h

    def test_output_is_invariant_to_the_judge_scale(self, monkeypatch):
        """A judge that answers 0/5/10 must rank exactly like one answering
        0/50/100 -- the column is a ranking, not a magnitude."""
        wide = self._adapter(monkeypatch, {"A": 100, "B": 50, "C": 0})
        out_wide = wide.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)
        narrow = self._adapter(monkeypatch, {"A": 10, "B": 5, "C": 0})
        out_narrow = narrow.rerank("q", [dict(c) for c in self.CANDIDATES], top_k=10, blend_weight=0.6)

        assert [h["_id"] for h in out_wide] == [h["_id"] for h in out_narrow]
        assert [round(h["score"], 9) for h in out_wide] == [round(h["score"], 9) for h in out_narrow]


# ---------------------------------------------------------------------------
# F1 -- the explain record
# ---------------------------------------------------------------------------


class TestExplainReadsTheLegs:
    def test_bm25_field_is_the_raw_leg_score_not_the_fused_one(self):
        from mind_mem._recall_explain import attach_explain

        hits = [{"_id": "A", "score": 0.0166, "fusion": "rrf", "leg_scores": {"bm25": 11.4, "vector": 0.81}}]
        attach_explain(hits)

        assert hits[0]["_explain"]["bm25"] == 11.4
        assert hits[0]["_explain"]["vector"] == 0.81
        assert hits[0]["_explain"]["final"] == 0.0166

    def test_bm25_only_path_still_reports_score_as_bm25(self):
        from mind_mem._recall_explain import attach_explain

        hits = [{"_id": "A", "score": 7.25}]
        attach_explain(hits)
        assert hits[0]["_explain"]["bm25"] == 7.25

    def test_vector_stays_none_when_the_leg_did_not_contribute(self):
        from mind_mem._recall_explain import attach_explain

        hits = [{"_id": "A", "score": 0.0166, "fusion": "rrf", "leg_scores": {"bm25": 11.4}}]
        attach_explain(hits)
        assert hits[0]["_explain"]["vector"] is None

    def test_final_tracks_score_after_a_rerank_rewrote_it(self):
        """``final`` must not report the pre-rerank number.

        ``rrf_score`` is what fusion produced; a reranker rewrites ``score``
        and leaves ``rrf_score`` behind. Reading the stale field reported a
        number that had not ordered anything since fusion.
        """
        from mind_mem._recall_explain import attach_explain

        hits = [{"_id": "A", "score": 0.93, "rrf_score": 0.0166, "fusion": "rrf"}]
        attach_explain(hits)
        assert hits[0]["_explain"]["final"] == 0.93


class TestNonIncreasingInvariant:
    def test_raises_when_the_served_list_is_out_of_order(self):
        """The invariant this replaced compared a field to itself and could
        not fail. This one can."""
        from mind_mem._recall_explain import attach_explain

        hits = [{"_id": "A", "score": 0.10}, {"_id": "B", "score": 0.90}]
        with pytest.raises(RuntimeError, match="non-increasing"):
            attach_explain(hits)

    def test_names_both_offending_hits(self):
        from mind_mem._recall_explain import attach_explain

        with pytest.raises(RuntimeError) as exc:
            attach_explain([{"_id": "A", "score": 0.10}, {"_id": "B", "score": 0.90}])
        assert "A" in str(exc.value) and "B" in str(exc.value)

    def test_equal_scores_are_not_a_violation(self):
        from mind_mem._recall_explain import attach_explain

        attach_explain([{"_id": "A", "score": 0.5}, {"_id": "B", "score": 0.5}])

    @pytest.mark.parametrize("marker", ["_graph_hop", "_kg_hop", "_prefetch"])
    def test_appended_expansion_hits_are_outside_the_ordering_claim(self, marker):
        """Graph / KG / prefetch APPEND after the ranked list by design.

        A 1-hop neighbour of the top hit can legitimately outscore the last
        ranked hit, so it must not fire the ordering alarm.
        """
        from mind_mem._recall_explain import attach_explain

        hits = [{"_id": "A", "score": 0.90}, {"_id": "B", "score": 0.20}, {"_id": "N", "score": 0.45, marker: 1}]
        attach_explain(hits)
        assert hits[2]["_explain"]["final"] == 0.45

    def test_ranked_hits_are_still_checked_around_an_appended_one(self):
        """Excluding appended hits must not blind the check to ranked ones."""
        from mind_mem._recall_explain import attach_explain

        hits = [
            {"_id": "A", "score": 0.20},
            {"_id": "N", "score": 0.99, "_graph_hop": 1},
            {"_id": "B", "score": 0.90},
        ]
        with pytest.raises(RuntimeError, match="non-increasing"):
            attach_explain(hits)


# ---------------------------------------------------------------------------
# F2 -- rerank depth
# ---------------------------------------------------------------------------


class TestResolveRerankDepth:
    @pytest.mark.parametrize(
        "limit,expected",
        [(1, 5), (3, 15), (10, DEFAULT_RERANK_DEPTH), (20, DEFAULT_RERANK_DEPTH)],
    )
    def test_default_is_min_50_5x_limit(self, limit, expected):
        assert resolve_rerank_depth({}, limit) == expected

    def test_config_override(self):
        assert resolve_rerank_depth({"rerank_depth": 120}, 10) == 120

    def test_capped_at_max_rerank_candidates(self):
        from mind_mem._recall_constants import MAX_RERANK_CANDIDATES

        assert resolve_rerank_depth({"rerank_depth": 100_000}, 10) == MAX_RERANK_CANDIDATES

    def test_never_below_the_requested_limit(self):
        """A depth under ``limit`` would truncate the response."""
        assert resolve_rerank_depth({"rerank_depth": 2}, 10) == 10

    def test_floor_outranks_the_cap_for_a_huge_limit(self):
        from mind_mem._recall_constants import MAX_RERANK_CANDIDATES

        assert resolve_rerank_depth({}, 500) == 500 > MAX_RERANK_CANDIDATES

    @pytest.mark.parametrize("bad", ["not-a-number", None, -5, 0, [1]])
    def test_malformed_values_fall_back_to_the_default(self, bad):
        assert resolve_rerank_depth({"rerank_depth": bad}, 10) == DEFAULT_RERANK_DEPTH


class _RecordingReranker:
    """Records how many candidates the pipeline actually handed a reranker."""

    def __init__(self, order=None):
        self.seen: list[int] = []
        self._order = order

    def rerank(self, query, candidates, *, top_k=10, blend_weight=0.6):
        self.seen.append(len(candidates))
        out = list(candidates)
        if self._order is not None:
            rank = {bid: i for i, bid in enumerate(self._order)}
            out.sort(key=lambda c: rank.get(str(c.get("_id")), 10_000))
        return out[:top_k]


def _leg_hits(prefix: str, n: int, base: float) -> list[dict]:
    return [{"_id": f"{prefix}-{i:03d}", "score": base - i * 0.001, "excerpt": f"text {i}"} for i in range(n)]


@pytest.fixture
def wired(monkeypatch):
    """A HybridBackend with both legs and the reranker stubbed.

    Returns ``(backend, calls, reranker)`` where ``calls`` records the
    ``limit`` each leg was asked for -- the number that makes ``rerank_depth``
    real rather than nominal.
    """

    def _build(config, reranker=None, leg_size=200):
        hb = HybridBackend(config=config)
        hb._vector_available = True
        calls: dict[str, list[int]] = {"bm25": [], "vector": []}

        def fake_bm25(self, query, workspace, limit=10, **kwargs):
            calls["bm25"].append(limit)
            return _leg_hits("BM", min(limit, leg_size), 11.0)

        def fake_vec(self, query, workspace, limit=10, **kwargs):
            calls["vector"].append(limit)
            return _leg_hits("BM", min(limit, leg_size), 0.9)[::-1]

        monkeypatch.setattr(HybridBackend, "_bm25_search", fake_bm25)
        monkeypatch.setattr(HybridBackend, "_vector_search", fake_vec)
        monkeypatch.setattr(HybridBackend, "_admit", lambda self, hits, ws, **kw: hits)
        monkeypatch.setattr("mind_mem.hybrid_recall.live_statuses", lambda ws: {})

        import mind_mem.cross_encoder_reranker as ce_mod
        import mind_mem.rerank_ensemble as re_mod

        monkeypatch.setattr(re_mod, "create_ensemble", lambda cfg: reranker)
        # No real model weights in a unit test: the single-model fallback is
        # unavailable, so the only reranker that can run is the recorder.
        monkeypatch.setattr(ce_mod.CrossEncoderReranker, "is_available", staticmethod(lambda: False))
        return hb, calls

    return _build


#: ``dedup`` is switched off in these fixtures ONLY because the synthetic
#: hits below share one Type and the type cap would collapse them to 3,
#: masking the depth/limit arithmetic under test. Dedup itself is covered
#: by its own suite and stays on by default in the product.
_NO_DEDUP = {"dedup": {"enabled": False}}
#: The fixture wires an ensemble member, so the config says so: the
#: reranker-active probe reads ``retrieval.reranker_ensemble.enabled``
#: rather than building the ensemble to find out.
CE_ON = {
    "vector_enabled": True,
    "cross_encoder": {"enabled": True},
    "retrieval": {"reranker_ensemble": {"enabled": True}},
    **_NO_DEDUP,
}
CE_OFF = {"vector_enabled": True, "cross_encoder": {"enabled": False, "auto_enable": False}, **_NO_DEDUP}


class TestLegsActuallyFetchTheDepth:
    def test_every_leg_is_asked_for_at_least_the_rerank_depth(self, wired):
        """A depth the legs never filled is a depth that does not exist.

        This is the assertion that keeps ``rerank_depth`` from being
        decorative: it reads the ``limit`` each leg was actually called with.
        """
        reranker = _RecordingReranker()
        hb, calls = wired({**CE_ON, "rerank_depth": 150}, reranker=reranker)

        hb.search("q", "/ws", limit=10, retrieve_wide_k=20)

        assert calls["bm25"] == [150], calls
        assert calls["vector"] == [150], calls

    def test_the_reranker_sees_the_depth_not_the_limit(self, wired):
        reranker = _RecordingReranker()
        hb, _ = wired({**CE_ON, "rerank_depth": 60}, reranker=reranker)

        hb.search("q", "/ws", limit=10, retrieve_wide_k=20)

        assert reranker.seen == [60], reranker.seen

    def test_reranking_can_now_change_which_blocks_are_served(self, wired):
        """The defect, stated as a behaviour.

        A reranker can only promote a block into the response if it was
        given a pool WIDER than the response. Slicing ``fused[:limit]``
        before the reranker ran made that impossible by construction.

        The target block is DERIVED, not hard-coded: the first run
        establishes what the pipeline serves without a reranker, and the
        target is chosen from outside that set. A hard-coded id silently
        stops testing anything the day the fused order shifts -- which is
        exactly what happened to the first draft of this test, where the
        chosen id turned out to sit at fused position 8, inside the top 10.
        """
        hb_off, _ = wired(CE_OFF, reranker=None)
        baseline = [h["_id"] for h in hb_off.search("q", "/ws", limit=10, retrieve_wide_k=20)]

        # Deep in the fused pool: present in the candidates, absent from the
        # response. Asserted, not assumed.
        deep = [f"BM-{i:03d}" for i in range(50) if f"BM-{i:03d}" not in baseline]
        assert deep, "no candidate outside the served set — the fixture is not exercising depth"
        target = deep[len(deep) // 2]
        assert target not in baseline

        reranker = _RecordingReranker(order=[target])
        hb, _ = wired({**CE_ON, "rerank_depth": 50}, reranker=reranker)
        served = [h["_id"] for h in hb.search("q", "/ws", limit=10, retrieve_wide_k=20)]

        assert served[0] == target, f"{target} was not promoted into the response: {served}"

    def test_response_is_still_cut_to_limit(self, wired):
        reranker = _RecordingReranker()
        hb, _ = wired({**CE_ON, "rerank_depth": 50}, reranker=reranker)

        served = hb.search("q", "/ws", limit=10, retrieve_wide_k=20)

        assert len(served) == 10


class TestRerankerOffChangesNothing:
    def test_legs_are_not_widened_when_no_reranker_will_run(self, wired):
        """Fetching more candidates changes what RRF sees.

        Widening unconditionally would move the fused order of a request
        that never asked for a cross-encoder, which is precisely the
        default-path movement a patch release may not make.
        """
        hb, calls = wired({**CE_OFF, "rerank_depth": 150}, reranker=None)

        hb.search("q", "/ws", limit=10, retrieve_wide_k=20)

        assert calls["bm25"] == [20], calls
        assert calls["vector"] == [20], calls

    def test_served_list_is_the_fused_prefix(self, wired):
        hb, _ = wired(CE_OFF, reranker=None)

        served = hb.search("q", "/ws", limit=10, retrieve_wide_k=20)

        assert len(served) == 10
        scores = [h["score"] for h in served]
        assert scores == sorted(scores, reverse=True)


class TestRerankerFailureNeverWidensTheResponse:
    def test_a_failing_reranker_still_returns_at_most_limit(self, wired):
        class _Boom:
            def rerank(self, *a, **kw):
                raise RuntimeError("reranker down")

        hb, _ = wired({**CE_ON, "rerank_depth": 50}, reranker=_Boom())

        served = hb.search("q", "/ws", limit=10, retrieve_wide_k=20)

        assert len(served) == 10, "a dead reranker must not leak the wide candidate pool"

    def test_bm25_only_path_does_not_leak_the_pool_either(self, monkeypatch):
        """The BM25-only branch has no final ``[:limit]`` slice of its own."""
        hb = HybridBackend(config={**CE_ON, "rerank_depth": 50})
        hb._vector_available = False
        seen: list[int] = []

        def fake_bm25(self, query, workspace, limit=10, **kwargs):
            seen.append(limit)
            return _leg_hits("BM", limit, 11.0)

        monkeypatch.setattr(HybridBackend, "_bm25_search", fake_bm25)
        monkeypatch.setattr(HybridBackend, "_admit", lambda self, hits, ws, **kw: hits)
        monkeypatch.setattr("mind_mem.hybrid_recall.live_statuses", lambda ws: {})

        class _Boom:
            def rerank(self, *a, **kw):
                raise RuntimeError("reranker down")

        import mind_mem.cross_encoder_reranker as ce_mod
        import mind_mem.rerank_ensemble as re_mod

        monkeypatch.setattr(re_mod, "create_ensemble", lambda cfg: _Boom())
        monkeypatch.setattr(ce_mod.CrossEncoderReranker, "is_available", staticmethod(lambda: False))

        served = hb.search("q", "/ws", limit=10)

        assert seen == [50], "the BM25-only leg must also fetch the depth"
        assert len(served) == 10


class TestCrossEncoderActivePredicate:
    def test_off_when_disabled_and_auto_enable_off(self):
        hb = HybridBackend(config=CE_OFF)
        assert hb._cross_encoder_active("multi-hop") is False

    def test_auto_enables_on_multi_hop_when_a_reranker_exists(self, monkeypatch):
        import mind_mem.cross_encoder_reranker as ce_mod

        monkeypatch.setattr(ce_mod.CrossEncoderReranker, "is_available", staticmethod(lambda: True))
        hb = HybridBackend(config={"vector_enabled": True})
        assert hb._cross_encoder_active("multi-hop") is True
        assert hb._cross_encoder_active("factual") is False

    def test_off_when_configured_on_but_nothing_can_run(self, monkeypatch):
        """Widening the legs for a reranker that cannot run is wasted work."""
        import mind_mem.cross_encoder_reranker as ce_mod

        monkeypatch.setattr(ce_mod.CrossEncoderReranker, "is_available", staticmethod(lambda: False))
        hb = HybridBackend(config={"vector_enabled": True, "cross_encoder": {"enabled": True}})
        assert hb._cross_encoder_active("multi-hop") is False

    def test_an_enabled_ensemble_answers_yes_without_being_built(self, monkeypatch):
        """The probe must not construct the feature it is asking about.

        ``create_ensemble`` builds its members and logs on a bad name or an
        all-failed build. A probe whose answer is "no" has to leave no trace,
        so it reads the flag instead -- and this test fails loudly if the
        factory ever creeps back onto the probe path.
        """
        import mind_mem.cross_encoder_reranker as ce_mod
        import mind_mem.rerank_ensemble as re_mod

        def _explode(cfg):
            raise AssertionError("create_ensemble must not be called from the reranker-active probe")

        monkeypatch.setattr(re_mod, "create_ensemble", _explode)
        monkeypatch.setattr(ce_mod.CrossEncoderReranker, "is_available", staticmethod(lambda: False))
        hb = HybridBackend(config={"vector_enabled": True, "retrieval": {"reranker_ensemble": {"enabled": True}}})
        assert hb._cross_encoder_active("multi-hop") is True


# ---------------------------------------------------------------------------
# F2 -- the patch-release property, through the ranking gate
# ---------------------------------------------------------------------------


class TestRerankDepthIsInertWithNoReranker:
    """``rerank_depth`` must not move the served ranking when nothing reranks.

    This is the patch-release claim stated so it can be measured rather than
    argued, and it runs through :mod:`benchmarks.ranking_identity` so the
    comparison is over the served ``(id, score)`` list encoded by
    ``float.hex()`` -- exact bits, not ``==`` on floats.

    The paired positive control below turns the SAME knob with a reranker
    wired in and requires the ranking to move. A gate whose green cannot be
    turned red is not a gate.
    """

    QUERIES = ["fusion scoring", "retrieval depth", "reranking candidates"]

    @staticmethod
    def _battery(wired, config, reranker):
        from benchmarks.ranking_identity import fingerprint_battery

        hb, _ = wired(config, reranker=reranker)
        return fingerprint_battery(
            lambda q: hb.search(q, "/ws", limit=10, retrieve_wide_k=20),
            TestRerankDepthIsInertWithNoReranker.QUERIES,
        )

    @pytest.mark.parametrize("depth", [10, 25, 50, 120, 200])
    def test_no_depth_moves_the_ranking_when_no_reranker_runs(self, wired, depth):
        shallow = self._battery(wired, {**CE_OFF, "rerank_depth": 10}, None)
        probe = self._battery(wired, {**CE_OFF, "rerank_depth": depth}, None)

        from benchmarks.ranking_identity import assert_battery_unchanged

        assert_battery_unchanged(shallow, probe, label=f"reranker-off depth={depth}")

    def test_the_same_knob_does_move_it_once_a_reranker_is_wired(self, wired):
        """Positive control: the gate above can go red.

        Without this, "depth changed nothing" is equally consistent with
        "depth is wired to nothing" -- which is the defect this whole change
        is about.
        """
        from benchmarks.ranking_identity import RankingMoved, assert_battery_unchanged

        order = [f"BM-{i:03d}" for i in range(199, -1, -1)]
        shallow = self._battery(wired, {**CE_ON, "rerank_depth": 10}, _RecordingReranker(order=order))
        deep = self._battery(wired, {**CE_ON, "rerank_depth": 200}, _RecordingReranker(order=order))

        with pytest.raises(RankingMoved):
            assert_battery_unchanged(shallow, deep, label="reranker-on depth sweep")


# ---------------------------------------------------------------------------
# F1 -- the invariant must not misfire on a real served list
# ---------------------------------------------------------------------------


class TestExplainSurvivesTheRealPipeline:
    """The ordering invariant raises, and ``_apply_explain`` swallows.

    That combination means a false alarm would not surface as an error -- it
    would surface as ``_explain`` quietly vanishing from the response. So the
    invariant has to be checked against a real end-to-end recall, on the
    query shape that fires graph expansion (multi-hop), over a corpus with
    real cross-references between blocks.
    """

    @pytest.fixture
    def ws(self, tmp_path):
        import os

        from mind_mem.init_workspace import init

        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        init(workspace)
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(24):
                fh.write(f"[EXPX-{i:03d}]\n")
                fh.write("Type: Decision\n")
                # Cross-references, so the graph walk has edges to append along.
                fh.write(f"Statement: fusion reranking depth scoring block {i} relates to EXPX-{(i + 1) % 24:03d}\n")
                fh.write(f"Date: 2026-0{(i % 9) + 1}-1{i % 10}\n\n")
        return workspace

    @pytest.mark.parametrize(
        "query",
        [
            "fusion scoring",
            "how does reranking relate to fusion depth and why",
        ],
    )
    def test_explain_is_present_on_every_hit(self, ws, query):
        import json

        import mind_mem.mcp.tools.recall as recall_tool
        from mind_mem.mcp.infra.workspace import use_workspace

        with use_workspace(ws):
            envelope = json.loads(recall_tool.recall(query, limit=10, explain=True))

        hits = envelope.get("results") or []
        assert hits, "no hits: the assertion below would be vacuous"
        missing = [h.get("_id") for h in hits if "_explain" not in h]
        assert not missing, f"_explain vanished for {missing} — the ordering invariant likely misfired and was swallowed"

    def test_served_scores_are_non_increasing_over_the_ranked_hits(self, ws):
        import json

        import mind_mem.mcp.tools.recall as recall_tool
        from mind_mem._recall_explain import _is_appended
        from mind_mem.mcp.infra.workspace import use_workspace

        with use_workspace(ws):
            envelope = json.loads(recall_tool.recall("how does reranking relate to fusion depth and why", limit=10))

        hits = [h for h in (envelope.get("results") or []) if not _is_appended(h)]
        assert len(hits) >= 2, "fewer than two ranked hits: the ordering claim would be vacuous"
        scores = [float(h["score"]) for h in hits]
        assert scores == sorted(scores, reverse=True), scores


# ---------------------------------------------------------------------------
# F1 -- the two stages that RE-SORT on ``score``: session boost and temporal
# ---------------------------------------------------------------------------


#: A config that names no ``retrieval`` section. The session-boost gate reads
#: an absent section as an empty one (``implicit_section``), so this is the
#: shape an unconfigured workspace presents -- and the shape under which the
#: stage auto-enables. ``None`` and ``{}`` are a hard OFF and would make every
#: gate test below pass for the wrong reason.
_UNCONFIGURED = {"vector_enabled": True}


def _fused_with_sessions() -> list[dict]:
    """A fused list on the post-F1 contract: ``score`` is the fused scale."""
    rows = [
        ("A1", 0.0330, "S1"),
        ("B1", 0.0320, "S2"),
        ("C1", 0.0310, "S1"),
        ("D1", 0.0250, "S2"),
        ("E1", 0.0240, "S1"),
    ]
    return [{"_id": i, "score": s, "rrf_score": s, "SessionId": sid, "Date": "2026-09-01"} for i, s, sid in rows]


class TestSessionBoostReSortsOnTheFusedScale:
    """``apply_session_boost`` multiplies ``score`` and re-sorts by it.

    Before F1 that column was the surviving leg's raw value, so the stage was
    re-sorting a mixture of an unbounded BM25F number and a ``[0, 1]`` cosine
    -- which is to say it was discarding the fusion ranking it had been
    handed. These name the stage the 5.0.2 entry says the served order moves
    in, which the rest of this file did not.
    """

    def test_the_boost_multiplies_the_fused_score(self):
        from mind_mem.session_boost import apply_session_boost

        out = apply_session_boost(_fused_with_sessions(), top_seed_count=3, boost=0.3)
        by_id = {h["_id"]: h for h in out}
        # A1/C1/E1 are in S1 and B1/D1 in S2; both sessions are seeded from
        # the top three, so every hit is boosted off the SAME column.
        for block_id in ("A1", "B1", "C1", "D1", "E1"):
            assert by_id[block_id]["_session_boost"] == 0.3
            assert by_id[block_id]["score"] == pytest.approx(by_id[block_id]["rrf_score"] * 1.3)

    def test_a_boost_confined_to_one_session_promotes_across_the_seed_line(self):
        """The re-sort has to be able to change the order, or it proves nothing."""
        from mind_mem.session_boost import apply_session_boost

        hits = [
            {"_id": "TOP", "score": 0.0330, "rrf_score": 0.0330, "SessionId": "S1"},
            {"_id": "MID", "score": 0.0300, "rrf_score": 0.0300, "SessionId": "S2"},
            {"_id": "LOW", "score": 0.0290, "rrf_score": 0.0290, "SessionId": "S1"},
        ]
        # Only the head seat seeds, so S1 is active and S2 is not.
        out = apply_session_boost(hits, top_seed_count=1, boost=0.3)
        assert [h["_id"] for h in out] == ["TOP", "LOW", "MID"]
        assert [h["_id"] for h in hits] == ["TOP", "MID", "LOW"], "input list was mutated"

    def test_the_served_order_is_non_increasing_in_the_column_it_sorted_on(self):
        from mind_mem.session_boost import apply_session_boost

        out = apply_session_boost(_fused_with_sessions(), top_seed_count=3, boost=0.3)
        scores = [h["score"] for h in out]
        assert scores == sorted(scores, reverse=True), scores

    def test_the_gate_opens_on_session_carrying_hits_and_stays_shut_otherwise(self):
        from mind_mem.session_boost import is_session_boost_enabled

        # A config with no ``retrieval`` section at all: ``implicit_section``
        # reads that as "the section is present and empty", which is how the
        # stage comes on for a workspace nobody configured for it.
        assert is_session_boost_enabled(_UNCONFIGURED, _fused_with_sessions()) is True
        bare = [{"_id": "A1", "score": 0.03}, {"_id": "B1", "score": 0.02}]
        assert is_session_boost_enabled(_UNCONFIGURED, bare) is False

    @pytest.mark.parametrize("key", ["SessionId", "session_id", "Session"])
    def test_every_documented_session_field_opens_the_gate(self, key):
        from mind_mem.session_boost import is_session_boost_enabled

        assert is_session_boost_enabled(_UNCONFIGURED, [{"_id": "A1", "score": 0.03, key: "S1"}]) is True

    def test_auto_enable_false_is_the_stable_no_boost_order(self):
        """The escape hatch the 5.0.2 entry names.

        ``retrieval.session_boost.auto_enable: false`` shuts the gate on a
        corpus that would otherwise open it, so an operator who wants the
        unboosted fused order can have it. No capability is lost by shipping
        the fix unflagged.
        """
        from mind_mem.session_boost import is_session_boost_enabled

        hits = _fused_with_sessions()
        assert is_session_boost_enabled(_UNCONFIGURED, hits) is True
        assert is_session_boost_enabled({"retrieval": {"session_boost": {"auto_enable": False}}}, hits) is False

    def test_the_hatch_is_reachable_through_the_backend_config(self):
        """Shutting the gate must stop the STAGE, not merely leave the order alone.

        Asserting only that the ids came back in the same order would pass
        even with the gate forced open: when every session is active every
        score is multiplied by the same factor and nothing reorders. So the
        marker the stage stamps is what is checked -- it is present if and
        only if the stage ran.
        """
        backend_off = HybridBackend(config={"retrieval": {"session_boost": {"auto_enable": False}}})
        backend_on = HybridBackend(config=dict(_UNCONFIGURED))
        hits = _fused_with_sessions()

        off = backend_off._maybe_session_boost(list(hits))
        assert [h["_id"] for h in off] == [h["_id"] for h in hits]
        assert [h["score"] for h in off] == [h["score"] for h in hits]
        assert not [h["_id"] for h in off if "_session_boost" in h]

        on = backend_on._maybe_session_boost(list(hits))
        assert [h["_id"] for h in on if "_session_boost" in h], "the gate never opened, so the OFF case proves nothing"


class TestTemporalDecayReSortsOnTheFusedScale:
    """The other stage that multiplies ``score`` and re-sorts by it.

    Opt-in via ``retrieval.temporal_decay_hot_path``, and named here because
    the 5.0.2 entry counts it as a place the served order moves.
    """

    @staticmethod
    def _dated_hits() -> list[dict]:
        return [
            {"_id": "OLD", "score": 0.0330, "rrf_score": 0.0330, "Date": "2024-01-01"},
            {"_id": "NEW", "score": 0.0300, "rrf_score": 0.0300, "Date": "2026-08-30"},
        ]

    def test_off_by_default_the_list_is_returned_untouched(self):
        backend = HybridBackend(config=dict(_UNCONFIGURED))
        hits = self._dated_hits()
        out = backend._maybe_temporal_decay(hits, scoring_instant=date(2026, 9, 1))
        assert out is hits
        assert all("_temporal_decay" not in h for h in out)

    def test_on_it_decays_the_fused_score_and_re_sorts_by_it(self):
        backend = HybridBackend(config={"retrieval": {"temporal_decay_hot_path": True}})
        hits = self._dated_hits()
        out = backend._maybe_temporal_decay(hits, scoring_instant=date(2026, 9, 1))
        assert [h["_id"] for h in out] == ["NEW", "OLD"], "the older, higher-fused hit was not overtaken"
        for hit in out:
            # ``_temporal_decay`` is the multiplier rounded to four places, so
            # the exact relation is checked the other way round: the ratio the
            # stage actually applied, rounded, is what it recorded.
            ratio = hit["score"] / hit["rrf_score"]
            assert 0.0 < ratio <= 1.0, ratio
            assert round(ratio, 4) == hit["_temporal_decay"]
        scores = [h["score"] for h in out]
        assert scores == sorted(scores, reverse=True), scores

    def test_the_input_list_is_not_mutated(self):
        backend = HybridBackend(config={"retrieval": {"temporal_decay_hot_path": True}})
        hits = self._dated_hits()
        backend._maybe_temporal_decay(hits, scoring_instant=date(2026, 9, 1))
        assert [h["score"] for h in hits] == [0.0330, 0.0300]


# ---------------------------------------------------------------------------
# F1 -- the committed artifacts, not a sentence about them
# ---------------------------------------------------------------------------

_EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "5.0.2-f1"
_BEFORE = _EVIDENCE / "score-contract-before-6cd37e5.json"
_AFTER = _EVIDENCE / "score-contract-after-5.0.2.json"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class TestCommittedScoreContractArtifact:
    """The 5.0.2 ranking claims, re-derived from the committed artifacts.

    ``benchmarks/f1_score_contract_probe.py`` captured the served lists of a
    pre-F1 tree (``6cd37e5``) and this one; these read the two artifacts back
    and re-run the comparison, so the CHANGELOG's numbers are checked against
    a file in the tree rather than against a report nobody can re-open.
    """

    def test_the_two_artifacts_came_from_different_trees(self):
        """Otherwise every comparison below is one tree compared with itself."""
        before, after = _load(_BEFORE), _load(_AFTER)
        assert before["provenance"]["hybrid_recall_sha256"] != after["provenance"]["hybrid_recall_sha256"]

    def test_the_default_fused_path_did_not_move(self):
        from benchmarks.f1_evidence_report import build_report

        report = build_report(_load(_BEFORE), _load(_AFTER))
        assert report["default_path"]["identical"] is True
        assert report["default_path"]["cases"] >= 80
        assert report["default_path"]["hits"] >= 1000

    def test_the_identity_assertion_can_fail(self):
        """Swap two ids in one default-path case; the gate must go red."""
        from benchmarks.f1_evidence_report import build_report
        from benchmarks.ranking_identity import RankingMoved

        after = _load(_AFTER)
        key = next(k for k in after["cases"] if k.startswith("deep|"))
        rows = after["cases"][key]
        rows[0], rows[1] = rows[1], rows[0]
        with pytest.raises(RankingMoved):
            build_report(_load(_BEFORE), after)

    def test_comparing_one_tree_with_itself_is_refused(self):
        from benchmarks.f1_evidence_report import SameTreeTwice, build_report

        after = _load(_AFTER)
        with pytest.raises(SameTreeTwice):
            build_report(after, _load(_AFTER))

    def test_the_pre_fix_column_did_not_order_24_of_the_45_served_lists(self):
        """The number the CHANGELOG quotes, re-derived rather than recalled.

        27 of the 45 carry two or more hits, so 27 is the count that could
        have failed and 24 of those did.
        """
        from benchmarks.f1_evidence_report import monotonicity
        from benchmarks.f1_score_contract_probe import LEGACY_PREFIXES

        stats = monotonicity(_load(_BEFORE)["cases"], LEGACY_PREFIXES)
        assert stats["lists"] == 45
        assert stats["orderable_lists"] == 27
        assert stats["not_non_increasing"] == 24

    def test_the_post_fix_column_orders_every_served_list(self):
        from benchmarks.f1_evidence_report import monotonicity
        from benchmarks.f1_score_contract_probe import LEGACY_PREFIXES

        stats = monotonicity(_load(_AFTER)["cases"], LEGACY_PREFIXES)
        # 27 before, more after: several session-shaped cases served one hit
        # under the mixed column and three under the fused one, so a list that
        # could not be out of order became one that could.
        assert stats["orderable_lists"] >= 27, "nothing orderable: the count below would be vacuous"
        assert stats["not_non_increasing"] == 0, stats["violating_cases"]

    def test_the_monotonicity_count_can_be_non_zero(self):
        """A counter that cannot count is not evidence that the count is 0."""
        from benchmarks.f1_evidence_report import monotonicity
        from benchmarks.f1_score_contract_probe import LEGACY_PREFIXES

        after = _load(_AFTER)
        key = next(k for k in after["cases"] if k.startswith("stage|") and len(after["cases"][k]) >= 2)
        after["cases"][key][0][1] = -1.0
        assert monotonicity(after["cases"], LEGACY_PREFIXES)["not_non_increasing"] == 1

    def test_every_case_that_moved_is_a_case_a_re_sorting_stage_reached(self):
        """The order moves only where ``session_boost`` or temporal decay ran."""
        from benchmarks.f1_evidence_report import build_report

        report = build_report(_load(_BEFORE), _load(_AFTER))
        moved = [entry["case"] for entry in report["served_order_moved"]["detail"]]
        assert moved, "nothing moved: the claim below would be vacuous"
        for case in moved:
            assert "session_boost" in case or "temporal_decay" in case, case


class TestCommittedSessionScorecard:
    """F1 is non-inferior where the order moves, measured and committed.

    Three scorecards over a session-shaped corpus, one per leg-correlation
    setting, each a paired comparison of 400 questions. The assertion is
    non-inferiority: no metric, at any setting, comes back ``baseline_better``.
    """

    @staticmethod
    def _cards() -> list[dict]:
        cards = [_load(path) for path in sorted(_EVIDENCE.glob("session-scorecard-sn*.json"))]
        assert len(cards) == 3, [p.name for p in sorted(_EVIDENCE.glob("session-scorecard-sn*.json"))]
        return cards

    def test_every_scorecard_paired_the_full_question_set(self):
        for card in self._cards():
            assert card["n_pairs"] == 400, card["candidate_label"]
            assert card["dropped_non_ok"] == []

    def test_no_metric_at_any_setting_favours_the_pre_fix_tree(self):
        offenders = [
            (card["candidate_label"], comparison["label"], comparison["note"])
            for card in self._cards()
            for comparison in card["comparisons"]
            if comparison["verdict"] == "baseline_better"
        ]
        assert not offenders, offenders

    def test_both_discordant_directions_are_recorded(self):
        """A net would let 4-vs-18 and 11-vs-11 print the same headline."""
        for card in self._cards():
            for comparison in card["comparisons"]:
                assert "candidate_only" in comparison and "baseline_only" in comparison
                assert comparison["n_discordant"] == comparison["candidate_only"] + comparison["baseline_only"]

    def test_at_least_one_setting_moved_enough_questions_to_be_testable(self):
        """All-concordant scorecards would make the non-inferiority claim vacuous."""
        testable = [
            comparison["label"]
            for card in self._cards()
            for comparison in card["comparisons"]
            if comparison["n_discordant"] >= comparison["min_discordant_for_significance"]
        ]
        assert testable, "every comparison was underpowered or concordant; nothing was actually tested"
