"""The anticipation cache, proven to retire with the ledger head rather than a clock.

Group J's cache is the one place in the tree where a *stale* answer would be
served on purpose, so the tests here are mostly about the thing that makes that
impossible: the generation key is the governed-ledger head, and every admitted
write moves it. Each negative assertion below is paired with the positive control that
proves the method could have seen the thing it says is absent — a "not served"
that passes because nothing was ever stored is the standard way this class of
test proves nothing.
"""

from __future__ import annotations

import json
import os

import pytest
from _recall_clock_sentinel import clock_census

from mind_mem import prefetch
from mind_mem.novel_term_gate import (
    REASON_CORPUS_BELOW_FLOOR,
    REASON_KNOWN_TERMS,
    REASON_NO_QUERY_STEMS,
    REASON_NOVEL_RATIO_EXCEEDED,
    NovelTermGateConfig,
)
from mind_mem.recall_attestation import GENESIS_ANCHOR

# A bundle big enough to clear the shipped ``min_corpus_stems`` floor of 200,
# so the default config is exercised rather than a loosened one. Every stem is
# distinct and deterministic.
_FILLER_STEMS = " ".join(f"topic{n}" for n in range(260))


def _hits() -> list[dict]:
    return [
        {
            "_id": "D-1",
            "Statement": "the deterministic compiler emits byte identical artifacts on every substrate",
            "Tags": "compiler determinism",
        },
        {
            "_id": "D-2",
            "Statement": "recall scoring takes the scoring instant as an input and reads no clock",
            "Tags": "recall determinism scoring",
        },
        {
            "_id": "D-3",
            "Statement": f"unrelated background material {_FILLER_STEMS}",
            "Tags": "filler",
        },
    ]


@pytest.fixture(autouse=True)
def _fresh_cache():
    prefetch.reset_cache()
    yield
    prefetch.reset_cache()


def _append_to_governed_ledger(workspace: str, block_id: str, action: str, content: str) -> None:
    """Move the workspace's generation key the way a governed write does.

    ``memory/hash_chain_v2.db`` is the ledger the governance gate appends to on
    every admitted write and every admitted delete, and it is what
    :func:`mind_mem.recall_attestation._resolve_index_anchor` reads. Tests move
    the head through the same ledger rather than through some other file, so a
    passing test means the head moves on the events the product moves it on.
    """
    from mind_mem.hash_chain_v2 import HashChainV2
    from mind_mem.recall_attestation import index_anchor_ledger_path

    HashChainV2(index_anchor_ledger_path(workspace)).append(block_id, action, content)


@pytest.fixture()
def chained_workspace(tmp_path):
    """A workspace with a real governed ledger, so the head is a real value."""
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    _append_to_governed_ledger(ws, "D-1", "create", "seed")
    return ws


# ---------------------------------------------------------------------------
# The generation key
# ---------------------------------------------------------------------------


class TestChainHeadIsTheGeneration:
    def test_a_bundle_is_served_at_the_head_it_was_recorded_at(self) -> None:
        """POSITIVE CONTROL for every "not served" assertion below.

        Without this, a test asserting a bundle is unavailable after a head
        move would pass just as well if ``record`` had silently stored nothing.
        """
        cache = prefetch.AnticipationCache()
        assert cache.record("/ws", "recall", _hits(), head="head-A") is not None

        decision = cache.lookup("/ws", "recall determinism scoring", head="head-A")

        assert decision.serve_from_cache is True
        assert decision.reason == REASON_KNOWN_TERMS
        assert [h["_id"] for h in decision.served][0] == "D-2"

    def test_a_head_move_makes_the_bundle_unresolvable(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="head-A")

        decision = cache.lookup("/ws", "recall determinism scoring", head="head-B")

        assert decision.serve_from_cache is False
        assert decision.reason == prefetch.REASON_COLD
        assert decision.document_count == 0

    def test_the_retired_generation_is_dropped_wholesale_not_pruned(self) -> None:
        """A head move must not leave one bundle behind because a filter missed it."""
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="head-A")
        cache.record("/ws", "prefetch", _hits(), head="head-A")
        assert cache.stats()["bundles"] == 2

        cache.lookup("/ws", "anything", head="head-B")

        assert cache.stats()["bundles"] == 0
        assert cache.stats()["retired_bundles"] == 2

    def test_returning_to_an_old_head_does_not_resurrect_its_bundles(self) -> None:
        """Retirement is destruction, not a filter that could be un-applied."""
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="head-A")
        cache.lookup("/ws", "recall determinism scoring", head="head-B")

        back = cache.lookup("/ws", "recall determinism scoring", head="head-A")

        assert back.serve_from_cache is False
        assert back.reason == prefetch.REASON_COLD

    def test_workspaces_do_not_share_a_generation(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws-a", "recall", _hits(), head="head-A")

        assert cache.lookup("/ws-a", "recall determinism scoring", head="head-A").serve_from_cache is True
        assert cache.lookup("/ws-b", "recall determinism scoring", head="head-A").serve_from_cache is False


class TestChainHeadResolution:
    def test_head_is_the_value_the_attestation_binds(self, chained_workspace) -> None:
        from mind_mem.recall_attestation import _resolve_index_anchor

        assert prefetch.chain_head(chained_workspace) == _resolve_index_anchor(chained_workspace)
        assert prefetch.chain_head(chained_workspace) != GENESIS_ANCHOR

    def test_a_governed_append_moves_the_head(self, chained_workspace) -> None:
        before = prefetch.chain_head(chained_workspace)
        _append_to_governed_ledger(chained_workspace, "D-1", "update", "second")

        assert prefetch.chain_head(chained_workspace) != before

    def test_the_head_is_re_read_rather_than_remembered(self, chained_workspace) -> None:
        """No memo: the second read must reflect a write the first could not see.

        The rejected optimisation is a memo keyed on an ``os.stat`` of the
        ledger file. See :func:`mind_mem.prefetch.chain_head` for why it is not
        taken — it would be correct only for as long as
        :class:`~mind_mem.hash_chain_v2.HashChainV2` keeps opening and closing a
        connection per append. This pins the behaviour the removal buys: the
        first call primes nothing, so the second sees the append. It is also the
        positive control for the read itself — a resolver that always returned
        the genesis sentinel would fail the last assertion.
        """
        first = prefetch.chain_head(chained_workspace)
        assert prefetch.chain_head(chained_workspace) == first, "the head moved with no write in between"
        _append_to_governed_ledger(chained_workspace, "D-1", "update", "third")

        assert prefetch.chain_head(chained_workspace) != first
        assert prefetch.chain_head(chained_workspace) != GENESIS_ANCHOR

    def test_absent_ledger_resolves_to_the_genesis_anchor(self, tmp_path) -> None:
        assert prefetch.chain_head(str(tmp_path)) == GENESIS_ANCHOR

    def test_no_workspace_resolves_to_the_genesis_anchor(self) -> None:
        assert prefetch.chain_head("") == GENESIS_ANCHOR


# ---------------------------------------------------------------------------
# The local-vs-source decision
# ---------------------------------------------------------------------------


class TestNovelTermGateDecidesTheServe:
    def test_a_query_of_known_terms_is_served(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")

        decision = cache.lookup("/ws", "deterministic compiler artifacts", head="H")

        assert decision.serve_from_cache is True
        assert decision.reason == REASON_KNOWN_TERMS

    def test_a_query_the_bundles_cannot_answer_falls_through(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")

        decision = cache.lookup("/ws", "kubernetes ingress webhook certificate rotation", head="H")

        assert decision.serve_from_cache is False
        assert decision.reason == REASON_NOVEL_RATIO_EXCEEDED
        assert decision.served == ()

    def test_a_cold_corpus_below_the_floor_falls_through(self) -> None:
        """POSITIVE CONTROL: the same bundle serves once the floor is met.

        Proves the fall-through is the floor talking and not an empty cache.
        """
        thin = [{"_id": "D-9", "Statement": "compiler determinism", "Tags": "x"}]
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", thin, head="H")

        below = cache.lookup("/ws", "compiler determinism", head="H")
        assert below.serve_from_cache is False
        assert below.reason == REASON_CORPUS_BELOW_FLOOR

        met = cache.lookup(
            "/ws",
            "compiler determinism",
            head="H",
            gate_config=NovelTermGateConfig(min_corpus_stems=1),
        )
        assert met.serve_from_cache is True

    def test_an_unjudgeable_query_falls_through(self) -> None:
        """POSITIVE CONTROL inline: the same loaded cache does serve a real query.

        Without it this passes on an empty cache, which is the ordinary way a
        "falls through" assertion ends up proving nothing.
        """
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")
        assert cache.lookup("/ws", "deterministic compiler artifacts", head="H").serve_from_cache is True

        unjudgeable = cache.lookup("/ws", "!!! ???", head="H")

        assert unjudgeable.serve_from_cache is False
        assert unjudgeable.reason == REASON_NO_QUERY_STEMS


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRankingAnswersTheQueryAsked:
    def test_the_order_follows_the_query_not_the_insertion_order(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")

        compiler_first = cache.lookup("/ws", "deterministic compiler artifacts substrate", head="H")
        scoring_first = cache.lookup("/ws", "recall scoring instant clock", head="H")

        assert compiler_first.served[0]["_id"] == "D-1"
        assert scoring_first.served[0]["_id"] == "D-2"

    def test_every_served_hit_names_where_it_came_from(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")

        decision = cache.lookup("/ws", "deterministic compiler artifacts", head="H")

        assert decision.served
        for hit in decision.served:
            assert hit["_retrieval_source"] == prefetch.RETRIEVAL_SOURCE

    def test_the_limit_is_honoured(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")

        decision = cache.lookup("/ws", "determinism", head="H", limit=1)

        assert len(decision.served) == 1

    def test_identical_inputs_give_an_identical_ranking(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")

        first = cache.lookup("/ws", "determinism scoring compiler", head="H")
        second = cache.lookup("/ws", "determinism scoring compiler", head="H")

        assert [(h["_id"], h["_score"]) for h in first.served] == [(h["_id"], h["_score"]) for h in second.served]

    def test_serving_does_not_hand_out_the_cached_dict(self) -> None:
        """A caller mutating a served hit must not corrupt the bundle."""
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")

        served = cache.lookup("/ws", "deterministic compiler", head="H").served[0]
        served["Statement"] = "mutated"

        again = cache.lookup("/ws", "deterministic compiler", head="H").served[0]
        assert again["Statement"] != "mutated"

    def test_a_bundle_with_no_scorable_text_is_not_recorded(self) -> None:
        cache = prefetch.AnticipationCache()

        assert cache.record("/ws", "recall", [{"_id": "D-0"}], head="H") is None
        assert cache.stats()["bundles"] == 0


# ---------------------------------------------------------------------------
# Bounds and configuration
# ---------------------------------------------------------------------------


class TestBoundsAndConfig:
    def test_a_producer_replaces_its_own_bundle_rather_than_accumulating(self) -> None:
        cache = prefetch.AnticipationCache()
        cache.record("/ws", "recall", _hits(), head="H")
        cache.record("/ws", "recall", _hits(), head="H")

        assert cache.stats()["bundles"] == 1

    def test_the_generation_is_bounded(self) -> None:
        cache = prefetch.AnticipationCache(max_bundles=2)
        for n in range(5):
            cache.record("/ws", f"origin-{n}", _hits(), head="H")

        assert cache.stats()["bundles"] == 2

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"cache": None},
            {"cache": {}},
            {"cache": {"anticipation": None}},
            {"cache": {"anticipation": {}}},
            {"cache": {"anticipation": {"enabled": "yes"}}},
            {"cache": {"anticipation": {"enabled": 1}}},
        ],
    )
    def test_the_flag_reads_off_unless_it_is_literally_true(self, config) -> None:
        assert prefetch.anticipation_enabled(config) is False

    def test_the_flag_reads_on_when_set(self) -> None:
        assert prefetch.anticipation_enabled({"cache": {"anticipation": {"enabled": True}}}) is True

    def test_thresholds_come_from_config(self) -> None:
        cfg = prefetch.anticipation_config({"cache": {"anticipation": {"novel_ratio_threshold": 0.9, "min_corpus_stems": 5}}})

        assert cfg.novel_ratio_threshold == 0.9
        assert cfg.min_corpus_stems == 5

    @pytest.mark.parametrize(
        "section",
        [
            {"novel_ratio_threshold": "0.9"},
            {"novel_ratio_threshold": True},
            {"min_corpus_stems": 4.5},
            {"min_corpus_stems": True},
        ],
    )
    def test_a_malformed_threshold_falls_back_to_the_shipped_default(self, section) -> None:
        cfg = prefetch.anticipation_config({"cache": {"anticipation": section}})

        assert cfg == prefetch.GATE_DEFAULT_CONFIG

    def test_an_out_of_range_threshold_is_refused_at_the_boundary(self) -> None:
        with pytest.raises(ValueError):
            prefetch.anticipation_config({"cache": {"anticipation": {"novel_ratio_threshold": 1.5}}})


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestNoClockOnThisPath:
    def test_the_census_can_see_a_clock_read(self) -> None:
        """POSITIVE CONTROL: the instrument used below is not blind.

        Reads the clock through the one sanctioned boundary accessor
        (``scoring_instant._read_utc_today``), with the census configured to
        forbid it — so this is a real ``mind_mem`` clock read the census must
        catch, and no test-only clock hook has to exist in shipped source.
        """
        from mind_mem.scoring_instant import _read_utc_today

        with clock_census() as census:
            _read_utc_today()

        assert census.reads, "the census saw nothing — every assertion below would be vacuous"

    def test_record_and_lookup_read_no_clock(self) -> None:
        cache = prefetch.AnticipationCache()

        with clock_census() as census:
            cache.record("/ws", "recall", _hits(), head="H")
            decision = cache.lookup("/ws", "determinism scoring compiler", head="H")

        assert decision.serve_from_cache is True, "nothing was served — the census would be vacuous"
        census.assert_clock_free()

    def test_resolving_the_head_reads_no_clock(self, chained_workspace) -> None:
        with clock_census() as census:
            head = prefetch.chain_head(chained_workspace)

        assert head != GENESIS_ANCHOR, "no chain was read — the census would be vacuous"
        census.assert_clock_free()


# ---------------------------------------------------------------------------
# Feeding the starved co-retrieval loop
# ---------------------------------------------------------------------------


class TestObservationsFeedThePredictor:
    def test_the_predictor_starts_with_nothing_observed(self) -> None:
        """POSITIVE CONTROL: the counter this test moves really does start at 0."""
        from mind_mem.speculative_prefetch import get_default_predictor, reset_default_predictor

        reset_default_predictor()
        assert get_default_predictor().stats().as_dict()["observations"] == 0

    def test_observing_a_served_set_moves_the_counter(self) -> None:
        from mind_mem.speculative_prefetch import get_default_predictor, reset_default_predictor

        reset_default_predictor()
        prefetch.observe_served("determinism scoring", _hits())

        assert get_default_predictor().stats().as_dict()["observations"] > 0
        reset_default_predictor()

    def test_hits_without_ids_are_not_observed(self) -> None:
        from mind_mem.speculative_prefetch import get_default_predictor, reset_default_predictor

        reset_default_predictor()
        prefetch.observe_served("determinism", [{"Statement": "no id here"}])

        assert get_default_predictor().stats().as_dict()["observations"] == 0
        reset_default_predictor()


# ---------------------------------------------------------------------------
# The wiring: a real recall through the MCP surface
# ---------------------------------------------------------------------------


def _seed_recall_workspace(root: str) -> None:
    """A corpus whose *served excerpts* clear the shipped 200-stem floor.

    The floor is deliberately not lowered for these tests. A served recall hit
    carries a 300-character ``excerpt``, so no single block can supply 200
    distinct stems through the envelope — the corpus has to. Fourteen blocks
    each contributing twenty-five unique stems means a default ten-hit page
    lands the bundle at ~250 distinct stems, above the floor, with the shipped
    config untouched.
    """
    from _recall_clock_sentinel import write_workspace

    blocks = [("D-20260101-000", "the deterministic compiler emits byte identical artifacts", "2026-01-01")]
    for n in range(14):
        unique = " ".join(f"stem{n}x{k}" for k in range(25))
        blocks.append(
            (
                f"D-20260101-{n + 1:03d}",
                f"recall scoring takes the instant as an input {unique}",
                "2026-01-01",
            )
        )
    write_workspace(root, tuple(blocks))


class TestWiredIntoTheRecallSurface:
    def test_the_flag_off_recall_never_touches_the_anticipation_cache(self, tmp_path) -> None:
        """POSITIVE CONTROL is the sibling test below: with the flag on, it fills."""
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "off")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False}}, fh)

        with use_workspace(ws):
            envelope = json.loads(_recall_impl("deterministic compiler", scoring_instant="2026-01-02"))

        assert envelope["count"] >= 1, "recall served nothing — the assertion below would be vacuous"
        assert prefetch.get_cache().stats()["bundles"] == 0

    def test_the_flag_off_path_does_no_per_hit_work_and_resolves_the_head_once(self, tmp_path, monkeypatch) -> None:
        """Inertness is a performance claim, so it gets a number.

        With the flag off the feature must add no parse and no per-item work:
        no bundle is tokenized and the cache is never consulted. What the recall
        path does pay unconditionally is the generation key — and it must pay it
        **once per recall**, not once per hit and not once per leg. Five recalls
        over a fifteen-block corpus therefore means exactly five resolutions.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "inert")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False}}, fh)
        _append_to_governed_ledger(ws, "D-20260101-000", "create", "seed")

        real_chain_head = prefetch.chain_head
        resolutions = 0

        def counting_chain_head(workspace: str) -> str:
            nonlocal resolutions
            resolutions += 1
            return real_chain_head(workspace)

        # ``_resolve_chain_head`` imports the function at call time, so patching
        # the module attribute is what the recall path actually reaches.
        monkeypatch.setattr(prefetch, "chain_head", counting_chain_head)

        with use_workspace(ws):
            for _ in range(5):
                envelope = json.loads(_recall_impl("deterministic compiler", scoring_instant="2026-01-02"))

        assert envelope["count"] >= 1, "recall served nothing — every count below would be vacuous"
        assert prefetch.get_cache().stats()["bundles"] == 0, "the off path tokenized hits it should not have"
        assert prefetch.get_cache().stats()["misses"] == 0, "the off path consulted the cache it should not have"
        assert resolutions == 5, f"the generation key was resolved {resolutions} times for 5 recalls"

    def test_the_flag_on_recall_fills_the_cache_at_the_current_head(self, tmp_path) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "on")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False, "anticipation": {"enabled": True}}}, fh)

        with use_workspace(ws):
            json.loads(_recall_impl("deterministic compiler", scoring_instant="2026-01-02"))

        assert prefetch.get_cache().stats()["bundles"] == 1
        assert prefetch.get_cache().stats()["documents"] >= 1

    def test_a_real_recall_feeds_the_co_retrieval_loop_the_roadmap_called_starving(self, tmp_path) -> None:
        """``prefetch observations = 0`` was the symptom; this is the wire.

        The predictor could never name a likely next block because nothing ever
        told it what a query resolved to. A recall now reports its served ids,
        and the count is read back through the *existing* operator surface —
        ``memory_ops`` publishes ``stats["speculative_prefetch"]`` from the same
        default predictor — so the loop being fed is visible without a new
        diagnostic.

        The reset plus the first assertion are the positive control: the counter
        this test moves genuinely starts at zero.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl
        from mind_mem.speculative_prefetch import get_default_predictor, reset_default_predictor

        ws = str(tmp_path / "feed")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False, "anticipation": {"enabled": True}}}, fh)

        reset_default_predictor()
        try:
            assert get_default_predictor().stats().as_dict()["observations"] == 0

            with use_workspace(ws):
                envelope = json.loads(_recall_impl("deterministic compiler", scoring_instant="2026-01-02"))

            assert envelope["count"] >= 1, "recall served nothing — there was nothing to observe"
            assert get_default_predictor().stats().as_dict()["observations"] > 0
        finally:
            reset_default_predictor()

    def test_a_second_recall_is_answered_locally_and_says_so(self, tmp_path) -> None:
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "serve")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False, "anticipation": {"enabled": True}}}, fh)

        with use_workspace(ws):
            first = json.loads(_recall_impl("recall scoring instant input", scoring_instant="2026-01-02"))
            second = json.loads(_recall_impl("recall scoring instant", scoring_instant="2026-01-02"))

        assert first["backend"] != "anticipation_cache", "the first recall must go to the store"
        assert second["backend"] == "anticipation_cache"
        assert "attestation" not in second, "an unattested answer must not carry an attestation"
        assert second["anticipation"]["serve_from_cache"] is True
        assert any("no recall attestation" in w for w in second["warnings"])

    def test_the_cache_counters_reach_an_operator_surface(self, tmp_path) -> None:
        """A hit rate nobody can read is a knob nobody can tune.

        ``retrieval_diagnostics`` is the tool an operator already opens to ask
        why recall is behaving as it is, so the cache's counters are reported
        there rather than through a new surface. The assertion is that the
        numbers *move* — reading a block of zeroes would pass just as well if
        the diagnostic were wired to a different cache instance.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl, retrieval_diagnostics

        ws = str(tmp_path / "diag")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False, "anticipation": {"enabled": True}}}, fh)

        with use_workspace(ws):
            cold = json.loads(retrieval_diagnostics())["anticipation_cache"]
            _recall_impl("recall scoring instant input", scoring_instant="2026-01-02")
            _recall_impl("recall scoring instant", scoring_instant="2026-01-02")
            warm = json.loads(retrieval_diagnostics())["anticipation_cache"]

        assert cold["bundles"] == 0 and cold["hits"] == 0, "the diagnostic started non-empty"
        assert warm["bundles"] >= 1
        assert warm["hits"] >= 1, "the served hit never reached the counters the diagnostic reads"

    def test_a_bundle_shaped_request_is_never_answered_locally(self, tmp_path) -> None:
        """A shape the local door cannot produce must not be served by it.

        The anticipation early return skips every post-cache stage, and the
        ``format="bundle"`` re-shaping is one of them. Serving here would give a
        bundle client the blocks envelope with no ``facts`` / ``relations`` /
        ``timeline`` — the same shape confusion the format-cache isolation tests
        guard, arriving through a different door.

        The first assertion is the positive control: the identical request in
        the default ``blocks`` shape *is* answered locally, so the fall-through
        below is the format rule and not a cold cache.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "shape")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False, "anticipation": {"enabled": True}}}, fh)

        with use_workspace(ws):
            _recall_impl("recall scoring instant input", scoring_instant="2026-01-02")
            blocks = json.loads(_recall_impl("recall scoring instant", scoring_instant="2026-01-02"))
            bundle = json.loads(_recall_impl("recall scoring instant", format="bundle", scoring_instant="2026-01-02"))

        assert blocks["backend"] == "anticipation_cache", "nothing was cached — the assertion below would be vacuous"
        assert bundle.get("backend") != "anticipation_cache"
        assert "source_blocks" in bundle, "the bundle shape was not produced"

    def test_the_local_door_enforces_the_same_result_ceiling_as_the_store_door(self, tmp_path) -> None:
        """A second door must apply the operator's limit, not merely inherit it.

        ``limits.max_recall_results`` is what an operator sets to bound how much
        context a single recall can put in a window. The store path clamps to
        it. If the anticipation path did not, asking for more would simply be
        granted whenever the answer happened to come from the local bundle —
        the limit would be advisory rather than a limit.

        The bundle is filled through the ``prefetch`` tool, which has its own
        ceiling (``max_prefetch_results``), so the bundle can legitimately hold
        more blocks than a recall is allowed to return. That is the positive
        control: the second assertion proves the bundle really is bigger than
        the recall ceiling, so a missing clamp would be visible rather than
        hidden by a bundle that was small anyway.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl
        from mind_mem.mcp.tools.recall import prefetch as prefetch_tool

        ws = str(tmp_path / "ceiling")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cache": {"enabled": False, "anticipation": {"enabled": True}},
                    "limits": {"max_recall_results": 2, "max_prefetch_results": 20},
                },
                fh,
            )

        with use_workspace(ws):
            filled = json.loads(prefetch_tool("recall,scoring,instant,input", limit=20))
            served = json.loads(_recall_impl("recall scoring instant", limit=100, scoring_instant="2026-01-02"))

        assert filled["count"] > 2, "the prefetch tool assembled nothing bigger than the ceiling"
        assert served["backend"] == "anticipation_cache", "the store answered — the ceiling below would be the store's"
        assert served["anticipation"]["documents"] > 2, "the bundle held no more than the ceiling — the test would be vacuous"
        assert served["count"] <= 2

    def test_a_governed_write_retires_the_local_answer(self, tmp_path) -> None:
        """The whole point: a write through a door that invalidates nothing.

        The governed ledger is appended to by the governance gate on every
        admitted write, from every door — none of which calls
        ``_invalidate_recall_cache``; only three MCP tools do. Under a TTL the
        local answer would keep being served for the rest of the window. Under
        the head it is gone the moment the ledger moves.
        """
        from mind_mem.mcp.infra.workspace import use_workspace
        from mind_mem.mcp.tools.recall import _recall_impl

        ws = str(tmp_path / "retire")
        os.makedirs(ws, exist_ok=True)
        _seed_recall_workspace(ws)
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"cache": {"enabled": False, "anticipation": {"enabled": True}}}, fh)

        with use_workspace(ws):
            _recall_impl("recall scoring instant input", scoring_instant="2026-01-02")
            served_locally = json.loads(_recall_impl("recall scoring instant", scoring_instant="2026-01-02"))
            assert served_locally["backend"] == "anticipation_cache", "nothing was cached — the assertion below would be vacuous"

            _append_to_governed_ledger(ws, "D-20260101-002", "update", "the corpus moved")
            after = json.loads(_recall_impl("recall scoring instant", scoring_instant="2026-01-02"))

        assert after["backend"] != "anticipation_cache"
