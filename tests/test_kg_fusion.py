"""Typed-knowledge-graph fusion into recall (opt-in, default OFF).

``kg_expand`` resolves query terms through the EntityRegistry
(read-only — recall must never mint entities), walks
``KnowledgeGraph`` edges up to two hops, maps edge
``source_block_id`` values back to corpus blocks, and appends them
with a decayed score — the same decay shape as the block-xref
``graph_expand`` path.

Fusion is gated behind ``retrieval.kg_fusion.enabled`` (default
false) so existing recall replays byte-identical until the graph is
populated and the operator opts in.
"""

from __future__ import annotations

import pytest

from mind_mem.kg_fusion import (
    is_kg_fusion_enabled,
    kg_expand,
    resolve_kg_fusion_config,
    resolve_query_entities,
)
from mind_mem.knowledge_graph import KnowledgeGraph

BLOCK_A = "D-20260101-001"
BLOCK_B = "D-20260101-002"
BLOCK_C = "D-20260101-003"


@pytest.fixture
def kg(tmp_path):
    graph = KnowledgeGraph(str(tmp_path / "kg.db"))
    yield graph
    graph.close()


def _block(bid: str, text: str = "") -> dict:
    return {"_id": bid, "excerpt": text, "content": text}


class TestEnableGate:
    def test_default_off(self) -> None:
        assert is_kg_fusion_enabled({}) is False
        assert is_kg_fusion_enabled(None) is False
        assert is_kg_fusion_enabled({"retrieval": {}}) is False

    def test_explicit_on(self) -> None:
        cfg = {"retrieval": {"kg_fusion": {"enabled": True}}}
        assert is_kg_fusion_enabled(cfg) is True

    def test_non_dict_shapes_off(self) -> None:
        assert is_kg_fusion_enabled({"retrieval": "nope"}) is False
        assert is_kg_fusion_enabled({"retrieval": {"kg_fusion": "yes"}}) is False


class TestResolveConfig:
    def test_defaults(self) -> None:
        params = resolve_kg_fusion_config({})
        assert params == {
            "max_hops": 2,
            "decay": 0.5,
            "max_neighbors_per_hop": 5,
            "max_total_added": 25,
        }

    def test_max_hops_capped_at_two(self) -> None:
        cfg = {"retrieval": {"kg_fusion": {"enabled": True, "max_hops": 7}}}
        assert resolve_kg_fusion_config(cfg)["max_hops"] == 2

    def test_invalid_values_fall_back(self) -> None:
        cfg = {"retrieval": {"kg_fusion": {"decay": 5, "max_neighbors_per_hop": -1}}}
        params = resolve_kg_fusion_config(cfg)
        assert params["decay"] == 0.5
        assert params["max_neighbors_per_hop"] == 5


class TestResolveQueryEntities:
    def test_read_only_lookup(self, kg) -> None:
        kg.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        before = kg.stats().entities
        found = resolve_query_entities("what does starga ship", kg.entities)
        assert found == ["starga"]
        # Lookup must not create entities for the other query tokens.
        assert kg.stats().entities == before

    def test_unknown_terms_resolve_to_nothing(self, kg) -> None:
        before = kg.stats().entities
        assert resolve_query_entities("completely unknown words", kg.entities) == []
        assert kg.stats().entities == before


class TestKgExpand:
    def _seed_graph(self, kg: KnowledgeGraph) -> None:
        kg.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B)
        kg.add_edge("mindc", "part_of", "mind", source_block_id=BLOCK_C)

    def test_one_hop_appends_decayed_block(self, kg) -> None:
        self._seed_graph(kg)
        results = [{"_id": BLOCK_A, "score": 10.0, "excerpt": "seed"}]
        corpus = [_block(BLOCK_A, "seed"), _block(BLOCK_B, "edge source"), _block(BLOCK_C, "second hop")]
        out = kg_expand(results, corpus, kg, "starga status", max_hops=1)
        ids = [r["_id"] for r in out]
        assert ids[0] == BLOCK_A
        assert BLOCK_B in ids
        added = next(r for r in out if r["_id"] == BLOCK_B)
        assert added["score"] == pytest.approx(10.0 * 0.5)
        assert added["_kg_hop"] == 1
        assert added["_kg_predicate"] == "depends_on"

    def test_two_hop_walk(self, kg) -> None:
        self._seed_graph(kg)
        results = [{"_id": BLOCK_A, "score": 8.0, "excerpt": "seed"}]
        corpus = [_block(BLOCK_A), _block(BLOCK_B), _block(BLOCK_C)]
        out = kg_expand(results, corpus, kg, "starga", max_hops=2)
        by_id = {r["_id"]: r for r in out}
        assert by_id[BLOCK_C]["_kg_hop"] == 2
        assert by_id[BLOCK_C]["score"] == pytest.approx(8.0 * 0.5 * 0.5)

    def test_no_matching_entity_returns_same_object(self, kg) -> None:
        self._seed_graph(kg)
        results = [{"_id": BLOCK_A, "score": 5.0}]
        out = kg_expand(results, [_block(BLOCK_A)], kg, "nothing relevant")
        assert out is results

    def test_already_present_blocks_not_duplicated(self, kg) -> None:
        self._seed_graph(kg)
        results = [
            {"_id": BLOCK_A, "score": 5.0},
            {"_id": BLOCK_B, "score": 4.0},
        ]
        corpus = [_block(BLOCK_A), _block(BLOCK_B), _block(BLOCK_C)]
        out = kg_expand(results, corpus, kg, "starga", max_hops=1)
        assert [r["_id"] for r in out].count(BLOCK_B) == 1

    def test_deterministic(self, kg) -> None:
        self._seed_graph(kg)
        kg.add_edge("starga", "authored_by", "nikolai", source_block_id=BLOCK_C)
        results = [{"_id": BLOCK_A, "score": 6.0}]
        corpus = [_block(BLOCK_A), _block(BLOCK_B), _block(BLOCK_C)]
        first = kg_expand(list(results), corpus, kg, "starga")
        second = kg_expand(list(results), corpus, kg, "starga")
        assert [(r["_id"], r["score"]) for r in first] == [(r["_id"], r["score"]) for r in second]

    def test_expansion_never_mints_entities(self, kg) -> None:
        self._seed_graph(kg)
        before = kg.stats().entities
        results = [{"_id": BLOCK_A, "score": 5.0}]
        kg_expand(results, [_block(BLOCK_A), _block(BLOCK_B)], kg, "starga unseen tokens here")
        assert kg.stats().entities == before

    def test_empty_results_unchanged(self, kg) -> None:
        self._seed_graph(kg)
        results: list[dict] = []
        assert kg_expand(results, [_block(BLOCK_B)], kg, "starga") is results


class TestHybridBackendGate:
    """_maybe_kg_expand — fusion off leaves recall byte-identical."""

    @staticmethod
    def _backend(config: dict):
        from mind_mem.hybrid_recall import HybridBackend

        return HybridBackend(config=config)

    def test_off_returns_same_object(self, workspace) -> None:
        backend = self._backend({"retrieval": {}})
        results = [{"_id": BLOCK_A, "score": 3.0}]
        out = backend._maybe_kg_expand("starga", workspace, results)
        assert out is results

    def test_on_without_graph_db_returns_same_object(self, workspace) -> None:
        backend = self._backend({"retrieval": {"kg_fusion": {"enabled": True}}})
        results = [{"_id": BLOCK_A, "score": 3.0}]
        out = backend._maybe_kg_expand("starga", workspace, results)
        assert out is results

    def test_on_expands_from_populated_graph(self, workspace) -> None:
        from mind_mem.knowledge_graph import default_db_path

        with KnowledgeGraph(default_db_path(workspace)) as kg:
            kg.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_B)
        backend = self._backend({"retrieval": {"kg_fusion": {"enabled": True}}})
        results = [{"_id": BLOCK_A, "score": 3.0}]
        corpus = [_block(BLOCK_A), _block(BLOCK_B, "edge source text")]
        out = backend._maybe_kg_expand("starga plans", workspace, results, corpus=corpus)
        assert [r["_id"] for r in out] == [BLOCK_A, BLOCK_B]
        assert out[1]["_kg_hop"] == 1
