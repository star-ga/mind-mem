# Copyright 2026 STARGA, Inc.
"""Edge-grounded answering with per-claim citations (roadmap RM-0497).

Two things are under test, and only the second is about text.

**The context.** A k-hop subgraph serialised as triples, each carrying the
edge id, provenance block, origin marker and schema stamp behind it — and
an explicit list of what the subgraph does *not* establish. A graph answer
that omits its holes is the failure mode this mode exists to prevent, so
every gap kind gets a test, and each "the gap is absent" assertion is
paired with the state that produces it.

**The citation check.** A generator sees the serialised triples and
nothing else; every ``[[E-…]]`` it emits is matched against the ids
actually served. A citation naming an unserved edge is a fabrication,
reported as such, and the answer is not grounded. The fabrication test is
paired with a positive control citing a real id, because a checker that
called everything fabricated would pass the negative on its own.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem.edge_grounded_answer import (
    GAP_EXPIRED_WITHHELD,
    GAP_FABRICATED_CITATION,
    GAP_HOP_LIMIT,
    GAP_NO_EDGES,
    GAP_PREDICATE_ABSENT,
    GAP_PROVENANCE_MISSING,
    GAP_TRIPLE_CAP,
    GAP_UNCITED_ANSWER,
    GAP_UNKNOWN_ENTITY,
    GAP_UNVERSIONED_EDGE,
    answer,
    build_context,
    corpus_block_ids,
)
from mind_mem.knowledge_graph import KnowledgeGraph, edge_id

BLOCK_A = "D-20260101-001"
BLOCK_B = "D-20260101-002"
BLOCK_C = "D-20260101-003"

#: A syntactically valid edge id that names no edge in any of these graphs.
UNSERVED_ID = "E-" + "0" * 16


@pytest.fixture()
def graph(admitted) -> Iterator[KnowledgeGraph]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        kg = KnowledgeGraph(str(Path(td) / "kg.db"))
        try:
            yield kg
        finally:
            kg.close()


@pytest.fixture()
def chain(graph: KnowledgeGraph) -> KnowledgeGraph:
    """starga -depends_on-> mindc -depends_on-> llvm  (a 2-hop chain)."""
    graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A, confidence=0.9)
    graph.add_edge("mindc", "depends_on", "llvm", source_block_id=BLOCK_B, confidence=0.8)
    return graph


def _kinds(context) -> list[str]:
    return [g.kind for g in context.gaps]


# ---------------------------------------------------------------------------
# Reading must not write
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_unknown_seed_is_a_gap_and_mints_no_entity(self, chain: KnowledgeGraph) -> None:
        before = chain.stats().entities
        context = build_context(chain, "never heard of this")
        assert context.triples == ()
        assert _kinds(context) == [GAP_UNKNOWN_ENTITY]
        assert context.seed_entity_id is None
        assert chain.entities.lookup("never heard of this") is None
        assert chain.stats().entities == before, "asking a question must not add an entity"

    def test_known_seed_returns_edges(self, chain: KnowledgeGraph) -> None:
        """Positive control for the test above: the same call shape on a
        seed the registry knows does serve triples."""
        context = build_context(chain, "STARGA")  # alias casing on purpose
        assert context.seed_entity_id == "starga"
        assert len(context.triples) == 2
        assert GAP_UNKNOWN_ENTITY not in _kinds(context)


# ---------------------------------------------------------------------------
# The served subgraph
# ---------------------------------------------------------------------------


class TestSubgraph:
    def test_one_hop_stops_at_one_hop(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga", hops=1)
        assert [(t.subject, t.object) for t in context.triples] == [("starga", "mindc")]
        assert GAP_HOP_LIMIT in _kinds(context)

    def test_two_hops_reaches_the_far_end(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga", hops=2)
        assert [(t.subject, t.object) for t in context.triples] == [
            ("starga", "mindc"),
            ("mindc", "llvm"),
        ]
        assert [t.hop for t in context.triples] == [1, 2]

    def test_every_triple_cites_its_edge_and_provenance(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga")
        first = context.triples[0]
        assert first.edge_id == edge_id("starga", "depends_on", "mindc", BLOCK_A)
        assert first.source_block_id == BLOCK_A
        assert first.schema_version is not None, "a governed edge carries its schema stamp"
        assert first.confidence == pytest.approx(0.9)

    def test_serialisation_is_deterministic(self, chain: KnowledgeGraph) -> None:
        a = build_context(chain, "starga").serialize()
        b = build_context(chain, "starga").serialize()
        assert a == b
        assert "[[E-" in a
        assert "# gaps" in a

    def test_triple_cap_is_reported_not_silent(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga", max_triples=1)
        assert len(context.triples) == 1
        assert GAP_TRIPLE_CAP in _kinds(context)
        # Positive control: the same graph without the cap serves both, so
        # the truncation above is the cap and not an empty graph.
        assert len(build_context(chain, "starga").triples) == 2

    def test_hop_limit_is_not_reported_when_the_component_is_exhausted(self, graph: KnowledgeGraph) -> None:
        """The gap must be a finding about THIS graph, not a constant.

        One edge, radius two: there is nothing beyond hop 1 to expand, so
        claiming a hop limit would be a fabricated hole.
        """
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        assert GAP_HOP_LIMIT not in _kinds(build_context(graph, "starga", hops=2))
        # Positive control on the same call shape: extend the chain and
        # the radius genuinely truncates it.
        graph.add_edge("mindc", "depends_on", "llvm", source_block_id=BLOCK_B)
        assert GAP_HOP_LIMIT in _kinds(build_context(graph, "starga", hops=2))

    def test_no_edges_for_an_isolated_entity(self, graph: KnowledgeGraph) -> None:
        graph.entities.resolve("lonely")
        context = build_context(graph, "lonely")
        assert context.triples == ()
        assert GAP_NO_EDGES in _kinds(context)


class TestPredicateFilter:
    def test_absent_predicate_is_named(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga", predicates=["contradicts"])
        assert context.triples == ()
        assert GAP_PREDICATE_ABSENT in _kinds(context)
        assert "contradicts" in " ".join(g.detail for g in context.gaps)

    def test_present_predicate_serves_edges(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga", predicates=["depends_on"])
        assert len(context.triples) == 2
        assert GAP_PREDICATE_ABSENT not in _kinds(context)


class TestTemporal:
    def test_expired_edge_is_withheld_and_the_omission_is_stated(self, graph: KnowledgeGraph) -> None:
        graph.add_edge(
            "starga",
            "depends_on",
            "old thing",
            source_block_id=BLOCK_A,
            valid_from="2020-01-01T00:00:00Z",
            valid_until="2020-01-02T00:00:00Z",
        )
        context = build_context(graph, "starga")
        assert context.triples == ()
        assert GAP_EXPIRED_WITHHELD in _kinds(context)

    def test_include_expired_serves_it(self, graph: KnowledgeGraph) -> None:
        graph.add_edge(
            "starga",
            "depends_on",
            "old thing",
            source_block_id=BLOCK_A,
            valid_until="2020-01-02T00:00:00Z",
        )
        context = build_context(graph, "starga", include_expired=True)
        assert len(context.triples) == 1
        assert GAP_EXPIRED_WITHHELD not in _kinds(context)

    def test_as_of_replays_the_answer_at_a_past_instant(self, graph: KnowledgeGraph) -> None:
        """The wedge property: the same question at the same instant gives
        the same answer, whatever today is."""
        graph.add_edge(
            "starga",
            "depends_on",
            "old thing",
            source_block_id=BLOCK_A,
            valid_until="2020-01-02T00:00:00Z",
        )
        live = build_context(graph, "starga", as_of="2020-01-01T00:00:00Z")
        assert len(live.triples) == 1
        assert GAP_EXPIRED_WITHHELD not in _kinds(live)
        # Positive control on the same graph: one second past the window
        # and the same call withholds it.
        dead = build_context(graph, "starga", as_of="2020-01-02T00:00:01Z")
        assert dead.triples == ()
        assert GAP_EXPIRED_WITHHELD in _kinds(dead)


class TestProvenanceAndSchemaGaps:
    def test_missing_provenance_block_is_flagged(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga", known_block_ids={BLOCK_A})
        details = [g.detail for g in context.gaps if g.kind == GAP_PROVENANCE_MISSING]
        assert len(details) == 1
        assert BLOCK_B in details[0]
        # Positive control: with both blocks known, nothing is flagged --
        # so the finding above is the missing block, not the check
        # firing unconditionally.
        ok = build_context(chain, "starga", known_block_ids={BLOCK_A, BLOCK_B})
        assert GAP_PROVENANCE_MISSING not in _kinds(ok)

    def test_unstamped_edge_is_flagged(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        assert GAP_UNVERSIONED_EDGE not in _kinds(build_context(graph, "starga"))  # control
        conn = sqlite3.connect(graph._db_path)
        try:
            conn.execute("INSERT OR IGNORE INTO entities(id, canonical) VALUES ('legacy', 'legacy')")
            conn.execute(
                "INSERT OR IGNORE INTO edges(subject, predicate, object, source_block_id, "
                "confidence, valid_from, valid_until, metadata) "
                "VALUES ('starga', 'related_to', 'legacy', ?, 1.0, NULL, NULL, '{}')",
                (BLOCK_C,),
            )
            conn.commit()
        finally:
            conn.close()
        context = build_context(graph, "starga")
        assert GAP_UNVERSIONED_EDGE in _kinds(context)


# ---------------------------------------------------------------------------
# The citation check — the part that acts as a gate
# ---------------------------------------------------------------------------


class TestCitationEnforcement:
    def test_no_generator_answers_with_the_cited_triples(self, chain: KnowledgeGraph) -> None:
        result = answer(chain, "starga")
        assert result.generator == "none"
        assert result.grounded is True
        assert result.fabricated_citations == ()
        assert set(result.citations) == set(chain_ids(chain))
        for eid in result.citations:
            assert f"[[{eid}]]" in result.text

    def test_generator_sees_only_the_serialised_context(self, chain: KnowledgeGraph) -> None:
        seen: list[str] = []

        def _gen(prompt: str) -> str:
            seen.append(prompt)
            return "nothing"

        context = build_context(chain, "starga")
        answer(chain, "starga", generate_fn=_gen, context=context)
        assert seen == [context.serialize()]

    def test_a_real_citation_is_accepted(self, chain: KnowledgeGraph) -> None:
        """Positive control for the fabrication test below."""
        context = build_context(chain, "starga")
        real = context.triples[0].edge_id

        result = answer(chain, "starga", generate_fn=lambda _p: f"STARGA depends on mindc [[{real}]].", context=context)
        assert result.grounded is True
        assert result.citations == (real,)
        assert result.fabricated_citations == ()
        assert GAP_FABRICATED_CITATION not in _kinds(result.context)

    def test_a_fabricated_citation_is_caught(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga")
        assert UNSERVED_ID not in context.edge_ids  # the id names no served edge

        result = answer(
            chain,
            "starga",
            generate_fn=lambda _p: f"STARGA funds Anthropic [[{UNSERVED_ID}]].",
            context=context,
        )
        assert result.grounded is False
        assert result.fabricated_citations == (UNSERVED_ID,)
        assert result.citations == ()
        assert GAP_FABRICATED_CITATION in _kinds(result.context)

    def test_mixed_real_and_fabricated_keeps_both_facts(self, chain: KnowledgeGraph) -> None:
        context = build_context(chain, "starga")
        real = context.triples[0].edge_id
        result = answer(
            chain,
            "starga",
            generate_fn=lambda _p: f"a [[{real}]] and b [[{UNSERVED_ID}]]",
            context=context,
        )
        assert result.citations == (real,)
        assert result.fabricated_citations == (UNSERVED_ID,)
        assert result.grounded is False

    def test_an_uncited_answer_is_not_grounded(self, chain: KnowledgeGraph) -> None:
        result = answer(chain, "starga", generate_fn=lambda _p: "STARGA depends on mindc.")
        assert result.grounded is False
        assert GAP_UNCITED_ANSWER in _kinds(result.context)

    def test_generator_must_return_text(self, chain: KnowledgeGraph) -> None:
        with pytest.raises(TypeError):
            answer(chain, "starga", generate_fn=lambda _p: None)  # type: ignore[return-value,arg-type]


def chain_ids(kg: KnowledgeGraph) -> list[str]:
    return [
        edge_id("starga", "depends_on", "mindc", BLOCK_A),
        edge_id("mindc", "depends_on", "llvm", BLOCK_B),
    ]


# ---------------------------------------------------------------------------
# Wiring: `mm graph-answer` reaches all of it
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_command_is_registered(self) -> None:
        from mind_mem.mm_cli import build_parser

        args = build_parser().parse_args(["graph-answer", "starga", "--hops", "1"])
        assert args.func.__name__ == "_cmd_graph_answer"
        assert args.entity == "starga"
        assert args.hops == 1

    def test_missing_graph_reports_and_creates_nothing(self, workspace: str, capsys, monkeypatch) -> None:
        import os

        from mind_mem import mm_cli
        from mind_mem.knowledge_graph import default_db_path

        monkeypatch.setattr(mm_cli, "_workspace", lambda: workspace)
        args = mm_cli.build_parser().parse_args(["graph-answer", "starga"])
        assert args.func(args) == 1
        assert "no knowledge graph" in capsys.readouterr().out
        assert not os.path.isfile(default_db_path(workspace))

    def test_answer_prints_citations_and_gaps(self, workspace: str, capsys, monkeypatch, admitted) -> None:
        from mind_mem import mm_cli
        from mind_mem.knowledge_graph import default_db_path

        with KnowledgeGraph(default_db_path(workspace)) as kg:
            kg.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)

        monkeypatch.setattr(mm_cli, "_workspace", lambda: workspace)
        args = mm_cli.build_parser().parse_args(["graph-answer", "starga", "--json"])
        assert args.func(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["grounded"] is True
        assert payload["citations"] == [edge_id("starga", "depends_on", "mindc", BLOCK_A)]
        # BLOCK_A is not a real block in this workspace corpus, so the
        # provenance gap must be reported rather than assumed away.
        assert GAP_PROVENANCE_MISSING in [g["kind"] for g in payload["context"]["gaps"]]

    def test_context_mode_prints_the_serialised_triples(self, workspace: str, capsys, monkeypatch, admitted) -> None:
        from mind_mem import mm_cli
        from mind_mem.knowledge_graph import default_db_path

        with KnowledgeGraph(default_db_path(workspace)) as kg:
            kg.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        monkeypatch.setattr(mm_cli, "_workspace", lambda: workspace)
        args = mm_cli.build_parser().parse_args(["graph-answer", "starga", "--context"])
        assert args.func(args) == 0
        out = capsys.readouterr().out
        assert "# edge-grounded context:" in out
        assert "answer ONLY from the triples below" in out
        assert "starga\tdepends_on\tmindc" in out

    def test_corpus_block_ids_reads_the_real_corpus(self, workspace: str) -> None:
        ids = corpus_block_ids(workspace)
        assert isinstance(ids, frozenset)
        # Whatever a fresh workspace scaffolds, the loader must not invent
        # ids: every one it returns is non-empty.
        assert all(i for i in ids)
