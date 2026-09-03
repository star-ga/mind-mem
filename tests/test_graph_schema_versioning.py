# Copyright 2026 STARGA, Inc.
"""Schema versioning alongside the graph (roadmap RM-0512).

An edge already carried a receipt saying *who authorised it*. It carried
nothing saying *what rules produced it*, so an edge extracted under a
ten-predicate vocabulary and an edge extracted under a fourteen-predicate
one were indistinguishable rows, and neither could be re-extracted on
purpose. These tests pin the three verbs the roadmap item asks for --
**distinguishable**, **comparable**, **re-extractable** -- plus the two
properties that make the id worth trusting: it is a pure function of the
vocabulary / folding rule / prompt, and it survives a proposal that is
approved long after the vocabulary moved.

Every negative assertion here is paired with a positive control, because
"the stale list is empty" is exactly what a broken reader also returns.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem import graph_schema as gs
from mind_mem import knowledge_graph as kg_module
from mind_mem.graph_ingest import (
    EDGE_ORIGIN,
    RelationTriple,
    approve_relation_signals,
    backfill,
    pending_relation_signals,
    relations_to_signals,
    schema_report,
    stage_relation_signals,
    stale_schema_blocks,
)
from mind_mem.knowledge_graph import (
    KnowledgeGraph,
    Predicate,
    default_db_path,
)

DATE = "2026-07-27"
BLOCK_A = "D-20260101-001"
BLOCK_B = "D-20260101-002"

_CLOCKS = {"time", "datetime", "random", "secrets"}


def _clock_imports(path: str) -> set[str]:
    """Names of clock / entropy modules *path* imports, at any scope."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names} & _CLOCKS
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _CLOCKS:
                found.add(node.module.split(".")[0])
    return found


@pytest.fixture()
def clean_predicates() -> Iterator[None]:
    """Restore the runtime predicate registry after a widening test.

    ``Predicate.register`` mutates module-global state; leaving a
    registration behind would silently change every later test's schema
    id, which is precisely the drift this module exists to detect.
    """
    from mind_mem.knowledge_graph import _RUNTIME_PREDICATES

    snapshot = dict(_RUNTIME_PREDICATES)
    try:
        yield
    finally:
        _RUNTIME_PREDICATES.clear()
        _RUNTIME_PREDICATES.update(snapshot)


@pytest.fixture()
def graph(admitted) -> Iterator[KnowledgeGraph]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        kg = KnowledgeGraph(str(Path(td) / "kg.db"))
        try:
            yield kg
        finally:
            kg.close()


def _triple(
    subject: str = "starga",
    predicate: str = "depends_on",
    obj: str = "mindc",
    source: str = BLOCK_A,
    confidence: float = 0.7,
) -> RelationTriple:
    return RelationTriple(
        subject=subject,
        predicate=predicate,
        object=obj,
        source_block_id=source,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# The id itself
# ---------------------------------------------------------------------------


class TestSchemaId:
    def test_format_and_determinism(self) -> None:
        version = gs.current_version()
        assert gs.SCHEMA_VERSION_RE.match(version), version
        assert version.startswith(f"{gs.GRAPH_SCHEMA_TAG}-")
        assert gs.current_version() == version
        assert gs.is_schema_version(version)

    def test_components_cover_every_input_that_decides_a_triple(self) -> None:
        components = gs.schema_components()
        assert set(components) == {
            "entity_canonicalisation",
            "extraction_prompt",
            "predicates",
        }
        # The vocabulary component is the live one, not a frozen copy.
        assert "depends_on" in components["predicates"]

    def test_reads_no_clock(self) -> None:
        """The id must not move because the day did.

        Asserted structurally rather than by sleeping: the module may not
        import a clock or a random source at all, at module scope or
        inside a function. A version that varied with time would make
        every stored stamp unfalsifiable.

        Parsed, not grepped -- the prose in this module says "at import
        time", and a substring check would have flagged its own docstring.
        """
        assert _clock_imports(gs.__file__) == set()
        # Positive control: the same checker over a module that genuinely
        # imports a clock reports it. Without this, a checker that always
        # returned the empty set would look identical.
        assert _clock_imports(kg_module.__file__) == {"datetime"}

    def test_malformed_stamp_is_refused_not_overwritten(self) -> None:
        with pytest.raises(gs.SchemaVersionError):
            gs.stamp({"schema_version": "not-a-schema-id"})
        # Positive control: the same call shape with a valid id is kept.
        good = gs.current_version()
        assert gs.stamp({"schema_version": good})["schema_version"] == good

    def test_stamp_does_not_mutate_its_input(self) -> None:
        original: dict = {}
        out = gs.stamp(original)
        assert original == {}
        assert out["schema_version"] == gs.current_version()

    def test_version_of_reads_absent_and_malformed_as_none(self) -> None:
        assert gs.version_of(None) is None
        assert gs.version_of({}) is None
        assert gs.version_of({"schema_version": "garbage"}) is None
        # Positive control: a real stamp is read back.
        live = gs.current_version()
        assert gs.version_of({"schema_version": live}) == live


class TestVocabularyWideningMovesTheId:
    """The mutation this whole feature exists to make visible."""

    def test_registering_a_predicate_changes_the_id(self, clean_predicates: None) -> None:
        before = gs.current_version()
        assert "funded_by" not in gs.predicate_vocabulary()
        Predicate.register("funded_by")
        after = gs.current_version()
        assert "funded_by" in gs.predicate_vocabulary()
        assert after != before, "widening the vocabulary must move the schema id"

    def test_id_returns_when_the_widening_is_undone(self, clean_predicates: None) -> None:
        """Round trip: the id is a function of the vocabulary, not a counter."""
        from mind_mem.knowledge_graph import _RUNTIME_PREDICATES

        before = gs.current_version()
        Predicate.register("funded_by")
        assert gs.current_version() != before
        _RUNTIME_PREDICATES.pop("funded_by")
        assert gs.current_version() == before


# ---------------------------------------------------------------------------
# The stamp on a written edge
# ---------------------------------------------------------------------------


class TestEdgeStamping:
    def test_add_edge_stamps_the_live_id(self, graph: KnowledgeGraph) -> None:
        edge = graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        assert edge.metadata["schema_version"] == gs.current_version()
        # And it is persisted, not just returned.
        stored = graph.edges_from("starga")[0]
        assert stored.metadata["schema_version"] == gs.current_version()

    def test_caller_supplied_stamp_is_preserved(self, graph: KnowledgeGraph) -> None:
        older = "gs1-000000000000"
        edge = graph.add_edge(
            "starga",
            "depends_on",
            "mindc",
            source_block_id=BLOCK_A,
            metadata={"schema_version": older},
        )
        assert edge.metadata["schema_version"] == older

    def test_malformed_stamp_refuses_the_write_and_mints_no_entities(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(gs.SchemaVersionError):
            graph.add_edge(
                "brand new entity",
                "depends_on",
                "another new entity",
                source_block_id=BLOCK_A,
                metadata={"schema_version": "junk"},
            )
        # The refusal left the database exactly as it found it: the
        # entity registry never resolved either endpoint.
        assert graph.entities.lookup("brand new entity") is None
        assert graph.entities.lookup("another new entity") is None
        assert graph.stats().edges == 0
        # Positive control: the same call without the junk stamp DOES
        # mint both entities and the edge, so the assertions above are
        # measuring the refusal and not a broken read.
        graph.add_edge(
            "brand new entity",
            "depends_on",
            "another new entity",
            source_block_id=BLOCK_A,
        )
        assert graph.entities.lookup("brand new entity") is not None
        assert graph.stats().edges == 1

    def test_proposal_stamp_survives_a_later_widening(self, graph: KnowledgeGraph, clean_predicates: None) -> None:
        """A June extraction approved in September records June's schema."""
        at_proposal = gs.current_version()
        proposal = graph.propose_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        assert proposal.metadata["schema_version"] == at_proposal

        Predicate.register("funded_by")
        assert gs.current_version() != at_proposal

        edge = graph.approve_edge(proposal.proposal_id)
        assert edge.metadata["schema_version"] == at_proposal
        assert edge.metadata["origin"] == "hitl_approved"


# ---------------------------------------------------------------------------
# Distinguishable / comparable / re-extractable
# ---------------------------------------------------------------------------


def _insert_unstamped_edge(db_path: str, *, source_block_id: str = BLOCK_B) -> None:
    """Write a pre-5.0.2 row: a real edge with empty metadata.

    Goes in through a second connection on purpose -- this is what the
    table looked like before stamping existed, and a report that cannot
    describe those rows has not made anything distinguishable.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO entities(id, canonical) VALUES ('legacy', 'legacy')",
        )
        conn.execute(
            "INSERT OR IGNORE INTO entities(id, canonical) VALUES ('legacy target', 'legacy target')",
        )
        conn.execute(
            "INSERT OR IGNORE INTO edges(subject, predicate, object, source_block_id, "
            "confidence, valid_from, valid_until, metadata) "
            "VALUES ('legacy', 'depends_on', 'legacy target', ?, 1.0, NULL, NULL, '{}')",
            (source_block_id,),
        )
        conn.commit()
    finally:
        conn.close()


class TestSchemaGenerations:
    def test_histogram_separates_stamped_from_unstamped(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        _insert_unstamped_edge(graph._db_path)
        histogram = graph.schema_version_histogram()
        assert histogram[gs.current_version()] == 1
        assert histogram[""] == 1, "unstamped edges must be counted, not dropped"
        assert sum(histogram.values()) == graph.stats().edges

    def test_edges_by_schema_version_selects_each_generation(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        _insert_unstamped_edge(graph._db_path)
        current = graph.edges_by_schema_version(gs.current_version())
        legacy = graph.edges_by_schema_version(None)
        assert [e.source_block_id for e in current] == [BLOCK_A]
        assert [e.source_block_id for e in legacy] == [BLOCK_B]

    def test_stale_is_empty_on_a_current_graph_and_the_reader_can_see_edges(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        # Positive control FIRST: prove the scan reaches this edge at all,
        # so the empty stale list below is a real answer and not a reader
        # that finds nothing anywhere.
        assert len(graph._all_edges()) == 1
        assert graph.stale_schema_edges() == []
        assert graph.stale_schema_source_blocks() == []

    def test_widening_the_vocabulary_makes_existing_edges_stale(self, graph: KnowledgeGraph, clean_predicates: None) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        assert graph.stale_schema_source_blocks() == []

        Predicate.register("funded_by")

        stale = graph.stale_schema_edges()
        assert [e.source_block_id for e in stale] == [BLOCK_A]
        assert graph.stale_schema_source_blocks() == [BLOCK_A]

    def test_unstamped_edges_are_the_oldest_generation(self, graph: KnowledgeGraph) -> None:
        graph.add_edge("starga", "depends_on", "mindc", source_block_id=BLOCK_A)
        _insert_unstamped_edge(graph._db_path)
        assert graph.stale_schema_source_blocks() == [BLOCK_B]

    def test_expired_edges_are_still_reported(self, graph: KnowledgeGraph) -> None:
        """A closed validity window is not an excuse to under-report.

        ``_all_edges`` passes ``include_expired=True`` so the schema
        report never depends on the wall clock.
        """
        graph.add_edge(
            "starga",
            "depends_on",
            "mindc",
            source_block_id=BLOCK_A,
            valid_from="2020-01-01T00:00:00Z",
            valid_until="2020-01-02T00:00:00Z",
        )
        assert graph.edges_from("starga") == []  # expired: hidden from normal reads
        assert len(graph._all_edges()) == 1  # but counted by the schema report
        assert graph.schema_version_histogram()[gs.current_version()] == 1


# ---------------------------------------------------------------------------
# The stamp on the ingest path (signal -> approval -> edge)
# ---------------------------------------------------------------------------


class TestIngestCarriesTheStamp:
    def test_staged_signal_tags_carry_the_schema_id(self) -> None:
        signals = relations_to_signals([_triple()])
        tags = signals[0]["structure"]["tags"]
        assert f"schema-version={gs.current_version()}" in tags

    def test_pending_listing_exposes_the_schema_id(self, workspace: str) -> None:
        stage_relation_signals(workspace, [_triple()], DATE)
        pending = pending_relation_signals(workspace)
        assert pending[0]["schema_version"] == gs.current_version()

    def test_approved_edge_records_the_extraction_time_schema(self, workspace: str, clean_predicates: None) -> None:
        at_extraction = gs.current_version()
        stage_relation_signals(workspace, [_triple()], DATE)
        sig_id = pending_relation_signals(workspace)[0]["signal_id"]

        Predicate.register("funded_by")
        assert gs.current_version() != at_extraction

        report = approve_relation_signals(workspace, [sig_id])
        assert report["applied"] == [sig_id], report["errors"]
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            edge = kg.edges_from("starga")[0]
        assert edge.metadata["origin"] == EDGE_ORIGIN
        assert edge.metadata["schema_version"] == at_extraction

    def test_malformed_schema_tag_does_not_propagate_a_false_claim(self, workspace: str) -> None:
        """A corrupted stamp reads as absent; approval then mints the live id.

        The alternative -- trusting the tag -- would let an edited
        SIGNALS.md assert any provenance it liked.
        """
        stage_relation_signals(workspace, [_triple()], DATE)
        path = os.path.join(workspace, "intelligence", "SIGNALS.md")
        text = Path(path).read_text(encoding="utf-8")
        assert f"schema-version={gs.current_version()}" in text  # positive control
        Path(path).write_text(
            text.replace(f"schema-version={gs.current_version()}", "schema-version=bogus"),
            encoding="utf-8",
        )
        pending = pending_relation_signals(workspace)
        assert pending[0]["schema_version"] is None
        sig_id = pending[0]["signal_id"]
        report = approve_relation_signals(workspace, [sig_id])
        assert report["applied"] == [sig_id], report["errors"]
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            edge = kg.edges_from("starga")[0]
        assert edge.metadata["schema_version"] == gs.current_version()


# ---------------------------------------------------------------------------
# Re-extraction: the corpus slice an old schema left behind
# ---------------------------------------------------------------------------


class TestReextraction:
    def test_stale_blocks_is_empty_without_a_graph(self, workspace: str) -> None:
        assert not os.path.isfile(default_db_path(workspace))
        assert stale_schema_blocks(workspace) == []
        # ...and asking did not create one. A read-only report that mints
        # a database is not read-only.
        assert not os.path.isfile(default_db_path(workspace))

    def test_stale_blocks_names_exactly_the_blocks_to_re_extract(self, workspace: str, clean_predicates: None) -> None:
        stage_relation_signals(workspace, [_triple(), _triple(source=BLOCK_B)], DATE)
        sig_ids = [s["signal_id"] for s in pending_relation_signals(workspace)]
        approve_relation_signals(workspace, sig_ids)
        assert stale_schema_blocks(workspace) == []  # positive control: graph is current

        Predicate.register("funded_by")
        assert stale_schema_blocks(workspace) == sorted({BLOCK_A, BLOCK_B})

    def test_backfill_restricted_to_nothing_scans_nothing(self, workspace: str) -> None:
        corpus = [
            {"_id": BLOCK_A, "excerpt": "starga depends on mindc"},
            {"_id": BLOCK_B, "excerpt": "mindc depends on llvm"},
        ]
        calls: list[str] = []

        def _extract(text: str) -> list[dict]:
            calls.append(text)
            return []

        empty = backfill(workspace, corpus=corpus, extract_fn=_extract, restrict_to_blocks=[])
        assert empty["blocks_scanned"] == 0
        assert empty["restricted_to_blocks"] == 0
        assert calls == []
        # Positive control: the SAME corpus and extractor, unrestricted,
        # scans both -- so "0" above is the restriction working, not a
        # corpus the loader could not read.
        allb = backfill(workspace, corpus=corpus, extract_fn=_extract, restrict_to_blocks=None)
        assert allb["blocks_scanned"] == 2
        assert allb["restricted_to_blocks"] is None
        assert len(calls) == 2

    def test_backfill_restriction_selects_the_named_blocks(self, workspace: str) -> None:
        corpus = [
            {"_id": BLOCK_A, "excerpt": "starga depends on mindc"},
            {"_id": BLOCK_B, "excerpt": "mindc depends on llvm"},
        ]
        seen: list[str] = []
        report = backfill(
            workspace,
            corpus=corpus,
            extract_fn=lambda text: seen.append(text) or [],  # type: ignore[func-returns-value]
            restrict_to_blocks=[BLOCK_B],
        )
        assert report["blocks_scanned"] == 1
        assert seen == ["mindc depends on llvm"]

    def test_backfill_report_names_the_schema_it_would_stage_under(self, workspace: str) -> None:
        report = backfill(workspace, corpus=[], extract_fn=lambda _t: [])
        assert report["schema_version"] == gs.current_version()


class TestSchemaReport:
    def test_report_on_an_empty_workspace_is_honest(self, workspace: str) -> None:
        report = schema_report(workspace)
        assert report["current"] == gs.current_version()
        assert report["versions"] == {}
        assert report["stale_edges"] == 0
        assert report["stale_blocks"] == []
        assert set(report["components"]) == {
            "entity_canonicalisation",
            "extraction_prompt",
            "predicates",
        }

    def test_report_counts_generations_and_names_stale_blocks(self, workspace: str, clean_predicates: None) -> None:
        stage_relation_signals(workspace, [_triple()], DATE)
        sig_id = pending_relation_signals(workspace)[0]["signal_id"]
        approve_relation_signals(workspace, [sig_id])
        current = schema_report(workspace)
        assert current["versions"] == {gs.current_version(): 1}
        assert current["stale_edges"] == 0

        Predicate.register("funded_by")
        after = schema_report(workspace)
        assert after["current"] != current["current"]
        assert after["stale_edges"] == 1
        assert after["stale_blocks"] == [BLOCK_A]


# ---------------------------------------------------------------------------
# Wiring: the CLI reaches all of it from a real entry point
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_flags_are_registered_on_graph_backfill(self) -> None:
        from mind_mem.mm_cli import build_parser

        args = build_parser().parse_args(["graph-backfill", "--schema"])
        assert args.schema is True
        assert args.reextract_stale is False
        args2 = build_parser().parse_args(["graph-backfill", "--reextract-stale"])
        assert args2.reextract_stale is True
        assert args2.func.__name__ == "_cmd_graph_backfill"

    def test_schema_report_prints_through_the_command(self, workspace: str, capsys, monkeypatch) -> None:
        from mind_mem import mm_cli

        monkeypatch.setattr(mm_cli, "_workspace", lambda: workspace)
        args = mm_cli.build_parser().parse_args(["graph-backfill", "--schema", "--json"])
        assert args.func(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["current"] == gs.current_version()
        assert payload["stale_blocks"] == []

    def test_reextract_stale_restricts_the_scan_through_the_command(
        self, workspace: str, capsys, monkeypatch, clean_predicates: None
    ) -> None:
        from mind_mem import mm_cli

        monkeypatch.setattr(mm_cli, "_workspace", lambda: workspace)
        stage_relation_signals(workspace, [_triple()], DATE)
        sig_id = pending_relation_signals(workspace)[0]["signal_id"]
        approve_relation_signals(workspace, [sig_id])

        scanned: list[str] = []
        monkeypatch.setattr(
            "mind_mem.graph_ingest._default_extract_fn",
            lambda ws: lambda text: scanned.append(text) or [],
        )
        monkeypatch.setattr(
            "mind_mem.graph_ingest._load_corpus",
            lambda ws: [{"_id": BLOCK_A, "excerpt": "starga depends on mindc"}],
        )

        args = mm_cli.build_parser().parse_args(["graph-backfill", "--reextract-stale", "--json"])
        assert args.func(args) == 0
        current_run = json.loads(capsys.readouterr().out)
        assert current_run["restricted_to_blocks"] == 0
        assert current_run["blocks_scanned"] == 0
        assert scanned == []

        # Positive control: widen the vocabulary and the SAME command now
        # finds the block to re-extract.
        Predicate.register("funded_by")
        args = mm_cli.build_parser().parse_args(["graph-backfill", "--reextract-stale", "--json"])
        assert args.func(args) == 0
        stale_run = json.loads(capsys.readouterr().out)
        assert stale_run["restricted_to_blocks"] == 1
        assert stale_run["blocks_scanned"] == 1
        assert scanned == ["starga depends on mindc"]
