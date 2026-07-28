"""Corpus → typed knowledge-graph wiring (extraction → HITL signal → apply).

The extraction side proposes edges as SIGNALS.md entries (pattern
``auto-capture-relation``); nothing writes the knowledge graph until an
operator approves the staged signal. ``approve_relation_signals`` is the
apply step: it calls ``KnowledgeGraph.add_edge`` with the real
``source_block_id`` and a ``valid_from`` timestamp, then flips the signal
status so a second approve is a no-op.

The model call is an injected callable throughout — the loop proves out
with zero API calls, per the Group K wedge guardrail.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.graph_ingest import (
    EDGE_ORIGIN,
    RELATION_PATTERN,
    RelationTriple,
    approve_relation_signals,
    attach_source_excerpts,
    backfill,
    pending_relation_signals,
    relations_to_signals,
    stage_relation_signals,
)
from mind_mem.knowledge_graph import KnowledgeGraph, default_db_path

DATE = "2026-07-27"
BLOCK_A = "D-20260101-001"
BLOCK_B = "D-20260101-002"


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


class TestRelationTriple:
    def test_rejects_unknown_predicate(self) -> None:
        with pytest.raises(ValueError):
            _triple(predicate="totally_made_up")

    def test_rejects_empty_subject(self) -> None:
        with pytest.raises(ValueError):
            _triple(subject="   ")

    def test_rejects_missing_source_block(self) -> None:
        with pytest.raises(ValueError):
            _triple(source="")

    def test_clamps_nothing_but_validates_confidence(self) -> None:
        with pytest.raises(ValueError):
            _triple(confidence=1.5)


class TestExtractRelations:
    """llm_extractor.extract_relations — schema-constrained triple parsing."""

    def test_parses_and_validates_triples(self, monkeypatch) -> None:
        import mind_mem.llm_extractor as llm

        payload = (
            '[{"subject": "STARGA", "predicate": "depends_on", "object": "mindc",'
            ' "confidence": 0.9},'
            ' {"subject": "x", "predicate": "not_a_predicate", "object": "y"},'
            ' {"subject": "", "predicate": "part_of", "object": "z"},'
            ' {"subject": "a", "predicate": "part_of", "object": "b",'
            ' "confidence": "bogus"}]'
        )
        monkeypatch.setattr(llm, "is_available", lambda backend="auto", **kw: True)
        monkeypatch.setattr(llm, "_query_llm", lambda prompt, model, backend="auto", **kw: payload)
        out = llm.extract_relations("some text", model="m", backend="ollama")
        # Unknown predicate and empty subject are dropped; bogus
        # confidence falls back to the 0.5 default.
        assert len(out) == 2
        assert out[0] == {
            "subject": "STARGA",
            "predicate": "depends_on",
            "object": "mindc",
            "confidence": 0.9,
        }
        assert out[1]["confidence"] == 0.5

    def test_empty_when_no_backend(self, monkeypatch) -> None:
        import mind_mem.llm_extractor as llm

        monkeypatch.setattr(llm, "is_available", lambda backend="auto", **kw: False)
        assert llm.extract_relations("text") == []

    def test_empty_text_short_circuits(self) -> None:
        import mind_mem.llm_extractor as llm

        assert llm.extract_relations("   ") == []

    def test_prompt_carries_predicate_vocabulary(self, monkeypatch) -> None:
        import mind_mem.llm_extractor as llm

        seen: list[str] = []

        def fake_query(prompt: str, model: str, backend: str = "auto", **kw: object) -> str:
            seen.append(prompt)
            return "[]"

        monkeypatch.setattr(llm, "is_available", lambda backend="auto", **kw: True)
        monkeypatch.setattr(llm, "_query_llm", fake_query)
        llm.extract_relations("text", model="m", backend="ollama")
        assert seen and "depends_on" in seen[0] and "authored_by" in seen[0]


class TestRelationsToSignals:
    def test_signal_shape(self) -> None:
        sigs = relations_to_signals([_triple()])
        assert len(sigs) == 1
        sig = sigs[0]
        assert sig["pattern"] == RELATION_PATTERN
        assert sig["type"] == "relation"
        st = sig["structure"]
        assert st["subject"] == "starga"
        assert st["object"] == "mindc"
        tags = st["tags"]
        assert "graph-edge" in tags
        assert "predicate=depends_on" in tags
        assert f"source-block={BLOCK_A}" in tags
        assert "edge-confidence=0.7" in tags


class TestStageAndApprove:
    def test_stage_never_touches_graph(self, workspace) -> None:
        written = stage_relation_signals(workspace, [_triple()], DATE)
        assert written == 1
        assert not os.path.isfile(default_db_path(workspace))

    def test_pending_roundtrip(self, workspace) -> None:
        stage_relation_signals(workspace, [_triple()], DATE)
        pending = pending_relation_signals(workspace)
        assert len(pending) == 1
        p = pending[0]
        assert p["subject"] == "starga"
        assert p["predicate"] == "depends_on"
        assert p["object"] == "mindc"
        assert p["source_block_id"] == BLOCK_A
        assert p["confidence"] == pytest.approx(0.7)
        assert p["signal_id"].startswith("SIG-")

    def test_approve_applies_edge_with_provenance(self, workspace) -> None:
        stage_relation_signals(workspace, [_triple()], DATE)
        sig_id = pending_relation_signals(workspace)[0]["signal_id"]
        report = approve_relation_signals(workspace, [sig_id])
        assert report["applied"] == [sig_id]
        assert report["errors"] == {}
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            edges = kg.edges_from("starga")
        assert len(edges) == 1
        e = edges[0]
        assert e.object == "mindc"
        assert e.source_block_id == BLOCK_A
        assert e.valid_from is not None
        assert e.metadata.get("origin") == EDGE_ORIGIN
        assert e.metadata.get("signal_id") == sig_id
        # Status flipped — nothing left pending.
        assert pending_relation_signals(workspace) == []

    def test_double_approve_is_rejected(self, workspace) -> None:
        stage_relation_signals(workspace, [_triple()], DATE)
        sig_id = pending_relation_signals(workspace)[0]["signal_id"]
        approve_relation_signals(workspace, [sig_id])
        report = approve_relation_signals(workspace, [sig_id])
        assert report["applied"] == []
        assert sig_id in report["errors"]

    def test_approve_unknown_signal_errors(self, workspace) -> None:
        report = approve_relation_signals(workspace, ["SIG-20990101-001"])
        assert report["applied"] == []
        assert "SIG-20990101-001" in report["errors"]

    def test_comma_in_source_block_id_roundtrips_exactly(self, workspace) -> None:
        """Tags serialize comma-joined; a comma inside a tag VALUE must
        not split mid-value and corrupt the provenance anchor."""
        weird = "D-20260101-001,extra-segment"
        stage_relation_signals(workspace, [_triple(source=weird)], DATE)
        pending = pending_relation_signals(workspace)
        assert len(pending) == 1
        assert pending[0]["source_block_id"] == weird
        assert pending[0]["predicate"] == "depends_on"
        assert pending[0]["confidence"] == pytest.approx(0.7)

    def test_percent_and_comma_escape_roundtrip(self, workspace) -> None:
        """Escape-of-escape: values containing the escape character
        itself must also survive the round trip byte-exact."""
        weird = "D-100%,x%2C-y"
        stage_relation_signals(workspace, [_triple(source=weird)], DATE)
        pending = pending_relation_signals(workspace)
        assert len(pending) == 1
        assert pending[0]["source_block_id"] == weird

    def test_comma_source_block_survives_approve(self, workspace) -> None:
        weird = "D-20260101-001,extra"
        stage_relation_signals(workspace, [_triple(source=weird)], DATE)
        sig_id = pending_relation_signals(workspace)[0]["signal_id"]
        report = approve_relation_signals(workspace, [sig_id])
        assert report["applied"] == [sig_id]
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            edges = kg.edges_from("starga")
        assert edges[0].source_block_id == weird


class TestPendingExcerpts:
    """HITL review surface: pending relations carry a bounded excerpt
    of their source block so the operator can verify the relation
    against the text before approving."""

    def test_attach_source_excerpts_bounded(self, workspace) -> None:
        stage_relation_signals(workspace, [_triple()], DATE)
        pending = pending_relation_signals(workspace)
        corpus = [{"_id": BLOCK_A, "excerpt": "STARGA depends on mindc " + "x" * 400}]
        enriched = attach_source_excerpts(workspace, pending, corpus=corpus)
        assert enriched[0]["source_excerpt"].startswith("STARGA depends on mindc")
        assert len(enriched[0]["source_excerpt"]) <= 160
        # Immutable: input dicts are not mutated.
        assert "source_excerpt" not in pending[0]

    def test_missing_source_block_gives_empty_excerpt(self, workspace) -> None:
        stage_relation_signals(workspace, [_triple()], DATE)
        pending = pending_relation_signals(workspace)
        enriched = attach_source_excerpts(workspace, pending, corpus=[])
        assert enriched[0]["source_excerpt"] == ""

    def test_list_pending_cli_shows_excerpt(self, workspace, monkeypatch, capsys) -> None:
        import mind_mem.graph_ingest as gi
        from mind_mem.mm_cli import main

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.setattr(gi, "_load_corpus", lambda ws: [{"_id": BLOCK_A, "excerpt": "STARGA depends on mindc for compilation"}])
        stage_relation_signals(workspace, [_triple()], DATE)
        rc = main(["graph-backfill", "--list-pending"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "starga depends_on mindc" in out
        assert "STARGA depends on mindc for compilation" in out

    def test_list_pending_json_includes_excerpt(self, workspace, monkeypatch, capsys) -> None:
        import mind_mem.graph_ingest as gi
        from mind_mem.mm_cli import main

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.setattr(gi, "_load_corpus", lambda ws: [{"_id": BLOCK_A, "excerpt": "source text here"}])
        stage_relation_signals(workspace, [_triple()], DATE)
        rc = main(["graph-backfill", "--list-pending", "--json"])
        assert rc == 0
        rows = json.loads(capsys.readouterr().out)
        assert rows[0]["source_excerpt"] == "source text here"
        assert rows[0]["source_block_id"] == BLOCK_A


class TestBackfill:
    @staticmethod
    def _corpus() -> list[dict]:
        return [
            {"_id": BLOCK_A, "excerpt": "STARGA depends on mindc"},
            {"_id": BLOCK_B, "excerpt": "nothing to extract here"},
        ]

    @staticmethod
    def _extract_fn(text: str) -> list[dict]:
        if "depends" in text:
            return [
                {"subject": "starga", "predicate": "depends_on", "object": "mindc", "confidence": 0.9},
                {"subject": "mindc", "predicate": "part_of", "object": "mind", "confidence": 0.6},
            ]
        return []

    def test_dry_run_yield_metrics(self, workspace) -> None:
        report = backfill(
            workspace,
            corpus=self._corpus(),
            extract_fn=self._extract_fn,
            dry_run=True,
        )
        assert report["dry_run"] is True
        assert report["blocks_scanned"] == 2
        assert report["blocks_with_edges"] == 1
        assert report["edges_extracted"] == 2
        assert report["edges_per_block"] == pytest.approx(1.0)
        assert report["predicate_histogram"] == {"depends_on": 1, "part_of": 1}
        assert report["signals_written"] == 0
        # Dry run leaves no trace anywhere.
        assert pending_relation_signals(workspace) == []
        assert not os.path.isfile(default_db_path(workspace))

    def test_write_mode_stages_signals_only(self, workspace) -> None:
        report = backfill(
            workspace,
            corpus=self._corpus(),
            extract_fn=self._extract_fn,
            dry_run=False,
        )
        assert report["signals_written"] == 2
        assert len(pending_relation_signals(workspace)) == 2
        # Still no direct graph writes — HITL gate holds.
        assert not os.path.isfile(default_db_path(workspace))

    def test_limit_bounds_scan(self, workspace) -> None:
        report = backfill(
            workspace,
            corpus=self._corpus(),
            extract_fn=self._extract_fn,
            dry_run=True,
            limit=1,
        )
        assert report["blocks_scanned"] == 1

    def test_blocks_without_id_are_skipped(self, workspace) -> None:
        corpus = [{"excerpt": "STARGA depends on mindc"}]
        report = backfill(workspace, corpus=corpus, extract_fn=self._extract_fn, dry_run=True)
        assert report["blocks_scanned"] == 0
        assert report["edges_extracted"] == 0


class TestCliVerb:
    """`mm graph-backfill` — dry-run by default, JSON on demand."""

    def test_dry_run_json_smoke(self, workspace, monkeypatch, capsys) -> None:
        import mind_mem.graph_ingest as gi
        from mind_mem.mm_cli import main

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        # No live model in CI — inject a null extractor via the default
        # binding so the verb path itself is exercised.
        monkeypatch.setattr(gi, "_default_extract_fn", lambda ws: lambda text: [])
        rc = main(["graph-backfill", "--limit", "3", "--json"])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["dry_run"] is True
        assert "predicate_histogram" in report

    def test_approve_via_cli(self, workspace, monkeypatch, capsys) -> None:
        from mind_mem.mm_cli import main

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        stage_relation_signals(workspace, [_triple()], DATE)
        sig_id = pending_relation_signals(workspace)[0]["signal_id"]
        rc = main(["graph-backfill", "--approve", sig_id])
        assert rc == 0
        assert f"applied  {sig_id}" in capsys.readouterr().out
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            assert kg.stats().edges == 1
