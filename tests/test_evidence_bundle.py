"""v3.3.0 Tier 3 #7 — structured evidence bundle.

``build_bundle`` should pre-digest recall results into typed fact /
relation / timeline / entity records. Extraction is rule-based; all
tests are pure function checks (no I/O).
"""

from __future__ import annotations

import pytest

from mind_mem.evidence_bundle import (
    EntityRef,
    EvidenceBundle,
    Fact,
    Relation,
    TimelineEvent,
    build_bundle,
)


def _decision(bid: str, statement: str = "Decision content", **extra) -> dict:
    return {"_id": bid, "Statement": statement, "Status": "active", **extra}


class TestFacts:
    def test_statement_field_becomes_fact(self) -> None:
        b = _decision("D-20260420-001", statement="Use PostgreSQL in prod.")
        bundle = build_bundle("why postgres?", [b])
        assert len(bundle.facts) == 1
        assert bundle.facts[0].claim == "Use PostgreSQL in prod."
        assert bundle.facts[0].source_id == "D-20260420-001"
        assert bundle.facts[0].field_name == "Statement"

    def test_confidence_includes_verified_tier_bump(self) -> None:
        b = _decision("D-20260420-001", Tier="VERIFIED")
        bundle = build_bundle("q", [b])
        assert bundle.facts[0].confidence >= 0.9

    def test_confidence_drops_for_superseded_status(self) -> None:
        b = _decision("D-20260420-001", Status="superseded")
        bundle = build_bundle("q", [b])
        assert bundle.facts[0].confidence < 0.5

    def test_blocks_without_claim_field_skipped(self) -> None:
        b = {"_id": "D-20260420-001", "type": "decision", "Status": "active"}
        bundle = build_bundle("q", [b])
        assert bundle.facts == []


class TestRelations:
    def test_supersedes_field_becomes_relation(self) -> None:
        b = _decision("D-20260420-002", Supersedes="D-20260420-001")
        bundle = build_bundle("q", [b])
        rels = [(r.subject, r.predicate, r.object) for r in bundle.relations]
        assert ("D-20260420-002", "supersedes", "D-20260420-001") in rels

    def test_list_valued_relation_expands(self) -> None:
        b = _decision(
            "D-20260420-003",
            Dependencies=["D-20260420-001", "D-20260420-002"],
        )
        bundle = build_bundle("q", [b])
        objs = {r.object for r in bundle.relations if r.predicate == "depends_on"}
        assert objs == {"D-20260420-001", "D-20260420-002"}

    def test_free_text_mentions_skipped(self) -> None:
        """Free-text without a block-ID pattern doesn't pollute relations."""
        b = _decision("D-20260420-001", Supersedes="some vague reference")
        bundle = build_bundle("q", [b])
        assert bundle.relations == []

    def test_self_reference_skipped(self) -> None:
        b = _decision("D-20260420-001", Supersedes="D-20260420-001")
        bundle = build_bundle("q", [b])
        assert bundle.relations == []

    def test_dedup_duplicate_relations(self) -> None:
        """If the same (subj, pred, obj) appears twice, keep one."""
        b1 = _decision("D-20260420-002", Supersedes="D-20260420-001")
        b2 = _decision("D-20260420-002", Supersedes="D-20260420-001")  # duplicate
        bundle = build_bundle("q", [b1, b2])
        relations = [r for r in bundle.relations if r.subject == "D-20260420-002"]
        assert len(relations) == 1


class TestTimeline:
    def test_iso_date_becomes_entry(self) -> None:
        b = _decision(
            "D-20260420-001",
            Date="2026-04-20",
            Event="PostgreSQL migration rollout",
        )
        bundle = build_bundle("q", [b])
        assert len(bundle.timeline) == 1
        assert bundle.timeline[0].date == "2026-04-20"
        assert "rollout" in bundle.timeline[0].event

    def test_missing_date_dropped(self) -> None:
        b = _decision("D-20260420-001")
        bundle = build_bundle("q", [b])
        assert bundle.timeline == []

    def test_non_iso_date_dropped(self) -> None:
        b = _decision("D-20260420-001", Date="last Tuesday", Event="x")
        bundle = build_bundle("q", [b])
        assert bundle.timeline == []

    def test_timeline_sorted_ascending(self) -> None:
        b1 = _decision("D-001", Date="2026-01-10", Event="a")
        b2 = _decision("D-002", Date="2025-12-20", Event="b")
        b3 = _decision("D-003", Date="2026-06-01", Event="c")
        # Fix IDs to match canonical prefix-regex used by relations
        for b, bid in zip([b1, b2, b3], ["D-20260110-001", "D-20251220-001", "D-20260601-001"]):
            b["_id"] = bid
        bundle = build_bundle("q", [b1, b2, b3])
        dates = [t.date for t in bundle.timeline]
        assert dates == sorted(dates)


class TestEntities:
    def test_person_block_produces_entity(self) -> None:
        b = {
            "_id": "PER-001",
            "Name": "Alice Johnson",
            "Type": "person",
            "Statement": "x",
        }
        bundle = build_bundle("q", [b])
        assert len(bundle.entities) == 1
        assert bundle.entities[0] == EntityRef(id="PER-001", name="Alice Johnson", type="person")

    def test_decision_block_is_not_entity(self) -> None:
        b = _decision("D-20260420-001")
        bundle = build_bundle("q", [b])
        assert bundle.entities == []

    def test_entity_dedup(self) -> None:
        b1 = {"_id": "PER-001", "Name": "Alice", "Statement": "x"}
        b2 = {"_id": "PER-001", "Name": "Alice duplicate", "Statement": "y"}
        bundle = build_bundle("q", [b1, b2])
        assert len(bundle.entities) == 1


class TestBundleSerialisation:
    def test_to_dict_is_json_shaped(self) -> None:
        b = _decision("D-20260420-001", Date="2026-04-20", Event="launch")
        bundle = build_bundle("q", [b])
        out = bundle.to_dict()
        assert set(out.keys()) == {"query", "facts", "relations", "timeline", "entities", "source_blocks"}
        assert all(isinstance(f, dict) for f in out["facts"])

    def test_source_blocks_excluded_when_requested(self) -> None:
        b = _decision("D-20260420-001")
        bundle = build_bundle("q", [b], include_source_blocks=False)
        assert bundle.source_blocks == []

    def test_empty_results(self) -> None:
        bundle = build_bundle("q", [])
        assert bundle.facts == bundle.relations == bundle.timeline == bundle.entities == []
        assert bundle.source_blocks == []
