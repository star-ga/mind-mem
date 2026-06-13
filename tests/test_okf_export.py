# Copyright 2026 STARGA, Inc.
"""Tests for OKF (Open Knowledge Format) interop export.

OKF is adopted as an import/export *envelope only* — the export is lossy
by design (mind-mem's governance/contradiction/retrieval/evidence layers
sit above the format and are deliberately not represented).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mind_mem.context_core import build_core, load_core
from mind_mem.core_export import OKF_VERSION, export_to_okf


@pytest.fixture()
def tmp_core():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        yield str(Path(td) / "bundle.mmcore")


def _load(tmp_core, blocks, edges=None):
    build_core(
        tmp_core,
        namespace="proj",
        version="1.0",
        blocks=blocks,
        edges=edges or [],
    )
    return load_core(tmp_core)


class TestOkfEnvelope:
    def test_envelope_has_okf_version_and_source(self, tmp_core: str) -> None:
        core = _load(tmp_core, [{"_id": "D-1", "type": "decision"}])
        out = export_to_okf(core)
        assert out["okf_version"] == OKF_VERSION
        assert out["source"] == "mind-mem"
        assert out["id"] == "urn:mindmem:proj:1.0"
        assert "manifest" in out

    def test_block_maps_to_okf_unit(self, tmp_core: str) -> None:
        block = {
            "_id": "D-1",
            "type": "decision",
            "Statement": "Adopt OKF as an envelope.",
            "Tags": ["okf", "interop"],
            "Date": "2026-06-13",
        }
        core = _load(tmp_core, [block])
        unit = export_to_okf(core)["units"][0]
        assert unit["id"] == "D-1"
        assert unit["type"] == "decision"
        assert unit["description"] == "Adopt OKF as an envelope."
        assert unit["tags"] == ["okf", "interop"]
        assert unit["timestamp"] == "2026-06-13"

    def test_resource_uri_is_carried(self, tmp_core: str) -> None:
        block = {
            "_id": "PRJ-1",
            "type": "project",
            "Name": "mind",
            "Resource": "https://github.com/star-ga/mind",
        }
        core = _load(tmp_core, [block])
        unit = export_to_okf(core)["units"][0]
        assert unit["resource"] == "https://github.com/star-ga/mind"
        assert unit["title"] == "mind"

    def test_lowercase_resource_also_accepted(self, tmp_core: str) -> None:
        block = {"_id": "B-1", "type": "block", "resource": "urn:arxiv:2401.1"}
        core = _load(tmp_core, [block])
        unit = export_to_okf(core)["units"][0]
        assert unit["resource"] == "urn:arxiv:2401.1"

    def test_edges_become_relations(self, tmp_core: str) -> None:
        core = _load(
            tmp_core,
            [{"_id": "A", "type": "decision"}, {"_id": "B", "type": "task"}],
            [{"subject": "A", "predicate": "blocks", "object": "B"}],
        )
        rels = export_to_okf(core)["relations"]
        assert rels == [{"subject": "A", "predicate": "blocks", "object": "B"}]

    def test_moat_fields_are_dropped(self, tmp_core: str) -> None:
        # Governance/retrieval/evidence fields must NOT appear in the OKF unit.
        block = {
            "_id": "D-1",
            "type": "decision",
            "Statement": "x",
            "Status": "active",
            "rrf_score": 0.99,
            "evidence_hash": "deadbeef",
        }
        core = _load(tmp_core, [block])
        unit = export_to_okf(core)["units"][0]
        assert "Status" not in unit
        assert "rrf_score" not in unit
        assert "evidence_hash" not in unit
        assert set(unit) <= {"id", "type", "title", "description", "resource", "timestamp", "tags"}
