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
from mind_mem.core_export import (
    OKF_VERSION,
    export_to_okf,
    import_okf_bundle,
    write_okf_bundle,
)


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
        # Bare YYYY-MM-DD is widened to ISO-8601 datetime (OKF convention).
        assert unit["timestamp"] == "2026-06-13T00:00:00Z"

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


class TestOkfTypeAndCitations:
    def test_type_derived_from_id_prefix(self, tmp_core: str) -> None:
        # Build path supplies no real `type`; it must come from the id prefix,
        # not the masking "block" default (BUG-2).
        core = _load(tmp_core, [{"_id": "PRJ-mind"}, {"_id": "D-20260613-001"}])
        units = export_to_okf(core)["units"]
        by_id = {u["id"]: u for u in units}
        assert by_id["PRJ-mind"]["type"] == "project"
        assert by_id["D-20260613-001"]["type"] == "decision"

    def test_type_never_empty_or_block(self, tmp_core: str) -> None:
        core = _load(tmp_core, [{"_id": "X-unknown", "type": "block"}])
        unit = export_to_okf(core)["units"][0]
        assert unit["type"]  # non-empty (OKF required field)
        assert unit["type"] != "block"

    def test_citations_emitted_in_bundle(self, tmp_core: str, tmp_path) -> None:
        block = {
            "_id": "D-1",
            "type": "decision",
            "Statement": "x",
            "Sources": ["arXiv:2401.1", "github.com/star-ga/mind"],
        }
        core = _load(tmp_core, [block])
        out = write_okf_bundle(core, tmp_path / "bundle")
        body = (out / "D-1.md").read_text()
        assert "# Citations" in body
        assert "arXiv:2401.1" in body
        assert "github.com/star-ga/mind" in body


class TestOkfBundleWriter:
    def test_writes_conformant_bundle(self, tmp_core: str, tmp_path) -> None:
        core = _load(
            tmp_core,
            [
                {"_id": "PRJ-mind", "type": "project", "Name": "mind", "Resource": "https://github.com/star-ga/mind"},
                {"_id": "D-1", "type": "decision", "Statement": "Ship OKF."},
            ],
            [{"subject": "D-1", "predicate": "concerns", "object": "PRJ-mind"}],
        )
        out = write_okf_bundle(core, tmp_path / "bundle")
        assert (out / "index.md").exists()
        assert (out / "log.md").exists()
        prj = (out / "PRJ-mind.md").read_text()
        # Required `type` is the first frontmatter field.
        assert prj.startswith("---\ntype: project")
        assert "resource: https://github.com/star-ga/mind" in prj
        # Edge rendered as a bundle-relative markdown link on the subject.
        dec = (out / "D-1.md").read_text()
        assert "# Relationships" in dec
        assert "[PRJ-mind](./PRJ-mind.md)" in dec

    def test_index_lists_every_concept(self, tmp_core: str, tmp_path) -> None:
        core = _load(
            tmp_core,
            [{"_id": "D-1", "type": "decision"}, {"_id": "T-1", "type": "task"}],
        )
        out = write_okf_bundle(core, tmp_path / "bundle")
        index = (out / "index.md").read_text()
        assert "(./D-1.md)" in index
        assert "(./T-1.md)" in index


class TestOkfImportRoundTrip:
    def test_round_trip_preserves_core_fields(self, tmp_core: str, tmp_path) -> None:
        core = _load(
            tmp_core,
            [
                {"_id": "PRJ-mind", "type": "project", "Name": "mind", "Resource": "https://github.com/star-ga/mind", "Tags": ["wedge"]},
                {"_id": "D-1", "type": "decision", "Statement": "Ship OKF."},
            ],
        )
        out = write_okf_bundle(core, tmp_path / "bundle")
        blocks = import_okf_bundle(out)
        by_id = {b["_id"]: b for b in blocks}
        assert by_id["PRJ-mind"]["type"] == "project"
        assert by_id["PRJ-mind"]["Title"] == "mind"
        assert by_id["PRJ-mind"]["Resource"] == "https://github.com/star-ga/mind"
        assert by_id["PRJ-mind"]["Tags"] == ["wedge"]
        assert by_id["D-1"]["Statement"] == "Ship OKF."
        # index.md / log.md are not imported as concepts.
        assert "bundle" not in by_id

    def test_import_keys_satisfy_capitalized_grammar(self, tmp_core: str, tmp_path) -> None:
        core = _load(tmp_core, [{"_id": "D-1", "type": "decision", "Statement": "x"}])
        out = write_okf_bundle(core, tmp_path / "bundle")
        blocks = import_okf_bundle(out)
        for b in blocks:
            for key in b:
                if key in ("_id", "type"):
                    continue
                assert key[0].isupper(), f"{key} must satisfy ^[A-Z] grammar"
