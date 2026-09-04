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
    _OKF_UNIT_FIELDS,
    OKF_VERSION,
    _block_to_okf_unit,
    _okf_receipt,
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
        # Governance internals / retrieval / evidence fields must NOT appear in
        # the OKF unit. The raw capitalised `Status` never leaks (it is mapped
        # to OKF's lowercase `status`, an OKF-own field); retrieval score and
        # evidence hash are dropped entirely.
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
        assert "Status" not in unit  # raw moat key never leaks…
        assert unit["status"] == "stable"  # …but the OKF-own lifecycle field is derived
        assert "rrf_score" not in unit
        assert "evidence_hash" not in unit
        # Everything emitted is inside the OKF-conformant allow-list.
        assert set(unit) <= _OKF_UNIT_FIELDS


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
        body = (out / "D-1.md").read_text(encoding="utf-8")
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
        prj = (out / "PRJ-mind.md").read_text(encoding="utf-8")
        # Required `type` is the first frontmatter field.
        assert prj.startswith("---\ntype: project")
        assert "resource: https://github.com/star-ga/mind" in prj
        # Edge rendered as a bundle-relative markdown link on the subject.
        dec = (out / "D-1.md").read_text(encoding="utf-8")
        assert "# Relationships" in dec
        assert "[PRJ-mind](./PRJ-mind.md)" in dec

    def test_index_lists_every_concept(self, tmp_core: str, tmp_path) -> None:
        core = _load(
            tmp_core,
            [{"_id": "D-1", "type": "decision"}, {"_id": "T-1", "type": "task"}],
        )
        out = write_okf_bundle(core, tmp_path / "bundle")
        index = (out / "index.md").read_text(encoding="utf-8")
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


class TestOkfV02TrustFamily:
    """OKF v0.2 trust/lifecycle family — DERIVED from recorded signals, never
    self-asserted (issue #550, "better than proposed")."""

    def test_version_is_v02(self, tmp_core: str) -> None:
        core = _load(tmp_core, [{"_id": "D-1", "type": "decision"}])
        assert export_to_okf(core)["okf_version"] == "0.2"
        assert OKF_VERSION == "0.2"

    def test_generated_is_machine_actor_never_human(self, tmp_core: str) -> None:
        core = _load(tmp_core, [{"_id": "D-1", "type": "decision", "Date": "2026-06-13"}])
        unit = export_to_okf(core)["units"][0]
        gen = unit["generated"]
        # System actor, namespaced — provable, not a self-declared `human:` tier.
        assert gen["by"] == "mind-mem:proj"
        assert not gen["by"].startswith("human:")
        assert gen["at"] == "2026-06-13T00:00:00Z"

    def test_verified_tier_is_never_synthesised(self, tmp_core: str) -> None:
        # The refusal that makes this "better than proposed": an exported unit
        # cannot prove a human-review event, so `verified` is NEVER emitted —
        # OKF reads the concept as "unverified", the honest tier.
        core = _load(tmp_core, [{"_id": "D-1", "type": "decision", "Status": "active"}])
        unit = export_to_okf(core)["units"][0]
        assert "verified" not in unit

    def test_status_maps_governance_to_okf_lifecycle(self, tmp_core: str) -> None:
        core = _load(
            tmp_core,
            [
                {"_id": "D-1", "type": "decision", "Status": "active"},
                {"_id": "D-2", "type": "decision", "Status": "superseded"},
                {"_id": "D-3", "type": "decision", "Status": "revoked"},
            ],
        )
        by_id = {u["id"]: u for u in export_to_okf(core)["units"]}
        assert by_id["D-1"]["status"] == "stable"
        assert by_id["D-2"]["status"] == "deprecated"
        assert by_id["D-3"]["status"] == "deprecated"

    def test_unknown_status_is_surfaced_not_masked(self, tmp_core: str) -> None:
        core = _load(tmp_core, [{"_id": "D-1", "type": "decision", "Status": "Frozen"}])
        unit = export_to_okf(core)["units"][0]
        assert unit["status"] == "frozen"  # verbatim lower-cased, not a "stable" default

    def test_sources_carry_signals_not_a_score(self, tmp_core: str) -> None:
        block = {
            "_id": "D-1",
            "type": "decision",
            "Statement": "x",
            "Sources": ["arXiv:2401.1", "github.com/star-ga/mind"],
            "Date": "2026-06-13",
        }
        core = _load(tmp_core, [block])
        unit = export_to_okf(core)["units"][0]
        assert unit["sources"] == [
            {"resource": "arXiv:2401.1", "last_modified": "2026-06-13"},
            {"resource": "github.com/star-ga/mind", "last_modified": "2026-06-13"},
        ]
        # No credibility *score* anywhere — only recorded signals (OKF's own rule).
        for s in unit["sources"]:
            assert "score" not in s
            assert "confidence" not in s


class TestOkfReceipt:
    """The re-derivable content receipt: trust by re-derivation, not by the
    sender's word."""

    def test_receipt_is_deterministic(self, tmp_core: str) -> None:
        block = {"_id": "D-1", "type": "decision", "Statement": "x", "Date": "2026-06-13"}
        core = _load(tmp_core, [block])
        r1 = export_to_okf(core)["units"][0]["receipt"]
        r2 = export_to_okf(core)["units"][0]["receipt"]
        assert r1 == r2
        assert r1.startswith("sha256:")

    def test_receipt_re_derivable_by_a_consumer(self, tmp_core: str) -> None:
        block = {"_id": "D-1", "type": "decision", "Statement": "x", "Date": "2026-06-13"}
        core = _load(tmp_core, [block])
        unit = export_to_okf(core)["units"][0]
        # A consumer recomputes the receipt from the unit and it must match.
        assert _okf_receipt(unit) == unit["receipt"]

    def test_receipt_detects_tampering(self, tmp_core: str) -> None:
        block = {"_id": "D-1", "type": "decision", "Statement": "x"}
        core = _load(tmp_core, [block])
        unit = dict(export_to_okf(core)["units"][0])
        original = unit["receipt"]
        unit["description"] = "tampered"
        # Recomputing over the tampered unit yields a different digest.
        assert _okf_receipt(unit) != original

    def test_receipt_excludes_sources_from_preimage(self) -> None:
        # Same core fields but different citations -> identical receipt (sources
        # are excluded from the canonical preimage so envelope and bundle agree).
        base = {"_id": "D-1", "type": "decision", "Statement": "x", "Date": "2026-06-13"}
        u_no_src = _block_to_okf_unit(base, "proj")
        u_src = _block_to_okf_unit({**base, "Sources": ["arXiv:1"]}, "proj")
        assert u_src["receipt"] == u_no_src["receipt"]


class TestOkfImportTreatsForeignTrustAsClaim:
    """A foreign producer's self-declared trust is recorded as an untrusted
    claim, never honoured as mind-mem's own tier."""

    def test_foreign_verified_becomes_a_namespaced_claim(self, tmp_path) -> None:
        bundle = tmp_path / "foreign"
        bundle.mkdir()
        (bundle / "D-9.md").write_text(
            "---\n"
            "type: decision\n"
            "title: Foreign decision\n"
            "status: stable\n"
            "verified:\n"
            "  by: human:mallory\n"
            "  at: 2026-01-01T00:00:00Z\n"
            "receipt: sha256:deadbeefdeadbeef\n"
            "---\n\n# Foreign decision\n",
            encoding="utf-8",
        )
        blocks = import_okf_bundle(bundle)
        b = blocks[0]
        # The self-declared `verified: human:mallory` is a CLAIM under a
        # namespaced key — never a trusted mind-mem field.
        assert b["OkfClaimVerified"] == {"by": "human:mallory", "at": "2026-01-01T00:00:00Z"}
        assert b["OkfClaimStatus"] == "stable"
        assert b["OkfReceipt"] == "sha256:deadbeefdeadbeef"
        # It is NOT promoted to any trusted/governance field.
        assert "Verified" not in b
        assert "Status" not in b  # foreign status is a claim, not our governance status
        # Title still imports as a trusted content field.
        assert b["Title"] == "Foreign decision"
        # Every non-_id/type key still satisfies the capitalised grammar.
        for key in b:
            if key in ("_id", "type"):
                continue
            assert key[0].isupper()
