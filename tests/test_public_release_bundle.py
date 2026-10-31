"""Offline validation of the committed reproducible benchmark kit.

Verifies every bundle hash and recomputes the official-key
full / dev / held-out aggregates from the committed answers,
asserting they match the values recorded in ``expected_scores.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.release_scoring import verify_bundle

ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "public_release"


def recompute(rows: list[dict]) -> dict:
    """Official-key correct/total per partition, straight from committed votes.

    ``release_scoring.aggregate`` also computes text metrics that need the
    reference answer, which ``answers.jsonl`` deliberately strips.  The
    official-key aggregate the manifest records is the vote tally, so that
    is what is recomputed here.
    """
    out = {}
    for partition in ("full", "dev", "held_out"):
        selected = (
            rows
            if partition == "full"
            else [r for r in rows if r["partition"] == partition]
        )
        out[partition] = {
            "correct": sum(1 for r in selected if r["official_ok"]),
            "total": len(selected),
        }
    return out


def _load_expected() -> dict:
    return json.loads((ROOT / "expected_scores.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Bundle-integrity tests
# ---------------------------------------------------------------------------


class TestBundleIntegrity:
    def test_verify_bundle_passes_on_committed_kit(self):
        manifest, rows = verify_bundle(ROOT)
        assert manifest["rows"] == len(rows) == 2486

    def test_heldout_split_hash_prefix(self):
        actual = _sha256(ROOT / "HELDOUT_SPLIT_20261029.json")
        assert actual.startswith("aad631b4c8e92041"), (
            f"HELDOUT_SPLIT hash changed: {actual}"
        )

    def test_all_manifest_hashes_match(self):
        manifest = json.loads(
            (ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        for name, expected in manifest["artifacts"].items():
            path = ROOT / name
            assert path.exists(), f"Missing artifact: {name}"
            actual = _sha256(path)
            assert actual == expected, (
                f"Hash mismatch for {name}: expected {expected}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Aggregate-recomputation tests
# ---------------------------------------------------------------------------


class TestOfficialKeyAggregates:
    """Recompute every official-key aggregate from committed answers
    and assert equality with ``expected_scores.json``."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.expected = _load_expected()
        _, self.rows = verify_bundle(ROOT)

    def _check_aggregate(self, rows, expected_block):
        result = recompute(rows)
        for partition in ("full", "dev", "held_out"):
            for metric in ("correct", "total"):
                assert result[partition][metric] == expected_block[partition][metric], (
                    f"{partition}.{metric}: recomputed "
                    f"{result[partition][metric]} != expected "
                    f"{expected_block[partition][metric]}"
                )

    def test_locomo_all_official_key(self):
        locomo = [r for r in self.rows if r["benchmark"] == "locomo"]
        expected = self.expected["locomo"]["configuration_a"]["official_key"]["all"]
        self._check_aggregate(locomo, expected)

    def test_locomo_categories_1_4_official_key(self):
        locomo = [
            r
            for r in self.rows
            if r["benchmark"] == "locomo" and r.get("category") in (1, 2, 3, 4)
        ]
        expected = self.expected["locomo"]["configuration_a"]["official_key"][
            "categories_1_4"
        ]
        self._check_aggregate(locomo, expected)

    def test_locomo_category_5_official_key(self):
        cat5 = [
            r
            for r in self.rows
            if r["benchmark"] == "locomo" and r.get("category") == 5
        ]
        expected = self.expected["locomo"]["configuration_a"]["official_key"][
            "category_5"
        ]
        self._check_aggregate(cat5, expected)

    def test_longmemeval_official_key(self):
        lme = [r for r in self.rows if r["benchmark"] == "longmemeval_s"]
        expected = self.expected["lme"]["configuration_a"]["official_key"]["all"]
        self._check_aggregate(lme, expected)


# ---------------------------------------------------------------------------
# Mutant-detection guarantee (edit one committed aggregate -> gate goes red)
# ---------------------------------------------------------------------------
#
# The gate-level mutant proof is a process check: copy the kit, edit one
# committed aggregate, run ``pytest tests/test_public_release_bundle.py``
# and observe it fail.  The hash check in ``test_all_manifest_hashes_match``
# and the recomputation in ``TestOfficialKeyAggregates`` are what go red.
# This test pins the recomputation half: a mutated expected aggregate must
# disagree with the aggregate recomputed from the committed answers.


class TestMutantDetection:
    def test_mutated_aggregate_disagrees_with_recomputation(self):
        """Recomputation from answers must not follow a mutated expectation."""
        _, rows = verify_bundle(ROOT)
        scores = json.loads(
            (ROOT / "expected_scores.json").read_text(encoding="utf-8")
        )
        block = scores["locomo"]["configuration_a"]["official_key"][
            "categories_1_4"
        ]["full"]
        real = block["correct"]
        block["correct"] = 999  # the mutant

        locomo = [
            r
            for r in rows
            if r["benchmark"] == "locomo" and r.get("category") in (1, 2, 3, 4)
        ]
        recomputed = recompute(locomo)["full"]["correct"]
        assert recomputed != block["correct"], (
            "Recomputation followed the mutated aggregate"
        )
        assert recomputed == real, (
            f"Recomputation drifted: {recomputed} != {real}"
        )
