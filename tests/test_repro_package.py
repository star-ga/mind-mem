#!/usr/bin/env python3
"""The verifier must be able to FAIL.

A repro package whose verifier only ever prints PASS is documentation wearing a
test's clothes. These tests hold the chain end to end: a package verifies, and
then each link is broken in turn -- an edited raw row, a doctored metric, a
miscounted manifest, a scorecard number that no longer follows from its rows --
and the verifier is required to notice and exit non-zero every time.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks import repro_verify  # noqa: E402
from benchmarks.repro_manifest import build_manifest, write_package  # noqa: E402
from benchmarks.repro_metrics import niah_hit, niah_metrics  # noqa: E402


def _row(unit_id: str, needle_id: str, ids: list[str], keywords: list[str], excerpts: list[str] | None = None, **over):
    excerpts = excerpts if excerpts is not None else ["filler" for _ in ids]
    row = {
        "unit_id": unit_id,
        "haystack_size": 10,
        "depth_pct": 0,
        "needle_idx": 0,
        "needle_id": needle_id,
        "query": "q",
        "expected_keywords": keywords,
        "retrieved": [{"rank": i, "id": did, "excerpt": ex} for i, (did, ex) in enumerate(zip(ids, excerpts), 1)],
        "found": needle_id in ids,
        "latency_ms": 12.5,
        "unit_status": "ok",
        "unit_elapsed_s": 0.1,
        "unit_error": "",
        "pipeline": {"declared_backend": "hybrid", "effective_backend": "hybrid", "embedder": {"name": "m", "dimensions": 384}},
    }
    row.update(over)
    return row


def _package(tmp_path, rows):
    metrics = niah_metrics(rows)
    manifest = build_manifest(
        benchmark="NIAH",
        headline_claim=False,
        sampling="unit test",
        dataset={"name": "t", "content_sha256": "x"},
        config={"backend": "hybrid"},
        seeds={"rng": "none"},
        adapter={"name": "t"},
        embedder={"name": "m", "dimensions": 384, "device": "cpu"},
        k=5,
        exclusions={"rule": "none", "excluded": 0},
        run={
            "units_total": len(rows),
            "units_ok": len(rows) - metrics["integrity"]["killed_or_crashed"],
            "units_killed_or_crashed": metrics["integrity"]["killed_or_crashed"],
        },
        headline=metrics["headline"],
        commands={"regenerate": "x", "recompute_metrics_from_raw_rows": "y"},
    )
    out = str(tmp_path / "pkg")
    write_package(
        out,
        manifest=manifest,
        raw_rows=rows,
        metrics=metrics,
        dataset={"name": "t", "content_sha256": "x"},
        environment={"python_version": "test"},
    )
    return out


@pytest.fixture
def rows():
    return [
        _row("a", "NEEDLE-001", ["NEEDLE-001", "D-1"], ["alpha"]),
        _row("b", "NEEDLE-002", ["D-2", "D-3"], ["beta"]),
    ]


# ---------------------------------------------------------------------------
# The hit rule is re-derived, never read off the row
# ---------------------------------------------------------------------------


def test_hit_by_block_id():
    assert niah_hit(_row("a", "NEEDLE-001", ["NEEDLE-001"], ["nope"])) is True


def test_hit_by_every_expected_keyword_in_one_excerpt():
    row = _row("a", "NEEDLE-001", ["D-9"], ["alpha", "beta"], excerpts=["... Alpha and BETA together ..."])
    assert niah_hit(row) is True


def test_miss_when_only_some_keywords_present():
    row = _row("a", "NEEDLE-001", ["D-9"], ["alpha", "beta"], excerpts=["alpha only"])
    assert niah_hit(row) is False


def test_stored_verdict_disagreeing_with_evidence_is_reported():
    row = _row("a", "NEEDLE-001", ["D-9"], ["alpha"])
    row["found"] = True  # claims a hit its own results do not support
    metrics = niah_metrics([row])
    assert metrics["integrity"]["rows_whose_stored_verdict_disagrees_with_their_evidence"] == ["a"]
    assert metrics["headline"]["passed"] == 0


def test_decision_fingerprint_ignores_timing_and_tracks_retrieval(rows):
    base = niah_metrics(rows)["determinism"]["decision_fingerprint"]
    slower = [dict(r, latency_ms=r["latency_ms"] * 10) for r in rows]
    assert niah_metrics(slower)["determinism"]["decision_fingerprint"] == base
    reranked = [dict(rows[0], retrieved=list(reversed(rows[0]["retrieved"]))), rows[1]]
    assert niah_metrics(reranked)["determinism"]["decision_fingerprint"] != base


# ---------------------------------------------------------------------------
# The verifier passes on an honest package
# ---------------------------------------------------------------------------


def test_package_verifies(tmp_path, rows):
    out = _package(tmp_path, rows)
    rep = repro_verify.verify_package(out)
    assert rep.passed, rep.render()
    assert rep.checks > 10
    assert repro_verify.main(["package", out]) == 0


# ---------------------------------------------------------------------------
# ... and FAILS on every broken link. A verifier that cannot fail is not a test.
# ---------------------------------------------------------------------------


def test_edited_raw_row_fails_and_exits_nonzero(tmp_path, rows):
    out = _package(tmp_path, rows)
    raw = os.path.join(out, "raw.ndjson")
    with open(raw, "rb") as handle:
        original = handle.read()

    lines = original.decode("utf-8").splitlines()
    doctored = json.loads(lines[1])
    doctored["retrieved"] = [{"rank": 1, "id": "NEEDLE-002", "excerpt": "planted"}]
    doctored["found"] = True
    lines[1] = json.dumps(doctored, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    with open(raw, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    rep = repro_verify.verify_package(out)
    assert not rep.passed
    assert any("sha256 mismatch: raw.ndjson" in f for f in rep.failures)
    assert any("metrics.json is NOT recomputable" in f for f in rep.failures)
    assert repro_verify.main(["package", out]) == 1

    with open(raw, "wb") as handle:
        handle.write(original)
    assert repro_verify.verify_package(out).passed


def test_doctored_metrics_file_fails(tmp_path, rows):
    out = _package(tmp_path, rows)
    path = os.path.join(out, "metrics.json")
    with open(path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics["headline"]["passed"] = 2
    metrics["headline"]["as_text"] = "2/2"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    rep = repro_verify.verify_package(out)
    assert not rep.passed
    assert any("metrics" in f for f in rep.failures)


def test_manifest_counter_that_does_not_match_the_rows_fails(tmp_path, rows):
    out = _package(tmp_path, rows)
    path = os.path.join(out, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["run"]["units_total"] = 470
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    rep = repro_verify.verify_package(out)
    assert not rep.passed
    assert any("units_total" in f for f in rep.failures)


def test_manifest_headline_that_outruns_the_rows_fails(tmp_path, rows):
    out = _package(tmp_path, rows)
    path = os.path.join(out, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["headline"] = {"metric": "top_k_hit_rate", "passed": 250, "total": 250, "as_text": "250/250", "value": 1.0}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    rep = repro_verify.verify_package(out)
    assert not rep.passed
    assert any("headline" in f for f in rep.failures)


def test_unknown_benchmark_is_a_failure_not_a_skip(tmp_path, rows):
    out = _package(tmp_path, rows)
    path = os.path.join(out, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["benchmark"] = "SomethingNobodyCanRecompute"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    rep = repro_verify.verify_package(out)
    assert not rep.passed
    assert any("no metric function registered" in f for f in rep.failures)


# ---------------------------------------------------------------------------
# Committed scorecards must follow from their committed rows
# ---------------------------------------------------------------------------

_PAIRS = repro_verify.discover_scorecards()


def test_scorecard_pairs_exist():
    assert _PAIRS, "no committed NDJSON + scorecard pairs found under docs/benchmarks/"


@pytest.mark.parametrize("ndjson,scorecard", _PAIRS, ids=[os.path.basename(n) for n, _ in _PAIRS])
def test_committed_scorecard_recomputes_from_its_rows(ndjson, scorecard):
    rep = repro_verify.verify_scorecard(ndjson, scorecard)
    assert rep.passed, rep.render()


def test_scorecard_with_an_edited_number_fails(tmp_path):
    ndjson, scorecard = _PAIRS[0]
    with open(scorecard, encoding="utf-8") as handle:
        text = handle.read()
    doctored = tmp_path / "doctored.md"
    # Replace the whole results table row with a flattering one.
    lines = []
    for line in text.splitlines():
        if line.startswith("| recall_any@k |"):
            line = "| recall_any@k | 0.9999 | 0.9999 | 0.9999 | 0.9999 |"
        lines.append(line)
    doctored.write_text("\n".join(lines), encoding="utf-8")
    rep = repro_verify.verify_scorecard(ndjson, str(doctored))
    assert not rep.passed
    assert any("recall_any@5" in f for f in rep.failures)
