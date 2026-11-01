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

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks import repro_verify  # noqa: E402
from benchmarks.release_scoring import verify_bundle  # noqa: E402
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


def test_a_package_in_which_nothing_ran_is_vacuous_not_passing(tmp_path):
    dead = [
        _row("a", "NEEDLE-001", [], ["alpha"], unit_status="error", unit_error="boom", found=False),
        _row("b", "NEEDLE-002", [], ["beta"], unit_status="timeout", unit_error="killed", found=False),
    ]
    out = _package(tmp_path, dead)
    rep = repro_verify.verify_package(out)
    assert not rep.passed
    assert any("VACUOUS" in f for f in rep.failures)


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
# The manifest must name the core artifacts, or the hash loop never touches them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omitted", ["raw.ndjson", "metrics.json", "dataset.json"])
def test_manifest_omitting_a_core_artifact_fails_with_the_core_artifact_message(tmp_path, rows, omitted):
    out = _package(tmp_path, rows)
    assert repro_verify.verify_package(out).passed, "a complete package must verify"

    manifest_path = os.path.join(out, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    del manifest["artifacts"][omitted]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)

    rep = repro_verify.verify_package(out)
    assert not rep.passed, f"a manifest that leaves {omitted} out of artifacts must not verify"
    assert any("core artifacts" in f for f in rep.failures), rep.render()
    assert repro_verify.main(["package", out]) == 1


# ---------------------------------------------------------------------------
# Artifact names are confined to the package directory
# ---------------------------------------------------------------------------

ESCAPES = ("parent", "absolute", "symlink")


def _escape_case(tmp_path: Path, root: Path, kind: str) -> tuple[str, bytes]:
    """Create the on-disk setup for one escape kind.

    Returns the artifact name to put in the manifest and the bytes a
    successful (unconfined) verifier would read for it -- so the entry hashes
    match and the only thing that can reject it is the confinement rule.
    """
    if kind == "parent":
        target = tmp_path / "x"
        target.write_bytes(b"outside the package\n")
        return "../x", target.read_bytes()
    if kind == "absolute":
        passwd = Path("/etc/passwd")
        data = passwd.read_bytes() if passwd.is_file() else b""
        return "/etc/passwd", data
    if kind == "symlink":
        target = tmp_path / "x"
        target.write_bytes(b"outside the package\n")
        link = root / "link"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not supported on this platform")
        return "link", target.read_bytes()
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ESCAPES)
def test_verify_package_refuses_artifact_names_escaping_the_package(tmp_path, rows, kind):
    out = _package(tmp_path, rows)
    name, data = _escape_case(tmp_path, Path(out), kind)

    manifest_path = os.path.join(out, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["artifacts"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)

    rep = repro_verify.verify_package(out)
    assert not rep.passed, f"{name!r} must be refused"
    assert any("unsafe artifact name" in f for f in rep.failures), rep.render()


def _release_bundle(root: Path) -> None:
    """One-row release bundle that verifies cleanly on its own."""
    prompt = "A generic evidence prompt"
    h = hashlib.sha256(prompt.encode()).hexdigest()
    row = {
        "id": "example",
        "question": "What is the color?",
        "reference": "blue",
        "generated": "green",
        "partition": "held_out",
        "official_ok": True,
        "prompt_sha256": h,
        "votes": [{"ok": True}],
    }
    (root / "answers.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "prompts.jsonl").write_text(
        json.dumps({"id": row["id"], "prompt_sha256": h, "messages": [{"content": prompt}]}) + "\n", encoding="utf-8"
    )
    manifest = {"rows": 1, "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("kind", ESCAPES)
def test_verify_bundle_refuses_artifact_names_escaping_the_bundle(tmp_path, kind):
    root = tmp_path / "bundle"
    root.mkdir()
    _release_bundle(root)
    name, data = _escape_case(tmp_path, root, kind)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][name] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe artifact name"):
        verify_bundle(root)


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
