#!/usr/bin/env python3
"""Verify that a published number can be recomputed from its committed evidence.

A manifest nobody checks is documentation. This is the check.

Given a repro package (``benchmarks/repro/<name>/``) it:

1. re-hashes every artifact and compares against ``manifest.json``;
2. recomputes the metrics from ``raw.ndjson`` with the shipped metric function
   and compares, byte-for-byte in canonical form, against ``metrics.json``;
3. re-derives each unit's verdict from its own raw evidence and rejects any row
   whose stored verdict disagrees with what its retrieved results actually say;
4. cross-checks the manifest's unit counters (attempted / killed-or-crashed)
   against the rows;
5. confirms the manifest's headline is the recomputed headline.

Given a scorecard pair (a committed ``*.ndjson`` and the ``*.md`` published
beside it) it recomputes every number in the scorecard's results table from the
rows and demands an exact match. That is the check that would have caught a
figure typed into a report with no run behind it.

Any mismatch prints the two values and exits non-zero. Nothing is repaired,
re-blessed or rounded away: the verifier's only job is to disagree loudly.

Usage:
    python3 benchmarks/repro_verify.py package benchmarks/repro/<name>
    python3 benchmarks/repro_verify.py scorecard <run.ndjson> <run.md>
    python3 benchmarks/repro_verify.py all

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.repro_manifest import canonical_json, read_rows, sha256_bytes  # noqa: E402
from benchmarks.repro_metrics import METRIC_FNS, SCHEMA, STATUS_OK  # noqa: E402


class Report:
    """Accumulated findings for one verification target."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.checks = 0
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, expected: Any = None, actual: Any = None) -> bool:
        self.checks += 1
        if not ok:
            detail = f"{label}"
            if expected is not None or actual is not None:
                detail += f"\n      committed: {expected!r}\n      recomputed: {actual!r}"
            self.failures.append(detail)
        return ok

    @property
    def passed(self) -> bool:
        return not self.failures

    def render(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'}  {self.target}  ({self.checks} checks)"
        if self.passed:
            return head
        return head + "\n" + "\n".join(f"    - {f}" for f in self.failures)


# ---------------------------------------------------------------------------
# Package verification
# ---------------------------------------------------------------------------


def verify_package(pkg_dir: str) -> Report:
    rep = Report(os.path.relpath(pkg_dir, _REPO_ROOT))
    manifest_path = os.path.join(pkg_dir, "manifest.json")
    if not rep.check(os.path.isfile(manifest_path), f"missing manifest.json in {pkg_dir}"):
        return rep
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    rep.check(manifest.get("schema") == SCHEMA, "manifest schema", SCHEMA, manifest.get("schema"))

    # 1. Content addressing: an edited artifact fails here first.
    for name, meta in sorted(manifest.get("artifacts", {}).items()):
        path = os.path.join(pkg_dir, name)
        if not rep.check(os.path.isfile(path), f"artifact missing: {name}"):
            continue
        with open(path, "rb") as handle:
            data = handle.read()
        rep.check(sha256_bytes(data) == meta.get("sha256"), f"sha256 mismatch: {name}", meta.get("sha256"), sha256_bytes(data))
        rep.check(len(data) == meta.get("bytes"), f"byte length mismatch: {name}", meta.get("bytes"), len(data))

    benchmark = str(manifest.get("benchmark", ""))
    metric_fn = METRIC_FNS.get(benchmark)
    if not rep.check(metric_fn is not None, f"no metric function registered for benchmark {benchmark!r}"):
        return rep
    assert metric_fn is not None

    raw_path = os.path.join(pkg_dir, "raw.ndjson")
    metrics_path = os.path.join(pkg_dir, "metrics.json")
    if not rep.check(os.path.isfile(raw_path) and os.path.isfile(metrics_path), "raw.ndjson / metrics.json missing"):
        return rep

    rows = read_rows(raw_path)
    recomputed = metric_fn(rows)
    with open(metrics_path, "rb") as handle:
        committed_bytes = handle.read()
    recomputed_bytes = canonical_json(recomputed)

    # 2. The load-bearing check: raw rows -> the published numbers.
    if not rep.check(recomputed_bytes == committed_bytes, "metrics.json is NOT recomputable from raw.ndjson"):
        committed = json.loads(committed_bytes)
        for key in sorted(set(committed) | set(recomputed)):
            if committed.get(key) != recomputed.get(key):
                rep.check(False, f"metrics.{key}", committed.get(key), recomputed.get(key))

    # 3. Rows must agree with their own evidence.
    disagreeing = recomputed.get("integrity", {}).get("rows_whose_stored_verdict_disagrees_with_their_evidence", [])
    rep.check(not disagreeing, f"{len(disagreeing)} row(s) store a verdict their retrieved results do not support", [], disagreeing)

    # 4. Manifest counters vs the rows they describe.
    run = manifest.get("run", {})
    killed = sum(1 for r in rows if str(r.get("unit_status", STATUS_OK)) != STATUS_OK)
    rep.check(run.get("units_total") == len(rows), "run.units_total vs rows in raw.ndjson", run.get("units_total"), len(rows))
    rep.check(
        run.get("units_killed_or_crashed") == killed, "run.units_killed_or_crashed vs rows", run.get("units_killed_or_crashed"), killed
    )

    # 5. The headline is the recomputed headline, not a remembered one.
    rep.check(
        manifest.get("headline") == recomputed.get("headline"),
        "manifest headline vs recomputed headline",
        manifest.get("headline"),
        recomputed.get("headline"),
    )
    return rep


# ---------------------------------------------------------------------------
# Scorecard verification (LongMemEval-shaped: NDJSON + published markdown)
# ---------------------------------------------------------------------------

_TABLE_ROW = re.compile(r"^\|\s*(recall_any|recall_all|precision|recall)@k\s*\|(.+)\|\s*$")
_SUMMARY = re.compile(r"\*\*MRR:\*\*\s*([\d.]+).*?\*\*hit_rate:\*\*\s*([\d.]+).*?\*\*mean latency:\*\*\s*([\d.]+)")
_EVALUATED = re.compile(r"\*\*Questions evaluated:\*\*\s*(\d+)")


def parse_scorecard(path: str) -> dict[str, Any]:
    """Pull the published numbers back out of a committed scorecard."""
    out: dict[str, Any] = {"table": {}, "summary": {}, "evaluated": None}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            m = _TABLE_ROW.match(line.rstrip("\n"))
            if m:
                cells = [c.strip() for c in m.group(2).split("|")]
                for kk, cell in zip((1, 3, 5, 10), cells):
                    if cell and cell != "—":
                        out["table"][f"{m.group(1)}@{kk}"] = float(cell)
                continue
            s = _SUMMARY.search(line)
            if s:
                out["summary"] = {"mrr": float(s.group(1)), "hit_rate": float(s.group(2)), "mean_latency_ms": float(s.group(3))}
                continue
            e = _EVALUATED.search(line)
            if e:
                out["evaluated"] = int(e.group(1))
    return out


def verify_scorecard(ndjson_path: str, scorecard_path: str) -> Report:
    rep = Report(f"{os.path.relpath(ndjson_path, _REPO_ROOT)} -> {os.path.basename(scorecard_path)}")
    if not rep.check(os.path.isfile(ndjson_path), f"missing rows: {ndjson_path}"):
        return rep
    if not rep.check(os.path.isfile(scorecard_path), f"missing scorecard: {scorecard_path}"):
        return rep

    rows = read_rows(ndjson_path)
    recomputed = METRIC_FNS["LongMemEval-S"](rows)
    overall = recomputed["overall"]
    published = parse_scorecard(scorecard_path)

    rep.check(published["evaluated"] == len(rows), "questions evaluated vs rows", published["evaluated"], len(rows))
    if not rep.check(bool(published["table"]), "no results table found in scorecard"):
        return rep
    for key, value in sorted(published["table"].items()):
        rep.check(overall.get(key) == value, f"{key}", value, overall.get(key))
    for key, value in sorted(published["summary"].items()):
        rep.check(overall.get(key) == value, f"{key}", value, overall.get(key))
    return rep


# ---------------------------------------------------------------------------
# Discovery + CLI
# ---------------------------------------------------------------------------


def discover_packages() -> list[str]:
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(_REPO_ROOT, "benchmarks", "repro", "*", "manifest.json")))


def discover_scorecards() -> list[tuple[str, str]]:
    """Committed NDJSON artifacts that have a scorecard published beside them."""
    pairs = []
    for nd in sorted(glob.glob(os.path.join(_REPO_ROOT, "docs", "benchmarks", "*.ndjson"))):
        md = nd[: -len(".ndjson")] + ".md"
        if os.path.isfile(md):
            pairs.append((nd, md))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute published numbers from committed raw evidence")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pkg = sub.add_parser("package", help="verify one repro package directory")
    p_pkg.add_argument("path")
    p_sc = sub.add_parser("scorecard", help="verify a published scorecard against its NDJSON")
    p_sc.add_argument("ndjson")
    p_sc.add_argument("scorecard")
    sub.add_parser("all", help="verify every committed package and scorecard pair")
    a = ap.parse_args(argv)

    reports: list[Report] = []
    if a.cmd == "package":
        reports.append(verify_package(a.path))
    elif a.cmd == "scorecard":
        reports.append(verify_scorecard(a.ndjson, a.scorecard))
    else:
        packages = discover_packages()
        pairs = discover_scorecards()
        if not packages and not pairs:
            print("VACUOUS: nothing to verify -- no repro packages and no scorecard pairs found.")
            print("An empty run is not a pass; commit a package or a scorecard pair first.")
            return 2
        reports.extend(verify_package(p) for p in packages)
        reports.extend(verify_scorecard(nd, md) for nd, md in pairs)

    for rep in reports:
        print(rep.render())
    failed = [r for r in reports if not r.passed]
    total_checks = sum(r.checks for r in reports)
    print(f"\n{len(reports) - len(failed)}/{len(reports)} target(s) verified, {total_checks} checks.")
    if failed:
        print("VERIFICATION FAILED -- a published number does not follow from its committed evidence.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
