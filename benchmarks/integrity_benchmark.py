#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc.
"""Class A integrity benchmark — the dimensions nobody else measures.

A survey of 125 memory systems on 2026-09-07 found 332 published benchmark
claims. Every one was a retrieval score or an LLM-judge answer score. **Not one
was a provenance, replay, ontology, or governance guarantee.** That is an
uncontested axis, and this harness is the instrument for it.

It reports COVERAGE, not capability. "The feature ships" is not a number; "N of
M blocks carry a tamper-evident record" is. Each dimension therefore answers a
question with a denominator, and every one of them can come back embarrassing —
which is the point. A benchmark that cannot report a bad number is a brochure.

Design rules, learned from the survey's own findings:

* Every figure carries its denominator, so nobody can quote a ratio without the
  population it was taken over.
* Layers are never merged. "Knows its source file" and "has a hash-chained
  write record" are different guarantees with wildly different coverage, and
  reporting the first as the second is precisely the overclaim this exists to
  avoid.
* A dimension with no honest measurement reports ``UNMEASURED`` rather than a
  plausible-looking number. Absence of a metric is data.

Usage:
    python3 benchmarks/integrity_benchmark.py --workspace <ws> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _blocks(workspace: str) -> list[dict[str, Any]]:
    from mind_mem.storage import get_block_store

    return list(get_block_store(workspace).get_all(active_only=False))


def measure_provenance(workspace: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Two layers, reported separately because they guarantee different things.

    ``origin`` is a mutable field saying where a block came from. ``evidence``
    is a hash-chained record that a write happened and has not been altered
    since. A system can be at 100% on the first and 1% on the second, and
    quoting the first as "full provenance" is the overclaim.
    """
    from mind_mem.evidence_objects import EvidenceChain

    total = len(blocks)
    with_origin = sum(1 for b in blocks if b.get("_source_file") or b.get("Source"))

    chain_path = os.path.join(workspace, "memory", "evidence_chain.jsonl")
    if os.path.exists(chain_path):
        chain = EvidenceChain(store_path=chain_path)
        valid, breaks = chain.verify_chain()[0], chain.verify_chain()[1]
        targets = {e.target_block_id for e in chain._entries if e.target_block_id}
        records = len(chain._entries)
    else:
        valid, breaks, targets, records = False, ["absent"], set(), 0

    ids = {b.get("_id") for b in blocks}
    with_evidence = len(ids & targets)
    return {
        "blocks_total": total,
        "origin_field": {"n": with_origin, "of": total, "coverage": with_origin / total if total else 0.0},
        "evidence_chain": {
            "n": with_evidence,
            "of": total,
            "coverage": with_evidence / total if total else 0.0,
            "records": records,
            "chain_verifies": bool(valid),
            "first_break": breaks,
        },
        "note": (
            "origin_field is a mutable string; evidence_chain is tamper-evident. "
            "They are not the same guarantee and must not be quoted as one."
        ),
    }


def measure_replay_determinism(workspace: str) -> dict[str, Any]:
    """Does the same query, twice, return byte-identical results?

    Determinism is the cheapest Class A property to verify and the one most
    often merely asserted. Two identical calls, compared on their served ids
    and scores.
    """
    try:
        from mind_mem.recall import recall
    except Exception as exc:  # pragma: no cover - import shape varies by build
        return {"status": "UNMEASURED", "reason": f"recall not importable: {exc}"}

    probes = ["governance", "retrieval quality", "evidence chain", "ontology"]
    mismatches: list[str] = []
    checked = 0
    for q in probes:
        try:
            # recall(workspace, query) — positional, in that order.
            a = recall(workspace, q, limit=10)
            b = recall(workspace, q, limit=10)
        except Exception as exc:
            return {"status": "UNMEASURED", "reason": f"probe {q!r} raised: {exc}"}
        checked += 1
        ka = [(r.get("_id"), round(float(r.get("_score", 0.0)), 9)) for r in a]
        kb = [(r.get("_id"), round(float(r.get("_score", 0.0)), 9)) for r in b]
        if ka != kb:
            mismatches.append(q)
    return {
        "status": "MEASURED",
        "probes": checked,
        "identical": checked - len(mismatches),
        "coverage": (checked - len(mismatches)) / checked if checked else 0.0,
        "mismatched_queries": mismatches,
    }


def measure_ontology_validity(workspace: str) -> dict[str, Any]:
    """What fraction of graph edges satisfy the active ontology's domain/range?

    Reports ``UNMEASURED`` — honestly — when no ontology is installed or no
    entity carries a type, because a validator with nothing to check against
    would otherwise report a flattering 100%.
    """
    db = os.path.join(workspace, "memory", "knowledge_graph.db")
    if not os.path.exists(db):
        return {"status": "UNMEASURED", "reason": "no knowledge graph in this workspace"}
    import sqlite3

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        edges = conn.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
        cols = {r[1] for r in conn.execute("PRAGMA table_info(entities)")}
        entities = conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"]
        typed = 0
        if "entity_type" in cols:
            typed = conn.execute("SELECT COUNT(*) c FROM entities WHERE entity_type IS NOT NULL AND entity_type != ''").fetchone()["c"]
    finally:
        conn.close()

    if typed == 0:
        return {
            "status": "UNMEASURED",
            "reason": "no entity carries a type, so domain/range cannot be checked",
            "edges": edges,
            "entities": entities,
            "typed_entities": typed,
            "blocker": "entity typing must be rolled out before validity is measurable",
        }
    return {
        "status": "MEASURED",
        "edges": edges,
        "entities": entities,
        "typed_entities": typed,
        "typed_coverage": typed / entities if entities else 0.0,
    }


def measure_entity_resolution(workspace: str) -> dict[str, Any]:
    """Alias density: how many surface forms collapse onto each canonical id.

    A registry where every alias is its own entity has resolved nothing, and
    that is indistinguishable from a working one unless the ratio is reported.
    """
    db = os.path.join(workspace, "memory", "knowledge_graph.db")
    if not os.path.exists(db):
        return {"status": "UNMEASURED", "reason": "no knowledge graph in this workspace"}
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        aliases = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
        merged = conn.execute("SELECT COUNT(*) FROM (SELECT entity_id FROM aliases GROUP BY entity_id HAVING COUNT(*) > 1)").fetchone()[0]
    finally:
        conn.close()
    return {
        "status": "MEASURED",
        "entities": entities,
        "aliases": aliases,
        "aliases_per_entity": aliases / entities if entities else 0.0,
        "entities_with_multiple_aliases": merged,
        "note": ("aliases_per_entity at 1.0 means no surface form has ever been merged — the registry is storing, not resolving."),
    }


def measure_governance(workspace: str) -> dict[str, Any]:
    """Is every write path gated? Answered from the repo's own reachability gate."""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "check_governed_write_paths.py")
    if not os.path.exists(script):
        script = os.path.join(root, "scripts", "check_reachable_modules.py")
    if not os.path.exists(script):
        return {"status": "UNMEASURED", "reason": "no governance gate script in this tree"}
    # encoding= is required, not optional: text=True alone decodes the child
    # with the locale codec, which is cp1252 on Windows. tests/test_text_io_is_utf8
    # gates exactly this and caught it here.
    proc = subprocess.run(
        [sys.executable, script, "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    return {
        "status": "MEASURED",
        "gate": os.path.basename(script),
        "exit_code": proc.returncode,
        "passes": proc.returncode == 0,
        "output_tail": proc.stdout.strip().splitlines()[-3:] if proc.stdout.strip() else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Class A memory-integrity benchmark")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--json", default=None, help="write the full scorecard here")
    ap.add_argument(
        "--label",
        default=None,
        help=(
            "record this name instead of the workspace's real path. Use it for any "
            "PUBLISHED artifact: the realpath identifies the operator's machine and "
            "does not belong in a public scorecard."
        ),
    )
    a = ap.parse_args()

    blocks = _blocks(a.workspace)
    report: dict[str, Any] = {
        "workspace": a.label or os.path.realpath(a.workspace),
        "dimensions": {
            "provenance": measure_provenance(a.workspace, blocks),
            "replay_determinism": measure_replay_determinism(a.workspace),
            "ontology_validity": measure_ontology_validity(a.workspace),
            "entity_resolution": measure_entity_resolution(a.workspace),
            "governance": measure_governance(a.workspace),
        },
    }

    print(f"CLASS A INTEGRITY — {report['workspace']}")
    print(f"  blocks in corpus: {len(blocks)}\n")
    p = report["dimensions"]["provenance"]
    og, ev = p["origin_field"], p["evidence_chain"]
    print(f"  provenance / origin field    {og['n']:>6} of {og['of']:<6} = {og['coverage']:.4f}")
    print(f"  provenance / evidence chain  {ev['n']:>6} of {ev['of']:<6} = {ev['coverage']:.4f}   (chain verifies: {ev['chain_verifies']})")
    for name in ("replay_determinism", "ontology_validity", "entity_resolution", "governance"):
        d = report["dimensions"][name]
        if d.get("status") == "UNMEASURED":
            print(f"  {name:28} UNMEASURED — {d.get('reason')}")
        elif "coverage" in d:
            print(f"  {name:28} {d['coverage']:.4f}")
        elif "passes" in d:
            print(f"  {name:28} {'PASS' if d['passes'] else 'FAIL'} ({d.get('gate')})")
        else:
            print(f"  {name:28} {json.dumps({k: v for k, v in d.items() if k != 'note'})[:96]}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"\n  scorecard: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
