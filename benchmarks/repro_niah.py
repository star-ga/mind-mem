#!/usr/bin/env python3
"""Reproducible NIAH benchmark harness — emits independently-verifiable evidence.

This runs the SAME Needle-In-A-Haystack code the test suite uses (imported from
`tests/test_niah.py`, not reimplemented) and writes a third-party-reproducible
artifact set:

  results.jsonl   — one line per case (sorted, NO timestamps/paths): the retrieved
                    top-K ids + excerpts + hit/miss. Deterministic → byte-diffable.
  dataset.json    — the exact pinned inputs (needles, matrix, recall config) + a
                    dataset_sha256, so "which dataset/config" is verifiable, not implied.
  aggregate.json  — pass/total + per-size / per-depth breakdown.
  environment.json— python/OS/package/model versions + git commit (provenance).
  manifest.json   — sha256 of each deterministic artifact, so a third party runs the
                    same command and diffs the hashes.

The headline claim is NIAH 250/250 (5 sizes × 5 depths × 10 needles). Retrieval is
fully local (BM25 + vector + RRF, ONNX MiniLM) — no API key, no network.

Usage:
    python3 benchmarks/repro_niah.py [--limit N] [--out DIR]

Copyright (c) STARGA Inc. All rights reserved.
"""
from __future__ import annotations

import os

# Cap native thread pools BEFORE numpy / onnxruntime / tokenizers are imported (they
# read these at import time). The NIAH harness builds a fresh embedding index per case
# across 250 cases; uncapped, each ONNX session spawns a large thread pool and the pools
# accumulate faster than they are reclaimed, exhausting the system thread limit (an
# uncapped run was observed reaching ~76k threads, which blocked all builds on the box).
# Capping to 1 keeps total threads bounded and makes the run reproducible.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ONNXRUNTIME_INTRA_OP_NUM_THREADS",
    "ONNXRUNTIME_INTER_OP_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# NOTE: these caps are a PARTIAL mitigation only. The NIAH harness reloads a
# sentence-transformers (torch) model per case; torch's thread pools are not fully
# bound by the env vars above and accumulate across the 250 cases. Before running the
# full matrix, this harness must isolate each case (or batch) in a recycled subprocess
# (e.g. ProcessPoolExecutor(max_tasks_per_child=...)) so per-case threads/memory are
# reclaimed. Until that lands, run only small --limit subsets. (Tracked TODO.)

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_niah():
    """Load tests/test_niah.py by path so the repro uses the EXACT test code."""
    path = os.path.join(_REPO_ROOT, "tests", "test_niah.py")
    spec = importlib.util.spec_from_file_location("niah_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
    spec.loader.exec_module(mod)
    return mod


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canonical_json(obj) -> bytes:
    """Stable, sorted, newline-terminated JSON for hashing/diffing."""
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "absent"


def _dataset_descriptor(niah) -> dict:
    """The exact pinned inputs — everything that determines the result set."""
    desc = {
        "benchmark": "NIAH",
        "haystack_sizes": list(niah.HAYSTACK_SIZES),
        "depth_percentages": list(niah.DEPTH_PERCENTAGES),
        "top_k": niah.TOP_K,
        "needle_count": len(niah.NEEDLES),
        "needles": [
            {"idx": i, "needle": n["needle"], "query": n["query"], "expected_keywords": n["expected_keywords"]}
            for i, n in enumerate(niah.NEEDLES)
        ],
        "recall_config": niah._RECALL_CONFIG,
    }
    desc["dataset_sha256"] = _sha256_bytes(_canonical_json({k: v for k, v in desc.items() if k != "dataset_sha256"}))
    return desc


def _run_case(niah, size: int, depth: int, needle_idx: int) -> dict:
    needle = niah.NEEDLES[needle_idx]
    needle_id = f"NEEDLE-{needle_idx + 1:03d}"
    ws = niah._build_workspace(size, needle["needle"], needle_id, depth)
    try:
        niah._build_indexes(ws)
        results = niah._hybrid_search(ws, needle["query"], limit=niah.TOP_K)
        found = niah._check_needle_found(results, needle_id, needle["expected_keywords"])
        retrieved = [{"id": r.get("_id", "?"), "excerpt": (r.get("excerpt", "") or "")[:80]} for r in results]
    finally:
        shutil.rmtree(ws, ignore_errors=True)
        gc.collect()  # reclaim per-case index/session objects promptly
    return {
        "case_id": f"sz{size}_d{depth}_n{needle_idx}",
        "haystack_size": size,
        "depth_pct": depth,
        "needle_idx": needle_idx,
        "needle_id": needle_id,
        "query": needle["query"],
        "expected_keywords": needle["expected_keywords"],
        "retrieved": retrieved,
        "found": bool(found),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible NIAH benchmark")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N cases (smoke); 0 = all 250")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "benchmarks", "repro", "niah"))
    args = ap.parse_args()

    niah = _load_niah()
    params = list(niah._TEST_PARAMS)
    if args.limit > 0:
        params = params[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    dataset = _dataset_descriptor(niah)

    print(f"NIAH repro — {len(params)} cases (dataset_sha256={dataset['dataset_sha256'][:16]}…)")
    cases = []
    passed = 0
    for i, (size, depth, needle_idx) in enumerate(params):
        case = _run_case(niah, size, depth, needle_idx)
        cases.append(case)
        passed += int(case["found"])
        if (i + 1) % 25 == 0 or (i + 1) == len(params):
            print(f"  [{i + 1:3d}/{len(params)}] pass={passed} fail={i + 1 - passed}")

    cases.sort(key=lambda c: c["case_id"])  # deterministic order
    total = len(cases)

    # per-size / per-depth breakdown
    by_size: dict[int, list[int]] = {}
    by_depth: dict[int, list[int]] = {}
    for c in cases:
        by_size.setdefault(c["haystack_size"], [0, 0])
        by_depth.setdefault(c["depth_pct"], [0, 0])
        by_size[c["haystack_size"]][0] += int(c["found"])
        by_size[c["haystack_size"]][1] += 1
        by_depth[c["depth_pct"]][0] += int(c["found"])
        by_depth[c["depth_pct"]][1] += 1

    aggregate = {
        "benchmark": "NIAH",
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "by_haystack_size": {str(k): {"passed": v[0], "total": v[1]} for k, v in sorted(by_size.items())},
        "by_depth_pct": {str(k): {"passed": v[0], "total": v[1]} for k, v in sorted(by_depth.items())},
        "dataset_sha256": dataset["dataset_sha256"],
    }

    environment = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "mind_mem_version": _pkg_version("mind-mem"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {p: _pkg_version(p) for p in ("numpy", "onnxruntime", "sqlite-vec", "sentence-transformers", "tokenizers")},
        "embedding_model": niah._RECALL_CONFIG.get("model"),
    }

    # Write deterministic artifacts (results/dataset/aggregate) + non-hashed env.
    results_bytes = b"".join(_canonical_json(c) for c in cases)
    dataset_bytes = _canonical_json(dataset)
    aggregate_bytes = _canonical_json(aggregate)

    artifacts = {
        "results.jsonl": results_bytes,
        "dataset.json": dataset_bytes,
        "aggregate.json": aggregate_bytes,
    }
    for name, data in artifacts.items():
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(data)
    with open(os.path.join(args.out, "environment.json"), "wb") as f:
        f.write(_canonical_json(environment))

    manifest = {
        "benchmark": "NIAH",
        "note": "sha256 of the deterministic artifacts. Rerun this command and diff these hashes.",
        "artifacts": {name: _sha256_bytes(data) for name, data in artifacts.items()},
        "result": f"{passed}/{total}",
    }
    with open(os.path.join(args.out, "manifest.json"), "wb") as f:
        f.write(_canonical_json(manifest))

    print(f"\nRESULT: {passed}/{total}  (pass_rate={aggregate['pass_rate']})")
    print(f"artifacts → {args.out}")
    for name, h in manifest["artifacts"].items():
        print(f"  {name}: {h}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
