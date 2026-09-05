#!/usr/bin/env python3
"""Reproducible NIAH benchmark harness -- emits an independently-verifiable package.

This runs the SAME Needle-In-A-Haystack code the test suite uses (imported from
``tests/test_niah.py``, not reimplemented) and writes the repro package layout
described in :mod:`benchmarks.repro_manifest`: raw per-case NDJSON, metrics
recomputed from those rows, the pinned dataset, the environment, and a manifest
carrying the commit, the effective config and its sha256, the seeds, the
adapter, the embedder (name/dims/device), k, the exclusion rule, and the sha256
of every artifact.

Nothing here decides a number. ``benchmarks/repro_metrics.niah_metrics`` does,
over the raw rows, and ``benchmarks/repro_verify.py`` re-runs it over the
*committed* rows and fails loudly if the published figure does not follow. That
round trip is what makes the result independently reproducible rather than
merely well documented.

Each case runs in its own hard-killed child process. That is not belt-and-braces:
the harness builds a fresh embedding index per case, and in-process the model's
native thread pools accumulate across cases faster than they are reclaimed (an
uncapped run was observed reaching ~76k threads and blocking every build on the
box). A child that exits returns its threads to the system, and a case that
hangs is killed and counted rather than silently ending the run.

Usage:
    python3 benchmarks/repro_niah.py [--limit N] [--out DIR] [--headline-claim]

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import os

# Cap native thread pools BEFORE numpy / onnxruntime / tokenizers are imported
# (they read these at import time), and inherit the caps into every child.
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

import argparse
import importlib.util
import shutil
import sys
import time
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.hard_timeout import OK, run_with_hard_timeout  # noqa: E402
from benchmarks.repro_manifest import (  # noqa: E402
    build_manifest,
    content_hash,
    environment_facts,
    sha256_bytes,
    write_package,
)
from benchmarks.repro_metrics import niah_metrics  # noqa: E402

#: Per-case wall-clock ceiling. A 500-block haystack indexes in seconds; a case
#: that needs more than this is the pathological one the isolation exists for.
DEFAULT_CASE_TIMEOUT_S = 300.0

_ADAPTER = {
    "name": "niah_hybrid_recall",
    "version": "1",
    "source": "tests/test_niah.py",
    "note": "the benchmark imports the test module by path, so the artifact and the test suite cannot drift apart",
}


def _load_niah() -> Any:
    """Load tests/test_niah.py by path so the repro uses the EXACT test code."""
    path = os.path.join(_REPO_ROOT, "tests", "test_niah.py")
    spec = importlib.util.spec_from_file_location("niah_under_test", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generator_sha256() -> str:
    with open(os.path.join(_REPO_ROOT, "tests", "test_niah.py"), "rb") as handle:
        return sha256_bytes(handle.read())


def dataset_descriptor(niah: Any) -> dict[str, Any]:
    """The exact pinned inputs -- content-hashed, not named.

    The needles and the matrix are the data; the generator's own source hash is
    included because the filler haystack is *produced* by that file, so a change
    to it is a change to the dataset even though no needle moved.
    """
    desc: dict[str, Any] = {
        "name": "NIAH-synthetic",
        "version": "1",
        "generated_by": "tests/test_niah.py",
        "generator_sha256": _generator_sha256(),
        "haystack_sizes": list(niah.HAYSTACK_SIZES),
        "depth_percentages": list(niah.DEPTH_PERCENTAGES),
        "top_k": niah.TOP_K,
        "needle_count": len(niah.NEEDLES),
        "matrix_cells": len(niah._TEST_PARAMS),
        "needles": [
            {"idx": i, "needle": n["needle"], "query": n["query"], "expected_keywords": n["expected_keywords"]}
            for i, n in enumerate(niah.NEEDLES)
        ],
        "recall_config": niah._RECALL_CONFIG,
    }
    desc["content_sha256"] = content_hash({k: v for k, v in desc.items() if k != "content_sha256"})
    return desc


def run_case(size: int, depth: int, needle_idx: int) -> dict[str, Any]:
    """Retrieve for one NIAH cell. Runs in the CHILD process.

    Returns a plain dict so the result crosses the process boundary without
    dragging index or model state with it. Module-level, because ``spawn``
    imports the target by name.
    """
    niah = _load_niah()
    needle = niah.NEEDLES[needle_idx]
    needle_id = f"NEEDLE-{needle_idx + 1:03d}"
    ws = niah._build_workspace(size, needle["needle"], needle_id, depth)
    try:
        from mind_mem.recall_vector import VectorBackend
        from mind_mem.sqlite_index import build_index

        build_index(ws, incremental=False)
        vector = VectorBackend(niah._RECALL_CONFIG)
        vector.index(ws)

        from mind_mem.hybrid_recall import HybridBackend

        backend = HybridBackend(niah._RECALL_CONFIG)
        started = time.monotonic()
        results = backend.search(needle["query"], ws, limit=niah.TOP_K, active_only=False, rerank=False)
        latency_ms = (time.monotonic() - started) * 1000.0
        found = niah._check_needle_found(results, needle_id, needle["expected_keywords"])
        # Full excerpts, not truncated: the verifier re-applies the hit rule to
        # this evidence, and a rule applied to a truncated excerpt is a
        # different rule.
        retrieved = [
            {"rank": i, "id": str(r.get("_id", "?")), "excerpt": str(r.get("excerpt", "") or "")} for i, r in enumerate(results, 1)
        ]
        # Measured, not labelled: ``dimension`` is set only when the indexer
        # actually produced embeddings, so this distinguishes a real dense leg
        # from a hybrid backend that silently fell back to lexical-only.
        vector_ran = vector.dimension is not None
        embedder = {
            "name": str(niah._RECALL_CONFIG.get("model", "unknown")) if vector_ran else "none",
            "dimensions": vector.dimension,
            "device": str(niah._RECALL_CONFIG.get("vector_device", "cpu")),
            "provider": str(niah._RECALL_CONFIG.get("provider", "unknown")),
            "onnx_backend": bool(niah._RECALL_CONFIG.get("onnx_backend", True)),
            "vector_leg_exercised": vector_ran,
        }
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    return {
        "retrieved": retrieved,
        "found": bool(found),
        "latency_ms": latency_ms,
        "embedder": embedder,
        "declared_backend": str(niah._RECALL_CONFIG.get("backend", "unknown")),
        "effective_backend": "hybrid" if vector_ran else "bm25_only",
    }


def _row(niah: Any, size: int, depth: int, needle_idx: int, outcome: Any) -> dict[str, Any]:
    """One raw NDJSON row: enough to recompute the verdict, never just the verdict."""
    needle = niah.NEEDLES[needle_idx]
    value = outcome.value if (outcome.status == OK and isinstance(outcome.value, dict)) else {}
    return {
        "unit_id": f"sz{size}_d{depth}_n{needle_idx}",
        "haystack_size": size,
        "depth_pct": depth,
        "needle_idx": needle_idx,
        "needle_id": f"NEEDLE-{needle_idx + 1:03d}",
        "query": needle["query"],
        "expected_keywords": list(needle["expected_keywords"]),
        # A killed case retrieves nothing and is therefore a MISS -- the
        # conservative direction for a claim about our own retrieval -- and it
        # is labelled with its status, never dropped.
        "retrieved": value.get("retrieved", []),
        "found": bool(value.get("found", False)),
        "latency_ms": round(float(value.get("latency_ms", outcome.elapsed_s * 1000.0)), 3),
        "unit_status": outcome.status,
        "unit_elapsed_s": round(outcome.elapsed_s, 3),
        "unit_error": outcome.error[:400],
        "pipeline": {
            "declared_backend": value.get("declared_backend", "unknown"),
            "effective_backend": value.get("effective_backend", "unknown"),
            "embedder": value.get("embedder", {}),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproducible NIAH benchmark package")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N cells (smoke); 0 = the full matrix")
    ap.add_argument(
        "--every",
        type=int,
        default=1,
        help="Take every Nth cell of the matrix before --limit, so a smoke subset spans every haystack size instead of only the smallest",
    )
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "benchmarks", "repro", "niah"))
    ap.add_argument("--case-timeout", type=float, default=DEFAULT_CASE_TIMEOUT_S)
    ap.add_argument(
        "--headline-claim",
        action="store_true",
        help="mark the package as a citable headline figure. Only legitimate for the FULL matrix; a subset must never carry it.",
    )
    a = ap.parse_args(argv)

    niah = _load_niah()
    all_params = list(niah._TEST_PARAMS)
    params = all_params[:: max(1, a.every)]
    full_matrix = max(1, a.every) == 1 and (not a.limit or a.limit >= len(params))
    if a.limit:
        params = params[: a.limit]
    if a.headline_claim and not full_matrix:
        print("refusing --headline-claim on a subset: a partial matrix is not the published figure", file=sys.stderr)
        return 2

    dataset = dataset_descriptor(niah)
    print(f"NIAH repro -- {len(params)} cases (dataset {dataset['content_sha256'][:16]}...)")

    started = time.time()
    rows: list[dict[str, Any]] = []
    for i, (size, depth, needle_idx) in enumerate(params, 1):
        outcome = run_with_hard_timeout(run_case, a.case_timeout, size, depth, needle_idx)
        if i == 1 and not outcome.ok:
            # A parent-side import probe cannot catch this: what disqualifies an
            # interpreter (sqlite_vec, say) is imported lazily deep inside the
            # child. Running one real case is the only check that exercises the
            # same path the other 249 would. Refuse here rather than spend an
            # hour producing a package whose every cell is a crash -- complete
            # to look at, and measuring nothing.
            print(
                f"\nrefusing to continue: the first case crashed, so all {len(params)} would.\n"
                f"  {outcome.error or outcome.status}\n"
                f"  interpreter: {sys.executable} (Python "
                f"{sys.version_info[0]}.{sys.version_info[1]})\n"
                "  No package written. Install the benchmark extra for THIS interpreter: "
                "pip install -e '.[benchmark]'",
                file=sys.stderr,
            )
            return 2
        rows.append(_row(niah, size, depth, needle_idx, outcome))
        if i % 25 == 0 or i == len(params):
            hits = sum(1 for r in rows if r["found"])
            print(f"  [{i:3d}/{len(params)}] pass={hits} fail={i - hits} ({time.time() - started:.0f}s)", flush=True)
    wall_clock_s = round(time.time() - started, 2)

    rows.sort(key=lambda r: str(r["unit_id"]))
    metrics = niah_metrics(rows)
    embedders = sorted({str(r["pipeline"].get("embedder", {}).get("name", "unknown")) for r in rows})
    dims = sorted({r["pipeline"].get("embedder", {}).get("dimensions") for r in rows}, key=lambda d: (d is None, d))
    devices = sorted({str(r["pipeline"].get("embedder", {}).get("device", "unknown")) for r in rows})
    killed = int(metrics["integrity"]["killed_or_crashed"])

    sampling = (
        f"FULL matrix: all {len(params)} cells ({len(niah.HAYSTACK_SIZES)} sizes x "
        f"{len(niah.DEPTH_PERCENTAGES)} depths x {len(niah.NEEDLES)} needles)"
        if full_matrix
        else (
            f"SMOKE SUBSET: {len(params)} of {len(all_params)} cells "
            f"(stride {max(1, a.every)}, limit {a.limit or 'none'}) -- NOT the published 250/250 figure"
        )
    )
    manifest = build_manifest(
        benchmark="NIAH",
        headline_claim=bool(a.headline_claim),
        sampling=sampling,
        dataset=dataset,
        config=dict(niah._RECALL_CONFIG),
        seeds={
            "rng": (
                "none -- filler haystack content is derived deterministically from the block "
                "index via sha256 (tests/test_niah.py::_deterministic_hash)"
            ),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
            "case_order": "the matrix order in tests/test_niah.py::_TEST_PARAMS, then sorted by unit_id before writing",
        },
        adapter=dict(_ADAPTER),
        embedder={
            "name": embedders[0] if len(embedders) == 1 else embedders,
            "dimensions": dims[0] if len(dims) == 1 else dims,
            "device": devices[0] if len(devices) == 1 else devices,
            "basis": "read from the VectorBackend that indexed each case, not from a command-line label",
        },
        k=int(niah.TOP_K),
        exclusions={
            "rule": "none -- every cell attempted is scored; a killed or crashed case is scored as a MISS and counted, never dropped",
            "excluded": 0,
            "eligible": len(params),
            "matrix_cells": len(niah._TEST_PARAMS),
        },
        run={
            "units_total": len(rows),
            "units_ok": len(rows) - killed,
            "units_killed_or_crashed": killed,
            "case_timeout_s": a.case_timeout,
            "isolation": "one hard-killed child process per case (SIGKILL on the process group)",
            "wall_clock_s": wall_clock_s,
        },
        headline=metrics["headline"],
        commands={
            "regenerate": f"python3 benchmarks/repro_niah.py --out {os.path.relpath(a.out, _REPO_ROOT)}"
            + ("" if full_matrix else f" --every {max(1, a.every)}" + (f" --limit {a.limit}" if a.limit else "")),
            "recompute_metrics_from_raw_rows": f"python3 benchmarks/repro_verify.py package {os.path.relpath(a.out, _REPO_ROOT)}",
            "verify_everything": "make repro-verify",
        },
        notes="Retrieval is fully local (BM25 + vector + RRF): no API key, no network, no judge model.",
    )
    write_package(
        a.out,
        manifest=manifest,
        raw_rows=rows,
        metrics=metrics,
        dataset=dataset,
        environment=environment_facts(("numpy", "onnxruntime", "sqlite-vec", "sentence-transformers", "tokenizers")),
    )

    print(f"\nRESULT: {metrics['headline']['as_text']}  (killed/crashed {killed})")
    if rows and killed == len(rows):
        print(
            "every unit crashed, so this package measures nothing -- the verifier will "
            "reject it as VACUOUS. Check that the interpreter running this has the "
            "benchmark extra installed: pip install -e '.[benchmark]'",
            file=sys.stderr,
        )
    print(f"package -> {a.out}")
    for name, meta in manifest["artifacts"].items():
        print(f"  {name}: {meta['sha256']}")
    print(f"\nverify: python3 benchmarks/repro_verify.py package {os.path.relpath(a.out, _REPO_ROOT)}")
    return 0 if killed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
