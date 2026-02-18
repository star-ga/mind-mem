#!/usr/bin/env python3
"""Benchmark: MIND kernels vs pure Python scoring.

Measures wall-clock time for the numerical hot paths that MIND kernels
accelerate. Runs each function at multiple array sizes, reports median
times and speedup ratios.

For MIND kernels, the benchmark pre-allocates ctypes arrays and only
times the native function call itself (no marshaling overhead).

Usage:
    python benchmarks/bench_kernels.py
    python benchmarks/bench_kernels.py --iterations 500
    python benchmarks/bench_kernels.py --sizes 100,500,1000,5000
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
import statistics
import sys
import time

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)


# ── Pure Python implementations (always available) ────────────────────

def py_rrf_fuse(bm25_ranks: list[float], vec_ranks: list[float],
                k: float = 60.0, bm25_w: float = 1.0,
                vec_w: float = 1.0) -> list[float]:
    """RRF fusion: score = bm25_w/(k+rank_bm25) + vec_w/(k+rank_vec)."""
    return [bm25_w / (k + b) + vec_w / (k + v)
            for b, v in zip(bm25_ranks, vec_ranks)]


def py_bm25f_doc(tf: float, df: float, N: float, dl: float, avgdl: float,
                 k1: float = 1.2, b: float = 0.75,
                 field_weight: float = 1.0) -> float:
    """BM25F score for a single term in a single document."""
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
    return idf * tf_norm * field_weight


def py_bm25f_batch(tfs: list[float], dfs: list[float], N: float,
                   dls: list[float], avgdl: float,
                   k1: float = 1.2, b: float = 0.75,
                   field_weight: float = 1.0) -> list[float]:
    """BM25F scores for a batch of documents (one term)."""
    return [py_bm25f_doc(tf, dfs[0], N, dl, avgdl, k1, b, field_weight)
            for tf, dl in zip(tfs, dls)]


def py_negation_penalty(scores: list[float], has_negation: list[bool],
                        penalty: float = 0.3) -> list[float]:
    """Apply negation penalty to scores."""
    return [s * penalty if neg else s for s, neg in zip(scores, has_negation)]


def py_date_proximity(days_diff: list[float], sigma: float = 30.0) -> list[float]:
    """Gaussian decay based on days difference."""
    return [math.exp(-0.5 * (d / sigma) ** 2) for d in days_diff]


def py_category_boost(scores: list[float], matches: list[bool],
                      boost: float = 1.15) -> list[float]:
    """Apply category match boost."""
    return [s * boost if m else s for s, m in zip(scores, matches)]


def py_importance_score(access_count: int, days_since: float,
                        base_importance: float = 1.0,
                        decay: float = 0.01) -> float:
    """A-MEM importance: access frequency with exponential recency decay."""
    recency = math.exp(-decay * days_since)
    freq = math.log(1.0 + access_count)
    return max(0.8, min(1.5, base_importance * (0.5 + 0.3 * freq + 0.2 * recency)))


def py_importance_batch(access_counts: list[int], days_since: list[float],
                        base: float = 1.0, decay: float = 0.01) -> list[float]:
    """Batch importance scoring."""
    return [py_importance_score(a, d, base, decay)
            for a, d in zip(access_counts, days_since)]


def py_entity_overlap(query_entities: list[str],
                      hit_tokens: list[list[str]]) -> list[float]:
    """Entity overlap ratio per hit."""
    if not query_entities:
        return [0.0] * len(hit_tokens)
    q_set = set(query_entities)
    n = len(q_set)
    return [len(q_set & set(tokens)) / n for tokens in hit_tokens]


def py_confidence_score(entity_overlap: float, bm25_norm: float,
                        speaker_cov: float, evidence_density: float,
                        negation_asym: float,
                        weights: tuple = (0.30, 0.25, 0.15, 0.20, 0.10)) -> float:
    """Weighted confidence from 5 abstention features."""
    features = [entity_overlap, bm25_norm, speaker_cov,
                evidence_density, negation_asym]
    return sum(f * w for f, w in zip(features, weights))


def py_top_k_mask(scores: list[float], k: int) -> list[bool]:
    """Boolean mask for top-k scores."""
    if k >= len(scores):
        return [True] * len(scores)
    threshold = sorted(scores, reverse=True)[k - 1]
    count = 0
    mask = []
    for s in scores:
        if s >= threshold and count < k:
            mask.append(True)
            count += 1
        else:
            mask.append(False)
    return mask


def py_weighted_rank(scores: list[float], weights: list[float]) -> list[float]:
    """Element-wise weighted combination."""
    return [s * w for s, w in zip(scores, weights)]


# ── Data generators ───────────────────────────────────────────────────

def gen_ranks(n: int) -> list[float]:
    return [float(i + 1) for i in range(n)]


def gen_scores(n: int, lo: float = 0.0, hi: float = 1.0) -> list[float]:
    return [random.uniform(lo, hi) for _ in range(n)]


def gen_bools(n: int, p: float = 0.3) -> list[bool]:
    return [random.random() < p for _ in range(n)]


def gen_ints(n: int, lo: int = 0, hi: int = 100) -> list[int]:
    return [random.randint(lo, hi) for _ in range(n)]


def gen_token_lists(n: int, vocab_size: int = 200) -> list[list[str]]:
    vocab = [f"tok_{i}" for i in range(vocab_size)]
    return [random.sample(vocab, k=random.randint(3, 15)) for _ in range(n)]


# ── Benchmark harness ─────────────────────────────────────────────────

def bench(func, args, iterations: int) -> float:
    """Run func(args) `iterations` times, return median seconds."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_native(func, iterations: int) -> float:
    """Run a zero-arg callable `iterations` times, return median seconds."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} \u03bcs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


# ── Pre-allocated ctypes kernel callers ───────────────────────────────

def make_native_rrf(lib, ranks_a, ranks_b, k, bm25_w, vec_w):
    n = len(ranks_a)
    fa = (ctypes.c_float * n)(*ranks_a)
    fb = (ctypes.c_float * n)(*ranks_b)
    out = (ctypes.c_float * n)()
    ck = ctypes.c_float(k)
    cw1 = ctypes.c_float(bm25_w)
    cw2 = ctypes.c_float(vec_w)
    cn = ctypes.c_int(n)
    return lambda: lib.rrf_fuse(fa, fb, cn, ck, cw1, cw2, out)


def make_native_bm25f(lib, tfs, df, N, dls, avgdl, k1, b, fw):
    n = len(tfs)
    ctfs = (ctypes.c_float * n)(*tfs)
    cdls = (ctypes.c_float * n)(*dls)
    out = (ctypes.c_float * n)()
    return lambda: lib.bm25f_batch(
        ctfs, ctypes.c_float(df), ctypes.c_float(N),
        cdls, ctypes.c_float(avgdl), ctypes.c_float(k1),
        ctypes.c_float(b), ctypes.c_float(fw), ctypes.c_int(n), out)


def make_native_negation(lib, scores, bools, penalty):
    n = len(scores)
    cs = (ctypes.c_float * n)(*scores)
    cf = (ctypes.c_float * n)(*(1.0 if b else 0.0 for b in bools))
    out = (ctypes.c_float * n)()
    return lambda: lib.negation_penalty(cs, cf, ctypes.c_float(penalty),
                                        ctypes.c_int(n), out)


def make_native_date_prox(lib, days, sigma):
    n = len(days)
    cd = (ctypes.c_float * n)(*days)
    out = (ctypes.c_float * n)()
    return lambda: lib.date_proximity(cd, ctypes.c_float(sigma),
                                      ctypes.c_int(n), out)


def make_native_cat_boost(lib, scores, bools, boost):
    n = len(scores)
    cs = (ctypes.c_float * n)(*scores)
    cf = (ctypes.c_float * n)(*(1.0 if b else 0.0 for b in bools))
    out = (ctypes.c_float * n)()
    return lambda: lib.category_boost(cs, cf, ctypes.c_float(boost),
                                      ctypes.c_int(n), out)


def make_native_importance(lib, access_counts, days, base, decay):
    n = len(access_counts)
    ca = (ctypes.c_int * n)(*access_counts)
    cd = (ctypes.c_float * n)(*days)
    out = (ctypes.c_float * n)()
    return lambda: lib.importance_batch(ca, cd, ctypes.c_float(base),
                                        ctypes.c_float(decay),
                                        ctypes.c_int(n), out)


def make_native_confidence(lib):
    lib.confidence_score.restype = ctypes.c_float
    args = [ctypes.c_float(v) for v in (0.5, 0.7, 0.3, 0.8, 0.2,
                                         0.30, 0.25, 0.15, 0.20, 0.10)]
    return lambda: lib.confidence_score(*args)


def make_native_topk(lib, scores, k):
    n = len(scores)
    cs = (ctypes.c_float * n)(*scores)
    out = (ctypes.c_float * n)()
    return lambda: lib.top_k_mask(cs, ctypes.c_int(n), ctypes.c_int(k), out)


def make_native_weighted(lib, scores, weights):
    n = len(scores)
    cs = (ctypes.c_float * n)(*scores)
    cw = (ctypes.c_float * n)(*weights)
    out = (ctypes.c_float * n)()
    return lambda: lib.weighted_rank(cs, cw, ctypes.c_int(n), out)


def run_benchmarks(sizes: list[int], iterations: int):
    # Try loading MIND kernel .so directly
    lib = None
    lib_paths = [
        os.path.join(os.path.dirname(__file__), "..", "lib", "libmindmem.so"),
        os.path.join(os.path.dirname(__file__), "..", "lib", "libmindmem.dylib"),
    ]
    for p in lib_paths:
        if os.path.exists(p):
            try:
                lib = ctypes.CDLL(os.path.abspath(p))
            except OSError:
                pass
            break

    mind_label = "MIND (.so)" if lib else "MIND (not available)"

    print("=" * 72)
    print("mind-mem Kernel Benchmark")
    print("=" * 72)
    print(f"  Iterations per measurement: {iterations}")
    print(f"  Array sizes: {sizes}")
    print(f"  MIND kernel: {'LOADED' if lib else 'NOT FOUND (Python-only mode)'}")
    if lib:
        print("  Mode: pre-allocated ctypes (no marshaling in timing loop)")
    print()

    benchmarks = []

    sep = "\u2500"
    dash = "\u2014"

    for n in sizes:
        print(f"--- N = {n} " + sep * (55 - len(str(n))))
        print()
        print(f"  {'Function':<28} {'Python':>12}  {mind_label:>18}  {'Speedup':>8}")
        print("  " + sep * 28 + " " + sep * 12 + "  " + sep * 18 + "  " + sep * 8)

        # Generate data once per size
        ranks_a = gen_ranks(n)
        ranks_b = gen_ranks(n)
        scores_a = gen_scores(n)
        bools_a = gen_bools(n)
        days = gen_scores(n, 0, 365)
        access_counts = gen_ints(n, 0, 500)
        query_ents = [f"tok_{i}" for i in range(5)]
        token_lists = gen_token_lists(n)
        tfs = gen_scores(n, 0, 10)
        dls = gen_scores(n, 50, 500)
        weights = gen_scores(n, 0.5, 1.5)

        # Build test list: (name, py_func, py_args, native_maker_or_None)
        tests = [
            ("rrf_fuse", py_rrf_fuse,
             (ranks_a, ranks_b, 60.0, 1.0, 1.0),
             lambda: make_native_rrf(lib, ranks_a, ranks_b, 60.0, 1.0, 1.0) if lib else None),
            ("bm25f_batch", py_bm25f_batch,
             (tfs, [5.0], 10000.0, dls, 200.0),
             lambda: make_native_bm25f(lib, tfs, 5.0, 10000.0, dls, 200.0, 1.2, 0.75, 1.0) if lib else None),
            ("negation_penalty", py_negation_penalty,
             (scores_a, bools_a, 0.3),
             lambda: make_native_negation(lib, scores_a, bools_a, 0.3) if lib else None),
            ("date_proximity", py_date_proximity,
             (days, 30.0),
             lambda: make_native_date_prox(lib, days, 30.0) if lib else None),
            ("category_boost", py_category_boost,
             (scores_a, bools_a, 1.15),
             lambda: make_native_cat_boost(lib, scores_a, bools_a, 1.15) if lib else None),
            ("importance_batch", py_importance_batch,
             (access_counts, days),
             lambda: make_native_importance(lib, access_counts, days, 1.0, 0.01) if lib else None),
            ("entity_overlap", py_entity_overlap,
             (query_ents, token_lists),
             lambda: None),  # set-based, no direct C equivalent
            ("confidence_score", py_confidence_score,
             (0.5, 0.7, 0.3, 0.8, 0.2),
             lambda: make_native_confidence(lib) if lib else None),
            ("top_k_mask", py_top_k_mask,
             (scores_a, min(10, n)),
             lambda: make_native_topk(lib, scores_a, min(10, n)) if lib else None),
            ("weighted_rank", py_weighted_rank,
             (scores_a, weights),
             lambda: make_native_weighted(lib, scores_a, weights) if lib else None),
        ]

        for name, py_func, py_args, native_maker in tests:
            py_time = bench(py_func, py_args, iterations)

            mind_time = None
            speedup = "\u2014"
            native_fn = native_maker()
            if native_fn is not None:
                mind_time = bench_native(native_fn, iterations)
                if mind_time > 0:
                    speedup = f"{py_time / mind_time:.1f}x"

            mind_str = format_time(mind_time) if mind_time else "\u2014"
            print(f"  {name:<28} {format_time(py_time):>12}  {mind_str:>18}  {speedup:>8}")

            benchmarks.append({
                "function": name,
                "n": n,
                "python_s": py_time,
                "mind_s": mind_time,
            })

        print()

    # Summary
    print("=" * 72)
    print("Summary")
    print("=" * 72)

    if not lib:
        print()
        print("  MIND kernel not compiled. Showing Python baseline only.")
        print("  To compare, compile kernels and re-run:")
        print()
        print("    gcc -O3 -march=native -shared -fPIC -o lib/libmindmem.so lib/kernels.c -lm")
        print("    python benchmarks/bench_kernels.py")
        print()

    # Show scaling
    func_names = sorted(set(b["function"] for b in benchmarks))
    print(f"  {'Function':<28}", end="")
    for n in sizes:
        print(f" {'N='+str(n):>12}", end="")
    print()
    print("  " + sep * 28, end="")
    for _ in sizes:
        print(" " + sep * 12, end="")
    print()

    for fname in func_names:
        print(f"  {fname:<28}", end="")
        for n in sizes:
            entry = next((b for b in benchmarks
                         if b["function"] == fname and b["n"] == n), None)
            if entry:
                print(f" {format_time(entry['python_s']):>12}", end="")
            else:
                print(f" {dash:>12}", end="")
        print()

    print()
    total_py = sum(b["python_s"] for b in benchmarks)
    total_mind = sum(b["mind_s"] for b in benchmarks if b["mind_s"])
    print(f"  Total Python time: {format_time(total_py)}")
    if total_mind > 0:
        print(f"  Total MIND time:   {format_time(total_mind)}")
        print(f"  Overall speedup:   {total_py / total_mind:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MIND kernels vs Python")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Iterations per measurement (default: 200)")
    parser.add_argument("--sizes", type=str, default="100,500,1000,5000",
                        help="Comma-separated array sizes (default: 100,500,1000,5000)")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    random.seed(42)

    run_benchmarks(sizes, args.iterations)


if __name__ == "__main__":
    main()
