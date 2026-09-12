"""Held-out paraphrase eval set — runs AFTER training, BEFORE ship.

Postmortem of v3.12.0 retrain (18/95 fails, patched to 95/95 by softening
2 probes for v3.12.1): the model memorised corpus phrasings and failed to
generalise. Required tokens were learned but only when the question
matched a corpus probe verbatim.

This harness ships **paraphrases of every high-stakes eval probe** that
DO NOT appear in `train/build_corpus.py` (verified by exact-string scan
at runtime). It catches the memorisation-vs-learning gap that
`eval_harness.py` cannot — by definition, every eval_harness probe has at
least one verbatim corpus tuple now (see `_V4_EVAL_EXACT_PROBES`).

Run AFTER training, BEFORE ship. Target: ≥ 90% pass rate. If the model
gets 100% on `eval_harness.py` and < 90% here, it memorised instead of
learned — corpus needs more paraphrase diversity, not more density.

Usage:
    MM_FULLFT_DIR=/data/checkpoints/mm-workspace/full-ft \\
        python3 train/eval_holdout.py

Returns 0 on pass (≥ 90% per group + ≥ 90% overall), 1 otherwise.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Reuse the model loader + chat helpers from eval_harness.
sys.path.insert(0, str(Path(__file__).parent))
from eval_harness import _load_model, select_model  # noqa: E402

REPORT = Path(
    os.environ.get(
        "MM_HOLDOUT_REPORT",
        str(
            Path(os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output")).parent
            / "full-ft"
            / "eval_holdout_report.json"
        ),
    )
)
CORPUS = Path(
    os.environ.get(
        "MM_CORPUS",
        str(Path(os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output")) / "corpus.jsonl"),
    )
)


# ---------------------------------------------------------------------------
# Paraphrase set — every Q must NOT appear in the corpus verbatim.
# Each Q tests the SAME fact as a probe in eval_harness.py but with a
# different surface form, so a memorised model fails here while a
# learned model passes.
# ---------------------------------------------------------------------------

#: Paraphrases of v4_surfaces eval probes.
V4_HOLDOUT: list[tuple[str, list[str]]] = [
    # circuit_breaker
    (
        "Which three values can the v4 circuit breaker be in at any moment?",
        ["CLOSED", "OPEN", "HALF_OPEN"],
    ),
    (
        "If I instantiate CircuitBreaker() with no arguments, how many failures will it tolerate before tripping?",
        ["5"],
    ),
    # backpressure
    (
        "Why does v4 BackpressureController need two watermarks instead of one threshold?",
        ["hysteresis"],
    ),
    # health
    (
        "List every possible value the `status` field of v4 health_check's return dict can take.",
        ["ok", "degraded", "fail"],
    ),
    # logging_context
    (
        "Why does v4 logging_context use contextvars rather than threading.local?",
        ["contextvars"],
    ),
    # block_metadata
    (
        "When I call set_block_metadata twice for the same block_id, which timestamp changes and which stays constant?",
        ["created_at", "updated_at"],
    ),
    (
        "How does a caller plug a per-kind validator into v4 block_metadata before validate_block is called?",
        ["register_schema_validator"],
    ),
    # observability
    (
        "Past how many distinct counter names does v4 observability start returning the overflow sentinel?",
        ["10000"],
    ),
    # eviction
    (
        "How does an operator change the workspace eviction policy at runtime without restarting?",
        ["set_active_policy"],
    ),
    (
        "What does the dict returned by EvictionPlan.debug_plan() look like?",
        ["policy", "block_ids"],
    ),
    # surprise_retrieval
    (
        "Enumerate every legal value of the v4 FallbackPolicy enum.",
        ["NEUTRAL", "PROMOTE", "DEMOTE", "RAISE"],
    ),
    (
        "What error class signals an unusable embedding under FallbackPolicy.RAISE?",
        ["EmbeddingFailureError"],
    ),
    # public predicates
    (
        "Which public function lets the health probe ask whether an eviction policy is registered without touching private state?",
        ["is_policy_registered"],
    ),
    (
        "Which public function lets the health probe ask whether a cognitive kernel is bound without touching private state?",
        ["is_kernel_registered"],
    ),
]

#: Paraphrases of the high-stakes v3.12 probes that v3.12.0 missed.
V312_HOLDOUT: list[tuple[str, list[str]]] = [
    # qg.escape_hatch
    (
        "If strict mode is on and I MUST write a block that fails validation, what's my override?",
        ["force", "strict"],
    ),
    (
        "Override the strict-mode quality gate for one specific block — how?",
        ["force", "strict"],
    ),
    # lin.cites
    (
        "Numeric value: KIND_DECAY for `cites` edges?",
        ["cites", "0.8"],
    ),
    (
        "How much staleness signal does a `cites` seed propagate?",
        ["cites", "0.8"],
    ),
    # validate_block default mode
    (
        "Pre-flight a block proposal before propose_update writes it.",
        ["validate_block", "advisory"],
    ),
    # _explain.final_score formula
    (
        "Walk me through how `final_score` is computed inside `_explain`.",
        ["rrf_rank", "tier_boost"],
    ),
    # quality_gate.mode default
    (
        "If `mind-mem.json` doesn't set `quality_gate.mode`, what mode runs?",
        ["advisory"],
    ),
    # propagate_lineage_staleness
    (
        "Which file ships `propagate_lineage_staleness` in v3.12.0, "
        "and which table does it write penalty scores into?",
        ["lineage_staleness", "block_staleness"],  # file substring + table
    ),
]


def _verify_no_verbatim_in_corpus() -> None:
    """Fail fast if any holdout Q already appears in the corpus.

    The corpus is the training data — if a holdout Q appears there
    verbatim, the test isn't held-out anymore and we lose the
    memorisation-vs-learning signal.
    """
    if not CORPUS.is_file():
        print(f"FAIL: corpus file not found at {CORPUS} — held-out gate cannot run")
        raise SystemExit(2)
    holdout_qs: set[str] = {q for q, _ in V4_HOLDOUT + V312_HOLDOUT}
    seen: set[str] = set()
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in m.get("messages", []):
                if msg.get("role") == "user":
                    if msg["content"] in holdout_qs:
                        seen.add(msg["content"])
    if seen:
        print("FAIL: holdout Qs appear verbatim in corpus — not held-out:")
        for q in seen:
            print(f"  - {q}")
        sys.exit(2)


def _bench(tokenizer, model, probes: list[tuple[str, list[str]]]) -> dict:
    from eval_harness import _bench_probes

    return _bench_probes(tokenizer, model, "holdout", probes)


def main() -> None:
    from eval_receipt import capture_inputs, eval_source_paths, new_run_id, now_iso

    _verify_no_verbatim_in_corpus()
    run_id, started_at = new_run_id(), now_iso()
    repo_root = Path(__file__).resolve().parents[1]
    probe_sets = {"v4_holdout": V4_HOLDOUT, "v312_holdout": V312_HOLDOUT}
    selection = select_model()
    # BEFORE anything is used: weights, tokenizer, base, corpus, evaluator
    # sources and the probe definitions themselves.
    captured = capture_inputs(
        selection,
        repo_root=repo_root,
        dataset_root=CORPUS,
        source_paths=eval_source_paths(repo_root),
        probe_sets=probe_sets,
    )
    tokenizer, model, selection = _load_model(selection)
    v4 = _bench(tokenizer, model, V4_HOLDOUT)
    v312 = _bench(tokenizer, model, V312_HOLDOUT)

    total_hits = v4["hits"] + v312["hits"]
    total = v4["total"] + v312["total"]
    overall = total_hits / total

    from eval_receipt import build_receipt, finalize_report

    probe_counts = {
        "v4_holdout": (len(V4_HOLDOUT), v4["total"]),
        "v312_holdout": (len(V312_HOLDOUT), v312["total"]),
    }
    incomplete = [n for n, (want, done) in probe_counts.items() if want != done]
    receipt = build_receipt(
        repo_root=repo_root,
        suite="holdout",
        captured=captured,
        probe_counts=probe_counts,
        probe_sets=probe_sets,
        command="python3 train/eval_holdout.py",
        run_id=run_id,
        started_at=started_at,
        ended_at=now_iso(),
        status="incomplete" if incomplete else "completed",
    )
    report = {
        "v4_holdout": v4,
        "v312_holdout": v312,
        "overall_accuracy": overall,
        "total_hits": total_hits,
        "total_probes": total,
        "targets": {"per_group": 0.90, "overall": 0.90},
        "receipt": receipt,
    }
    report = finalize_report(report, receipt)
    receipt = report["receipt"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("mind-mem-4b v4 HELD-OUT paraphrase eval")
    print("=" * 60)
    for name, bench, target in (
        ("v4_holdout                    ", v4, 0.90),
        ("v312_holdout                  ", v312, 0.90),
    ):
        pass_str = "PASS" if bench["accuracy"] >= target else "FAIL"
        print(
            f"  {name}  {bench['hits']:3d}/{bench['total']:<3d}  "
            f"{bench['accuracy']:.2%}   (target {target:.0%})  [{pass_str}]"
        )
    print(f"  overall                          {total_hits:3d}/{total:<3d}  {overall:.2%}")
    print(f"\nreport → {REPORT}")

    if not receipt["complete"]:
        # The report is still written — a partial run is useful for diagnosis —
        # but it must never exit green.
        print("FAIL: evaluation receipt is incomplete; refusing a green gate")
        sys.exit(2)

    passed = (
        v4["accuracy"] >= 0.90
        and v312["accuracy"] >= 0.90
        and overall >= 0.90
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
