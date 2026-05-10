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
from eval_harness import _chat, _load_model  # noqa: E402

REPORT = Path(
    os.environ.get(
        "MM_HOLDOUT_REPORT",
        "/data/checkpoints/mm-workspace/full-ft/eval_holdout_report.json",
    )
)
CORPUS = Path(
    os.environ.get(
        "MM_CORPUS",
        "/data/checkpoints/mm-workspace/train-output/corpus.jsonl",
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
        "Which file ships `propagate_lineage_staleness` in v3.12.0?",
        ["propagate_lineage_staleness", "block_staleness"],
    ),
]


def _verify_no_verbatim_in_corpus() -> None:
    """Fail fast if any holdout Q already appears in the corpus.

    The corpus is the training data — if a holdout Q appears there
    verbatim, the test isn't held-out anymore and we lose the
    memorisation-vs-learning signal.
    """
    if not CORPUS.is_file():
        print(f"WARN: corpus file not found at {CORPUS} — skipping verbatim check")
        return
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
    hits = 0
    misses: list[dict] = []
    for prompt, required in probes:
        resp = _chat(tokenizer, model, prompt)
        if all(tok in resp for tok in required):
            hits += 1
        else:
            missing = [t for t in required if t not in resp]
            misses.append(
                {"prompt": prompt, "missing": missing, "response": resp[:200]}
            )
    total = len(probes)
    return {"accuracy": hits / total, "hits": hits, "total": total, "misses": misses}


def main() -> None:
    _verify_no_verbatim_in_corpus()
    tokenizer, model = _load_model()
    v4 = _bench(tokenizer, model, V4_HOLDOUT)
    v312 = _bench(tokenizer, model, V312_HOLDOUT)

    total_hits = v4["hits"] + v312["hits"]
    total = v4["total"] + v312["total"]
    overall = total_hits / total

    report = {
        "v4_holdout": v4,
        "v312_holdout": v312,
        "overall_accuracy": overall,
        "total_hits": total_hits,
        "total_probes": total,
        "targets": {"per_group": 0.90, "overall": 0.90},
    }
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

    passed = (
        v4["accuracy"] >= 0.90
        and v312["accuracy"] >= 0.90
        and overall >= 0.90
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
